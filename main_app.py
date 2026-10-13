import io
import os
import queue
import logging
import threading
import warnings
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from scipy.signal import resample_poly
from transformers import BertTokenizer, BertModel

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = exc

from diarization_engine import (
    DiarizationEngine,
    StreamingVADSegmenter,
    DiarizedSegment,
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

TARGET = ["neutral", "joy", "anger", "sadness", "surprise"]
EMOTION_EMOJIS = {
    "neutral": "😐", "joy": "😊", "anger": "😠",
    "sadness": "😢", "surprise": "😲",
}
EMOTION_COLORS = {
    "neutral": "#6c757d", "joy": "#28a745", "anger": "#dc3545",
    "sadness": "#17a2b8", "surprise": "#fd7e14",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000


class Model(nn.Module):
    """PRJ3 BERT + MFCC gated-fusion baseline used by the existing PRJ4 app."""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.t_proj = nn.Linear(768, 128)
        self.a_proj = nn.Linear(15, 128)
        self.w_t = nn.Linear(128, 1)
        self.w_a = nn.Linear(128, 1)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5),
        )

    def forward(self, ids, mask, aud):
        text_features = self.t_proj(
            self.bert(input_ids=ids, attention_mask=mask).pooler_output
        )
        audio_features = self.a_proj(aud)
        weights = torch.softmax(
            torch.cat([self.w_t(text_features), self.w_a(audio_features)], dim=1),
            dim=1,
        )
        fused = weights[:, 0:1] * text_features + weights[:, 1:2] * audio_features
        return self.fc(fused)


@st.cache_resource
def load_emotion_model():
    model = Model().to(DEVICE)
    checkpoint_path = "best3.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "best3.pt was not found in the project working directory. "
            "Place the trained checkpoint beside main_app.py."
        )
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base", device=DEVICE)


@st.cache_resource
def load_diarization_engine():
    return DiarizationEngine.load_models("tmpdir_spkrec")


def extract_mfcc_features(y, sr=SAMPLE_RATE):
    try:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()
        features = np.concatenate([mfcc, [0.0, rms]])
        features = (features - features.mean()) / (features.std() + 1e-6)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception:
        return torch.zeros((1, 15), dtype=torch.float32, device=DEVICE)


def audio_tensor_to_wav_bytes(audio_tensor, sr=SAMPLE_RATE):
    buffer = io.BytesIO()
    audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak
    sf.write(buffer, audio_np, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def analyze_utterance(audio_tensor, turn_id, start_sec, end_sec,
                      diarizer, whisper_model, emotion_model, tokenizer):
    """Runs the expensive inference stage; called only by the background worker."""
    segment = diarizer.process_utterance(
        audio_tensor=audio_tensor,
        turn_id=turn_id,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    audio_np = segment.audio.detach().cpu().numpy().astype(np.float32)
    transcription = whisper_model.transcribe(
        audio_np, fp16=(DEVICE.type == "cuda"), verbose=False
    )
    segment.transcript = transcription.get("text", "").strip() or "[No speech transcribed]"

    encoded = tokenizer(
        segment.transcript,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    ).to(DEVICE)
    audio_features = extract_mfcc_features(audio_np, sr=SAMPLE_RATE)

    with torch.no_grad():
        logits = emotion_model(encoded["input_ids"], encoded["attention_mask"], audio_features)
        probabilities = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

    label_index = int(np.argmax(probabilities))
    segment.emotion = TARGET[label_index]
    segment.emotion_confidence = float(probabilities[label_index])
    segment.emotion_distribution = {
        TARGET[i]: float(probabilities[i]) for i in range(len(TARGET))
    }
    return segment


def inference_worker(work_queue, result_queue, diarizer, whisper_model,
                     emotion_model, tokenizer):
    """Never touches Streamlit UI; recording can continue while this worker infers."""
    while True:
        item = work_queue.get()
        try:
            if item is None:
                return
            audio, start_sec, end_sec, turn_id = item
            result = analyze_utterance(
                audio, turn_id, start_sec, end_sec, diarizer,
                whisper_model, emotion_model, tokenizer,
            )
            result_queue.put({"ok": True, "segment": result})
        except Exception as exc:
            result_queue.put({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "turn_id": item[3] if item else None,
            })
        finally:
            work_queue.task_done()


def frame_to_16k_mono(frame):
    """Convert a PyAV audio frame to mono float32 at 16 kHz."""
    array = np.asarray(frame.to_ndarray())
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        audio = array.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = array.astype(np.float32)
    if audio.ndim == 2:
        # PyAV generally returns (channels, samples) for planar audio.
        audio = audio.mean(axis=0)

    source_rate = int(frame.sample_rate or SAMPLE_RATE)
    if source_rate != SAMPLE_RATE and audio.size:
        # Typical browser input is 48 kHz; polyphase resampling is fast and stable.
        from math import gcd
        divisor = gcd(source_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, source_rate // divisor
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def render_results(container, results, errors):
    with container.container():
        st.subheader("Live dialogue timeline")
        if not results and not errors:
            st.info("Start speaking. Results appear after each utterance ends.")
            return

        counts = Counter(
            seg.emotion for seg in results if seg.emotion in TARGET
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Processed utterances", len(results))
        m2.metric("Speakers detected", len({seg.speaker_id for seg in results}))
        dominant = counts.most_common(1)[0][0] if counts else "—"
        m3.metric("Most frequent emotion", dominant.upper())

        for seg in reversed(results):
            emoji = EMOTION_EMOJIS.get(seg.emotion, "🎭")
            color = EMOTION_COLORS.get(seg.emotion, "#6c757d")
            with st.container(border=True):
                left, middle, right = st.columns([1.25, 3.5, 1.5])
                with left:
                    st.markdown(f"**{seg.speaker_id}**")
                    st.caption(f"{seg.start_sec:.1f}s – {seg.end_sec:.1f}s")
                    if seg.is_new_speaker:
                        st.caption("First appearance")
                with middle:
                    st.write(seg.transcript)
                    st.markdown(
                        f"<span style='background:{color}22;color:{color};"
                        f"padding:4px 8px;border-radius:6px;font-weight:600'>"
                        f"{emoji} {seg.emotion.upper()} "
                        f"({seg.emotion_confidence * 100:.1f}%)</span>",
                        unsafe_allow_html=True,
                    )
                with right:
                    with st.expander("Emotion scores"):
                        st.json(seg.emotion_distribution or {})
                    st.audio(audio_tensor_to_wav_bytes(seg.audio), format="audio/wav")

        for error in errors[-5:]:
            st.error(error)


st.set_page_config(
    page_title="PRJ4 — Real-time Emotion & Speaker Tracker",
    page_icon="🎭",
    layout="wide",
)
st.title("Multimodal Emotion Recognition & Speaker Diarization")
st.caption(
    "Real-time mode processes each completed utterance in a background worker "
    "while microphone capture continues. WAV batch analysis remains available."
)
with st.sidebar:
    st.header("System")
    st.info(f"Compute device: **{DEVICE}**")
    if DEVICE.type == "cuda":
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    st.markdown("---")
    st.header("Pipeline parameters")
    vad_thresh = st.slider("VAD speech threshold", 0.1, 0.9, 0.5, 0.05)
    silence_dur = st.slider("Silence before finalizing (sec)", 0.5, 2.5, 1.1, 0.1)
    min_speech_dur = st.slider("Minimum utterance duration (sec)", 0.3, 1.5, 0.6, 0.1)
    spk_sim_thresh = st.slider("Speaker similarity threshold", 0.20, 0.70, 0.38, 0.02)
    st.caption("Lower speaker similarity threshold tends to merge more speakers.")

live_tab, batch_tab = st.tabs(["Live microphone", "WAV file analysis"])

with live_tab:
    st.markdown(
        "1. Load models once. 2. Start the microphone. 3. Speak naturally. "
        "A turn is finalized after the configured silence interval; the next turn "
        "can be recorded while the previous turn is being analyzed."
    )
    if WEBRTC_IMPORT_ERROR is not None:
        st.error("Real-time microphone support is not installed.")
        st.code("python -m pip install streamlit-webrtc av")
        st.caption(f"Import detail: {WEBRTC_IMPORT_ERROR}")
    else:
        if "live_models_ready" not in st.session_state:
            st.session_state.live_models_ready = False
        if "live_results" not in st.session_state:
            st.session_state.live_results = []
        if "live_errors" not in st.session_state:
            st.session_state.live_errors = []
        if "live_runtime" not in st.session_state:
            st.session_state.live_runtime = None

        if st.button("Load models for live analysis", type="primary"):
            st.session_state.live_models_ready = True

        if st.session_state.live_models_ready:
            try:
                with st.spinner("Loading VAD, speaker, Whisper and emotion models..."):
                    live_diarizer = load_diarization_engine()
                    live_whisper = load_whisper_model()
                    live_emotion, live_tokenizer = load_emotion_model()
                st.success("Models loaded. Start the microphone below.")
            except Exception as exc:
                st.session_state.live_models_ready = False
                st.error(f"Could not load models: {type(exc).__name__}: {exc}")
                live_diarizer = live_whisper = live_emotion = live_tokenizer = None

            if st.session_state.live_models_ready:
                live_diarizer.speaker_manager.threshold = spk_sim_thresh
                ctx = webrtc_streamer(
                    key="prj4-live-microphone",
                    mode=WebRtcMode.SENDONLY,
                    audio_receiver_size=1024,
                    media_stream_constraints={"video": False, "audio": True},
                    async_processing=True,
                )

                results_area = st.empty()
                status_area = st.empty()

                if ctx.state.playing and ctx.audio_receiver is not None:
                    runtime = st.session_state.live_runtime
                    if runtime is None or runtime.get("closed", False):
                        live_diarizer.speaker_manager.reset()
                        segmenter = StreamingVADSegmenter(
                            live_diarizer.vad_model,
                            threshold=vad_thresh,
                            silence_duration=silence_dur,
                            min_speech_duration=min_speech_dur,
                        )
                        work_queue = queue.Queue(maxsize=20)
                        result_queue = queue.Queue()
                        worker = threading.Thread(
                            target=inference_worker,
                            args=(
                                work_queue, result_queue, live_diarizer,
                                live_whisper, live_emotion, live_tokenizer,
                            ),
                            daemon=True,
                        )
                        worker.start()
                        runtime = {
                            "segmenter": segmenter,
                            "work_queue": work_queue,
                            "result_queue": result_queue,
                            "worker": worker,
                            "turn_id": 0,
                            "queued": 0,
                            "closed": False,
                        }
                        st.session_state.live_runtime = runtime

                    status_area.info(
                        "Microphone is live. Audio capture continues while the "
                        "background worker transcribes and classifies earlier turns."
                    )
                    try:
                        while ctx.state.playing:
                            try:
                                frames = ctx.audio_receiver.get_frames(timeout=0.2)
                            except queue.Empty:
                                frames = []

                            for frame in frames:
                                audio_16k = frame_to_16k_mono(frame)
                                utterances = runtime["segmenter"].push_audio(audio_16k)
                                for audio, start_sec, end_sec in utterances:
                                    runtime["turn_id"] += 1
                                    item = (audio, start_sec, end_sec, runtime["turn_id"])
                                    try:
                                        runtime["work_queue"].put_nowait(item)
                                        runtime["queued"] += 1
                                    except queue.Full:
                                        st.session_state.live_errors.append(
                                            "Inference queue is full; one utterance was dropped. "
                                            "Try a smaller Whisper model or faster hardware."
                                        )

                            updated = False
                            while True:
                                try:
                                    message = runtime["result_queue"].get_nowait()
                                except queue.Empty:
                                    break
                                updated = True
                                if message["ok"]:
                                    st.session_state.live_results.append(message["segment"])
                                else:
                                    st.session_state.live_errors.append(
                                        f"Turn {message.get('turn_id')}: {message['error']}"
                                    )
                            if updated:
                                render_results(
                                    results_area,
                                    st.session_state.live_results,
                                    st.session_state.live_errors,
                                )
                    finally:
                        # Finalize any utterance still in progress when capture stops.
                        for audio, start_sec, end_sec in runtime["segmenter"].flush():
                            runtime["turn_id"] += 1
                            try:
                                runtime["work_queue"].put(
                                    (audio, start_sec, end_sec, runtime["turn_id"]),
                                    timeout=1,
                                )
                            except queue.Full:
                                st.session_state.live_errors.append(
                                    "Final utterance could not be queued because the inference queue was full."
                                )
                        runtime["closed"] = True
                        try:
                            runtime["work_queue"].put(None, timeout=1)
                        except queue.Full:
                            st.session_state.live_errors.append(
                                "Could not signal the inference worker to stop cleanly."
                            )
                        runtime["worker"].join(timeout=8)
                        while True:
                            try:
                                message = runtime["result_queue"].get_nowait()
                            except queue.Empty:
                                break
                            if message["ok"]:
                                st.session_state.live_results.append(message["segment"])
                            else:
                                st.session_state.live_errors.append(
                                    f"Turn {message.get('turn_id')}: {message['error']}"
                                )
                        st.session_state.live_runtime = None
                        render_results(
                            results_area,
                            st.session_state.live_results,
                            st.session_state.live_errors,
                        )
                        status_area.info("Microphone stopped.")

                elif st.session_state.live_results:
                    render_results(
                        st.empty(), st.session_state.live_results,
                        st.session_state.live_errors,
                    )

                if st.button("Clear live results"):
                    st.session_state.live_results = []
                    st.session_state.live_errors = []
                    st.session_state.live_runtime = None
                    st.rerun()
        else:
            st.info("Load the models before starting the live microphone.")

with batch_tab:
    st.markdown("Upload a WAV conversation for the original whole-file workflow.")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"], key="batch_wav")
    manual_override = st.text_input(
        "Optional transcript override (only used if exactly one utterance is detected)",
        key="batch_transcript_override",
    )
    run_batch = st.button("Run batch analysis", type="primary")

    if run_batch:
        if uploaded_file is None:
            st.warning("Upload a WAV file first.")
        else:
            try:
                with st.spinner("Loading models..."):
                    diarizer = load_diarization_engine()
                    whisper_model = load_whisper_model()
                    emotion_model, tokenizer = load_emotion_model()

                diarizer.speaker_manager.threshold = spk_sim_thresh
                diarizer.speaker_manager.reset()
                with st.spinner("Reading and segmenting audio..."):
                    audio, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True)
                    duration = len(audio) / SAMPLE_RATE
                    segments = diarizer.process_full_audio(audio, sample_rate=SAMPLE_RATE)

                if not segments:
                    st.warning("No speech detected.")
                else:
                    emotion_counts = Counter()
                    progress = st.progress(0.0, text="Analyzing utterances...")
                    for index, segment in enumerate(segments):
                        if manual_override.strip() and len(segments) == 1:
                            segment.transcript = manual_override.strip()
                        else:
                            segment.transcript = whisper_model.transcribe(
                                segment.audio.numpy(),
                                fp16=(DEVICE.type == "cuda"),
                                verbose=False,
                            ).get("text", "").strip() or "[No speech transcribed]"

                        encoded = tokenizer(
                            segment.transcript,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                        ).to(DEVICE)
                        features = extract_mfcc_features(
                            segment.audio.numpy(), sr=SAMPLE_RATE
                        )
                        with torch.no_grad():
                            logits = emotion_model(
                                encoded["input_ids"], encoded["attention_mask"], features
                            )
                            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                        label_index = int(np.argmax(probabilities))
                        segment.emotion = TARGET[label_index]
                        segment.emotion_confidence = float(probabilities[label_index])
                        segment.emotion_distribution = {
                            TARGET[i]: float(probabilities[i])
                            for i in range(len(TARGET))
                        }
                        emotion_counts[segment.emotion] += 1
                        progress.progress((index + 1) / len(segments))

                    progress.empty()
                    st.success("Batch analysis complete.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Duration", f"{duration:.1f} sec")
                    m2.metric("Speakers", len({s.speaker_id for s in segments}))
                    m3.metric("Utterances", len(segments))
                    for segment in segments:
                        with st.container(border=True):
                            st.markdown(
                                f"**{segment.speaker_id}** · "
                                f"{segment.start_sec:.1f}s–{segment.end_sec:.1f}s"
                            )
                            st.write(segment.transcript)
                            st.markdown(
                                f"{EMOTION_EMOJIS.get(segment.emotion, '🎭')} "
                                f"**{segment.emotion.upper()}** — "
                                f"{segment.emotion_confidence * 100:.1f}%"
                            )
                            st.audio(audio_tensor_to_wav_bytes(segment.audio), format="audio/wav")
                    st.subheader("Emotion distribution")
                    st.bar_chart({emotion.capitalize(): emotion_counts.get(emotion, 0)
                                  for emotion in TARGET})
            except Exception as exc:
                st.error(f"Batch analysis failed: {type(exc).__name__}: {exc}")
