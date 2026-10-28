import io
import os
import logging
import warnings
import sys
from pathlib import Path
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import whisper

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = exc

try:
    import sounddevice  # noqa: F401  (only checking that PortAudio is usable)
    SD_IMPORT_ERROR = None
except Exception as exc:
    SD_IMPORT_ERROR = exc

from audio_capture import LiveSession, SoundDeviceSource, list_input_devices
from diarization_engine import (
    DiarizationEngine,
    StreamingVADSegmenter,
    DiarizedSegment,
)

# New trained multimodal model bundle. Keep its common2.py isolated from old code.
MODEL_BUNDLE_DIR = Path(__file__).resolve().parent / "model_bundle"
MODEL_CHECKPOINT_DIR = MODEL_BUNDLE_DIR / "runs" / "v2_rl_audio"
if not MODEL_CHECKPOINT_DIR.is_dir():
    raise FileNotFoundError(
        f"New emotion model directory not found: {MODEL_CHECKPOINT_DIR}. "
        "Extract prj4_emotion_model.tar.gz into model_bundle first."
    )
sys.path.insert(0, str(MODEL_BUNDLE_DIR))
from infer2 import EmotionContextEngine

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

TARGET = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
EMOTION_EMOJIS = {
    "neutral": "😐", "joy": "😊", "surprise": "😲",
    "anger": "😠", "sadness": "😢", "disgust": "🤢", "fear": "😨",
}
EMOTION_COLORS = {
    "neutral": "#6c757d", "joy": "#28a745", "surprise": "#fd7e14",
    "anger": "#dc3545", "sadness": "#17a2b8", "disgust": "#795548",
    "fear": "#6f42c1",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000


@st.cache_resource
def load_emotion_model():
    """Load RoBERTa + WavLM on CPU to conserve the laptop's 6 GB VRAM."""
    return EmotionContextEngine(
        ckpt_dir=str(MODEL_CHECKPOINT_DIR),
        device="cpu",
        audio_fp16=False,
    )


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("large", device=DEVICE)


@st.cache_resource
def load_diarization_engine():
    return DiarizationEngine.load_models("tmpdir_spkrec")


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
                      diarizer, whisper_model, emotion_engine):
    """Diarize, transcribe, then classify one completed utterance."""
    segment = diarizer.process_utterance(
        audio_tensor=audio_tensor,
        turn_id=turn_id,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    audio_np = segment.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
    transcription = whisper_model.transcribe(
        audio_np, fp16=(DEVICE.type == "cuda"), verbose=False
    )
    segment.transcript = transcription.get("text", "").strip() or "[No speech transcribed]"

    result = emotion_engine.update(
        speaker=str(segment.speaker_id),
        text=segment.transcript,
        audio=audio_np,
        commit=True,
    )
    segment.emotion = result["emotion"]
    segment.emotion_confidence = float(result["confidence"])
    segment.emotion_distribution = result["probs"]
    return segment


DEBUG_WAV_PATH = "debug_capture_16k.wav"


def segment_wav_bytes(seg):
    """Encode a segment to WAV once; re-encoding every segment on every UI refresh was slow."""
    cached = getattr(seg, "_wav_bytes", None)
    if cached is None:
        cached = audio_tensor_to_wav_bytes(seg.audio)
        seg._wav_bytes = cached
    return cached


def collect_from_session(session):
    """Move finished results/errors from the session's queues into st.session_state."""
    got_something = False
    for message in session.drain_results():
        got_something = True
        if message["ok"]:
            st.session_state.live_results.append(message["segment"])
        else:
            st.session_state.live_errors.append(
                f"Turn {message.get('turn_id')}: {message['error']}"
            )
    for error in session.drain_errors():
        got_something = True
        st.session_state.live_errors.append(error)
    return got_something


def stop_live_session():
    session = st.session_state.live_session
    if session is None:
        return
    with st.spinner("Finishing queued utterances..."):
        session.stop()
    collect_from_session(session)
    st.session_state.live_session = None


def render_capture_health():
    """Light fragment, refreshes every second while recording. Never re-renders the timeline."""
    session = st.session_state.live_session
    if session is None:
        return
    if collect_from_session(session):
        st.rerun()                      # full rerun -> timeline below shows the new turns
    if not session.running:
        st.session_state.live_session = None
        st.rerun()

    snap = session.snapshot()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Captured audio", f"{snap['audio_sec']:.0f} s")
    c2.metric("Input peak", f"{snap['peak']:.2f}")
    c3.metric("Timeline gaps", f"{snap['gap_events']} ({snap['gap_ms']:.0f} ms)")
    c4.metric("Utterances pending", snap["pending"])
    info = st.session_state.get("live_source_info")
    if info:
        st.caption(f"Source: {info} -> 16 kHz mono")
    if snap["clipped_samples"] > 0:
        st.warning("Input is clipping (hard distortion). Lower the microphone gain / input volume.")
    if snap.get("overflows", 0) or snap.get("dropped_blocks", 0):
        st.warning("PortAudio dropped input blocks - the machine could not keep up with capture.")
    if snap["gap_events"] > 0:
        st.warning(
            "Audio timeline gaps (dropped WebRTC packets or Opus silence suppression). "
            "They are filled with silence, but glitchy speech means: use Local microphone mode."
        )



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
        known_speakers = {seg.speaker_id for seg in results if seg.speaker_id != "Unknown"}
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
                    st.audio(segment_wav_bytes(seg), format="audio/wav")

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
        "keeps being captured while the previous one is analysed in the background."
    )
    for key, factory in (
        ("live_models_ready", lambda: False),
        ("live_results", list),
        ("live_errors", list),
        ("live_session", lambda: None),
        ("live_source_info", lambda: ""),
    ):
        if key not in st.session_state:
            st.session_state[key] = factory()

    if st.button("Load models for live analysis", type="primary"):
        st.session_state.live_models_ready = True

    if not st.session_state.live_models_ready:
        st.info("Load the models before starting the live microphone.")
    else:
        try:
            with st.spinner("Loading VAD, speaker, Whisper and the new emotion model..."):
                live_diarizer = load_diarization_engine()
                live_whisper = load_whisper_model()
                live_emotion = load_emotion_model()
            st.success("Models loaded.")
            models_ok = True
        except Exception as exc:
            st.session_state.live_models_ready = False
            st.error(f"Could not load models: {type(exc).__name__}: {exc}")
            models_ok = False

        if models_ok:
            def make_live_session(pull, on_close=None, extra_stats=None, debug=False):
                live_diarizer.speaker_manager.threshold = spk_sim_thresh
                live_diarizer.speaker_manager.reset()
                live_emotion.reset()
                segmenter = StreamingVADSegmenter(
                    live_diarizer.vad_model,
                    threshold=vad_thresh,
                    silence_duration=silence_dur,
                    min_speech_duration=min_speech_dur,
                )

                def analyze(audio, start_sec, end_sec, turn_id):
                    return analyze_utterance(
                        audio, turn_id, start_sec, end_sec,
                        live_diarizer, live_whisper, live_emotion,
                    )

                session = LiveSession(
                    pull, segmenter, analyze,
                    on_close=on_close, extra_stats=extra_stats,
                    debug_wav_path=DEBUG_WAV_PATH if debug else None,
                )
                session.start()
                return session

            active = st.session_state.live_session is not None
            modes = ["Local microphone (recommended)", "Browser microphone (WebRTC)"]
            mode = st.radio("Audio input", modes, horizontal=True, disabled=active,
                            key="live_input_mode")
            debug_dump = st.checkbox(
                f"Save the captured 16 kHz audio to {DEBUG_WAV_PATH} (to check capture quality)",
                disabled=active,
            )

            if mode == modes[0]:
                # Same capture idea as the original live_speaker_tracker.py: straight from
                # the sound card, no browser, no WebRTC, no Opus codec.
                if SD_IMPORT_ERROR is not None:
                    st.error("Local microphone support is not installed.")
                    st.code("python -m pip install sounddevice av")
                    st.caption(f"Import detail: {SD_IMPORT_ERROR}")
                else:
                    devices = list_input_devices()
                    labels = ["System default"] + [d["label"] for d in devices]
                    index_of = {d["label"]: d["index"] for d in devices}
                    choice = st.selectbox("Input device", labels, disabled=active)
                    b1, b2 = st.columns(2)
                    start_clicked = b1.button("Start microphone", type="primary", disabled=active)
                    stop_clicked = b2.button("Stop", disabled=not active)

                    if start_clicked:
                        source = None
                        try:
                            source = SoundDeviceSource(index_of.get(choice))
                            source.start()
                            st.session_state.live_session = make_live_session(
                                source.pull, on_close=source.stop,
                                extra_stats=source.stats, debug=debug_dump,
                            )
                            st.session_state.live_source_info = (
                                f"{source.device_name} @ {source.rate} Hz"
                            )
                        except Exception as exc:
                            if source is not None:
                                source.stop()
                            st.session_state.live_errors.append(
                                f"Could not open microphone: {type(exc).__name__}: {exc}"
                            )
                        st.rerun()
                    if stop_clicked:
                        stop_live_session()
                        st.rerun()
            else:
                if WEBRTC_IMPORT_ERROR is not None:
                    st.error("Browser microphone support is not installed.")
                    st.code("python -m pip install streamlit-webrtc av")
                    st.caption(f"Import detail: {WEBRTC_IMPORT_ERROR}")
                else:
                    st.caption(
                        "Browser audio travels through WebRTC and the lossy Opus codec. Use this "
                        "only when the browser and this server are on different machines."
                    )
                    ctx = webrtc_streamer(
                        key="prj4-live-microphone",
                        mode=WebRtcMode.SENDONLY,
                        audio_receiver_size=2048,
                        media_stream_constraints={
                            "video": False,
                            # The browser's default voice-call DSP (noise suppression, echo
                            # cancellation, auto gain) mangles speech and flattens the
                            # prosody the emotion model relies on. Turn it all off.
                            "audio": {
                                "echoCancellation": False,
                                "noiseSuppression": False,
                                "autoGainControl": False,
                                "channelCount": 1,
                            },
                        },
                        async_processing=True,
                    )
                    playing = bool(ctx.state.playing and ctx.audio_receiver is not None)
                    if playing and st.session_state.live_session is None:
                        receiver = ctx.audio_receiver
                        st.session_state.live_session = make_live_session(
                            lambda timeout: receiver.get_frames(timeout=timeout),
                            debug=debug_dump,
                        )
                        st.session_state.live_source_info = "Browser (WebRTC/Opus 48 kHz)"
                        st.rerun()
                    elif not playing and st.session_state.live_session is not None:
                        stop_live_session()
                        st.rerun()

            # Capture-health strip: its own fragment, so refreshing it never touches the timeline.
            st.fragment(
                render_capture_health,
                run_every=1.0 if st.session_state.live_session is not None else None,
            )()

    render_results(
        st.container(), st.session_state.live_results, st.session_state.live_errors
    )
    if st.button("Clear live results"):
        st.session_state.live_results = []
        st.session_state.live_errors = []
        st.rerun()


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
                    emotion_engine = load_emotion_model()

                diarizer.speaker_manager.threshold = spk_sim_thresh
                diarizer.speaker_manager.reset()
                emotion_engine.reset()
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

                        audio_np = segment.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
                        result = emotion_engine.update(
                            speaker=str(segment.speaker_id),
                            text=segment.transcript,
                            audio=audio_np,
                            commit=True,
                        )
                        segment.emotion = result["emotion"]
                        segment.emotion_confidence = float(result["confidence"])
                        segment.emotion_distribution = result["probs"]
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