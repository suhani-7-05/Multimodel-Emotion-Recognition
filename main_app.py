import os
import io
import tempfile
import logging
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import streamlit as st
import whisper
from transformers import BertTokenizer, BertModel

from diarization_engine import DiarizationEngine, DiarizedSegment

# --- PREVENT LOG CLUTTER ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONSTANTS ---
TARGET = ['neutral', 'joy', 'anger', 'sadness', 'surprise']
EMOTION_EMOJIS = {
    'neutral': '😐',
    'joy': '😊',
    'anger': '😠',
    'sadness': '😢',
    'surprise': '😲'
}
EMOTION_COLORS = {
    'neutral': '#6c757d',
    'joy': '#28a745',
    'anger': '#dc3545',
    'sadness': '#17a2b8',
    'surprise': '#fd7e14'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (PRJ3 BERT + MFCC FUSION BASELINE) ---
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.t_proj = nn.Linear(768, 128)
        self.a_proj = nn.Linear(15, 128)
        self.w_t = nn.Linear(128, 1)
        self.w_a = nn.Linear(128, 1)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, ids, mask, aud):
        t = self.bert(input_ids=ids, attention_mask=mask).pooler_output
        t = self.t_proj(t)
        a = self.a_proj(aud)
        s = torch.cat([self.w_t(t), self.w_a(a)], dim=1)
        w = torch.softmax(s, dim=1)
        f = w[:, 0:1] * t + w[:, 1:2] * a
        return self.fc(f)

# --- MODEL CACHING ---
@st.cache_resource
def load_emotion_model():
    model = Model().to(DEVICE)
    checkpoint_path = "best3.pt"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base", device=DEVICE)

@st.cache_resource
def load_diarization_engine():
    return DiarizationEngine.load_models("tmpdir_spkrec")

# --- AUDIO HELPERS ---
def extract_mfcc_features(y, sr=16000):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()
        feat = np.concatenate([mfcc, [0.0, rms]])
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception:
        return torch.zeros((1, 15), dtype=torch.float32).to(DEVICE)

def audio_tensor_to_wav_bytes(audio_tensor, sr=16000):
    """Converts a 1D audio tensor into playable WAV byte stream for st.audio."""
    buffer = io.BytesIO()
    audio_np = audio_tensor.detach().cpu().numpy()
    # Normalize volume to avoid clipping
    max_val = np.max(np.abs(audio_np)) + 1e-6
    if max_val > 1.0:
        audio_np = audio_np / max_val
    sf.write(buffer, audio_np, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Multimodal Emotion & Speaker Tracker",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Multimodal Emotion Recognition & Speaker Diarization")
st.markdown(
    "**End-to-End Assistive Pipeline**: "
    "**Silero VAD** (Speech Segmentation) ➔ "
    "**SpeechBrain ECAPA-TDNN** (Speaker Diarization) ➔ "
    "**Whisper ASR** (Speech-to-Text) ➔ "
    "**BERT + MFCC Gated Fusion** (Emotion Recognition)"
)

with st.sidebar:
    st.header("⚙️ System Status")
    st.info(f"Compute Device: **{DEVICE}**")
    if "cuda" in str(DEVICE):
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.caption("Running in CPU mode")

    st.markdown("---")
    st.header("🎛️ Pipeline Parameters")
    vad_thresh = st.slider("VAD Speech Threshold", 0.1, 0.9, 0.5, 0.05)
    silence_dur = st.slider("Pause Finalization (sec)", 0.5, 2.5, 1.1, 0.1)
    spk_sim_thresh = st.slider("Speaker Clustering Threshold", 0.20, 0.70, 0.38, 0.02)
    st.caption("Lower similarity creates more speakers; higher merges speakers.")

st.markdown("### 🎙️ 1. Input Audio (Live Microphone or File)")

tab_mic, tab_file = st.tabs(["🔴 Live Microphone Recording", "📁 Upload WAV File"])

with tab_mic:
    st.markdown("Click the **Microphone** icon to record. Speak naturally (multiple speakers can take turns with natural pauses), then click **Stop**.")
    mic_audio = st.audio_input("Record Conversation Live from Microphone:")

with tab_file:
    uploaded_file = st.file_uploader(
        "Upload a multi-speaker or continuous conversation (.wav):",
        type=["wav"]
    )

active_audio_source = None
source_name = ""

if mic_audio is not None:
    active_audio_source = mic_audio
    source_name = "Live Microphone Recording"
elif uploaded_file is not None:
    active_audio_source = uploaded_file
    source_name = f"Uploaded File ({uploaded_file.name})"

col_up1, col_up2 = st.columns([1, 2])
if active_audio_source is not None:
    with col_up1:
        st.audio(active_audio_source, format="audio/wav")
    with col_up2:
        st.success(f"**Ready for Analysis:** {source_name}")
        st.caption("Click the button below to segment turns, identify speakers, and detect emotions.")

manual_override = st.text_input(
    "Optional: Manual text override (applies only if processing a single short utterance):",
    placeholder="Leave blank to automatically transcribe with Whisper..."
)

analyze_btn = st.button("🚀 Run Full Dialogue & Emotion Analysis", type="primary", use_container_width=True)

if analyze_btn:
    if active_audio_source is None:
        st.warning("Please record audio using the microphone tab or upload a .wav audio file first.")
    else:
        # Load models
        with st.spinner("Loading AI models (VAD, ECAPA-TDNN, Whisper, BERT)..."):
            diarizer = load_diarization_engine()
            whisper_model = load_whisper_model()
            emotion_model, tokenizer = load_emotion_model()

        # Update dynamic parameters
        diarizer.speaker_manager.threshold = spk_sim_thresh
        diarizer.speaker_manager.reset()

        # Read audio with librosa at 16kHz
        with st.spinner("Loading and preprocessing audio stream..."):
            active_audio_source.seek(0)
            y, sr = librosa.load(active_audio_source, sr=16000)
            total_duration = len(y) / sr

        # Stage 1: VAD & Diarization
        with st.spinner("Stage 1/3: Slicing conversation with Silero VAD & clustering speakers with SpeechBrain..."):
            segments = diarizer.process_full_audio(y, sample_rate=16000)

        if not segments:
            st.warning("No speech activity detected in the audio file.")
        else:
            # Stage 2 & 3: Whisper ASR + Emotion Inference per segment
            progress_bar = st.progress(0, text="Analyzing dialogue turns...")
            total_turns = len(segments)

            emotion_counts = {e: 0 for e in TARGET}

            for idx, seg in enumerate(segments):
                progress_bar.progress((idx + 1) / total_turns, text=f"Processing Turn {idx+1}/{total_turns} ({seg.speaker_id})...")

                # Whisper Speech-to-Text (Direct in-memory numpy array, no ffmpeg required)
                seg_audio_np = seg.audio.numpy()
                if manual_override.strip() and total_turns == 1:
                    transcript = manual_override.strip()
                else:
                    asr_res = whisper_model.transcribe(seg_audio_np, fp16=(DEVICE.type == "cuda"))
                    transcript = asr_res["text"].strip()

                seg.transcript = transcript if transcript else "[No speech transcribed]"

                # Text Tokens for BERT
                enc = tokenizer(
                    seg.transcript,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=128
                ).to(DEVICE)

                # Audio MFCC Features
                aud_feat = extract_mfcc_features(seg_audio_np, sr=16000)

                # Multimodal Emotion Prediction
                with torch.no_grad():
                    logits = emotion_model(enc['input_ids'], enc['attention_mask'], aud_feat)
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_label = TARGET[pred_idx]
                    confidence = float(probs[pred_idx])

                seg.emotion = pred_label
                seg.emotion_confidence = confidence
                seg.emotion_distribution = {TARGET[i]: float(probs[i]) for i in range(len(TARGET))}
                emotion_counts[pred_label] += 1

            progress_bar.empty()

            # --- DISPLAY RESULTS ---
            st.success("✅ Complete Conversation Analysis Finished!")

            # 1. Metric Row
            unique_speakers = len(set(s.speaker_id for s in segments))
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Duration", f"{total_duration:.1f} sec")
            m2.metric("Distinct Speakers", f"{unique_speakers}")
            m3.metric("Dialogue Turns", f"{total_turns}")
            m4.metric("Dominant Affect", f"{EMOTION_EMOJIS[dominant_emotion]} {dominant_emotion.upper()}")

            st.divider()

            # 2. Dialogue Timeline View
            st.subheader("💬 Dialogue Turns & Affective Timeline")

            for seg in segments:
                emoji = EMOTION_EMOJIS.get(seg.emotion, "🎭")
                color = EMOTION_COLORS.get(seg.emotion, "#6c757d")

                with st.container():
                    c_spk, c_content, c_aud = st.columns([1.5, 3.5, 2])

                    with c_spk:
                        st.markdown(f"### 👤 {seg.speaker_id}")
                        st.caption(f"⏱️ **{seg.start_sec:.1f}s – {seg.end_sec:.1f}s** ({(seg.end_sec - seg.start_sec):.1f}s)")
                        if seg.is_new_speaker:
                            st.caption("✨ *First appearance*")

                    with c_content:
                        st.markdown(f"**Transcript:** {seg.transcript}")
                        st.markdown(
                            f"**Emotion:** <span style='background-color:{color}22; color:{color}; padding:4px 8px; border-radius:6px; font-weight:bold;'>"
                            f"{emoji} {seg.emotion.upper()} ({seg.emotion_confidence*100:.1f}%)</span>",
                            unsafe_allow_html=True
                        )

                    with c_aud:
                        wav_bytes = audio_tensor_to_wav_bytes(seg.audio, sr=16000)
                        st.audio(wav_bytes, format="audio/wav")

                    st.markdown("---")

            # 3. Overall Emotion Distribution
            st.subheader("📊 Aggregate Emotion Distribution in Conversation")
            col_chart, col_details = st.columns([2, 1])

            with col_chart:
                chart_data = {e.capitalize(): count for e, count in emotion_counts.items()}
                st.bar_chart(chart_data)

            with col_details:
                st.markdown("**Per-Turn Confidence Breakdown**")
                for seg in segments:
                    with st.expander(f"Turn {seg.turn_id} ({seg.speaker_id}) - {seg.emotion.upper()}"):
                        st.write(seg.emotion_distribution)