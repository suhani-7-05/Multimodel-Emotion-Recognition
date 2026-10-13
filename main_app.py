import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import BertTokenizer, BertModel
import whisper  # Integrated Whisper
import logging
import warnings
import os

# --- PREVENT LOG CLUTTER ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONSTANTS ---
TARGET = ['neutral', 'joy', 'anger', 'sadness', 'surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION ---
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

# --- UTILS & CACHING ---
@st.cache_resource
def load_emotion_model():
    model = Model().to(DEVICE)
    state_dict = torch.load("best3.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

@st.cache_resource
def load_whisper_model():
    # 'base' is good for RTX 3050 or CPU
    return whisper.load_model("base", device=DEVICE)

def get_audio_features(audio_bytes):
    try:
        # Librosa can load from bytes/buffer
        y, sr = librosa.load(audio_bytes, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()
        feat = np.concatenate([mfcc, [0.0, rms]])
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return torch.zeros((1, 15)).to(DEVICE)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Emotion Detector", page_icon="🎭")
st.title("🎭 Multimodal Emotion Recognition")
st.markdown("Predicting emotions using **Whisper** (Speech-to-Text), **BERT** (Text), and **MFCCs** (Audio).")

with st.sidebar:
    st.info(f"Running on: **{DEVICE}**")
    st.write("RTX 3050 detected" if "cuda" in str(DEVICE) else "Using CPU")

col1, col2 = st.columns(2)

with col2:
    audio_file = st.file_uploader("Upload Audio (WAV):", type=["wav"])
    if audio_file:
        st.audio(audio_file)

with col1:
    text_input = st.text_area("Utterance Text:", 
                              placeholder="Leave blank to use Whisper transcription...",
                              help="If blank, the app will transcribe your uploaded audio.")

if st.button("Analyze Emotion", use_container_width=True):
    if audio_file:
        with st.spinner("Processing Multimodal Input..."):
            # 1. Load Models
            emotion_model, tokenizer = load_emotion_model()
            whisper_model = load_whisper_model()
            
            # 2. Handle Transcription (if text is missing)
            final_text = text_input
            if not text_input.strip():
                st.info("Transcribing audio with Whisper...")
                # Save temp file for whisper
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_file.getbuffer())
                
                # Transcribe
                whisper_result = whisper_model.transcribe("temp_audio.wav", fp16=False)
                final_text = whisper_result["text"]
                st.success(f"**Whisper Transcribed:** {final_text}")

            # 3. Prep Text for BERT
            enc = tokenizer(final_text, return_tensors='pt', padding='max_length', 
                            truncation=True, max_length=128).to(DEVICE)
            
            # 4. Prep Audio for MFCCs
            # Reset buffer to start for librosa
            audio_file.seek(0)
            aud_feat = get_audio_features(audio_file)
            
            # 5. Inference
            with torch.no_grad():
                logits = emotion_model(enc['input_ids'], enc['attention_mask'], aud_feat)
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()

            # 6. Display Results
            st.divider()
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Detected Emotion", TARGET[pred_idx].upper())
            
            with res_col2:
                st.metric("Confidence Score", f"{confidence:.2%}")
                
            prob_dict = {TARGET[i]: float(probs[0][i]) for i in range(len(TARGET))}
            st.bar_chart(prob_dict)
    else:
        st.warning("Please upload an audio file first.")