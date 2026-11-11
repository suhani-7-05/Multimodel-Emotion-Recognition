import os
from pathlib import Path

from . import BACKEND_DIR, MODEL_BUNDLE_DIR

SAMPLE_RATE = 16000

MODEL_CHECKPOINT_DIR = Path(os.environ.get(
    "EMOTION_CKPT_DIR", MODEL_BUNDLE_DIR / "runs" / "v2_rl_audio"
))
SPEAKER_MODEL_DIR = Path(os.environ.get("SPEAKER_MODEL_DIR", BACKEND_DIR / "tmpdir_spkrec"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large")
DEBUG_WAV_PATH = BACKEND_DIR / "debug_capture_16k.wav"

# Load all models when the server starts instead of waiting for POST /api/models/load.
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "0") == "1"

# Origins allowed to call the API directly (the Vite dev server proxies, so this is
# only needed when the frontend is served from somewhere else).
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]

# How many utterance clips to keep in memory for playback.
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "500"))

TARGET_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

DEFAULT_SETTINGS = {
    "vad_threshold": 0.5,
    "silence_duration": 1.1,
    "min_speech_duration": 0.6,
    "speaker_threshold": 0.38,
}
