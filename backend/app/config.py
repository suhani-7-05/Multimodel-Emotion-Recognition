"""All runtime settings, overridable with environment variables (see README)."""
import os
from pathlib import Path

from . import BACKEND_DIR, MODEL_BUNDLE_DIR


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


SAMPLE_RATE = 16000

# ---------------------------------------------------------------- model locations
MODEL_CHECKPOINT_DIR = Path(_env("EMOTION_CKPT_DIR", str(MODEL_BUNDLE_DIR / "runs" / "v2_rl_audio")))
SPEAKER_MODEL_DIR = Path(_env("SPEAKER_MODEL_DIR", str(BACKEND_DIR / "tmpdir_spkrec")))
DEBUG_WAV_PATH = BACKEND_DIR / "debug_capture_16k.wav"

# ---------------------------------------------------------------- speech recognition
# large-v3-turbo: large-v3 encoder with a 4-layer decoder. Far faster than "large" with
# a small accuracy cost. "large-v3" is the accuracy-first option on a big GPU.
_LOCAL_TURBO = BACKEND_DIR / "models" / "faster-whisper-large-v3-turbo"
WHISPER_MODEL = _env("WHISPER_MODEL", str(_LOCAL_TURBO) if _LOCAL_TURBO.is_dir() else "large-v3-turbo")
ASR_ENGINE = _env("ASR_ENGINE", "auto")          # auto | faster | openai
ASR_LANGUAGE = _env("ASR_LANGUAGE", "en") or None
ASR_BEAM_SIZE = int(_env("ASR_BEAM_SIZE", "5"))

# ---------------------------------------------------------------- emotion model
# auto: GPU when it has room for everything (>= 12 GB, e.g. the 18 GB pod slice),
# otherwise CPU so Whisper and the LLM keep the laptop's 6 GB.
EMOTION_DEVICE = _env("EMOTION_DEVICE", "auto")  # auto | cuda | cpu
GPU_ROOMY_GB = float(_env("GPU_ROOMY_GB", "12"))

# ---------------------------------------------------------------- plain-language interpreter
LLM_ENABLED = _env("LLM_ENABLED", "1") == "1"
LLM_API = _env("LLM_API", "ollama")              # ollama | openai (vLLM, llama.cpp server, ...)
LLM_BASE_URL = _env("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = _env("LLM_MODEL", "qwen3:4b")
LLM_API_KEY = _env("LLM_API_KEY", "")
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "60"))
LLM_CONTEXT_TURNS = int(_env("LLM_CONTEXT_TURNS", "8"))
# Let reasoning models (Qwen3, ...) think before answering: better at sarcasm/hints, slower.
LLM_THINK = _env("LLM_THINK", "0") == "1"
# Ollama only: how many layers to put on the GPU ("" = Ollama decides, "0" = CPU only).
# On a 6 GB laptop a 7B model on the GPU starves Whisper (turns went from ~1 s to ~7 s);
# use 0 there. On the 18 GB pod leave it empty.
LLM_NUM_GPU = _env("LLM_NUM_GPU", "")
LLM_KEEP_ALIVE = _env("LLM_KEEP_ALIVE", "30m")

# ---------------------------------------------------------------- server
PRELOAD_MODELS = _env("PRELOAD_MODELS", "0") == "1"
CORS_ORIGINS = [o.strip() for o in _env(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
MAX_CLIPS = int(_env("MAX_CLIPS", "500"))

TARGET_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

DEFAULT_SETTINGS = {
    "vad_threshold": 0.5,
    "silence_duration": 1.1,
    "min_speech_duration": 0.6,
    "speaker_threshold": 0.38,
}
