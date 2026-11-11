"""FastAPI backend for the PRJ4 multimodal emotion recognition pipeline."""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_BUNDLE_DIR = BACKEND_DIR / "model_bundle"

# audio_capture.py / diarization_engine.py live in backend/, infer2.py + common2.py in
# backend/model_bundle/. Make both importable however uvicorn is launched.
for _p in (BACKEND_DIR, MODEL_BUNDLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
