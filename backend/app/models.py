"""Loads the heavy models once and guards the shared per-conversation state."""
import logging
import os
import threading
import warnings
from typing import Optional

import torch

from . import config

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRegistry:
    """
    Holds VAD + ECAPA (diarizer), Whisper and the emotion engine.

    The diarizer's speaker centroids and the emotion engine's dialogue history are
    per-conversation state, so only one live session or batch job may use them at a
    time: callers take `busy` (non-blocking) for the whole job.
    """

    def __init__(self):
        self.state = "idle"            # idle | loading | ready | error
        self.error: Optional[str] = None
        self.step: Optional[str] = None
        self.diarizer = None
        self.whisper = None
        self.emotion = None
        self.busy = threading.Lock()
        self.busy_with: Optional[str] = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load_async(self):
        with self._load_lock:
            if self.state in ("loading", "ready"):
                return
            self.state, self.error = "loading", None
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self):
        try:
            from diarization_engine import DiarizationEngine
            from infer2 import EmotionContextEngine
            import whisper

            if not config.MODEL_CHECKPOINT_DIR.is_dir():
                raise FileNotFoundError(
                    f"Emotion model directory not found: {config.MODEL_CHECKPOINT_DIR}. "
                    "Extract prj4_emotion_model.tar.gz into backend/model_bundle first."
                )
            self.step = "VAD + speaker model"
            self.diarizer = DiarizationEngine.load_models(str(config.SPEAKER_MODEL_DIR))
            self.step = f"Whisper ({config.WHISPER_MODEL})"
            self.whisper = whisper.load_model(config.WHISPER_MODEL, device=DEVICE)
            self.step = "emotion model"
            # RoBERTa + WavLM stay on CPU to leave the laptop's 6 GB VRAM for Whisper.
            self.emotion = EmotionContextEngine(
                ckpt_dir=str(config.MODEL_CHECKPOINT_DIR), device="cpu", audio_fp16=False,
            )
            self.step = None
            self.state = "ready"
            LOG.info("All models loaded")
        except Exception as exc:  # surfaced through /api/status
            LOG.exception("Model loading failed")
            self.step = None
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    # ------------------------------------------------------------------ jobs
    def acquire(self, job: str) -> bool:
        if not self.busy.acquire(blocking=False):
            return False
        self.busy_with = job
        return True

    def release(self):
        self.busy_with = None
        self.busy.release()

    def reset_conversation(self, speaker_threshold: float):
        self.diarizer.speaker_manager.threshold = speaker_threshold
        self.diarizer.speaker_manager.reset()
        self.emotion.reset()

    def status(self) -> dict:
        return {
            "models": self.state,
            "loading_step": self.step,
            "error": self.error,
            "device": DEVICE.type,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "whisper_model": config.WHISPER_MODEL,
            "busy_with": self.busy_with,
        }


registry = ModelRegistry()
