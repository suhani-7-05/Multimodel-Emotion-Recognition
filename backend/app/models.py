"""Loads the heavy models once and guards the shared per-conversation state."""
import logging
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from . import config

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch's intra-op thread count is PER THREAD, and Silero VAD sets it to 1 in whichever thread
# loads it. Model threads that never set it ran WavLM/RoBERTa on CPU up to ~6x slower, so every
# thread that runs a model calls use_model_threads() (pool initializer + analysis entry).
_DEFAULT_TORCH_THREADS = int(os.environ.get("TORCH_THREADS", torch.get_num_threads()))


def use_model_threads():
    if torch.get_num_threads() != _DEFAULT_TORCH_THREADS:
        torch.set_num_threads(_DEFAULT_TORCH_THREADS)


def _gpu_total_gb() -> float:
    if DEVICE.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def _emotion_device() -> str:
    if config.EMOTION_DEVICE in ("cuda", "cpu"):
        return config.EMOTION_DEVICE if DEVICE.type == "cuda" else "cpu"
    return "cuda" if _gpu_total_gb() >= config.GPU_ROOMY_GB else "cpu"


class ModelRegistry:
    """
    Holds VAD + ECAPA (diarizer), the ASR model and the emotion engine.

    Speaker centroids, ASR prompt history and the emotion engine's dialogue history are
    per-conversation state, so only one live session or batch job may use them at a time:
    callers take `busy` (non-blocking) for the whole job.
    """

    def __init__(self):
        self.state = "idle"            # idle | loading | ready | error
        self.error: Optional[str] = None
        self.step: Optional[str] = None
        self.diarizer = None
        self.asr = None
        self.emotion = None
        self.emotion_device = None
        # Speaker embedding, ASR and WavLM features run concurrently for each utterance.
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stage",
                                       initializer=use_model_threads)
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
            from .asr import ASR

            if not config.MODEL_CHECKPOINT_DIR.is_dir():
                raise FileNotFoundError(
                    f"Emotion model directory not found: {config.MODEL_CHECKPOINT_DIR}. "
                    "Extract prj4_emotion_model.tar.gz into backend/model_bundle first."
                )
            self.step = "VAD + speaker model"
            self.diarizer = DiarizationEngine.load_models(str(config.SPEAKER_MODEL_DIR))
            self.step = f"speech recognition ({config.WHISPER_MODEL})"
            self.asr = ASR(config.WHISPER_MODEL, DEVICE.type, engine=config.ASR_ENGINE,
                           language=config.ASR_LANGUAGE, beam_size=config.ASR_BEAM_SIZE)
            self.step = "emotion model"
            self.emotion_device = _emotion_device()
            on_gpu = self.emotion_device == "cuda"
            self.emotion = EmotionContextEngine(
                ckpt_dir=str(config.MODEL_CHECKPOINT_DIR), device=self.emotion_device,
                audio_fp16=on_gpu, text_fp16=on_gpu,
            )
            use_model_threads()
            self.step = "warming up"
            self._warm_up()
            self.step = None
            self.state = "ready"
            LOG.info("All models loaded (emotion model on %s, %d CPU threads)",
                     self.emotion_device, torch.get_num_threads())
        except Exception as exc:  # surfaced through /api/status
            LOG.exception("Model loading failed")
            self.step = None
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"

    def _warm_up(self):
        """Run every model once so the first real turn isn't slowed by lazy init (~9 s -> ~2 s)."""
        import numpy as np
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(24000) * 0.05).astype(np.float32)
        try:
            jobs = [self.pool.submit(self.diarizer.embed, torch.from_numpy(audio)),
                    self.pool.submit(self.asr.transcribe, audio),
                    self.pool.submit(self.emotion.audio_features, audio)]
            feats = jobs[2].result()
            jobs[0].result(), jobs[1].result()
            from .pipeline import measure_voice
            measure_voice(audio)  # librosa JIT-compiles yin/rms on first use (seconds)
            self.emotion.update("warmup", "Hello there.", audio_feats=feats, commit=False)
        except Exception as exc:  # warm-up is best effort
            LOG.warning("Warm-up failed: %s", exc)
        finally:
            self.emotion.reset()
            self.asr.reset()

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
        self.asr.reset()

    def status(self) -> dict:
        return {
            "models": self.state,
            "loading_step": self.step,
            "error": self.error,
            "device": DEVICE.type,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "gpu_memory_gb": round(_gpu_total_gb(), 1),
            "whisper_model": config.WHISPER_MODEL,
            "asr_engine": self.asr.engine if self.asr else None,
            "emotion_device": self.emotion_device,
            "busy_with": self.busy_with,
        }


registry = ModelRegistry()
