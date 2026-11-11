"""
Speech recognition with guards against Whisper's known failure modes.

* faster-whisper (CTranslate2) by default - same Whisper weights, several times faster
  than the PyTorch reference implementation. Falls back to openai-whisper if missing.
* Language is fixed (no per-clip language detection, which misfires on short clips).
* The previous turns are passed as `initial_prompt`, so names and topic words stay
  consistent across the conversation.
* Segments Whisper itself flags as non-speech, and the stock hallucinations it emits on
  noise ("Thanks for watching!"), are dropped instead of being fed to the emotion model.
"""
import logging
import os
import re
import sys
import threading
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger("asr")

# Phrases Whisper produces on silence/noise (learned from YouTube subtitles).
_ALWAYS_HALLUCINATED = re.compile(
    r"^\W*(thanks? (you )?(so much )?for watching|please subscribe|like and subscribe|"
    r"subtitles? (by|from)|transcribed by|amara\.org|\[?music\]?|♪+|\.+)\W*$",
    re.I,
)
# Real people say these too, so drop them only when Whisper also suspects non-speech.
_SUSPICIOUS = re.compile(r"^\W*(thank you|you|bye|okay)\W*$", re.I)


def _add_cuda12_dlls():
    """
    Windows only: CTranslate2 (faster-whisper) is built against CUDA 12 and looks for
    cublas64_12.dll, while a CUDA 13 PyTorch only ships cuBLAS 13. Put the two CUDA 12
    cuBLAS DLLs in backend/models/cuda12 (or point CUDA12_DLL_DIR elsewhere).
    Linux needs nothing here: pip's nvidia-cublas-cu12 wheel is found automatically.
    """
    if sys.platform != "win32":
        return
    from . import BACKEND_DIR
    d = os.environ.get("CUDA12_DLL_DIR") or str(BACKEND_DIR / "models" / "cuda12")
    if os.path.isdir(d):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


class ASR:
    def __init__(self, model_name: str, device: str, engine: str = "auto",
                 language: Optional[str] = "en", beam_size: int = 5, prompt_turns: int = 2):
        self.language = language
        self.beam_size = beam_size
        self._recent: deque = deque(maxlen=prompt_turns)
        self._lock = threading.Lock()
        self.engine = None

        if engine in ("auto", "faster"):
            try:
                _add_cuda12_dlls()
                from faster_whisper import WhisperModel
                compute = "float16" if device == "cuda" else "int8"
                # model_name may be a size ("large-v3-turbo") or a local CTranslate2 folder.
                # "auto" never downloads (a stalled download would hang startup on a slow or
                # offline network); it falls back to openai-whisper's cached .pt instead.
                self._model = WhisperModel(model_name, device=device, compute_type=compute,
                                           local_files_only=(engine == "auto"))
                self.engine = "faster-whisper"
            except Exception as exc:
                if engine == "faster":
                    raise
                LOG.warning("faster-whisper unavailable (%s); using openai-whisper", exc)
        if self.engine is None:
            import whisper
            from pathlib import Path
            name = Path(model_name).name if Path(model_name).is_dir() else model_name
            # openai-whisper names large-v3-turbo just "turbo".
            name = {"large-v3-turbo": "turbo"}.get(name.replace("faster-whisper-", ""), name)
            self._model = whisper.load_model(name, device=device)
            self._fp16 = device == "cuda"
            self.engine = "openai-whisper"
        self.model_name = model_name
        LOG.info("ASR: %s %s on %s", self.engine, model_name, device)

    def reset(self):
        self._recent.clear()

    def _prompt(self) -> Optional[str]:
        text = " ".join(self._recent).strip()
        return text[-200:] if text else None

    def transcribe(self, audio: np.ndarray) -> Tuple[str, Dict]:
        """Returns (text, info). Empty text means 'no reliable speech'."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak_db = 20 * np.log10(float(np.max(np.abs(audio))) + 1e-9) if audio.size else -120.0
        prompt = self._prompt()
        kept, dropped = [], 0
        with self._lock:  # one decode at a time per model
            if self.engine == "faster-whisper":
                segments, _ = self._model.transcribe(
                    audio, language=self.language, beam_size=self.beam_size,
                    initial_prompt=prompt, condition_on_previous_text=False,
                    without_timestamps=True, vad_filter=False,
                )
                segs = [(s.text, s.no_speech_prob, s.avg_logprob, s.compression_ratio) for s in segments]
            else:
                out = self._model.transcribe(
                    audio, language=self.language, fp16=self._fp16, beam_size=self.beam_size,
                    initial_prompt=prompt, condition_on_previous_text=False, verbose=None,
                )
                segs = [(s["text"], s["no_speech_prob"], s["avg_logprob"], s["compression_ratio"])
                        for s in out.get("segments", [])]
        for text, no_speech, logprob, ratio in segs:
            text = text.strip()
            if not text:
                continue
            # Whisper's own silence rule + repetition loops + stock hallucinations.
            if ((no_speech > 0.6 and logprob < -1.0) or ratio > 2.4
                    or _ALWAYS_HALLUCINATED.match(text)
                    or (_SUSPICIOUS.match(text) and (no_speech > 0.3 or peak_db < -38))):
                dropped += 1
                continue
            kept.append(text)
        text = " ".join(kept).strip()
        if text:
            self._recent.append(text)
        return text, {"segments": len(segs), "dropped": dropped, "peak_db": round(float(peak_db), 1)}
