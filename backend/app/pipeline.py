"""Per-utterance analysis (diarize -> Whisper -> emotion) and JSON serialisation."""
import io
import threading
import uuid
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import soundfile as sf

from . import config
from .models import DEVICE, ModelRegistry


# --------------------------------------------------------------------------- clips
class ClipStore:
    """Keeps recent utterance WAVs in memory so the frontend can play them by URL."""

    def __init__(self, limit: int):
        self._limit = limit
        self._clips: "OrderedDict[str, bytes]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, wav: bytes) -> str:
        clip_id = uuid.uuid4().hex
        with self._lock:
            self._clips[clip_id] = wav
            while len(self._clips) > self._limit:
                self._clips.popitem(last=False)
        return clip_id

    def get(self, clip_id: str) -> Optional[bytes]:
        with self._lock:
            return self._clips.get(clip_id)


clips = ClipStore(config.MAX_CLIPS)


def to_wav_bytes(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- analysis
def classify(segment, audio_np: np.ndarray, models: ModelRegistry, transcript: Optional[str] = None):
    """Transcribe (unless a transcript is given) and run the context-aware emotion model."""
    if transcript is None:
        out = models.whisper.transcribe(audio_np, fp16=(DEVICE.type == "cuda"), verbose=False)
        transcript = out.get("text", "").strip()
    segment.transcript = transcript or "[No speech transcribed]"
    result = models.emotion.update(
        speaker=str(segment.speaker_id), text=segment.transcript, audio=audio_np, commit=True,
    )
    segment.emotion = result["emotion"]
    segment.emotion_confidence = float(result["confidence"])
    segment.emotion_distribution = result["probs"]
    segment.sentiment = result.get("sentiment")
    segment.tone = result.get("tone")
    return segment


def analyze_utterance(audio_tensor, turn_id, start_sec, end_sec, models: ModelRegistry):
    """Live path: one VAD-finalised utterance -> fully analysed segment."""
    segment = models.diarizer.process_utterance(
        audio_tensor=audio_tensor, turn_id=turn_id, start_sec=start_sec, end_sec=end_sec,
    )
    audio_np = segment.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return classify(segment, audio_np, models)


def segment_to_dict(segment) -> Dict:
    audio_np = segment.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
    clip_id = clips.put(to_wav_bytes(audio_np))
    return {
        "turn_id": segment.turn_id,
        "speaker_id": segment.speaker_id,
        "start_sec": round(float(segment.start_sec), 3),
        "end_sec": round(float(segment.end_sec), 3),
        "similarity": round(float(segment.similarity), 4),
        "is_new_speaker": bool(segment.is_new_speaker),
        "transcript": segment.transcript,
        "emotion": segment.emotion,
        "confidence": float(segment.emotion_confidence),
        "distribution": segment.emotion_distribution or {},
        "sentiment": getattr(segment, "sentiment", None),
        "tone": getattr(segment, "tone", None),
        "audio_url": f"/api/clips/{clip_id}.wav",
    }
