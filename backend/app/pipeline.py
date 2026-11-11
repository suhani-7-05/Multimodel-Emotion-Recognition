"""
Per-utterance analysis and JSON serialisation.

    utterance audio --+--> ECAPA speaker embedding ----+
                      +--> Whisper transcript ---------+--> RoBERTa+WavLM fusion --> segment
                      +--> WavLM features + prosody ---+

The three independent stages run concurrently; only the fusion model waits for all of
them (it needs the speaker, the words and the voice features).
"""
import io
import math
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from . import config
from .models import ModelRegistry, use_model_threads


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


# --------------------------------------------------------------------------- prosody
def measure_voice(audio: np.ndarray) -> Dict:
    """Loudness (dBFS) and pitch movement (semitone std over voiced frames). Cheap."""
    import librosa

    sr = config.SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    out = {"loudness_db": round(20 * math.log10(rms + 1e-9), 1), "pitch_var_st": None}
    if audio.size < sr // 2:
        return out
    hop = 256
    f0 = librosa.yin(audio, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    frame_rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop)[0]
    n = min(len(f0), len(frame_rms))
    f0, frame_rms = f0[:n], frame_rms[:n]
    voiced = frame_rms > np.percentile(frame_rms, 40)
    if voiced.sum() >= 10:
        semis = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
        out["pitch_var_st"] = round(float(np.std(semis)), 2)
    return out


class Conversation:
    """Per-conversation state the models don't hold: each speaker's usual voice."""

    def __init__(self):
        self._hist: Dict[str, List[Dict]] = defaultdict(list)

    def describe(self, speaker: str, p: Dict) -> List[str]:
        """Plain cues comparing this turn with the speaker's earlier turns (needs >= 2)."""
        prev = self._hist[speaker]
        cues: List[str] = []
        if len(prev) >= 2:
            loud = np.mean([q["loudness_db"] for q in prev])
            if p["loudness_db"] > loud + 4:
                cues.append("louder than this speaker's usual")
            elif p["loudness_db"] < loud - 4:
                cues.append("quieter than this speaker's usual")
            rates = [q["words_per_sec"] for q in prev if q["words_per_sec"]]
            if rates and p["words_per_sec"]:
                if p["words_per_sec"] > 1.3 * np.mean(rates):
                    cues.append("faster speech than usual")
                elif p["words_per_sec"] < 0.75 * np.mean(rates):
                    cues.append("slower speech than usual")
            pitches = [q["pitch_var_st"] for q in prev if q["pitch_var_st"]]
            if pitches and p["pitch_var_st"]:
                if p["pitch_var_st"] < 0.65 * np.mean(pitches):
                    cues.append("flatter, more monotone voice than usual")
                elif p["pitch_var_st"] > 1.4 * np.mean(pitches):
                    cues.append("more animated pitch than usual")
        self._hist[speaker].append(p)
        return cues


# --------------------------------------------------------------------------- analysis
def _timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - t) * 1000


def _voice_job(models: ModelRegistry, audio_np: np.ndarray):
    feats = models.emotion.audio_features(audio_np)
    return feats, measure_voice(audio_np)


def analyze_utterance(audio_tensor, turn_id, start_sec, end_sec,
                      models: ModelRegistry, conv: Conversation):
    """One VAD-finalised utterance -> fully analysed segment (live and batch path)."""
    use_model_threads()  # this thread runs the fusion model (RoBERTa) below
    t0 = time.perf_counter()
    audio_tensor = audio_tensor.detach().float().cpu().flatten()
    audio_np = audio_tensor.numpy().astype(np.float32)
    duration = audio_np.size / config.SAMPLE_RATE

    f_spk = models.pool.submit(_timed, models.diarizer.embed, audio_tensor)
    f_asr = models.pool.submit(_timed, models.asr.transcribe, audio_np)
    f_voice = models.pool.submit(_timed, _voice_job, models, audio_np)

    emb, spk_ms = f_spk.result()
    speaker_id, sim, is_new, uncertain = models.diarizer.assign(emb, duration)
    (transcript, asr_info), asr_ms = f_asr.result()
    (feats, voice), voice_ms = f_voice.result()

    from diarization_engine import DiarizedSegment
    segment = DiarizedSegment(
        turn_id=turn_id, speaker_id=speaker_id, start_sec=float(start_sec), end_sec=float(end_sec),
        audio=audio_tensor, similarity=float(sim), is_new_speaker=bool(is_new),
        speaker_uncertain=bool(uncertain),
    )
    segment.transcript = transcript or "[No clear speech]"

    t = time.perf_counter()
    result = models.emotion.update(
        speaker=str(speaker_id), text=transcript, audio_feats=feats, commit=True,
    )
    emo_ms = (time.perf_counter() - t) * 1000

    words = len(transcript.split())
    prosody = {**voice, "duration_sec": round(duration, 2),
               "words_per_sec": round(words / duration, 2) if words and duration > 0 else None}
    segment.emotion = result["emotion"]
    segment.emotion_confidence = float(result["confidence"])
    segment.emotion_distribution = result["probs"]
    segment.extra = {
        "sentiment": result.get("sentiment"),
        "tone": result.get("tone"),
        "text_probs": result.get("text_probs"),
        "audio_probs": result.get("audio_probs"),
        "prosody": prosody,
        "cues": conv.describe(speaker_id, prosody),
        "asr": asr_info,
        "timings_ms": {
            "speaker": round(spk_ms), "asr": round(asr_ms), "voice": round(voice_ms),
            "emotion": round(emo_ms), "total": round((time.perf_counter() - t0) * 1000),
        },
    }
    return segment


def segment_to_dict(segment) -> Dict:
    clip_id = clips.put(to_wav_bytes(segment.audio.detach().cpu().numpy()))
    extra = getattr(segment, "extra", {}) or {}
    return {
        "turn_id": segment.turn_id,
        "speaker_id": segment.speaker_id,
        "speaker_uncertain": bool(getattr(segment, "speaker_uncertain", False)),
        "start_sec": round(float(segment.start_sec), 3),
        "end_sec": round(float(segment.end_sec), 3),
        "similarity": round(float(segment.similarity), 4),
        "is_new_speaker": bool(segment.is_new_speaker),
        "transcript": segment.transcript,
        "emotion": segment.emotion,
        "confidence": float(segment.emotion_confidence),
        "distribution": segment.emotion_distribution or {},
        **extra,
        "audio_url": f"/api/clips/{clip_id}.wav",
    }
