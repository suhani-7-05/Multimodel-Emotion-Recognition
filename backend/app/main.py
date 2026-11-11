"""
PRJ4 backend API.

Run from the backend/ folder:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
import io
import logging
from collections import Counter
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from . import config
from .interpreter import interpreter
from .live import run_live_session
from .models import registry
from .pipeline import Conversation, analyze_utterance, clips, segment_to_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if config.PRELOAD_MODELS:
        registry.load_async()
    yield


app = FastAPI(title="PRJ4 Emotion Recognition API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    try:
        import sounddevice  # noqa: F401
        server_mic = True
    except Exception:
        server_mic = False
    return {**registry.status(), "server_mic_available": server_mic, "llm": interpreter.status()}


@app.get("/api/config")
def get_config():
    return {"emotions": config.TARGET_EMOTIONS, "default_settings": config.DEFAULT_SETTINGS}


@app.post("/api/models/load")
def load_models():
    registry.load_async()
    return registry.status()


@app.get("/api/devices")
def devices():
    try:
        from audio_capture import list_input_devices
        return {"devices": list_input_devices()}
    except Exception as exc:
        raise HTTPException(503, f"Server microphone unavailable: {type(exc).__name__}: {exc}")


@app.get("/api/clips/{clip_id}.wav")
def get_clip(clip_id: str):
    wav = clips.get(clip_id)
    if wav is None:
        raise HTTPException(404, "Clip expired or not found")
    return Response(wav, media_type="audio/wav")


@app.post("/api/batch")
def batch_analysis(
    file: UploadFile = File(...),
    vad_threshold: float = Form(config.DEFAULT_SETTINGS["vad_threshold"]),
    silence_duration: float = Form(config.DEFAULT_SETTINGS["silence_duration"]),
    min_speech_duration: float = Form(config.DEFAULT_SETTINGS["min_speech_duration"]),
    speaker_threshold: float = Form(config.DEFAULT_SETTINGS["speaker_threshold"]),
):
    """Whole-file workflow. Uses the same segmenter and per-turn pipeline as live mode."""
    import librosa

    if not registry.ready:
        raise HTTPException(409, "Models are not loaded yet.")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if not registry.acquire("batch"):
        raise HTTPException(409, f"Pipeline busy ({registry.busy_with}). Try again when it finishes.")
    try:
        try:
            audio, _ = librosa.load(io.BytesIO(data), sr=config.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise HTTPException(400, f"Could not decode audio: {type(exc).__name__}: {exc}")
        registry.reset_conversation(speaker_threshold)
        utterances = registry.diarizer.segment_full_audio(
            audio, threshold=vad_threshold, silence_duration=silence_duration,
            min_speech_duration=min_speech_duration,
        )
        conv = Conversation()
        results = [
            segment_to_dict(analyze_utterance(utt, i + 1, start, end, registry, conv))
            for i, (utt, start, end) in enumerate(utterances)
        ]
        counts = Counter(r["emotion"] for r in results)
        return {
            "duration_sec": round(len(audio) / config.SAMPLE_RATE, 2),
            "speakers": len({r["speaker_id"] for r in results}),
            "segments": results,
            "emotion_counts": {e: counts.get(e, 0) for e in config.TARGET_EMOTIONS},
        }
    finally:
        registry.release()


class InterpretRequest(BaseModel):
    turns: List[Dict]
    target: Optional[int] = None


@app.post("/api/interpret")
def interpret(req: InterpretRequest):
    """Plain-language explanation of one turn, given the turns before it (stateless)."""
    try:
        return interpreter.interpret(req.turns, req.target)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Interpreter failed: {type(exc).__name__}: {exc}")


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await run_live_session(ws)
