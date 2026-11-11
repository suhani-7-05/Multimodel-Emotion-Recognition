"""
Live sessions over a single WebSocket (/ws/live).

Client -> server
    {"type": "start", "source": "server" | "browser", "device": int | null,
     "sample_rate": int, "format": "f32" | "s16" (browser only),
     "settings": {...}, "debug_wav": bool}
    binary frames: mono PCM at `sample_rate` in `format`   (browser source only;
                   the UI sends 16 kHz int16 = 32 KB/s, small enough for a remote backend)
    {"type": "stop"}

Server -> client
    {"type": "started", "source_info": str, "settings": {...}}
    {"type": "stats", ...capture health, level, speaking...}   (4x a second)
    {"type": "segment", "segment": {...}}              (one per analysed utterance)
    {"type": "error", "message": str}
    {"type": "stopping"}  then  {"type": "stopped"}

The session ends when the client sends "stop" or disconnects. Utterances already
queued are still analysed before "stopped" is sent.
"""
import asyncio
import json
import logging
import queue
import time
from typing import Dict, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from . import config
from .models import registry
from .pipeline import Conversation, analyze_utterance, segment_to_dict

LOG = logging.getLogger("live")


class BrowserSource:
    """Raw PCM pushed by the browser over the WebSocket - no WebRTC, no Opus."""

    def __init__(self, rate: int, fmt: str = "f32"):
        self.rate = int(rate)
        self.fmt = fmt if fmt in ("f32", "s16") else "f32"
        self.dropped_blocks = 0
        self._closed = False
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)

    def push(self, data: bytes):
        if self._closed or len(data) < 4:
            return
        if self.fmt == "s16":
            block = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2").astype(np.float32) / 32768.0
        else:
            block = np.frombuffer(data[: len(data) - len(data) % 4], dtype="<f4").copy()
        try:
            self._q.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1

    def stop(self):
        self._closed = True

    def pull(self, timeout: float):
        blocks = [self._q.get(timeout=timeout)]  # raises queue.Empty on timeout
        while len(blocks) < 128:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return [(np.concatenate(blocks), self.rate)]

    def stats(self) -> Dict:
        return {"overflows": 0, "dropped_blocks": self.dropped_blocks}


def merge_settings(raw: Optional[dict]) -> dict:
    settings = dict(config.DEFAULT_SETTINGS)
    for key, value in (raw or {}).items():
        if key in settings and value is not None:
            settings[key] = float(value)
    return settings


class _Client:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.connected = True

    async def send(self, payload: dict):
        if not self.connected:
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            self.connected = False


async def _flush(client: _Client, session):
    for message in session.drain_results():
        if message["ok"]:
            await client.send({"type": "segment", "segment": segment_to_dict(message["segment"])})
        else:
            await client.send({
                "type": "error",
                "message": f"Turn {message.get('turn_id')}: {message['error']}",
            })
    for error in session.drain_errors():
        await client.send({"type": "error", "message": error})


async def run_live_session(ws: WebSocket):
    from audio_capture import LiveSession, SoundDeviceSource
    from diarization_engine import StreamingVADSegmenter

    await ws.accept()
    client = _Client(ws)
    source = session = recv_task = None
    acquired = False
    try:
        start = await ws.receive_json()
        if start.get("type") != "start":
            await client.send({"type": "error", "message": "First message must be 'start'."})
            return
        if not registry.ready:
            await client.send({"type": "error", "message": "Models are not loaded yet."})
            return
        if not registry.acquire("live"):
            await client.send({
                "type": "error",
                "message": f"Pipeline busy ({registry.busy_with}). Stop it first.",
            })
            return
        acquired = True

        settings = merge_settings(start.get("settings"))
        browser = start.get("source") == "browser"
        try:
            if browser:
                source = BrowserSource(int(start.get("sample_rate") or 48000), start.get("format", "f32"))
                info = f"Browser microphone ({source.fmt} PCM @ {source.rate} Hz)"
            else:
                device = start.get("device")
                source = SoundDeviceSource(None if device is None else int(device))
                source.start()
                info = f"{source.device_name} @ {source.rate} Hz"
        except Exception as exc:
            if source is not None:
                source.stop()
            await client.send({
                "type": "error",
                "message": f"Could not open microphone: {type(exc).__name__}: {exc}",
            })
            return

        registry.reset_conversation(settings["speaker_threshold"])
        segmenter = StreamingVADSegmenter(
            registry.diarizer.vad_model,
            threshold=settings["vad_threshold"],
            silence_duration=settings["silence_duration"],
            min_speech_duration=settings["min_speech_duration"],
        )

        conv = Conversation()

        def analyze(audio, start_sec, end_sec, turn_id):
            return analyze_utterance(audio, turn_id, start_sec, end_sec, registry, conv)

        session = LiveSession(
            source.pull, segmenter, analyze,
            on_close=source.stop, extra_stats=source.stats,
            debug_wav_path=str(config.DEBUG_WAV_PATH) if start.get("debug_wav") else None,
        )
        session.start()
        await client.send({"type": "started", "source_info": info, "settings": settings})

        stop_requested = asyncio.Event()

        async def receiver():
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        client.connected = False
                        break
                    if msg.get("bytes") is not None:
                        if browser:
                            source.push(msg["bytes"])
                    elif msg.get("text"):
                        try:
                            data = json.loads(msg["text"])
                        except ValueError:
                            continue
                        if data.get("type") == "stop":
                            break
            except (WebSocketDisconnect, RuntimeError):
                client.connected = False
            finally:
                stop_requested.set()

        recv_task = asyncio.create_task(receiver())
        last_stats = 0.0
        while not stop_requested.is_set() and session.running:
            await _flush(client, session)
            now = time.monotonic()
            if now - last_stats >= 0.25:
                last_stats = now
                await client.send({"type": "stats", "source_info": info, **session.snapshot()})
            try:
                await asyncio.wait_for(stop_requested.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

        await client.send({"type": "stopping"})
        await asyncio.to_thread(session.stop)   # finishes queued utterances
        await _flush(client, session)
        await client.send({"type": "stats", "source_info": info, **session.snapshot()})
        await client.send({"type": "stopped"})
    except WebSocketDisconnect:
        client.connected = False
    except Exception as exc:
        LOG.exception("Live session failed")
        await client.send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        if recv_task is not None and not recv_task.done():
            recv_task.cancel()
        if session is not None:
            if session.running:
                await asyncio.to_thread(session.stop)
        elif source is not None:
            source.stop()
        if acquired:
            registry.release()
        if client.connected:
            try:
                await ws.close()
            except Exception:
                pass
