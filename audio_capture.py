"""
Live-capture plumbing for the PRJ4 real-time emotion pipeline.

Why this module exists
----------------------
The Streamlit script thread used to do three jobs at once: drain the WebRTC
queue, run Silero VAD, and re-render the whole results page (including
re-encoding every utterance to WAV). Whenever rendering stalled the capture
loop, frames piled up or were dropped, and every dropped frame silently
deleted samples from the waveform -> clicks / "robotic" speech.

Here capture runs on its own thread and never touches Streamlit:

    audio source -> [capture thread] resample(16 kHz mono) -> VAD segmenter
                                   -> work_q -> [worker thread] ASR + emotion
                                   -> result_q -> Streamlit (polls, renders)
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Dict, List, Optional

import av
import numpy as np

SAMPLE_RATE = 16000
LOG = logging.getLogger("audio_capture")


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def block_to_frame(block: np.ndarray, rate: int) -> av.AudioFrame:
    """Wrap a mono float32 block (e.g. from sounddevice) in a PyAV frame."""
    arr = np.ascontiguousarray(block, dtype=np.float32).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(arr, format="flt", layout="mono")
    frame.sample_rate = int(rate)
    return frame


class StreamResampler:
    """
    ONE persistent PyAV resampler for the whole session -> mono float32 @ 16 kHz.

    Used for both the local microphone and the WebRTC path so there is exactly
    one place where sample-rate conversion happens.

    Gap awareness: WebRTC frames carry an RTP-based `pts`. If frames are
    missing (dropped packets, queue overflow, Opus DTX), the resampled stream
    would otherwise just lose that time and splice two pieces of speech
    together. We insert the equivalent amount of silence instead, so the
    timeline and timestamps stay correct and the splice becomes a pause.
    """

    def __init__(self, target_rate: int = SAMPLE_RATE, max_gap_sec: float = 5.0):
        self.target_rate = int(target_rate)
        self.max_gap_sec = float(max_gap_sec)
        self._rs = self._new_resampler()
        self._next_pts: Optional[int] = None
        self.source_rate: Optional[int] = None
        self.frames = 0
        self.gap_events = 0
        self.gap_ms = 0.0

    def _new_resampler(self) -> av.AudioResampler:
        return av.AudioResampler(format="fltp", layout="mono", rate=self.target_rate)

    @staticmethod
    def _collect(frames) -> List[np.ndarray]:
        if frames is None:
            return []
        if not isinstance(frames, (list, tuple)):  # older PyAV returned one frame
            frames = [frames]
        out = []
        for f in frames:
            a = f.to_ndarray().reshape(-1)  # fltp mono -> (1, n)
            if a.size:
                out.append(a.astype(np.float32, copy=False))
        return out

    def process(self, frame: av.AudioFrame) -> np.ndarray:
        rate = int(frame.sample_rate or 0)
        if rate <= 0:
            raise ValueError(
                "Audio frame has no valid sample_rate; refusing to assume one "
                "(a wrong guess changes pitch/speed)."
            )
        self.source_rate = rate
        self.frames += 1
        pieces: List[np.ndarray] = []

        pts = frame.pts
        if pts is not None:
            if self._next_pts is not None:
                # modular diff so RTP timestamp wrap-around is harmless
                diff = (int(pts) - self._next_pts + 2**31) % 2**32 - 2**31
                if rate * 0.005 < diff <= int(self.max_gap_sec * rate):
                    # Emit what the filter still holds from BEFORE the gap, then
                    # the silence, then restart the filter cleanly so old and new
                    # audio never get mixed across the hole.
                    try:
                        pieces.extend(self._collect(self._rs.resample(None)))
                    except Exception:
                        pass
                    self._rs = self._new_resampler()
                    n = int(round(diff * self.target_rate / rate))
                    pieces.append(np.zeros(n, dtype=np.float32))
                    self.gap_events += 1
                    self.gap_ms += 1000.0 * diff / rate
            self._next_pts = int(pts) + int(frame.samples)

        pieces.extend(self._collect(self._rs.resample(frame)))
        if not pieces:
            return np.empty(0, dtype=np.float32)
        return np.clip(np.concatenate(pieces), -1.0, 1.0)

    def flush(self) -> np.ndarray:
        """Samples still buffered inside the resampler at end of stream."""
        try:
            pieces = self._collect(self._rs.resample(None))
        except Exception:
            pieces = []
        if not pieces:
            return np.empty(0, dtype=np.float32)
        return np.clip(np.concatenate(pieces), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Local microphone (same idea as the original live_speaker_tracker.py)
# --------------------------------------------------------------------------- #
def list_input_devices() -> List[Dict]:
    import sounddevice as sd

    devices = []
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            api = hostapis[dev["hostapi"]]["name"]
            devices.append({"index": idx, "label": f"{idx} · {dev['name']} ({api})"})
    return devices


class SoundDeviceSource:
    """
    Direct PortAudio capture - no browser, no WebRTC, no Opus.

    The device is opened at its *native* rate and we resample ourselves with
    PyAV, instead of asking Windows/MME to convert to 16 kHz (its converter is
    low quality). The PortAudio callback does nothing except copy the block into
    a queue, and 'high' latency gives PortAudio big buffers so Python hiccups
    don't cause input overflows.
    """

    def __init__(self, device: Optional[int] = None):
        import sounddevice as sd

        info = sd.query_devices(device, "input")
        self.device_name: str = info["name"]
        self.rate = int(round(info["default_samplerate"]))
        self.overflows = 0
        self.dropped_blocks = 0
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1000)
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            latency="high",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status and getattr(status, "input_overflow", False):
            self.overflows += 1
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_blocks += 1

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def pull(self, timeout: float):
        blocks = [self._q.get(timeout=timeout)]  # raises queue.Empty on timeout
        while len(blocks) < 128:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return [(np.concatenate(blocks), self.rate)]

    def stats(self) -> Dict:
        return {"overflows": self.overflows, "dropped_blocks": self.dropped_blocks}


# --------------------------------------------------------------------------- #
# Live session: capture thread + inference worker
# --------------------------------------------------------------------------- #
class LiveSession:
    """
    pull(timeout) -> list of av.AudioFrame or (ndarray, rate) tuples.
                     May raise queue.Empty.
    analyze(audio, start_sec, end_sec, turn_id) -> DiarizedSegment (slow).
    """

    def __init__(
        self,
        pull: Callable[[float], list],
        segmenter,
        analyze: Callable,
        *,
        on_close: Optional[Callable[[], None]] = None,
        extra_stats: Optional[Callable[[], Dict]] = None,
        max_pending: int = 20,
        debug_wav_path: Optional[str] = None,
    ):
        self._pull = pull
        self.segmenter = segmenter
        self._analyze = analyze
        self._on_close = on_close
        self._extra_stats = extra_stats
        self._debug_wav_path = debug_wav_path

        self.resampler = StreamResampler()
        self.work_q: "queue.Queue" = queue.Queue(maxsize=max_pending)
        self.result_q: "queue.Queue" = queue.Queue()
        self.error_q: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()

        self.turn_id = 0
        self.audio_samples = 0
        self.peak = 0.0
        self.clipped_samples = 0
        self.dropped_utterances = 0

        self._capture = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        self._worker = threading.Thread(target=self._worker_loop, name="inference", daemon=True)

    # ---- lifecycle ------------------------------------------------------- #
    def start(self):
        self._worker.start()
        self._capture.start()

    @property
    def running(self) -> bool:
        return self._capture.is_alive() or self._worker.is_alive()

    def stop(self, timeout: Optional[float] = None):
        """Stop the source, finish queued utterances, join both threads."""
        if self._on_close is not None:
            self._on_close()          # no new audio after this point
        self._stop.set()              # capture drains what's left, then exits
        self._capture.join(timeout)
        self._worker.join(timeout)    # worker exits after the None sentinel

    # ---- polling from the UI -------------------------------------------- #
    def drain_results(self) -> List[Dict]:
        out = []
        while True:
            try:
                out.append(self.result_q.get_nowait())
            except queue.Empty:
                return out

    def drain_errors(self) -> List[str]:
        out = []
        while True:
            try:
                out.append(self.error_q.get_nowait())
            except queue.Empty:
                return out

    def snapshot(self) -> Dict:
        snap = {
            "audio_sec": self.audio_samples / SAMPLE_RATE,
            "source_rate": self.resampler.source_rate,
            "peak": self.peak,
            "clipped_samples": self.clipped_samples,
            "gap_events": self.resampler.gap_events,
            "gap_ms": self.resampler.gap_ms,
            "pending": self.work_q.qsize(),
            "dropped_utterances": self.dropped_utterances,
        }
        if self._extra_stats is not None:
            try:
                snap.update(self._extra_stats())
            except Exception:
                pass
        return snap

    # ---- threads --------------------------------------------------------- #
    def _error(self, msg: str):
        LOG.error(msg)
        self.error_q.put(msg)

    def _capture_loop(self):
        wav = None
        try:
            if self._debug_wav_path:
                import soundfile as sf

                wav = sf.SoundFile(
                    self._debug_wav_path, "w", samplerate=SAMPLE_RATE,
                    channels=1, subtype="PCM_16",
                )
            while True:
                try:
                    items = self._pull(0.2)
                except queue.Empty:
                    items = []
                except Exception as exc:
                    self._error(f"Audio source failed: {type(exc).__name__}: {exc}")
                    break
                if not items:
                    if self._stop.is_set():
                        break
                    continue
                for item in items:
                    self._ingest(item, wav)

            self._push_audio(self.resampler.flush(), wav)
            for utt in self.segmenter.flush():
                self._enqueue(utt, final=True)
        except Exception as exc:
            self._error(f"Capture thread crashed: {type(exc).__name__}: {exc}")
        finally:
            if wav is not None:
                wav.close()
            self.work_q.put(None)  # tell the worker to finish and exit

    def _ingest(self, item, wav):
        frame = item if isinstance(item, av.AudioFrame) else block_to_frame(*item)
        self._push_audio(self.resampler.process(frame), wav)

    def _push_audio(self, audio: np.ndarray, wav):
        if audio.size == 0:
            return
        self.audio_samples += audio.size
        self.peak = max(self.peak, float(np.max(np.abs(audio))))
        self.clipped_samples += int(np.count_nonzero(np.abs(audio) >= 0.999))
        if wav is not None:
            wav.write(audio)
        for utt in self.segmenter.push_audio(audio):
            self._enqueue(utt)

    def _enqueue(self, utt, final: bool = False):
        audio, start_sec, end_sec = utt
        self.turn_id += 1
        item = (audio, start_sec, end_sec, self.turn_id)
        try:
            if final:
                self.work_q.put(item, timeout=5)
            else:
                self.work_q.put_nowait(item)
        except queue.Full:
            self.dropped_utterances += 1
            self._error(
                f"Inference queue full - utterance {self.turn_id} dropped. "
                "Use a smaller Whisper model or faster hardware."
            )

    def _worker_loop(self):
        while True:
            item = self.work_q.get()
            try:
                if item is None:
                    return
                audio, start_sec, end_sec, turn_id = item
                segment = self._analyze(audio, start_sec, end_sec, turn_id)
                self.result_q.put({"ok": True, "segment": segment})
            except Exception as exc:
                self.result_q.put({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "turn_id": item[3] if item else None,
                })
            finally:
                self.work_q.task_done()
