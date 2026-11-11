import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

SAMPLE_RATE = 16000
CHUNK_SIZE = 512              # 32ms frames @ 16kHz (Silero VAD standard)
VAD_THRESHOLD = 0.5           # Speech probability threshold
SILENCE_DURATION = 1.1        # Seconds of silence before finalizing a speech segment
MIN_SPEECH_DURATION = 0.6     # Discard noise/coughs under 0.6s
SIMILARITY_THRESHOLD = 0.38   # Cosine similarity threshold for ECAPA-TDNN speaker clustering
PREROLL_DURATION = 0.25       # Audio kept from BEFORE VAD fires (VAD lags the true onset)
TAIL_DURATION = 0.20          # Trailing audio kept after the last speech frame (natural decay)

@dataclass
class DiarizedSegment:
    turn_id: int
    speaker_id: str
    start_sec: float
    end_sec: float
    audio: torch.Tensor       # 1D float32 tensor @ 16kHz
    similarity: float
    is_new_speaker: bool
    transcript: str = ""      # Populated by Whisper ASR
    emotion: str = ""         # Populated by Emotion Model
    emotion_confidence: float = 0.0
    emotion_distribution: dict = None
    speaker_uncertain: bool = False   # short turn attributed to the nearest known voice

class SpeakerManager:
    """Tracks dynamic speaker centroids using SpeechBrain ECAPA-TDNN 192d embeddings."""
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.speakers = {}
        self.speaker_counter = 1

    def reset(self):
        self.speakers.clear()
        self.speaker_counter = 1

    def _embedding(self, audio_tensor: torch.Tensor, speaker_model: EncoderClassifier) -> torch.Tensor:
        """Create an L2-normalized ECAPA embedding for mono audio."""
        audio_tensor = audio_tensor.detach().float().flatten()
        if audio_tensor.numel() == 0:
            raise ValueError("Cannot create a speaker embedding from empty audio.")
        with torch.no_grad():
            emb = speaker_model.encode_batch(audio_tensor.unsqueeze(0)).squeeze()
            emb = emb / (torch.norm(emb, p=2) + 1e-8)
        return emb

    def nearest(self, emb: torch.Tensor) -> Tuple[Optional[str], float]:
        """Closest known speaker regardless of the threshold (None if nobody is registered)."""
        best_speaker, highest_sim = None, -1.0
        for spk_id, spk_data in self.speakers.items():
            sim = torch.dot(emb, spk_data["centroid"]).item()
            if sim > highest_sim:
                best_speaker, highest_sim = spk_id, sim
        return best_speaker, highest_sim

    def register_or_update(self, emb: torch.Tensor) -> Tuple[str, float, bool]:
        """Same decision as identify_or_register, from a precomputed embedding."""
        best_speaker, highest_sim = self.nearest(emb)
        if best_speaker is not None and highest_sim >= self.threshold:
            pool = self.speakers[best_speaker]["embeddings"]
            pool.append(emb)
            if len(pool) > 5:
                pool.pop(0)
            centroid = torch.stack(pool).mean(dim=0)
            self.speakers[best_speaker]["centroid"] = centroid / (torch.norm(centroid, p=2) + 1e-8)
            self.speakers[best_speaker]["count"] += 1
            return best_speaker, highest_sim, False
        new_id = f"Speaker {self.speaker_counter}"
        self.speaker_counter += 1
        self.speakers[new_id] = {"embeddings": [emb], "centroid": emb, "count": 1}
        return new_id, (highest_sim if highest_sim != -1.0 else 1.0), True

    def identify_existing(self, audio_tensor: torch.Tensor, speaker_model: EncoderClassifier) -> Tuple[Optional[str], float]:
        """Match a clip without registering a speaker or changing any centroid."""
        if not self.speakers:
            return None, -1.0
        emb = self._embedding(audio_tensor, speaker_model)
        best_speaker = None
        highest_sim = -1.0
        for spk_id, spk_data in self.speakers.items():
            sim = torch.dot(emb, spk_data["centroid"]).item()
            if sim > highest_sim:
                best_speaker, highest_sim = spk_id, sim
        if best_speaker is not None and highest_sim >= self.threshold:
            return best_speaker, highest_sim
        return None, highest_sim

    def identify_or_register(self, audio_tensor: torch.Tensor, speaker_model: EncoderClassifier) -> Tuple[str, float, bool]:
        emb = self._embedding(audio_tensor, speaker_model)

        best_speaker = None
        highest_sim = -1.0

        for spk_id, spk_data in self.speakers.items():
            sim = torch.dot(emb, spk_data["centroid"]).item()
            if sim > highest_sim:
                highest_sim = sim
                best_speaker = spk_id

        if highest_sim >= self.threshold and best_speaker is not None:
            # Update moving centroid with sliding pool of up to 5 embeddings
            spk_pool = self.speakers[best_speaker]["embeddings"]
            spk_pool.append(emb)
            if len(spk_pool) > 5:
                spk_pool.pop(0)

            new_centroid = torch.stack(spk_pool).mean(dim=0)
            self.speakers[best_speaker]["centroid"] = new_centroid / (torch.norm(new_centroid, p=2) + 1e-8)
            self.speakers[best_speaker]["count"] += 1
            return best_speaker, highest_sim, False
        else:
            new_id = f"Speaker {self.speaker_counter}"
            self.speaker_counter += 1
            self.speakers[new_id] = {
                "embeddings": [emb],
                "centroid": emb,
                "count": 1
            }
            return new_id, (highest_sim if highest_sim != -1.0 else 1.0), True


class StreamingVADSegmenter:
    """
    Incremental Silero VAD segmenter. Feed mono float32 audio sampled at 16 kHz.

    Utterances include a short PRE-ROLL (audio from before the VAD crossed its
    threshold) and a short TAIL. Without them the first consonant and the decay
    of the last word are chopped off, which sounds clipped/robotic when replayed
    and hurts both Whisper and the MFCC emotion features.
    """
    def __init__(
        self,
        vad_model,
        threshold: float = VAD_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
        min_speech_duration: float = MIN_SPEECH_DURATION,
        preroll_duration: float = PREROLL_DURATION,
        tail_duration: float = TAIL_DURATION,
    ):
        self.vad_model = vad_model
        self.threshold = float(threshold)
        self.silence_limit_frames = max(1, int(silence_duration * SAMPLE_RATE / CHUNK_SIZE))
        self.min_speech_frames = max(1, int(min_speech_duration * SAMPLE_RATE / CHUNK_SIZE))
        self.preroll_frames = max(0, int(preroll_duration * SAMPLE_RATE / CHUNK_SIZE))
        self.tail_frames = max(0, int(tail_duration * SAMPLE_RATE / CHUNK_SIZE))
        self.reset()

    def _clear_utterance(self):
        self.speech_chunks = []
        self.silence_counter = 0
        self.preroll_count = 0
        self.is_speaking = False
        self.preroll.clear()
        self.vad_model.reset_states()

    def drop_partial(self):
        """Discard a half-finished utterance (stream discontinuity) but keep the timeline."""
        self._clear_utterance()

    def reset(self):
        self.pending = np.empty(0, dtype=np.float32)
        self.preroll = deque(maxlen=self.preroll_frames)
        self.frame_idx = 0
        self.current_start_frame = 0
        self._clear_utterance()

    def _finalize(self):
        # speech_chunks = [pre-roll][speech ... last speech frame][trailing silence]
        speech_len = len(self.speech_chunks) - self.silence_counter
        active_frames = speech_len - self.preroll_count
        if active_frames < self.min_speech_frames:
            self._clear_utterance()
            return None

        keep = min(len(self.speech_chunks), speech_len + self.tail_frames)
        audio = torch.cat(self.speech_chunks[:keep]).detach().cpu().float()
        start_sec = self.current_start_frame * CHUNK_SIZE / SAMPLE_RATE
        end_sec = start_sec + len(audio) / SAMPLE_RATE
        self._clear_utterance()
        return audio, start_sec, end_sec

    def push_audio(self, audio: np.ndarray):
        """Return zero or more (audio_tensor, start_sec, end_sec) utterances."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            self.pending = np.concatenate((self.pending, audio))

        completed = []
        while self.pending.size >= CHUNK_SIZE:
            chunk_np = self.pending[:CHUNK_SIZE]
            self.pending = self.pending[CHUNK_SIZE:]
            chunk = torch.from_numpy(chunk_np.copy())

            with torch.no_grad():
                probability = float(self.vad_model(chunk, SAMPLE_RATE).item())

            if probability >= self.threshold:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.preroll_count = len(self.preroll)
                    self.current_start_frame = self.frame_idx - self.preroll_count
                    self.speech_chunks = list(self.preroll)
                    self.preroll.clear()
                self.speech_chunks.append(chunk)
                self.silence_counter = 0
            elif self.is_speaking:
                self.speech_chunks.append(chunk)
                self.silence_counter += 1
                if self.silence_counter > self.silence_limit_frames:
                    result = self._finalize()
                    if result is not None:
                        completed.append(result)
            else:
                self.preroll.append(chunk)

            self.frame_idx += 1

        return completed

    def flush(self):
        """Finalize speech still in progress when the user stops recording."""
        completed = []
        if self.pending.size:
            padded = np.pad(self.pending, (0, CHUNK_SIZE - self.pending.size))
            self.pending = np.empty(0, dtype=np.float32)
            completed.extend(self.push_audio(padded))
        if self.is_speaking:
            result = self._finalize()
            if result is not None:
                completed.append(result)
        return completed

# Short clips are retained in the timeline, but are not allowed to create or
# contaminate persistent speaker profiles. This affects streaming only.
STREAMING_MIN_SPEAKER_DURATION = 1.2


class DiarizationEngine:
    """Unified VAD & Speaker Diarization Engine for both batch audio files and live streams."""
    def __init__(self, vad_model, speaker_model):
        self.vad_model = vad_model
        self.speaker_model = speaker_model
        self.speaker_manager = SpeakerManager()

    @classmethod
    def load_models(cls, savedir: str = "tmpdir_spkrec"):
        # 1. Silero VAD. The pip package bundles the weights (works offline, e.g. on the
        #    college pod); torch.hub is only a fallback and needs internet the first time.
        try:
            from silero_vad import load_silero_vad
            vad_model = load_silero_vad()
        except ImportError:
            vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                force_reload=False, trust_repo=True,
            )
        vad_model.eval()

        # 2. SpeechBrain ECAPA-TDNN. If the folder already holds the model, load it from
        #    disk so no Hugging Face access is needed.
        from pathlib import Path
        local = Path(savedir) / "hyperparams.yaml"
        speaker_model = EncoderClassifier.from_hparams(
            source=str(Path(savedir).resolve()) if local.is_file() else "speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
        speaker_model.eval()
        return cls(vad_model, speaker_model)

    # ------------------------------------------------------------------ speaker assignment
    def embed(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """ECAPA embedding (the slow part). Thread-safe; can run in parallel with ASR."""
        return self.speaker_manager._embedding(audio_tensor, self.speaker_model)

    def assign(self, emb: torch.Tensor, duration_sec: float):
        """
        Decide the speaker for one utterance from its embedding.
        Returns (speaker_id, similarity, is_new, uncertain).

        Short turns (< STREAMING_MIN_SPEAKER_DURATION) give unreliable embeddings, so they
        never create a speaker or move a centroid. They are attributed to the nearest known
        voice and flagged uncertain when that voice is below the similarity threshold.
        """
        if duration_sec < STREAMING_MIN_SPEAKER_DURATION:
            nearest, sim = self.speaker_manager.nearest(emb)
            if nearest is None:
                return "Unknown", 0.0, False, True
            return nearest, max(sim, 0.0), False, sim < self.speaker_manager.threshold
        speaker_id, sim, is_new = self.speaker_manager.register_or_update(emb)
        return speaker_id, sim, is_new, False

    def process_utterance(self, audio_tensor: torch.Tensor, turn_id: int,
                          start_sec: float, end_sec: float) -> DiarizedSegment:
        """Assign a speaker to one already-segmented utterance while retaining speaker history."""
        audio_tensor = audio_tensor.detach().float().cpu().flatten()
        emb = self.embed(audio_tensor)
        speaker_id, sim, is_new, uncertain = self.assign(emb, audio_tensor.numel() / SAMPLE_RATE)
        return DiarizedSegment(
            turn_id=turn_id, speaker_id=speaker_id,
            start_sec=float(start_sec), end_sec=float(end_sec),
            audio=audio_tensor, similarity=float(sim),
            is_new_speaker=bool(is_new), speaker_uncertain=bool(uncertain),
        )

    # ------------------------------------------------------------------ whole files
    def segment_full_audio(self, audio_data, threshold: float = VAD_THRESHOLD,
                           silence_duration: float = SILENCE_DURATION,
                           min_speech_duration: float = MIN_SPEECH_DURATION):
        """
        Split a whole recording into utterances with the SAME streaming segmenter the live
        path uses (same pre-roll/tail, same settings), so a file and a live recording of the
        same speech produce the same turns. Returns [(audio_tensor, start_sec, end_sec)].
        """
        audio = audio_data.numpy() if isinstance(audio_data, torch.Tensor) else np.asarray(audio_data)
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        seg = StreamingVADSegmenter(self.vad_model, threshold=threshold,
                                    silence_duration=silence_duration,
                                    min_speech_duration=min_speech_duration)
        utterances = []
        step = SAMPLE_RATE  # feed 1 s at a time, exactly like a live stream
        for i in range(0, len(audio), step):
            utterances.extend(seg.push_audio(audio[i:i + step]))
        utterances.extend(seg.flush())
        if not utterances and len(audio) / SAMPLE_RATE >= min_speech_duration:
            # VAD found nothing distinct (very short or quiet file): keep it as one turn.
            utterances.append((torch.from_numpy(audio), 0.0, len(audio) / SAMPLE_RATE))
        return utterances

    def process_full_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE,
                           **segment_kwargs) -> List[DiarizedSegment]:
        """Segment a whole recording, then diarize every utterance in order (legacy helper)."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Resample to {SAMPLE_RATE} Hz first (got {sample_rate}).")
        return [
            self.process_utterance(audio, i + 1, start, end)
            for i, (audio, start, end) in enumerate(self.segment_full_audio(audio_data, **segment_kwargs))
        ]
