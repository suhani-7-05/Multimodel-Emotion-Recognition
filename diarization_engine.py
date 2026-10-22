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
        # 1. Silero VAD v5
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        vad_model.eval()

        # 2. SpeechBrain ECAPA-TDNN (with COPY_SKIP_CACHE for Windows)
        speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            local_strategy=LocalStrategy.COPY_SKIP_CACHE
        )
        speaker_model.eval()

        return cls(vad_model, speaker_model)


    def process_utterance(
        self,
        audio_tensor: torch.Tensor,
        turn_id: int,
        start_sec: float,
        end_sec: float,
    ) -> DiarizedSegment:
        """Assign a speaker to one already-segmented utterance while retaining speaker history."""
        audio_tensor = audio_tensor.detach().float().cpu().flatten()
        duration = audio_tensor.numel() / SAMPLE_RATE
        if duration < STREAMING_MIN_SPEAKER_DURATION:
            # Short/noisy turns may be matched to an existing speaker, but they
            # must never create a new ID or update a centroid.
            matched_id, similarity = self.speaker_manager.identify_existing(
                audio_tensor, self.speaker_model
            )
            speaker_id = matched_id or "Unknown"
            is_new = False
            similarity = float(similarity if similarity >= 0 else 0.0)
        else:
            speaker_id, similarity, is_new = self.speaker_manager.identify_or_register(
                audio_tensor, self.speaker_model
            )
        return DiarizedSegment(
            turn_id=turn_id,
            speaker_id=speaker_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            audio=audio_tensor,
            similarity=float(similarity),
            is_new_speaker=bool(is_new),
        )

    def process_full_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> List[DiarizedSegment]:
        """
        Segments a complete audio recording (e.g. from an uploaded WAV file)
        into speech utterances using Silero VAD, then diarizes each segment
        with ECAPA-TDNN.
        """
        if sample_rate != SAMPLE_RATE:
            # Resampling should be handled before or via librosa/torchaudio
            pass

        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.float()

        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)  # mono

        total_samples = len(audio_tensor)
        silence_limit_frames = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        min_speech_frames = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK_SIZE)

        segments: List[DiarizedSegment] = []
        is_speaking = False
        speech_chunks = []
        silence_counter = 0
        current_start_frame = 0
        frame_idx = 0

        self.vad_model.reset_states()

        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = audio_tensor[i:i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                # Pad last frame if needed
                chunk = F.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

            with torch.no_grad():
                prob = self.vad_model(chunk, SAMPLE_RATE).item()

            if prob > VAD_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    current_start_frame = frame_idx
                speech_chunks.append(chunk)
                silence_counter = 0
            elif is_speaking:
                speech_chunks.append(chunk)
                silence_counter += 1

                if silence_counter > silence_limit_frames:
                    is_speaking = False
                    speech_frames_count = len(speech_chunks) - silence_counter
                    if speech_frames_count >= min_speech_frames:
                        segment_audio = torch.cat(speech_chunks[:speech_frames_count])
                        start_sec = (current_start_frame * CHUNK_SIZE) / SAMPLE_RATE
                        end_sec = start_sec + (len(segment_audio) / SAMPLE_RATE)

                        spk_id, sim, is_new = self.speaker_manager.identify_or_register(
                            segment_audio, self.speaker_model
                        )

                        segments.append(DiarizedSegment(
                            turn_id=len(segments) + 1,
                            speaker_id=spk_id,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            audio=segment_audio,
                            similarity=sim,
                            is_new_speaker=is_new
                        ))

                    speech_chunks = []
                    silence_counter = 0
                    self.vad_model.reset_states()

            frame_idx += 1

        # Check if an ongoing utterance reached the end of the file
        if is_speaking:
            speech_frames_count = len(speech_chunks) - silence_counter
            if speech_frames_count >= min_speech_frames:
                segment_audio = torch.cat(speech_chunks[:speech_frames_count])
                start_sec = (current_start_frame * CHUNK_SIZE) / SAMPLE_RATE
                end_sec = start_sec + (len(segment_audio) / SAMPLE_RATE)
                spk_id, sim, is_new = self.speaker_manager.identify_or_register(
                    segment_audio, self.speaker_model
                )
                segments.append(DiarizedSegment(
                    turn_id=len(segments) + 1,
                    speaker_id=spk_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    audio=segment_audio,
                    similarity=sim,
                    is_new_speaker=is_new
                ))

        # Fallback: if VAD found no distinct segments (e.g. short audio or low volume),
        # return the full audio as one segment if longer than min duration
        if not segments and (total_samples / SAMPLE_RATE) >= MIN_SPEECH_DURATION:
            spk_id, sim, is_new = self.speaker_manager.identify_or_register(
                audio_tensor, self.speaker_model
            )
            segments.append(DiarizedSegment(
                turn_id=1,
                speaker_id=spk_id,
                start_sec=0.0,
                end_sec=total_samples / SAMPLE_RATE,
                audio=audio_tensor,
                similarity=sim,
                is_new_speaker=is_new
            ))

        return segments