import torch
import torch.nn.functional as F
import numpy as np
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

    def identify_or_register(self, audio_tensor: torch.Tensor, speaker_model: EncoderClassifier) -> Tuple[str, float, bool]:
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():
            emb = speaker_model.encode_batch(audio_tensor).squeeze()
            emb = emb / (torch.norm(emb, p=2) + 1e-8)

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
