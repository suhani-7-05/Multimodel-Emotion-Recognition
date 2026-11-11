import sys
import queue
import time
import torch
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# Audio Configuration (Tuned for conversational rhythm & laptop mics)
SAMPLE_RATE = 16000
CHUNK_SIZE = 512              # 32ms per frame (Silero requirement)
VAD_THRESHOLD = 0.5           # Speech probability threshold
SILENCE_DURATION = 1.1        # WAITS LONGER: 1.1 seconds of silence before finalizing utterance
MIN_SPEECH_DURATION = 0.6     # 0.6s minimum speech to be valid
SIMILARITY_THRESHOLD = 0.38   # Cleanly separates same speaker (~0.41-0.65) from different speaker (~0.15-0.25)

print("Initializing models, please wait...")

# 1. Load Silero VAD
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
vad_model.eval()

# 2. Load SpeechBrain ECAPA-TDNN
speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir_spkrec",
    local_strategy=LocalStrategy.COPY_SKIP_CACHE
)

print("\n>>> Models loaded! Setting up Live Speaker Tracker... <<<")

class SpeakerManager:
    def __init__(self, threshold=SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.speakers = {}
        self.speaker_counter = 1

    def identify_or_register(self, audio_tensor):
        with torch.no_grad():
            emb = speaker_model.encode_batch(audio_tensor.unsqueeze(0)).squeeze()
            emb = emb / torch.norm(emb, p=2)

        best_speaker = None
        highest_sim = -1.0
        scores_debug = []

        for spk_id, spk_data in self.speakers.items():
            sim = torch.dot(emb, spk_data["centroid"]).item()
            scores_debug.append(f"{spk_id}: {sim:.2f}")
            if sim > highest_sim:
                highest_sim = sim
                best_speaker = spk_id

        debug_info = ", ".join(scores_debug) if scores_debug else "None"

        if highest_sim >= self.threshold and best_speaker is not None:
            spk_pool = self.speakers[best_speaker]["embeddings"]
            spk_pool.append(emb)
            if len(spk_pool) > 5:
                spk_pool.pop(0)

            new_centroid = torch.stack(spk_pool).mean(dim=0)
            self.speakers[best_speaker]["centroid"] = new_centroid / torch.norm(new_centroid, p=2)
            self.speakers[best_speaker]["count"] += 1
            return best_speaker, highest_sim, False, debug_info
        else:
            new_id = f"Speaker {self.speaker_counter}"
            self.speaker_counter += 1
            self.speakers[new_id] = {
                "embeddings": [emb],
                "centroid": emb,
                "count": 1
            }
            return new_id, highest_sim if highest_sim != -1.0 else 1.0, True, debug_info

speaker_manager = SpeakerManager()
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status Warning]: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def main():
    silence_frames_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    min_speech_frames = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK_SIZE)

    is_speaking = False
    speech_buffer = []
    silence_counter = 0

    print("\n" + "="*65)
    print("LIVE AUDIO TRACKER RUNNING")
    print(f"Pause Wait Time: {SILENCE_DURATION}s | Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print("Speak naturally. Take normal pauses. Stop speaking when finished.")
    print("Press Ctrl+C to stop.")
    print("="*65 + "\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=CHUNK_SIZE, callback=audio_callback):
        try:
            while True:
                chunk = audio_queue.get()
                tensor_chunk = torch.from_numpy(chunk.squeeze())

                with torch.no_grad():
                    speech_prob = vad_model(tensor_chunk, SAMPLE_RATE).item()

                if speech_prob > VAD_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        print("[Speaking...]", end="\r", flush=True)
                    speech_buffer.append(tensor_chunk)
                    silence_counter = 0
                elif is_speaking:
                    speech_buffer.append(tensor_chunk)
                    silence_counter += 1

                    if silence_counter > silence_frames_limit:
                        is_speaking = False
                        total_speech_frames = len(speech_buffer) - silence_counter

                        if total_speech_frames >= min_speech_frames:
                            utterance_audio = torch.cat(speech_buffer[:total_speech_frames])
                            duration = len(utterance_audio) / SAMPLE_RATE

                            spk_id, sim, is_new, debug_scores = speaker_manager.identify_or_register(utterance_audio)

                            tag = "[NEW SPEAKER]" if is_new else "[RECOGNIZED] "
                            print(f"\n{tag} -> {spk_id} ({duration:.2f}s) | Score: {sim:.2f} | Comparisons: [{debug_scores}] | Active: {len(speaker_manager.speakers)}")
                        else:
                            print("\n[Utterance < 0.6s discarded as noise]")

                        speech_buffer = []
                        silence_counter = 0
                        vad_model.reset_states()

        except KeyboardInterrupt:
            print("\n\nSession stopped.")
            print(f"Final Count of Distinct Speakers: {len(speaker_manager.speakers)}")

if __name__ == "__main__":
    main()

