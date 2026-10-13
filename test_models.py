import os
import shutil
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# Clean up partial folder if previously created
if os.path.exists("tmpdir_spkrec"):
    shutil.rmtree("tmpdir_spkrec")

print("--- Testing Silero VAD v5 ---")
# 1. Download & load Silero VAD
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("Silero VAD loaded successfully!")

print("\n--- Testing SpeechBrain ECAPA-TDNN ---")
# 2. Download & load SpeechBrain model with COPY_SKIP_CACHE (Windows compatible)
speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir_spkrec",
    local_strategy=LocalStrategy.COPY_SKIP_CACHE
)
print("SpeechBrain ECAPA-TDNN loaded successfully!")

print("\n--- Running Dummy Forward Pass ---")
# 3. Create dummy 1.5 seconds of 16kHz audio
dummy_audio = torch.randn(1, 24000)  # 24,000 samples = 1.5s at 16kHz

# Test VAD on a 512-sample chunk (32ms at 16kHz)
vad_prob = vad_model(dummy_audio[:, :512], 16000).item()
print(f"Silero VAD speech probability on dummy chunk: {vad_prob:.4f}")

# Test Speaker Embedding (should produce a 192-dimensional vector)
embedding = speaker_model.encode_batch(dummy_audio)
print(f"SpeechBrain embedding shape: {embedding.shape} (Expected: [1, 1, 192])")

print("\n>>> ALL MODELS DOWNLOADED AND OPERATIONAL! <<<")
