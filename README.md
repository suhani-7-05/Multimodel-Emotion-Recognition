# Multimodel-Emotion-Recognition

Real-time conversation interpreter (PRJ4). Each spoken turn is:

1. split off by Silero VAD and given a speaker (ECAPA-TDNN),
2. transcribed (Whisper large-v3-turbo via faster-whisper),
3. classified by a context-aware RoBERTa + WavLM model (7 MELD emotions, plus what the
   words alone and the voice alone suggest, plus simple prosody cues),
4. explained in plain, literal language by a local LLM: what the speaker means, any
   sarcasm / hint / indirect request / polite refusal, and whether a reply is expected.

The UI ("Subtext") shows the conversation as a chat with a "What they mean" card under
every turn, can read explanations aloud, and has a larger-text mode. Three modes:

- **Live** - the microphone.
- **Recording** - upload an audio file, analysed in one go.
- **Watch** - open a film/TV clip; it plays normally (picture + sound) while its soundtrack
  is analysed live. Captions under the video show who spoke, the emotion and the plain
  meaning; the video can pause automatically on a line with a hidden meaning. The video file
  never leaves the browser - only its audio is streamed. Rewinding shows earlier results
  without re-analysing; jumping ahead starts analysing from the new point.

## Layout

```
backend/            FastAPI inference server (all models live here)
  app/              main.py (routes), live.py (WebSocket), pipeline.py (per-turn analysis),
                    asr.py (Whisper + guards), interpreter.py (LLM), models.py, config.py
  audio_capture.py  capture thread, resampling, LiveSession worker
  diarization_engine.py
  model_bundle/     emotion model code (infer2.py, common2.py) + runs/v2_rl_audio weights
  tools/            convert_whisper_ct2.py (offline Whisper -> faster-whisper conversion)
  models/           local model folders, not in git (see below)
frontend/           React + Vite UI (no ML code)
training/           training script, DGX job files, eval_meld.py
legacy/             previous Streamlit app and CLI speaker tracker
docs/               write-ups, report, result plots
```

Not in git: `backend/model_bundle/runs/v2_rl_audio/best.pt` (extract
`prj4_emotion_model.tar.gz` there) and `backend/models/`.

## Running on the laptop

```bash
conda activate audio_sentiment
pip install -r backend/requirements.txt
python backend/tools/convert_whisper_ct2.py --name large-v3-turbo   # once; uses ~/.cache/whisper/large-v3-turbo.pt
ollama pull qwen3:4b                                                  # once
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 - proxies /api and /ws to :8000
```

Windows only: faster-whisper needs the CUDA 12 cuBLAS DLLs (`cublas64_12.dll`,
`cublasLt64_12.dll`) in `backend/models/cuda12/` (copied from any CUDA 12 PyTorch install).
Without them, or without the converted model, the backend falls back to openai-whisper.

### Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `backend/models/faster-whisper-large-v3-turbo` if present, else `large-v3-turbo` | ASR model size or local folder |
| `ASR_ENGINE` | `auto` | `faster`, `openai`, or `auto` (faster-whisper if the model is on disk) |
| `EMOTION_DEVICE` | `auto` | `cuda` / `cpu`; auto = GPU only if it has >= 12 GB |
| `LLM_API` | `ollama` | `ollama` or `openai` (vLLM, llama.cpp server, ...) |
| `LLM_BASE_URL` | `http://127.0.0.1:11434` | LLM server |
| `LLM_MODEL` | `qwen3:4b` | model name on that server |
| `LLM_THINK` | `0` | `1` lets reasoning models think first (better, slower) |
| `LLM_NUM_GPU` | (Ollama decides) | `0` keeps the LLM off a small GPU so Whisper isn't starved |
| `PRELOAD_MODELS` | `0` | `1` loads everything at startup |
| `TORCH_THREADS` | torch default | CPU threads for the emotion model |

## Measured on the laptop (RTX 3050 6 GB, 16 threads)

| | Before | Now |
|---|---|---|
| ASR, 5 s clip | 5.5 s (Whisper large, spilled past 6 GB VRAM) | 0.4 s (turbo) |
| Per live turn, all stages | ~10 s (estimate from stages) | ~2.2 s |
| Per turn from a file | — | ~1.0 s |

Fixes behind this: turbo ASR; speaker / ASR / voice stages run in parallel; Silero VAD was
silently forcing PyTorch onto 1 CPU thread; VAD in the capture thread no longer competes for
all cores; warm-up at load time (librosa compiled itself on the first turn, +7 s).

## Evaluation

```bash
python training/eval_meld.py --data_dir /path/to/MELD.Raw              # gold transcripts
python training/eval_meld.py --data_dir ... --text live_asr --limit 500  # our ASR, end to end
python training/eval_meld.py --data_dir ... --no_audio                  # text-only ablation
```

Writes weighted/macro F1, per-class F1, calibration (ECE) and a confusion matrix to
`docs/results/eval/`, for the fused model and for the words-only and voice-only heads.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Models, device, ASR engine, LLM availability |
| POST | `/api/models/load` | Start loading models in the background |
| GET | `/api/config` | Emotion labels and default settings |
| GET | `/api/devices` | Microphones on the server |
| POST | `/api/batch` | Multipart `file` (+ VAD/speaker settings) → analysed turns |
| POST | `/api/interpret` | `{turns, target}` → plain-language explanation of one turn |
| GET | `/api/clips/{id}.wav` | Audio of one analysed turn |
| WS | `/ws/live` | Live session (protocol at the top of `backend/app/live.py`) |

Live input can come from the browser microphone (16 kHz int16 over the WebSocket, ~32 KB/s,
fine for a remote backend), the server's own microphone, or a playing video (Watch mode:
int16 at the browser's rate, ~96 KB/s; `cut` messages mark seeks so lines from both sides of
a jump are never merged).

## Known limitations

- The emotion model is trained on MELD (TV sitcom dialogue); expect lower accuracy on real
  conversations than on the MELD test set.
- It reads mostly the words, so sarcastic praise is usually labelled "joy".
- Small local LLMs (mistral 7B, qwen3:4b) still miss sarcasm that is only signalled by context,
  and qwen3:4b sometimes labels a line "sarcasm" but still writes the literal meaning.
  A larger model on the GPU server is needed for reliable explanations.
- Watch mode on the laptop: explanations arrive several seconds after a line, so the
  auto-pause lands a little after it (exactly on time when rewatching). Background music can
  be picked up as speech; raise "Speech detection strictness" for music-heavy scenes.
