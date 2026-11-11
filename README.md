# Multimodel-Emotion-Recognition

Real-time multimodal emotion recognition with speaker diarization (PRJ4).
Each utterance is segmented by Silero VAD, assigned to a speaker with ECAPA-TDNN,
transcribed with Whisper, and classified by a context-aware RoBERTa + WavLM model.

## Layout

```
backend/            FastAPI inference server (all models live here)
  app/              API: main.py (routes), live.py (WebSocket sessions), pipeline.py, models.py
  audio_capture.py  capture thread, resampling, LiveSession worker
  diarization_engine.py
  model_bundle/     emotion model code (infer2.py, common2.py) + runs/v2_rl_audio weights
frontend/           React + Vite UI (no ML code)
training/           PRJ3 training script and DGX job files (best3.pt lives here, not in git)
legacy/             previous Streamlit app and CLI speaker tracker
docs/               project write-ups and report; docs/results/ holds the PRJ3 result plots
```

`backend/model_bundle/runs/v2_rl_audio/best.pt` is not in git. If it's missing, extract
`prj4_emotion_model.tar.gz` (kept in `backend/model_bundle/`, also not in git) there.

## Running

Backend, using the `audio_sentiment` conda env:

```bash
conda activate audio_sentiment
pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and click **Load models**. The Vite dev server proxies
`/api` and `/ws` to the backend on port 8000.

### Backend environment variables

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `large` | Whisper size (`small`, `medium`, … for faster turns) |
| `PRELOAD_MODELS` | `0` | `1` loads all models at startup |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed origins when the UI is served from elsewhere |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Model state, device, whether a job is running |
| POST | `/api/models/load` | Start loading models in the background |
| GET | `/api/config` | Emotion labels and default pipeline settings |
| GET | `/api/devices` | Microphones available on the server |
| POST | `/api/batch` | Multipart `file` (+ `speaker_threshold`, `transcript_override`) → analysed segments |
| GET | `/api/clips/{id}.wav` | Audio of one analysed utterance |
| WS | `/ws/live` | Live session: see the protocol at the top of `backend/app/live.py` |

Live input can come from the **server microphone** (the machine running the backend,
recommended) or the **browser microphone**. The browser option sends raw PCM over the
WebSocket, so it also works when the UI and the backend run on different machines.

Only one live session or batch job runs at a time, because speaker profiles and
dialogue history are shared model state.
