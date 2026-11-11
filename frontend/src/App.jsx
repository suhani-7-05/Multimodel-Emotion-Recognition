import { useEffect, useState } from "react";
import { api } from "./api.js";
import { useLiveSession } from "./useLiveSession.js";
import Settings from "./components/Settings.jsx";
import LivePanel from "./components/LivePanel.jsx";
import BatchPanel from "./components/BatchPanel.jsx";

const DEFAULT_SETTINGS = {
  vad_threshold: 0.5,
  silence_duration: 1.1,
  min_speech_duration: 0.6,
  speaker_threshold: 0.38,
};

function ModelStatus({ status, backendError, onLoad }) {
  if (backendError) return <p className="error">Backend unreachable: {backendError}</p>;
  if (!status) return <p className="muted">Connecting to backend…</p>;
  const { models, loading_step, error } = status;
  return (
    <div className="model-status">
      <span className={`dot dot-${models}`} />
      <span>
        Models: <strong>{models}</strong>
        {loading_step && <span className="muted"> · loading {loading_step}…</span>}
      </span>
      {(models === "idle" || models === "error") && (
        <button className="primary" onClick={onLoad}>
          {models === "error" ? "Retry loading" : "Load models"}
        </button>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("live");
  const [status, setStatus] = useState(null);
  const [backendError, setBackendError] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const live = useLiveSession();

  useEffect(() => {
    api.config().then((c) => setSettings(c.default_settings)).catch(() => {});
  }, []);

  // Poll backend status; faster while models are loading.
  useEffect(() => {
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const s = await api.status();
        if (cancelled) return;
        setStatus(s);
        setBackendError("");
        timer = setTimeout(poll, s.models === "loading" ? 1000 : 4000);
      } catch (err) {
        if (cancelled) return;
        setBackendError(err.message);
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const loadModels = async () => {
    try {
      setStatus(await api.loadModels());
    } catch (err) {
      setBackendError(err.message);
    }
  };

  const modelsReady = status?.models === "ready";

  return (
    <div className="layout">
      <aside className="sidebar">
        <section className="panel">
          <h2>System</h2>
          <p>
            Compute device: <strong>{status?.device ?? "—"}</strong>
          </p>
          {status?.gpu && <p className="muted small">GPU: {status.gpu}</p>}
          {status && <p className="muted small">Whisper: {status.whisper_model}</p>}
          <ModelStatus status={status} backendError={backendError} onLoad={loadModels} />
        </section>
        <Settings settings={settings} onChange={setSettings} disabled={live.status !== "idle"} />
      </aside>

      <main className="main">
        <header>
          <h1>Multimodal Emotion Recognition &amp; Speaker Diarization</h1>
          <p className="muted">
            Each finished utterance is diarized, transcribed with Whisper and classified by the
            context-aware RoBERTa + WavLM model while capture continues.
          </p>
        </header>
        <nav className="tabs">
          <button className={tab === "live" ? "active" : ""} onClick={() => setTab("live")}>
            Live microphone
          </button>
          <button className={tab === "batch" ? "active" : ""} onClick={() => setTab("batch")}>
            Audio file analysis
          </button>
        </nav>
        <div hidden={tab !== "live"}>
          <LivePanel
            live={live}
            settings={settings}
            modelsReady={modelsReady}
            serverMic={status?.server_mic_available ?? true}
          />
        </div>
        <div hidden={tab !== "batch"}>
          <BatchPanel settings={settings} modelsReady={modelsReady && live.status === "idle"} />
        </div>
      </main>
    </div>
  );
}
