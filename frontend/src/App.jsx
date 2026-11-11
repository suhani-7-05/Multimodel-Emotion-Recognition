import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { useLiveSession } from "./hooks/useLiveSession.js";
import { useWatchSession } from "./hooks/useWatchSession.js";
import { useInterpreter } from "./hooks/useInterpreter.js";
import { interpretationSpeech, speak } from "./lib/speech.js";
import Conversation from "./components/Conversation.jsx";
import Insights from "./components/Insights.jsx";
import SessionCard from "./components/SessionCard.jsx";
import SettingsDrawer from "./components/SettingsDrawer.jsx";
import UploadCard from "./components/UploadCard.jsx";
import WatchView from "./components/WatchView.jsx";
import { GearIcon, TextSizeIcon } from "./components/Icons.jsx";

export const APP_NAME = "Subtext";

const DEFAULT_SETTINGS = { vad_threshold: 0.5, silence_duration: 1.1, min_speech_duration: 0.6, speaker_threshold: 0.38 };
const DEFAULT_PREFS = { source: "browser", device: null, explain: true, autoSpeak: false, onlySubtext: true, pauseOnSubtext: true, debugWav: false, easyRead: false };

function loadPrefs() {
  try {
    return { ...DEFAULT_PREFS, ...JSON.parse(localStorage.getItem("subtext.prefs") || "{}") };
  } catch {
    return DEFAULT_PREFS;
  }
}

function useBackendStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const s = await api.status();
        if (cancelled) return;
        setStatus(s);
        setError("");
        timer = setTimeout(poll, s.models === "loading" ? 1000 : 5000);
      } catch (err) {
        if (cancelled) return;
        setError(err.message);
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);
  return [status, setStatus, error];
}

function StatusPill({ status, error, onLoad }) {
  if (error) return <span className="pill pill-bad" title={error}><span className="pill-dot" /><span className="pill-text">Backend offline</span></span>;
  if (!status) return <span className="pill" title="Connecting…"><span className="pill-dot" /><span className="pill-text">Connecting…</span></span>;
  if (status.models === "ready") {
    const llm = status.llm;
    const llmOk = llm?.reachable && llm?.model_available;
    const text = `Ready${llmOk ? "" : " · no explanations"}`;
    return (
      <span className="pill pill-ok" title={llmOk ? `Ready · explanations by ${llm.model}` : text}>
        <span className="pill-dot" /><span className="pill-text">{text}</span>
      </span>
    );
  }
  if (status.models === "loading") {
    const text = `Loading ${status.loading_step || "models"}…`;
    return <span className="pill pill-wait" title={text}><span className="pill-dot" /><span className="pill-text">{text}</span></span>;
  }
  return (
    <button className="pill pill-action" onClick={onLoad} title={status.error || ""}>
      <span className="pill-text">{status.models === "error" ? "Loading failed — retry" : "Load models"}</span>
    </button>
  );
}

function LiveEmpty({ ready }) {
  return (
    <div className="empty">
      <h2>Understand what people really mean.</h2>
      <p>
        {APP_NAME} listens to a conversation, works out who is speaking and how they sound,
        and explains each turn in plain words — including sarcasm, hints and polite refusals.
      </p>
      <ol className="steps">
        <li><span>1</span>{ready ? "Models are loaded." : "Load the models (top right)."}</li>
        <li><span>2</span>Press the microphone and talk normally.</li>
        <li><span>3</span>After each pause, the turn and its meaning appear here.</li>
      </ol>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("live");
  const [status, setStatus, backendError] = useBackendStatus();
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [prefs, setPrefs] = useState(loadPrefs);
  const [drawer, setDrawer] = useState(false);
  const [liveSegments, setLiveSegments] = useState([]);
  const [upload, setUpload] = useState({ result: null, running: false, error: "" });

  useEffect(() => {
    try { localStorage.setItem("subtext.prefs", JSON.stringify(prefs)); } catch { /* private mode */ }
    document.documentElement.dataset.easyRead = prefs.easyRead ? "1" : "0";
  }, [prefs]);

  useEffect(() => {
    api.config().then((c) => setSettings(c.default_settings)).catch(() => {});
  }, []);

  const ready = status?.models === "ready";
  const llmOk = !!(status?.llm?.reachable && status?.llm?.model_available);

  const live = useLiveSession({ onSegment: (seg) => setLiveSegments((s) => [...s, seg]) });
  const onLiveResult = useCallback((_seg, interp) => {
    if (!prefs.autoSpeak) return;
    if (prefs.onlySubtext && (!interp.subtext || interp.subtext === "none")) return;
    speak(interpretationSpeech(interp), { interrupt: false });
  }, [prefs.autoSpeak, prefs.onlySubtext]);
  const liveInterp = useInterpreter(liveSegments, { enabled: prefs.explain && llmOk, onResult: onLiveResult });

  const uploadSegments = upload.result?.segments || [];
  const uploadInterp = useInterpreter(uploadSegments, { enabled: prefs.explain && llmOk });

  const [watchSegments, setWatchSegments] = useState([]);
  const watch = useWatchSession({ onSegment: (seg) => setWatchSegments((s) => [...s, seg]) });
  const watchInterp = useInterpreter(watchSegments, { enabled: prefs.explain && llmOk });
  const busy = live.status !== "idle" || watch.status !== "idle";

  const loadModels = async () => {
    try { setStatus({ ...(await api.loadModels()), llm: status?.llm }); } catch { /* status poll shows it */ }
  };

  const startLive = () => {
    liveInterp.reset();
    setLiveSegments([]);
    live.start({ source: prefs.source, device: prefs.device, settings, debugWav: prefs.debugWav });
  };

  const runUpload = async (file) => {
    uploadInterp.reset();
    setUpload({ result: null, running: true, error: "" });
    try {
      setUpload({ result: await api.batch(file, settings), running: false, error: "" });
    } catch (err) {
      setUpload({ result: null, running: false, error: err.message });
    }
  };

  const s = live.stats;
  const pendingLabel = mode === "live" && live.status !== "idle"
    ? s?.speaking ? "Listening…" : (s?.analyzing || s?.pending > 0 || live.status === "stopping") ? "Working on what was said…" : null
    : mode === "upload" && upload.running ? "Analysing the recording…" : null;

  const segments = mode === "live" ? liveSegments : uploadSegments;
  const interp = mode === "live" ? liveInterp : uploadInterp;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>“</span>
          <span>{APP_NAME}</span>
        </div>
        <nav className="segmented" aria-label="Mode">
          <button className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>Live</button>
          <button className={mode === "upload" ? "on" : ""} onClick={() => setMode("upload")}>Recording</button>
          <button className={mode === "watch" ? "on" : ""} onClick={() => setMode("watch")}>Watch</button>
        </nav>
        <div className="topbar-right">
          <StatusPill status={status} error={backendError} onLoad={loadModels} />
          <button className={`icon-btn ${prefs.easyRead ? "active" : ""}`} aria-pressed={prefs.easyRead}
            aria-label="Larger text" title="Larger text" onClick={() => setPrefs({ ...prefs, easyRead: !prefs.easyRead })}>
            <TextSizeIcon />
          </button>
          <button className="icon-btn" aria-label="Settings" title="Settings" onClick={() => setDrawer(true)}><GearIcon /></button>
        </div>
      </header>

      <div className="layout" hidden={mode !== "watch"}>
        <WatchView
          watch={watch}
          segments={watchSegments}
          interp={watchInterp}
          settings={settings}
          prefs={prefs}
          setPrefs={setPrefs}
          ready={ready && live.status === "idle"}
          onNewVideo={() => { setWatchSegments([]); watchInterp.reset(); watch.reset(); }}
        />
      </div>

      <div className="layout" hidden={mode === "watch"}>
        <main className="stream">
          <Conversation
            segments={segments}
            interpretations={interp.results}
            onRetry={interp.retry}
            pendingLabel={pendingLabel}
            empty={mode === "live" ? <LiveEmpty ready={ready} /> : (
              <div className="empty">
                <h2>Analyse a recorded conversation.</h2>
                <p>Upload an audio file on the right. Every turn is transcribed, given a speaker and an emotion, and explained in plain words.</p>
              </div>
            )}
          />
        </main>

        <aside className="rail">
          {mode === "live" ? (
            <SessionCard
              live={live}
              ready={ready && watch.status === "idle"}
              onStart={startLive}
              sourceLabel={prefs.source === "browser" ? "Using this browser’s microphone" : "Using the server’s microphone"}
            />
          ) : (
            <UploadCard ready={ready && !busy} running={upload.running} onRun={runUpload} error={upload.error} summary={upload.result} />
          )}
          <Insights segments={segments} interpretations={interp.results} />
          {segments.length > 0 && !(mode === "live" && live.status !== "idle") && (
            <button className="btn ghost block" onClick={() => {
              if (mode === "live") { setLiveSegments([]); liveInterp.reset(); }
              else { setUpload({ result: null, running: false, error: "" }); uploadInterp.reset(); }
            }}>Clear conversation</button>
          )}
          <p className="disclaimer small muted">
            Emotion and meaning are automatic estimates and can be wrong. Use them as hints, not facts.
          </p>
        </aside>
      </div>

      <SettingsDrawer
        open={drawer}
        onClose={() => setDrawer(false)}
        settings={settings}
        setSettings={setSettings}
        prefs={prefs}
        setPrefs={setPrefs}
        locked={busy}
        status={status}
      />
    </div>
  );
}
