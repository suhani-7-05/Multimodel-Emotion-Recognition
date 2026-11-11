import { useEffect, useState } from "react";
import { api } from "../api.js";
import { CloseIcon } from "./Icons.jsx";

const SLIDERS = [
  { key: "vad_threshold", label: "Speech detection sensitivity", hint: "Higher ignores more background noise.", min: 0.1, max: 0.9, step: 0.05 },
  { key: "silence_duration", label: "Pause that ends a turn", hint: "Seconds of silence before a turn is analysed.", min: 0.5, max: 2.5, step: 0.1, unit: "s" },
  { key: "min_speech_duration", label: "Shortest turn", hint: "Shorter sounds (coughs, clicks) are ignored.", min: 0.3, max: 1.5, step: 0.1, unit: "s" },
  { key: "speaker_threshold", label: "Speaker matching", hint: "Lower merges similar voices into one speaker.", min: 0.2, max: 0.7, step: 0.02 },
];

export default function SettingsDrawer({ open, onClose, settings, setSettings, prefs, setPrefs, locked, status }) {
  const [devices, setDevices] = useState([]);
  useEffect(() => {
    if (open && status?.server_mic_available) api.devices().then((d) => setDevices(d.devices)).catch(() => setDevices([]));
  }, [open, status?.server_mic_available]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const set = (patch) => setPrefs({ ...prefs, ...patch });

  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} aria-hidden />
      <aside className={`drawer ${open ? "open" : ""}`} aria-hidden={!open} aria-label="Settings" role="dialog">
        <header className="drawer-head">
          <h2>Settings</h2>
          <button className="icon-btn" aria-label="Close settings" onClick={onClose}><CloseIcon /></button>
        </header>

        <section>
          <h3>Microphone</h3>
          <fieldset className="radio-cards" disabled={locked}>
            <label className={prefs.source === "browser" ? "on" : ""}>
              <input type="radio" checked={prefs.source === "browser"} onChange={() => set({ source: "browser" })} />
              <strong>This browser</strong>
              <span className="muted small">Works with a remote server too.</span>
            </label>
            <label className={prefs.source === "server" ? "on" : ""}>
              <input type="radio" checked={prefs.source === "server"} disabled={!status?.server_mic_available} onChange={() => set({ source: "server" })} />
              <strong>Server’s microphone</strong>
              <span className="muted small">Only when the backend runs on this computer.</span>
            </label>
          </fieldset>
          {prefs.source === "server" && (
            <select value={prefs.device ?? ""} disabled={locked} onChange={(e) => set({ device: e.target.value === "" ? null : Number(e.target.value) })}>
              <option value="">System default</option>
              {devices.map((d) => <option key={d.index} value={d.index}>{d.label}</option>)}
            </select>
          )}
        </section>

        <section>
          <h3>Explanations</h3>
          <label className="switch">
            <input type="checkbox" checked={prefs.explain} onChange={(e) => set({ explain: e.target.checked })} />
            <span>Explain what each turn means</span>
          </label>
          <label className="switch">
            <input type="checkbox" checked={prefs.autoSpeak} onChange={(e) => set({ autoSpeak: e.target.checked })} />
            <span>Read explanations aloud automatically</span>
          </label>
          <label className="switch">
            <input type="checkbox" checked={prefs.onlySubtext} onChange={(e) => set({ onlySubtext: e.target.checked })} />
            <span>Only read aloud when there is a hidden meaning</span>
          </label>
          {status?.llm && (
            <p className="muted small">
              Language model: {status.llm.model}{" "}
              {!status.llm.reachable ? "— not reachable" : !status.llm.model_available ? "— not downloaded" : "— ready"}
            </p>
          )}
        </section>

        <section>
          <h3>Listening</h3>
          {SLIDERS.map((f) => (
            <label className="slider" key={f.key}>
              <span className="slider-head"><span>{f.label}</span><output className="num">{settings[f.key].toFixed(2)}{f.unit || ""}</output></span>
              <input type="range" min={f.min} max={f.max} step={f.step} value={settings[f.key]} disabled={locked}
                onChange={(e) => setSettings({ ...settings, [f.key]: parseFloat(e.target.value) })} />
              <span className="muted small">{f.hint}</span>
            </label>
          ))}
          <label className="switch">
            <input type="checkbox" checked={prefs.debugWav} disabled={locked} onChange={(e) => set({ debugWav: e.target.checked })} />
            <span>Save the captured audio on the server (for debugging)</span>
          </label>
        </section>

        {status && (
          <section>
            <h3>System</h3>
            <dl className="kv">
              <dt>GPU</dt><dd>{status.gpu || "none"}{status.gpu_memory_gb ? ` · ${status.gpu_memory_gb} GB` : ""}</dd>
              <dt>Speech to text</dt><dd>{status.asr_engine || "—"} · {status.whisper_model}</dd>
              <dt>Emotion model on</dt><dd>{status.emotion_device || "—"}</dd>
            </dl>
          </section>
        )}
      </aside>
    </>
  );
}
