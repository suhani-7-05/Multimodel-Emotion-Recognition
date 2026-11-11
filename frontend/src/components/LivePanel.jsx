import { useEffect, useState } from "react";
import { api } from "../api.js";
import { mostFrequent } from "../emotions.js";
import SegmentCard from "./SegmentCard.jsx";

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="muted">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function LivePanel({ live, settings, modelsReady, serverMic }) {
  const [source, setSource] = useState("server");
  const [devices, setDevices] = useState([]);
  const [device, setDevice] = useState("");
  const [debugWav, setDebugWav] = useState(false);
  const active = live.status !== "idle";

  useEffect(() => {
    if (source !== "server" || !serverMic) return;
    api.devices().then((d) => setDevices(d.devices)).catch(() => setDevices([]));
  }, [source, serverMic]);

  const startLive = () =>
    live.start({
      source,
      device: source === "server" && device !== "" ? Number(device) : null,
      settings,
      debugWav,
    });

  const { stats, segments, errors } = live;
  const dominant = mostFrequent(segments);

  return (
    <div className="stack">
      <section className="panel">
        <div className="controls">
          <fieldset className="radio" disabled={active}>
            <label>
              <input type="radio" checked={source === "server"} onChange={() => setSource("server")} />
              Server microphone (recommended)
            </label>
            <label>
              <input type="radio" checked={source === "browser"} onChange={() => setSource("browser")} />
              Browser microphone
            </label>
          </fieldset>

          {source === "server" && (
            <select value={device} onChange={(e) => setDevice(e.target.value)} disabled={active || !serverMic}>
              <option value="">System default</option>
              {devices.map((d) => (
                <option key={d.index} value={d.index}>
                  {d.label}
                </option>
              ))}
            </select>
          )}

          <label className="check">
            <input type="checkbox" checked={debugWav} disabled={active} onChange={(e) => setDebugWav(e.target.checked)} />
            Save captured 16 kHz audio on the server
          </label>

          <div className="buttons">
            <button className="primary" onClick={startLive} disabled={active || !modelsReady}>
              {live.status === "connecting" ? "Starting…" : "Start microphone"}
            </button>
            <button onClick={live.stop} disabled={live.status !== "running"}>
              {live.status === "stopping" ? "Finishing queued turns…" : "Stop"}
            </button>
            <button className="ghost" onClick={live.clear} disabled={active}>
              Clear results
            </button>
          </div>
        </div>
        {!modelsReady && <p className="muted">Load the models before starting the microphone.</p>}
        {source === "server" && !serverMic && (
          <p className="warn">The backend has no sounddevice/PortAudio, so use the browser microphone.</p>
        )}
        {source === "server" && (
          <p className="muted small">The server microphone is the mic of the machine running the backend.</p>
        )}

        {stats && (
          <>
            <div className="metrics">
              <Metric label="Captured audio" value={`${stats.audio_sec.toFixed(0)} s`} />
              <Metric label="Input peak" value={stats.peak.toFixed(2)} />
              <Metric label="Timeline gaps" value={`${stats.gap_events} (${stats.gap_ms.toFixed(0)} ms)`} />
              <Metric label="Utterances pending" value={stats.pending} />
            </div>
            <p className="muted small">Source: {live.sourceInfo} → 16 kHz mono</p>
            {stats.clipped_samples > 0 && (
              <p className="warn">Input is clipping. Lower the microphone gain / input volume.</p>
            )}
            {(stats.overflows > 0 || stats.dropped_blocks > 0) && (
              <p className="warn">Audio blocks were dropped. The machine could not keep up with capture.</p>
            )}
          </>
        )}
      </section>

      <section className="stack">
        <h2>Live dialogue timeline</h2>
        {segments.length === 0 && errors.length === 0 ? (
          <p className="empty">Start speaking. Results appear after each utterance ends.</p>
        ) : (
          <>
            <div className="metrics">
              <Metric label="Processed utterances" value={segments.length} />
              <Metric label="Speakers detected" value={new Set(segments.map((s) => s.speaker_id)).size} />
              <Metric label="Most frequent emotion" value={dominant ? dominant.toUpperCase() : "—"} />
            </div>
            {errors.slice(-5).map((e, i) => (
              <p className="error" key={i}>
                {e}
              </p>
            ))}
            {[...segments].reverse().map((seg) => (
              <SegmentCard key={seg.audio_url} seg={seg} />
            ))}
          </>
        )}
      </section>
    </div>
  );
}
