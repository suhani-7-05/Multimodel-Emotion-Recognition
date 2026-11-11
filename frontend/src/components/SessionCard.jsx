import { useEffect, useState } from "react";
import { MicIcon, StopIcon } from "./Icons.jsx";

function useElapsed(startedAt) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!startedAt) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [startedAt]);
  if (!startedAt) return "0:00";
  const s = Math.max(0, Math.floor((now - startedAt) / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export default function SessionCard({ live, ready, onStart, sourceLabel }) {
  const elapsed = useElapsed(live.startedAt);
  const running = live.status === "running";
  const busy = live.status === "connecting" || live.status === "stopping";
  // RMS -> 0..1 on a dB scale (-60 dBFS .. 0)
  const level = live.stats?.level ? Math.max(0, Math.min(1, (20 * Math.log10(live.stats.level + 1e-9) + 60) / 60)) : 0;
  const s = live.stats;

  return (
    <section className="card session">
      <button
        className={`mic-btn ${running ? "on" : ""}`}
        style={{ "--level": running ? level : 0 }}
        onClick={running ? live.stop : onStart}
        disabled={busy || (!running && !ready)}
        aria-label={running ? "Stop listening" : "Start listening"}
      >
        {running ? <StopIcon width={28} height={28} /> : <MicIcon width={30} height={30} />}
      </button>
      <div className="session-text">
        <strong>
          {live.status === "connecting" ? "Starting…" : live.status === "stopping" ? "Finishing last turns…" : running ? (s?.speaking ? "Hearing speech" : "Listening") : "Ready to listen"}
        </strong>
        <span className="muted small">{running ? `${elapsed} · ${live.sourceInfo}` : ready ? sourceLabel : "Load the models first"}</span>
      </div>
      {s && (s.clipped_samples > 0 || s.dropped_blocks > 0 || s.overflows > 0 || s.dropped_utterances > 0) && (
        <ul className="warnings">
          {s.clipped_samples > 0 && <li>Input is clipping — lower the mic volume.</li>}
          {(s.dropped_blocks > 0 || s.overflows > 0) && <li>Some audio was dropped; the computer is struggling to keep up.</li>}
          {s.dropped_utterances > 0 && <li>{s.dropped_utterances} turn(s) skipped because analysis fell behind.</li>}
        </ul>
      )}
      {live.errors.length > 0 && (
        <ul className="errors">{live.errors.slice(-3).map((e, i) => <li key={i}>{e}</li>)}</ul>
      )}
    </section>
  );
}
