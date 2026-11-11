import { useState } from "react";
import { api } from "../api.js";
import { EMOTIONS, meta } from "../emotions.js";
import SegmentCard from "./SegmentCard.jsx";

export default function BatchPanel({ settings, modelsReady }) {
  const [file, setFile] = useState(null);
  const [override, setOverride] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const run = async () => {
    if (!file) return setError("Choose an audio file first.");
    setRunning(true);
    setError("");
    setResult(null);
    try {
      setResult(await api.batch(file, settings, override.trim()));
    } catch (err) {
      setError(`Batch analysis failed: ${err.message}`);
    } finally {
      setRunning(false);
    }
  };

  const maxCount = result ? Math.max(1, ...Object.values(result.emotion_counts)) : 1;

  return (
    <div className="stack">
      <section className="panel">
        <p>Upload a recorded conversation to analyse the whole file at once.</p>
        <div className="controls">
          <input type="file" accept=".wav,audio/wav,audio/*" onChange={(e) => setFile(e.target.files[0] || null)} />
          <input
            type="text"
            className="grow"
            placeholder="Optional transcript override (used only if exactly one utterance is detected)"
            value={override}
            onChange={(e) => setOverride(e.target.value)}
          />
          <button className="primary" onClick={run} disabled={running || !modelsReady}>
            {running ? "Analysing…" : "Run batch analysis"}
          </button>
        </div>
        {!modelsReady && <p className="muted">Load the models first.</p>}
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="stack">
          {result.segments.length === 0 ? (
            <p className="empty">No speech detected.</p>
          ) : (
            <>
              <div className="metrics">
                <div className="metric"><span className="muted">Duration</span><strong>{result.duration_sec.toFixed(1)} s</strong></div>
                <div className="metric"><span className="muted">Speakers</span><strong>{result.speakers}</strong></div>
                <div className="metric"><span className="muted">Utterances</span><strong>{result.segments.length}</strong></div>
              </div>
              <div className="panel">
                <h2>Emotion distribution</h2>
                <div className="bars">
                  {EMOTIONS.map((e) => (
                    <div className="bar-row" key={e}>
                      <span className="bar-label">{e}</span>
                      <span className="bar-track">
                        <span
                          className="bar-fill"
                          style={{ width: `${(result.emotion_counts[e] / maxCount) * 100}%`, background: meta(e).color }}
                        />
                      </span>
                      <span className="bar-value">{result.emotion_counts[e]}</span>
                    </div>
                  ))}
                </div>
              </div>
              {result.segments.map((seg) => (
                <SegmentCard key={seg.audio_url} seg={seg} />
              ))}
            </>
          )}
        </section>
      )}
    </div>
  );
}
