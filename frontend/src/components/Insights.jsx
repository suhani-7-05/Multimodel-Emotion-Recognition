import { EMOTIONS, meta, speakerStyles } from "../lib/emotions.js";

function SpeakerTrails({ segments }) {
  const styles = speakerStyles(segments);
  const bySpeaker = {};
  for (const s of segments) (bySpeaker[s.speaker_id] ||= []).push(s);
  const ids = Object.keys(bySpeaker);
  if (!ids.length) return <p className="muted small">Speakers appear here once they talk.</p>;
  return (
    <ul className="trails">
      {ids.map((id) => (
        <li key={id} style={{ "--spk": styles[id].color }}>
          <span className="trail-name"><span className="spk-dot" />{id}</span>
          <span className="trail" aria-label={`Emotion over time for ${id}`}>
            {bySpeaker[id].slice(-16).map((s) => (
              <span
                key={s.audio_url}
                className="trail-dot"
                title={`${meta(s.emotion).label} · ${(s.confidence * 100).toFixed(0)}%`}
                style={{ background: meta(s.emotion).color, opacity: 0.35 + 0.65 * s.confidence }}
              />
            ))}
          </span>
        </li>
      ))}
    </ul>
  );
}

function MoodMix({ segments }) {
  const counts = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
  for (const s of segments) if (s.emotion in counts) counts[s.emotion] += 1;
  const total = segments.length || 1;
  const present = EMOTIONS.filter((e) => counts[e]);
  return (
    <>
      <div className="mix" role="img" aria-label="Share of each emotion">
        {present.map((e) => (
          <span key={e} style={{ flexGrow: counts[e], background: meta(e).color }} title={`${meta(e).label}: ${counts[e]}`} />
        ))}
      </div>
      <ul className="legend">
        {present.map((e) => (
          <li key={e}><span className="emo-dot" style={{ "--c": meta(e).color }} />{meta(e).label}<span className="num">{Math.round((counts[e] / total) * 100)}%</span></li>
        ))}
      </ul>
    </>
  );
}

function Perf({ segments, interpretations }) {
  const t = segments.map((s) => s.timings_ms).filter(Boolean);
  if (!t.length) return <p className="muted small">Timings appear after the first turn.</p>;
  const avg = (k) => Math.round(t.reduce((a, x) => a + (x[k] || 0), 0) / t.length);
  const llm = Object.values(interpretations).map((r) => r.latency).filter((x) => x != null);
  const rows = [
    ["Speaker ID", avg("speaker")],
    ["Speech to text", avg("asr")],
    ["Voice features", avg("voice")],
    ["Emotion model", avg("emotion")],
  ];
  const max = Math.max(...rows.map((r) => r[1]), 1);
  return (
    <>
      <dl className="perf">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd><span className="perf-bar" style={{ width: `${(v / max) * 100}%` }} /><span className="num">{v} ms</span></dd>
          </div>
        ))}
      </dl>
      <p className="small">Per turn <strong className="num">{avg("total")} ms</strong> <span className="muted">(stages overlap)</span>
        {llm.length > 0 && <> · explanation <strong className="num">{Math.round(llm.reduce((a, b) => a + b, 0) / llm.length)} ms</strong></>}</p>
    </>
  );
}

export default function Insights({ segments, interpretations }) {
  return (
    <>
      <section className="card">
        <h3>Speakers</h3>
        <SpeakerTrails segments={segments} />
      </section>
      {segments.length > 0 && (
        <section className="card">
          <h3>Emotion mix</h3>
          <MoodMix segments={segments} />
        </section>
      )}
      <section className="card">
        <h3>Speed</h3>
        <Perf segments={segments} interpretations={interpretations} />
      </section>
    </>
  );
}
