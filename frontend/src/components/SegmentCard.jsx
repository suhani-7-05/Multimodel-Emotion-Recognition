import { useState } from "react";
import { EMOTIONS, meta } from "../emotions.js";

export function EmotionBadge({ emotion, confidence }) {
  const { emoji, color } = meta(emotion);
  return (
    <span className="badge" style={{ color, background: `${color}22` }}>
      {emoji} {emotion?.toUpperCase()}
      {confidence != null && ` (${(confidence * 100).toFixed(1)}%)`}
    </span>
  );
}

export function ScoreBars({ distribution }) {
  return (
    <div className="bars">
      {EMOTIONS.map((e) => {
        const p = distribution?.[e] ?? 0;
        return (
          <div className="bar-row" key={e}>
            <span className="bar-label">{e}</span>
            <span className="bar-track">
              <span
                className="bar-fill"
                style={{ width: `${(p * 100).toFixed(1)}%`, background: meta(e).color }}
              />
            </span>
            <span className="bar-value">{(p * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default function SegmentCard({ seg }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="card segment">
      <div className="seg-speaker">
        <strong>{seg.speaker_id}</strong>
        <span className="muted">
          {seg.start_sec.toFixed(1)}s – {seg.end_sec.toFixed(1)}s
        </span>
        {seg.is_new_speaker && <span className="muted">First appearance</span>}
      </div>
      <div className="seg-body">
        <p className="transcript">{seg.transcript}</p>
        <div className="seg-tags">
          <EmotionBadge emotion={seg.emotion} confidence={seg.confidence} />
          {seg.tone && <span className="muted">tone: {seg.tone}</span>}
        </div>
        {open && <ScoreBars distribution={seg.distribution} />}
      </div>
      <div className="seg-side">
        <audio controls preload="none" src={seg.audio_url} />
        <button className="link" onClick={() => setOpen((o) => !o)}>
          {open ? "Hide scores" : "Emotion scores"}
        </button>
      </div>
    </article>
  );
}
