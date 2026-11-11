import { useEffect, useRef, useState } from "react";
import { EMOTIONS, SUBTEXT_LABELS, initials, meta, mismatch, speakerStyles, strength } from "../lib/emotions.js";
import { canSpeak, interpretationSpeech, speak } from "../lib/speech.js";
import { AlertIcon, ChevronIcon, ReplyIcon, SparkIcon, SpeakerIcon } from "./Icons.jsx";

function EmotionChip({ emotion, confidence }) {
  const m = meta(emotion);
  return (
    <span className="chip emo-chip" style={{ "--c": m.color }} title={`${strength(confidence)} · ${(confidence * 100).toFixed(0)}%`}>
      <span className="emo-dot" />
      {m.label}
      <span className="conf-track" aria-hidden><span className="conf-fill" style={{ width: `${confidence * 100}%` }} /></span>
      <span className="sr-only">{(confidence * 100).toFixed(0)} percent</span>
    </span>
  );
}

function Bars({ title, probs }) {
  if (!probs) return null;
  return (
    <div className="bars">
      <h4>{title}</h4>
      {EMOTIONS.map((e) => (
        <div className="bar-row" key={e}>
          <span>{meta(e).label}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${(probs[e] || 0) * 100}%`, background: meta(e).color }} /></span>
          <span className="num">{((probs[e] || 0) * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

function Details({ seg, interp }) {
  const t = seg.timings_ms;
  return (
    <div className="details">
      <div className="bars-grid">
        <Bars title="Combined" probs={seg.distribution} />
        <Bars title="Words only" probs={seg.text_probs} />
        <Bars title="Voice only" probs={seg.audio_probs} />
      </div>
      <div className="details-row">
        <audio controls preload="none" src={seg.audio_url} />
        <dl className="kv">
          {seg.tone && <><dt>Tone</dt><dd>{seg.tone}</dd></>}
          {seg.prosody?.words_per_sec != null && <><dt>Pace</dt><dd>{seg.prosody.words_per_sec} words/s</dd></>}
          <dt>Speaker match</dt><dd>{(seg.similarity * 100).toFixed(0)}%</dd>
          {t && <><dt>Processing</dt><dd>{t.total} ms <span className="muted">(speaker {t.speaker} · speech {t.asr} · voice {t.voice} · emotion {t.emotion})</span></dd></>}
          {interp?.latency != null && <><dt>Explanation</dt><dd>{interp.latency} ms</dd></>}
        </dl>
      </div>
    </div>
  );
}

function Interpretation({ state, onRetry }) {
  if (!state) return null;
  if (state.status === "loading") {
    return (
      <div className="interp interp-loading" aria-live="polite">
        <SparkIcon /> Working out what they mean…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="interp interp-error">
        <AlertIcon /> Couldn’t explain this turn. <span className="muted">{state.error}</span>
        <button className="btn-text" onClick={onRetry}>Try again</button>
      </div>
    );
  }
  const d = state.data;
  if (!d) return null;
  const subtext = SUBTEXT_LABELS[d.subtext];
  return (
    <div className={`interp certainty-${d.certainty}`} aria-live="polite">
      <div className="interp-head">
        <span className="interp-label">What they mean</span>
        {subtext && <span className="subtext-badge">{subtext}</span>}
        <span className={`certainty certainty-${d.certainty}`} title="How sure the explanation is">
          {d.certainty === "high" ? "Clear" : d.certainty === "medium" ? "Fairly sure" : "Not sure"}
        </span>
        {canSpeak() && (
          <button className="icon-btn small" aria-label="Read aloud" onClick={() => speak(interpretationSpeech(d))}>
            <SpeakerIcon />
          </button>
        )}
      </div>
      <p className="plain">{d.plain_meaning}</p>
      {subtext && d.subtext_explanation && <p className="subtext-text">{d.subtext_explanation}</p>}
      {d.feeling && <p className="feeling"><span className="muted">Feeling:</span> {d.feeling}</p>}
      {d.expects_reply && (
        <div className="reply">
          <ReplyIcon />
          <div>
            <strong>They’re waiting for a reply.</strong>
            {d.suggested_reply && <p>You could say: “{d.suggested_reply}”</p>}
          </div>
        </div>
      )}
      {d.evidence?.length > 0 && (
        <details className="why">
          <summary>Why?</summary>
          <ul>{d.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

function Turn({ seg, style, interp, onRetry, time, onSelect, active }) {
  const [open, setOpen] = useState(false);
  const mm = mismatch(seg);
  return (
    <li className={`turn side-${style.side} ${active ? "active" : ""}`} style={{ "--spk": style.color }}>
      <div className="avatar" aria-hidden>{initials(seg.speaker_id)}</div>
      <div className="turn-body">
        <div className="turn-meta">
          <strong>{seg.speaker_id}{seg.speaker_uncertain && <span className="muted" title="Short turn: speaker is a best guess"> ?</span>}</strong>
          {onSelect ? (
            <button className="time-link" onClick={() => onSelect(seg)} title="Play this line in the video">▶ {fmtTime(time)}</button>
          ) : (
            <span className="muted">{fmtTime(time)}</span>
          )}
        </div>
        <div className="bubble">
          <p className="transcript">{seg.transcript}</p>
          <div className="chips">
            <EmotionChip emotion={seg.emotion} confidence={seg.confidence} />
            {mm && (
              <span className="chip chip-warn" title={`Words suggest ${mm.words[0]}, voice suggests ${mm.voice[0]}`}>
                Words and voice differ
              </span>
            )}
            {(seg.cues || []).map((c) => <span className="chip" key={c}>{c}</span>)}
          </div>
          <button className="btn-text details-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            <ChevronIcon className={open ? "rot" : ""} /> {open ? "Hide details" : "Details"}
          </button>
          {open && <Details seg={seg} interp={interp} />}
        </div>
        <Interpretation state={interp} onRetry={onRetry} />
      </div>
    </li>
  );
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

/**
 * timeOf(seg)   - seconds shown next to the speaker (default: stream time)
 * onSelect(seg) - makes that time a button (Watch mode: jump the video there)
 * newestFirst   - newest turn at the top and no auto-scroll (Watch mode keeps the video in view)
 * activeKey     - audio_url of the turn to highlight
 */
export default function Conversation({ segments, interpretations, onRetry, pendingLabel, empty,
                                       timeOf = (s) => s.start_sec, onSelect, newestFirst = false, activeKey }) {
  const styles = speakerStyles(segments);
  const endRef = useRef(null);
  useEffect(() => {
    if (!newestFirst) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [segments.length, pendingLabel, newestFirst]);

  if (!segments.length && !pendingLabel) return empty;
  const pending = pendingLabel && (
    <li className="turn pending" aria-live="polite">
      <div className="avatar" aria-hidden>…</div>
      <div className="bubble pending-bubble"><span className="dots"><i /><i /><i /></span> {pendingLabel}</div>
    </li>
  );
  const turns = segments.map((seg) => (
    <Turn
      key={seg.audio_url}
      seg={seg}
      style={styles[seg.speaker_id]}
      interp={interpretations[seg.audio_url]}
      onRetry={() => onRetry(seg.audio_url)}
      time={timeOf(seg)}
      onSelect={onSelect}
      active={seg.audio_url === activeKey}
    />
  ));
  return (
    <ol className="conversation" aria-label="Conversation">
      {newestFirst ? <>{pending}{turns.reverse()}</> : <>{turns}{pending}</>}
      <li ref={endRef} className="end-anchor" aria-hidden />
    </ol>
  );
}


