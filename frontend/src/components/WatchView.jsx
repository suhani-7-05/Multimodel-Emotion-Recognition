import { useEffect, useMemo, useRef, useState } from "react";
import { SUBTEXT_LABELS, meta, speakerStyles } from "../lib/emotions.js";
import { interpretationSpeech, speak } from "../lib/speech.js";
import Conversation from "./Conversation.jsx";
import Insights from "./Insights.jsx";
import { SpeakerIcon, UploadIcon } from "./Icons.jsx";

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const isHidden = (d) => d && d.subtext && d.subtext !== "none" && d.certainty !== "low";

function Spotlight({ spot, onContinue }) {
  const { seg, data } = spot;
  return (
    <div className="spotlight" role="dialog" aria-label="Hidden meaning">
      <div className="spotlight-card">
        <div className="spotlight-head">
          <span className="subtext-badge">{SUBTEXT_LABELS[data.subtext] || "Hidden meaning"}</span>
          <span className="muted small">{seg.speaker_id} · {fmt(seg.media_start)}</span>
        </div>
        <p className="spotlight-quote">“{seg.transcript}”</p>
        <p className="plain">{data.plain_meaning}</p>
        {data.subtext_explanation && <p className="subtext-text">{data.subtext_explanation}</p>}
        <div className="spotlight-actions">
          <button className="btn" onClick={() => new Audio(seg.audio_url).play()}><SpeakerIcon /> Hear the line again</button>
          <button className="btn primary" onClick={onContinue} autoFocus>Continue watching</button>
        </div>
      </div>
    </div>
  );
}

function Caption({ seg, interp, color }) {
  if (!seg) return <div className="caption caption-empty muted">Lines appear here as they are understood.</div>;
  const m = meta(seg.emotion);
  const d = interp?.status === "done" ? interp.data : null;
  return (
    <div className="caption" style={{ "--spk": color }}>
      <div className="caption-line">
        <strong className="caption-speaker">{seg.speaker_id}</strong>
        <span className="caption-text">{seg.transcript}</span>
        <span className="chip emo-chip" style={{ "--c": m.color }}><span className="emo-dot" />{m.label}</span>
      </div>
      {d && (
        <p className="caption-meaning">
          {isHidden(d) && <span className="subtext-badge">{SUBTEXT_LABELS[d.subtext]}</span>} {d.plain_meaning}
        </p>
      )}
      {interp?.status === "loading" && <p className="caption-meaning muted">Working out what they mean…</p>}
    </div>
  );
}

export default function WatchView({ watch, segments, interp, settings, prefs, setPrefs, ready, onNewVideo }) {
  const videoRef = useRef(null);
  const inputRef = useRef(null);
  const [video, setVideo] = useState(null); // { url, name, size }
  const [now, setNow] = useState(0);
  const [spot, setSpot] = useState(null);
  const [drag, setDrag] = useState(false);
  const [muted, setMuted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const shownRef = useRef(new Set());
  const prevTimeRef = useRef(0);

  const ordered = useMemo(() => [...segments].sort((a, b) => a.media_start - b.media_start), [segments]);
  const styles = speakerStyles(segments);

  const open = (file) => {
    if (!file) return;
    if (watch.status !== "idle") watch.stop();
    if (video?.url) URL.revokeObjectURL(video.url);
    shownRef.current = new Set();
    setSpot(null);
    setNow(0);
    onNewVideo();
    setVideo({ url: URL.createObjectURL(file), name: file.name, size: file.size });
  };
  useEffect(() => () => video?.url && URL.revokeObjectURL(video.url), [video?.url]);

  // Start analysing whenever the film plays into territory not analysed yet.
  const maybeStart = () => {
    const el = videoRef.current;
    if (!el || el.paused || el.ended || !ready || watch.status !== "idle") return;
    if (el.currentTime >= watch.frontierRef.current - 0.1) watch.start(el, settings);
  };

  const onTime = () => {
    const el = videoRef.current;
    if (!el) return;
    const t = el.currentTime;
    // Rewatching: results are already known, so pause exactly at the end of a hidden-meaning line.
    if (prefs.pauseOnSubtext && !el.paused && t < watch.frontierRef.current) {
      const hit = ordered.find((s) => s.media_end > prevTimeRef.current && s.media_end <= t && isHidden(interp.results[s.audio_url]?.data));
      if (hit) {
        el.pause();
        setSpot({ seg: hit, data: interp.results[hit.audio_url].data });
      }
    }
    prevTimeRef.current = t;
    setNow(t);
    maybeStart();
  };

  // First watch: explanations arrive a few seconds after a line. Pause then if it hides something.
  useEffect(() => {
    for (const seg of segments) {
      const r = interp.results[seg.audio_url];
      if (r?.status !== "done" || shownRef.current.has(seg.audio_url)) continue;
      shownRef.current.add(seg.audio_url);
      const hidden = isHidden(r.data);
      if (prefs.autoSpeak && (!prefs.onlySubtext || hidden)) speak(interpretationSpeech(r.data), { interrupt: false });
      const el = videoRef.current;
      if (hidden && prefs.pauseOnSubtext && el && !el.paused) {
        el.pause();
        setSpot({ seg, data: r.data });
      }
    }
  }, [interp.results, segments, prefs.autoSpeak, prefs.onlySubtext, prefs.pauseOnSubtext]);

  const current = useMemo(() => {
    let best = null;
    for (const s of ordered) {
      if (s.media_start <= now + 0.05 && now <= s.media_end + 4) best = s;
    }
    return best;
  }, [ordered, now]);

  const seek = (seg) => {
    const el = videoRef.current;
    if (!el) return;
    setSpot(null);
    el.currentTime = Math.max(0, seg.media_start - 0.3);
    prevTimeRef.current = el.currentTime;
    el.play();
  };

  const behind = video && now < watch.frontierRef.current - 0.25;
  const s = watch.stats;
  const statusText =
    !video ? "" :
    !ready ? "Load the models to analyse the film" :
    watch.status === "connecting" ? "Connecting…" :
    watch.status === "stopping" ? "Finishing the last lines…" :
    behind || watch.rewatching ? "Rewatching — showing earlier results" :
    watch.status === "running" && !playing ? (s?.analyzing || s?.pending > 0 ? "Paused · still understanding the last line…" : "Paused") :
    watch.status === "running" ? (s?.speaking ? "Hearing dialogue" : s?.analyzing || s?.pending > 0 ? "Understanding the last line…" : "Listening to the film") :
    "Press play to start";

  if (!video) {
    return (
      <>
        <main className="stream">
          <div
            className={`empty watch-drop ${drag ? "drag" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); open(e.dataTransfer.files[0]); }}
          >
            <h2>Watch a scene, understand every line.</h2>
            <p>
              Open a film or TV clip. It plays normally with picture and sound, while Subtext listens
              to the dialogue and explains what each person means — including sarcasm and hints.
              The video never leaves your computer; only its sound is analysed.
            </p>
            <button className="btn primary" onClick={() => inputRef.current?.click()}><UploadIcon /> Open a video</button>
            <p className="muted small">or drop it here · MP4 or WebM work best</p>
            <input ref={inputRef} type="file" accept="video/*,.mkv" hidden onChange={(e) => open(e.target.files[0])} />
          </div>
        </main>
        <aside className="rail">
          <Insights segments={segments} interpretations={interp.results} />
        </aside>
      </>
    );
  }

  return (
    <>
      <main className="stream watch">
        <div className="stage">
          <video
            ref={videoRef}
            src={video.url}
            controls
            playsInline
            onPlay={() => { setPlaying(true); maybeStart(); }}
            onPause={() => setPlaying(false)}
            onTimeUpdate={onTime}
            onSeeked={() => { prevTimeRef.current = videoRef.current?.currentTime || 0; setNow(prevTimeRef.current); }}
            onEnded={() => watch.status === "running" && watch.stop()}
            onVolumeChange={() => setMuted(!!videoRef.current?.muted || videoRef.current?.volume === 0)}
          />
          {spot && <Spotlight spot={spot} onContinue={() => { setSpot(null); videoRef.current?.play(); }} />}
        </div>
        <Caption seg={current} interp={current && interp.results[current.audio_url]} color={current && styles[current.speaker_id]?.color} />
        <div className="status-strip" aria-live="polite">
          <span className={`pill-dot ${watch.status === "running" && playing && !behind ? "live" : ""}`} />
          {statusText}
          {watch.frontierRef.current > 0 && <span className="muted small"> · analysed up to {fmt(watch.frontierRef.current)}</span>}
        </div>
        <Conversation
          segments={segments}
          interpretations={interp.results}
          onRetry={interp.retry}
          timeOf={(seg) => seg.media_start}
          onSelect={seek}
          newestFirst
          activeKey={current?.audio_url}
          empty={<p className="muted watch-hint">Each line will appear here, newest first. Click a time to jump to it.</p>}
        />
      </main>

      <aside className="rail">
        <section className="card watch-file">
          <strong className="file-name" title={video.name}>{video.name}</strong>
          <span className="muted small">{(video.size / 1024 / 1024).toFixed(0)} MB · stays on this computer</span>
          <div className="buttons">
            <button className="btn" onClick={() => inputRef.current?.click()}>Open another</button>
            {watch.status === "running" && <button className="btn ghost" onClick={watch.stop}>Stop analysing</button>}
          </div>
          <input ref={inputRef} type="file" accept="video/*,.mkv" hidden onChange={(e) => open(e.target.files[0])} />
          <label className="switch">
            <input type="checkbox" checked={prefs.pauseOnSubtext} onChange={(e) => setPrefs({ ...prefs, pauseOnSubtext: e.target.checked })} />
            <span>Pause when a line has a hidden meaning</span>
          </label>
          <label className="switch">
            <input type="checkbox" checked={prefs.autoSpeak} onChange={(e) => setPrefs({ ...prefs, autoSpeak: e.target.checked })} />
            <span>Read explanations aloud</span>
          </label>
          {muted && <p className="warn-text small">The video is muted — Subtext can only analyse what it can hear.</p>}
          {watch.errors.length > 0 && <ul className="errors">{watch.errors.slice(-3).map((e, i) => <li key={i}>{e}</li>)}</ul>}
        </section>
        <Insights segments={segments} interpretations={interp.results} />
        <p className="disclaimer small muted">
          Film soundtracks mix music and effects with speech, so some lines may be missed or
          misheard. If music is picked up as speech, raise “Speech detection strictness” in Settings.
        </p>
      </aside>
    </>
  );
}
