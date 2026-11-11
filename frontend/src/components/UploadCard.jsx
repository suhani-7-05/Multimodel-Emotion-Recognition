import { useRef, useState } from "react";
import { UploadIcon } from "./Icons.jsx";

export default function UploadCard({ ready, running, onRun, error, summary }) {
  const [file, setFile] = useState(null);
  const [drag, setDrag] = useState(false);
  const input = useRef(null);

  const pick = (f) => f && setFile(f);

  return (
    <section className="card upload">
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      >
        <UploadIcon width={26} height={26} />
        <strong>{file ? file.name : "Drop a recording here"}</strong>
        <span className="muted small">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "or click to choose · WAV, MP3, M4A"}</span>
        <input ref={input} type="file" accept="audio/*,.wav" hidden onChange={(e) => pick(e.target.files[0])} />
      </div>
      <button className="btn primary block" disabled={!file || !ready || running} onClick={() => onRun(file)}>
        {running ? "Analysing…" : "Analyse recording"}
      </button>
      {!ready && <p className="muted small">Load the models first.</p>}
      {summary && (
        <p className="small">
          <strong className="num">{summary.duration_sec.toFixed(1)} s</strong> · {summary.speakers} speaker{summary.speakers === 1 ? "" : "s"} · {summary.segments.length} turns
        </p>
      )}
      {error && <p className="error-text">{error}</p>}
    </section>
  );
}
