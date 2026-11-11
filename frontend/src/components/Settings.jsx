const FIELDS = [
  { key: "vad_threshold", label: "VAD speech threshold", min: 0.1, max: 0.9, step: 0.05, live: true },
  { key: "silence_duration", label: "Silence before finalizing (s)", min: 0.5, max: 2.5, step: 0.1, live: true },
  { key: "min_speech_duration", label: "Minimum utterance (s)", min: 0.3, max: 1.5, step: 0.1, live: true },
  { key: "speaker_threshold", label: "Speaker similarity threshold", min: 0.2, max: 0.7, step: 0.02 },
];

export default function Settings({ settings, onChange, disabled }) {
  return (
    <section className="panel">
      <h2>Pipeline parameters</h2>
      {FIELDS.map((f) => (
        <label className="slider" key={f.key}>
          <span>
            {f.label}
            {f.live && <em className="muted"> · live only</em>}
          </span>
          <span className="slider-row">
            <input
              type="range"
              min={f.min}
              max={f.max}
              step={f.step}
              value={settings[f.key]}
              disabled={disabled}
              onChange={(e) => onChange({ ...settings, [f.key]: parseFloat(e.target.value) })}
            />
            <output>{settings[f.key].toFixed(2)}</output>
          </span>
        </label>
      ))}
      <p className="muted small">A lower speaker similarity threshold merges more speakers.</p>
    </section>
  );
}
