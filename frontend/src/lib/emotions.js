export const EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"];

// Muted, distinguishable hues; CSS variables so dark mode can retune them.
export const EMOTION_META = {
  neutral: { label: "Neutral", color: "var(--emo-neutral)" },
  joy: { label: "Joy", color: "var(--emo-joy)" },
  surprise: { label: "Surprise", color: "var(--emo-surprise)" },
  anger: { label: "Anger", color: "var(--emo-anger)" },
  sadness: { label: "Sadness", color: "var(--emo-sadness)" },
  disgust: { label: "Disgust", color: "var(--emo-disgust)" },
  fear: { label: "Fear", color: "var(--emo-fear)" },
};

export const meta = (e) => EMOTION_META[e] || { label: e || "—", color: "var(--emo-neutral)" };

export const strength = (p) => (p >= 0.65 ? "confident" : p >= 0.45 ? "likely" : "unclear");

export function top(probs) {
  if (!probs) return null;
  let best = null;
  for (const [e, p] of Object.entries(probs)) if (!best || p > best[1]) best = [e, p];
  return best;
}

/** Words and voice point to different emotions, both with some conviction. */
export function mismatch(seg) {
  const w = top(seg.text_probs);
  const v = top(seg.audio_probs);
  return !!(w && v && w[0] !== v[0] && w[1] >= 0.4 && v[1] >= 0.4) && { words: w, voice: v };
}

const SPEAKER_COLORS = [
  "var(--spk-1)", "var(--spk-2)", "var(--spk-3)", "var(--spk-4)", "var(--spk-5)", "var(--spk-6)",
];

/** Stable colour + side per speaker, by order of first appearance. */
export function speakerStyles(segments) {
  const map = {};
  for (const s of segments) {
    if (!(s.speaker_id in map)) {
      const n = Object.keys(map).length;
      map[s.speaker_id] = { color: SPEAKER_COLORS[n % SPEAKER_COLORS.length], side: n === 1 ? "right" : "left", index: n };
    }
  }
  return map;
}

export const initials = (id) => {
  const m = /(\d+)/.exec(id || "");
  return m ? `S${m[1]}` : "?";
};

export const SUBTEXT_LABELS = {
  none: null,
  sarcasm: "Sarcasm",
  joke: "Joking",
  hint: "A hint",
  indirect_request: "Indirect request",
  polite_refusal: "Polite “no”",
  exaggeration: "Exaggeration",
  criticism: "Criticism",
  other: "Hidden meaning",
};
