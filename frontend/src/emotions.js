export const EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"];

export const EMOTION_META = {
  neutral: { emoji: "😐", color: "#6c757d" },
  joy: { emoji: "😊", color: "#28a745" },
  surprise: { emoji: "😲", color: "#fd7e14" },
  anger: { emoji: "😠", color: "#dc3545" },
  sadness: { emoji: "😢", color: "#17a2b8" },
  disgust: { emoji: "🤢", color: "#795548" },
  fear: { emoji: "😨", color: "#6f42c1" },
};

export function meta(emotion) {
  return EMOTION_META[emotion] || { emoji: "🎭", color: "#6c757d" };
}

export function mostFrequent(segments) {
  const counts = {};
  for (const s of segments) counts[s.emotion] = (counts[s.emotion] || 0) + 1;
  let best = null;
  for (const [e, n] of Object.entries(counts)) if (!best || n > counts[best]) best = e;
  return best;
}
