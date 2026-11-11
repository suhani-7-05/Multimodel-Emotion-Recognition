// Read-aloud through the browser's built-in speech synthesis (no server, works offline
// on most systems). Prefers a natural-sounding English voice when one is installed.
let cachedVoice = null;

function pickVoice() {
  if (cachedVoice) return cachedVoice;
  const voices = window.speechSynthesis?.getVoices() || [];
  const en = voices.filter((v) => v.lang?.toLowerCase().startsWith("en"));
  cachedVoice =
    en.find((v) => /natural|neural|online/i.test(v.name)) ||
    en.find((v) => v.default) ||
    en[0] ||
    null;
  return cachedVoice;
}

export const canSpeak = () => typeof window !== "undefined" && "speechSynthesis" in window;

export function speak(text, { interrupt = true } = {}) {
  if (!canSpeak() || !text) return;
  if (interrupt) window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  const v = pickVoice();
  if (v) u.voice = v;
  u.rate = 0.95;
  window.speechSynthesis.speak(u);
}

export function interpretationSpeech(interp) {
  if (!interp) return "";
  const parts = [interp.plain_meaning];
  if (interp.subtext && interp.subtext !== "none" && interp.subtext_explanation) parts.push(interp.subtext_explanation);
  if (interp.expects_reply && interp.suggested_reply) parts.push(`You could say: ${interp.suggested_reply}`);
  return parts.join(" ");
}
