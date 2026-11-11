"""
Plain-language interpreter: turns a conversation + the emotion model's evidence into a
short, literal explanation of what the speaker most likely means.

The emotion model supplies *evidence* (fused emotion, what the words alone and the voice
alone suggest, prosody cues). The LLM does the pragmatic part the classifier cannot:
sarcasm, irony, hints, indirect requests, polite refusals. It is told the evidence is
probabilistic and must hedge when it is weak.

Backends: Ollama's native API (laptop) or any OpenAI-compatible server such as vLLM
(college pod). Both are asked for JSON that matches SCHEMA.
"""
import json
import logging
import time
from typing import Dict, List, Optional

import httpx

from . import config

LOG = logging.getLogger("interpreter")

SUBTEXT_TYPES = ["none", "sarcasm", "joke", "hint", "indirect_request", "polite_refusal",
                 "exaggeration", "criticism", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "plain_meaning": {"type": "string"},
        "feeling": {"type": "string"},
        "subtext": {"type": "string", "enum": SUBTEXT_TYPES},
        "subtext_explanation": {"type": "string"},
        "expects_reply": {"type": "boolean"},
        "suggested_reply": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "certainty": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["plain_meaning", "feeling", "subtext", "subtext_explanation",
                 "expects_reply", "suggested_reply", "evidence", "certainty"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You help a listener understand what people in a conversation really mean.
Some listeners find it hard to pick up tone of voice, sarcasm, irony, jokes, hints, indirect
requests or polite refusals. Your job is to explain the TARGET line in plain, literal language.

Write for clarity:
- Short sentences. Everyday words. No idioms, metaphors or sarcasm of your own.
- "plain_meaning": what the speaker most likely means and wants, in at most 30 words.
  If the line means exactly what it says, restate it simply.
- "feeling": how the speaker seems to feel, in at most 12 words, hedged ("seems", "probably").
- "subtext": the kind of hidden meaning, or "none".
- "subtext_explanation": one or two sentences on what is unsaid and how you can tell. Empty if none.
- "expects_reply": true if the speaker is waiting for an answer or an action.
- "suggested_reply": a short, kind, literal reply the listener could give. Empty if no reply is needed.
- "evidence": up to 3 short facts you relied on (words, context, voice cues).
- "certainty": high, medium or low.

Before answering, check the line against the situation:
1. Taken literally, does it fit what was just said? Would a person sincerely say this here?
   Praise or delight in response to something annoying or bad is usually sarcasm.
2. Is a statement really a request? ("It's late and I have an exam" after someone says they will
   play music usually means "please don't, or keep it quiet".)
3. Is a vague or delayed answer really a "no"? ("Maybe another time" often means no.)
4. If none of these apply, the line means what it says: subtext "none".

About the evidence:
- Emotion and voice estimates come from an automatic model trained on TV dialogue. They are
  often wrong. Treat "unclear" estimates as weak hints, never as facts.
- That model reads mostly the surface words, so it labels sarcastic praise ("Oh great", "Wonderful")
  as joy even when it is confident. A "joy" label never rules out sarcasm: judge sarcasm from what
  was said before. Bad news followed by delighted words is usually sarcasm.
- When the words and the voice disagree (for example positive words said in a flat or negative
  voice), consider sarcasm, but only claim it when the conversation supports it.
- Transcripts come from speech recognition and may contain mistakes.
- If the evidence is thin or conflicting, say so and set certainty to low.

Never diagnose, label or judge the speaker or the listener. Answer only with the JSON object."""


# --------------------------------------------------------------------------- prompt
def _top(probs: Optional[Dict]) -> Optional[tuple]:
    if not probs:
        return None
    e = max(probs, key=probs.get)
    return e, probs[e]


def _strength(p: float) -> str:
    return "confident" if p >= 0.65 else "likely" if p >= 0.45 else "unclear"


def _line(t: Dict) -> str:
    who = t.get("speaker_id", "Speaker ?") + (" (speaker uncertain)" if t.get("speaker_uncertain") else "")
    emo = t.get("emotion")
    conf = float(t.get("confidence") or 0)
    tag = f"emotion: {_strength(conf)} {emo} {conf:.2f}" if emo else "emotion: n/a"
    if t.get("tone"):
        tag += f"; tone: {t['tone']}"
    return f'{who}: "{t.get("transcript", "").strip()}"   [{tag}]'


# Worked examples (few-shot). Small local models follow the pattern far better with these.
_EXAMPLES = [
    (
        'Conversation so far (oldest first):\n'
        '[1] Speaker 1: "Sorry, I forgot to book the table, so we have nowhere to eat."   [emotion: likely sadness 0.50]\n\n'
        'TARGET [2] Speaker 2: "Oh, perfect. That\'s just great."   [emotion: unclear joy 0.40; tone: negative]\n\n'
        'Evidence for the TARGET line:\n- The words alone suggest: joy (0.70)\n- The voice alone suggests: anger (0.52)\n'
        '- NOTE: the words and the voice point to different emotions.',
        {"plain_meaning": "They are annoyed that the table was not booked. They do not think it is great.",
         "feeling": "Probably frustrated.", "subtext": "sarcasm",
         "subtext_explanation": "They said 'perfect' about a bad situation. People say positive words in an annoyed voice to show they are upset.",
         "expects_reply": True, "suggested_reply": "I'm sorry. I'll find another place for us now.",
         "evidence": ["Positive words about bad news", "Voice sounds angry, words sound happy"], "certainty": "high"},
    ),
    (
        'Conversation so far (oldest first):\n'
        '[1] Speaker 1: "Want to come to my place and watch a movie tonight?"   [emotion: likely joy 0.55]\n\n'
        'TARGET [2] Speaker 2: "Hmm, I\'m pretty tired this week. Maybe some other time."   [emotion: likely neutral 0.58; tone: neutral]\n\n'
        'Evidence for the TARGET line:\n- The words alone suggest: neutral (0.60)\n- The voice alone suggests: neutral (0.55)',
        {"plain_meaning": "They are saying no to the movie tonight.",
         "feeling": "Seems calm, maybe a bit uncomfortable.", "subtext": "polite_refusal",
         "subtext_explanation": "'Maybe some other time' is a polite way to say no. It may or may not mean they want to meet later.",
         "expects_reply": True, "suggested_reply": "No problem. Let me know if you want to do it another day.",
         "evidence": ["Gives a reason instead of yes", "'Maybe some other time' is a common soft no"], "certainty": "medium"},
    ),
    (
        'Conversation so far (oldest first):\n'
        '[1] Speaker 1: "What time does the train leave?"   [emotion: confident neutral 0.80]\n\n'
        'TARGET [2] Speaker 2: "At 6:15, from platform 2."   [emotion: confident neutral 0.82; tone: neutral]\n\n'
        'Evidence for the TARGET line:\n- The words alone suggest: neutral (0.85)\n- The voice alone suggests: neutral (0.70)',
        {"plain_meaning": "The train leaves at 6:15 from platform 2.",
         "feeling": "Seems calm.", "subtext": "none", "subtext_explanation": "",
         "expects_reply": False, "suggested_reply": "",
         "evidence": ["Direct answer to the question"], "certainty": "high"},
    ),
]


def build_messages(turns: List[Dict], target: int) -> List[Dict]:
    lo = max(0, target - config.LLM_CONTEXT_TURNS)
    history = turns[lo:target]
    cur = turns[target]

    lines = ["Conversation so far (oldest first):"]
    lines += [f"[{i + 1}] {_line(t)}" for i, t in enumerate(history, start=lo)] or ["(this is the first line)"]
    lines.append("")
    lines.append(f"TARGET [{target + 1}] {_line(cur)}")
    lines.append("")
    lines.append("Evidence for the TARGET line:")
    fused, words, voice = _top(cur.get("distribution")), _top(cur.get("text_probs")), _top(cur.get("audio_probs"))
    if fused:
        lines.append(f"- Combined estimate (words + voice + context): {_strength(fused[1])} {fused[0]} ({fused[1]:.2f})")
    if words:
        lines.append(f"- The words alone suggest: {words[0]} ({words[1]:.2f})")
    if voice:
        lines.append(f"- The voice alone suggests: {voice[0]} ({voice[1]:.2f})")
    if words and voice and words[0] != voice[0] and words[1] >= 0.4 and voice[1] >= 0.4:
        lines.append("- NOTE: the words and the voice point to different emotions.")
    sent = cur.get("sentiment") or {}
    if cur.get("tone") and sent:
        lines.append(f"- Overall tone: {cur['tone']} ({sent.get(cur['tone'], 0):.2f})")
    for cue in cur.get("cues") or []:
        lines.append(f"- Voice cue: {cue}")
    if cur.get("speaker_uncertain"):
        lines.append("- The speaker label is a guess (very short turn).")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prompt, answer in _EXAMPLES:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    messages.append({"role": "user", "content": "\n".join(lines)})
    return messages


# --------------------------------------------------------------------------- backends
class Interpreter:
    def __init__(self):
        self._client = httpx.Client(timeout=config.LLM_TIMEOUT)
        self._status_cache = (0.0, None)

    # ---- health
    def status(self) -> Dict:
        ts, cached = self._status_cache
        if cached is not None and time.monotonic() - ts < 10:
            return cached
        info = {"enabled": config.LLM_ENABLED, "api": config.LLM_API, "model": config.LLM_MODEL,
                "reachable": False, "model_available": False}
        if config.LLM_ENABLED:
            try:
                if config.LLM_API == "ollama":
                    r = self._client.get(f"{config.LLM_BASE_URL}/api/tags", timeout=3)
                    names = {m["name"] for m in r.json().get("models", [])}
                    info["model_available"] = (config.LLM_MODEL in names
                                               or f"{config.LLM_MODEL}:latest" in names)
                else:
                    r = self._client.get(f"{config.LLM_BASE_URL}/v1/models", headers=self._headers(), timeout=3)
                    info["model_available"] = config.LLM_MODEL in {m["id"] for m in r.json().get("data", [])}
                info["reachable"] = True
            except Exception as exc:
                info["error"] = f"{type(exc).__name__}: {exc}"
        self._status_cache = (time.monotonic(), info)
        return info

    def _headers(self) -> Dict:
        return {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}

    # ---- generation
    def _ollama(self, messages: List[Dict]) -> str:
        body = {
            "model": config.LLM_MODEL, "messages": messages, "stream": False,
            "format": SCHEMA, "think": config.LLM_THINK, "keep_alive": config.LLM_KEEP_ALIVE,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        }
        if config.LLM_NUM_GPU != "":
            body["options"]["num_gpu"] = int(config.LLM_NUM_GPU)
        r = self._client.post(f"{config.LLM_BASE_URL}/api/chat", json=body)
        if r.status_code == 400 and "think" in r.text:     # model without a thinking switch
            body.pop("think")
            r = self._client.post(f"{config.LLM_BASE_URL}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _openai(self, messages: List[Dict]) -> str:
        body = {
            "model": config.LLM_MODEL, "messages": messages, "temperature": 0.2,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "interpretation", "schema": SCHEMA, "strict": True}},
            # vLLM: Qwen3-style "thinking" switch (LLM_THINK). Ignored by other servers.
            "chat_template_kwargs": {"enable_thinking": config.LLM_THINK},
        }
        r = self._client.post(f"{config.LLM_BASE_URL}/v1/chat/completions", json=body, headers=self._headers())
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def interpret(self, turns: List[Dict], target: Optional[int] = None) -> Dict:
        if not config.LLM_ENABLED:
            raise RuntimeError("The interpreter is disabled (LLM_ENABLED=0).")
        if not turns:
            raise ValueError("No turns given.")
        target = len(turns) - 1 if target is None else target
        if not 0 <= target < len(turns):
            raise ValueError("target out of range")
        messages = build_messages(turns, target)
        t = time.perf_counter()
        raw = self._ollama(messages) if config.LLM_API == "ollama" else self._openai(messages)
        latency = (time.perf_counter() - t) * 1000
        data = json.loads(raw)
        if data.get("subtext") not in SUBTEXT_TYPES:
            data["subtext"] = "other"
        data["evidence"] = [str(e) for e in (data.get("evidence") or [])][:3]
        return {"interpretation": data, "latency_ms": round(latency), "model": config.LLM_MODEL}


interpreter = Interpreter()
