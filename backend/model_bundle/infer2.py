"""
Step 3 - live inference + LLM-ready emotion context.

    engine = EmotionContextEngine("/workspace/runs/rl_wavlm_s42")
    r = engine.update("Speaker 1", "I can't believe you did that!", audio_16k_float32)   # per finished utterance
    prompt_block = engine.format_for_llm(last_n=12)                                       # give this to the LLM

Plug-in point in the Streamlit app / FastAPI backend: call engine.update(...) where the old
BERT+MFCC model was called, using the diarizer's "Speaker N" label and the Whisper transcript.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from common2 import (ContextBuilder, EMOTIONS, MAX_SPEAKERS, SENTIMENTS, MultimodalERC, AudioFeatureExtractor,
                    clean_text, prepare_waveform, softmax_np)

LLM_AFFECT_PREAMBLE = (
    "Each line below is one utterance with an automatic estimate of the speaker's emotion, inferred from "
    "the words, the preceding conversation and the tone of voice. 'confident' means the model was clearly "
    "sure, 'likely' means moderately sure, 'unclear' means it was not - treat unclear tags as weak hints, "
    "never as facts. 'tone' is the overall positive/negative/neutral sentiment and is usually more "
    "reliable than the exact emotion. Emotions can only be one of: " + ", ".join(EMOTIONS) + "."
)


class EmotionContextEngine:
    def __init__(self, ckpt_dir: str, device: Optional[str] = None, audio_model: Optional[str] = None,
                 audio_fp16: bool = False):
        from transformers import AutoConfig, AutoTokenizer

        ckpt = Path(ckpt_dir)
        self.cfg = json.load(open(ckpt / "config.json"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        kw = {"add_prefix_space": True} if self.cfg.get("add_prefix_space") else {}
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer", use_fast=True, **kw)
        self.builder = ContextBuilder(self.tokenizer, self.cfg["max_len"], self.cfg["max_history"],
                                      self.cfg["max_utt_tokens"])
        audio_shape = tuple(self.cfg["audio_shape"]) if self.cfg["use_audio"] else None
        self.model = MultimodalERC(
            text_config=AutoConfig.from_pretrained(ckpt / "text_config"), audio_shape=audio_shape,
            hidden=self.cfg["hidden"], dropout=self.cfg["dropout"], load_pretrained=False)
        self.model.load_state_dict(torch.load(ckpt / "best.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.audio = None
        if self.cfg["use_audio"]:
            self.audio = AudioFeatureExtractor(audio_model or self.cfg["audio_model"], str(self.device),
                                               fp16=audio_fp16)
        self.reset()

    # ------------------------------------------------------------------ session state
    def reset(self):
        """Call at the start of a new conversation / recording session."""
        self._speaker_index: Dict[str, int] = {}
        self._hist: List[tuple] = []          # (speaker_idx, token_ids) oldest -> newest
        self.timeline: List[Dict] = []

    def _spk(self, label: str) -> int:
        key = str(label).strip().lower()
        if key not in self._speaker_index:
            self._speaker_index[key] = min(len(self._speaker_index), MAX_SPEAKERS - 1)
        return self._speaker_index[key]

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def update(self, speaker: str, text: str, audio: Optional[np.ndarray] = None,
               commit: bool = True) -> Dict:
        """Classify one finished utterance given everything said before it, then (optionally) add it
        to the history. `audio` = mono float32 waveform @16 kHz of exactly this utterance (or None)."""
        text = clean_text(text)
        spk = self._spk(speaker)
        ids = self.builder.encode_text(text)
        input_ids, ts, te = self.builder.build(self._hist, (spk, ids))
        x = torch.tensor([input_ids], device=self.device)
        attn = torch.ones_like(x)
        tmask = torch.zeros_like(x, dtype=torch.bool)
        tmask[0, ts:te] = True

        audio_t, ok = None, None
        if self.audio is not None and audio is not None and len(audio) > 0:
            feats = self.audio(prepare_waveform(audio, self.cfg.get("audio_max_seconds", 15.0)))
            audio_t = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)
            ok = torch.ones(1, dtype=torch.bool, device=self.device)

        out = self.model(x, attn, tmask, audio_t, ok)
        probs = softmax_np(out["emo"].float().cpu().numpy()[0] / self.temperature)
        sent = softmax_np(out["sent"].float().cpu().numpy()[0])
        order = np.argsort(-probs)
        result = {
            "speaker": speaker, "text": text,
            "emotion": EMOTIONS[order[0]], "confidence": float(probs[order[0]]),
            "top2": [(EMOTIONS[i], float(probs[i])) for i in order[:2]],
            "probs": {e: float(p) for e, p in zip(EMOTIONS, probs)},
            "sentiment": {s: float(p) for s, p in zip(SENTIMENTS, sent)},
            "tone": SENTIMENTS[int(sent.argmax())],
            "audio_used": audio_t is not None,
        }
        if commit:
            self._hist.append((spk, ids))
            self.timeline.append(result)
        return result

    # ------------------------------------------------------------------ LLM-facing output
    @staticmethod
    def _describe(r: Dict, hi: float = 0.65, lo: float = 0.45) -> str:
        (e1, p1), (e2, p2) = r["top2"]
        if p1 >= hi:
            return f"{e1} (confident, {p1:.2f})"
        if p1 >= lo:
            return f"likely {e1} ({p1:.2f})" + (f", possibly {e2} ({p2:.2f})" if p2 >= 0.2 else "")
        return f"unclear: {e1} ({p1:.2f}) or {e2} ({p2:.2f})"

    def format_for_llm(self, last_n: Optional[int] = 12, with_preamble: bool = False) -> str:
        rows = self.timeline[-last_n:] if last_n else self.timeline
        def tag(r):
            tone = r["tone"] if r["sentiment"][r["tone"]] >= 0.5 else None
            return self._describe(r) + (f"; tone: {tone}" if tone else "")

        lines = [f'{r["speaker"]}: "{r["text"]}"   [emotion: {tag(r)}]' for r in rows]
        body = "\n".join(lines)
        traj = self.trajectories(last_n)
        if traj:
            body += "\n\nEmotion trajectory per speaker (confident/likely readings only):\n" + traj
        return (LLM_AFFECT_PREAMBLE + "\n\n" + body) if with_preamble else body

    def trajectories(self, last_n: Optional[int] = 12, min_conf: float = 0.45) -> str:
        rows = self.timeline[-last_n:] if last_n else self.timeline
        per: "OrderedDict[str, List[str]]" = OrderedDict()
        for r in rows:
            if r["confidence"] >= min_conf:
                seq = per.setdefault(r["speaker"], [])
                if not seq or seq[-1] != r["emotion"]:
                    seq.append(r["emotion"])
        return "\n".join(f"- {s}: " + " -> ".join(seq) for s, seq in per.items() if seq)

    def to_json(self) -> str:
        return json.dumps(self.timeline, indent=2)
