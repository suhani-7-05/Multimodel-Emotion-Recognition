"""
Shared code for the conversation-aware multimodal emotion model (trained on MELD).

Design goals (driven by the live pipeline, where an LLM consumes the output):
  * CAUSAL      - an utterance is classified from the past turns + itself, never the future.
  * SPEAKER-AWARE - history lines are tagged "Speaker 1:", "Speaker 2:" ... exactly like the
                  live diarizer ("Speaker N", numbered by first appearance), so there is no
                  train/serve mismatch and no character-name leakage from MELD.
  * MULTIMODAL  - RoBERTa-style text encoder over the dialogue history + frozen speech-SSL
                  (WavLM) utterance features, fused with modality dropout and per-modality
                  auxiliary losses so neither branch can dominate.
  * CALIBRATED  - probabilities are temperature-scaled on dev, so "0.8" means about 80%.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Labels / constants
# --------------------------------------------------------------------------- #
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
SENTIMENTS = ["neutral", "positive", "negative"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
SENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}
SAMPLE_RATE = 16000
MAX_SPEAKERS = 12

SPLITS = {
    "train": ("train_sent_emo.csv", "train_wav"),
    "dev": ("dev_sent_emo.csv", "dev_wav"),
    "test": ("test_sent_emo.csv", "test_wav"),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
_CP1252_FIX = str.maketrans({
    "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x85": "...",
    "\x96": "-", "\x97": "-", "\xa0": " ",
})
_AUDIO_RE = re.compile(r"dia(\d+)_utt(\d+)", re.I)
_AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4"}


def clean_text(s: str) -> str:
    s = str(s).translate(_CP1252_FIX)
    return re.sub(r"\s+", " ", s).strip()


def index_audio_dir(d: Path) -> Dict[Tuple[int, int], str]:
    """Map (dialogue_id, utterance_id) -> audio path, from names like dia12_utt3.wav."""
    found: Dict[Tuple[int, int], str] = {}
    if not d.is_dir():
        return found
    for p in d.rglob("*"):
        if p.suffix.lower() in _AUDIO_EXT:
            m = _AUDIO_RE.search(p.stem)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key not in found or not found[key].lower().endswith(".wav"):
                    found[key] = str(p)
    return found


def load_split(root: str, split: str, verbose: bool = True, asr_dir: Optional[str] = None) -> pd.DataFrame:
    csv_name, wav_dir = SPLITS[split]
    csv_path = Path(root) / csv_name
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    need = ["Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}; has {list(df.columns)}")

    out = pd.DataFrame({
        "dlg": df["Dialogue_ID"].astype(int),
        "utt": df["Utterance_ID"].astype(int),
        "speaker": df["Speaker"].astype(str).str.strip().str.lower(),
        "text": df["Utterance"].map(clean_text),
        "emotion_id": df["Emotion"].astype(str).str.strip().str.lower().map(EMO2ID),
        "sentiment_id": df["Sentiment"].astype(str).str.strip().str.lower().map(SENT2ID),
    })
    if out["emotion_id"].isna().any() or out["sentiment_id"].isna().any():
        bad = df.loc[out["emotion_id"].isna() | out["sentiment_id"].isna(), ["Emotion", "Sentiment"]]
        raise ValueError(f"Unknown emotion/sentiment labels in {csv_path}:\n{bad.drop_duplicates()}")
    out["emotion_id"] = out["emotion_id"].astype(int)
    out["sentiment_id"] = out["sentiment_id"].astype(int)
    out = out.sort_values(["dlg", "utt"], kind="stable").reset_index(drop=True)
    out["uid"] = [f"dia{d}_utt{u}" for d, u in zip(out.dlg, out.utt)]

    index = index_audio_dir(Path(root) / wav_dir)
    out["wav_path"] = [index.get((d, u), "") for d, u in zip(out.dlg, out.utt)]
    out["file_ok"] = out["wav_path"] != ""

    # optional Whisper transcripts (from transcribe_meld.py); falls back to the gold text when Whisper
    # returned nothing, so "asr_text" is always usable once a transcript file exists
    out["asr_text"] = ""
    if asr_dir:
        p = Path(asr_dir) / f"{split}_asr.csv"
        if p.exists():
            a = pd.read_csv(p, keep_default_na=False)
            m = dict(zip(a["uid"], a["asr_text"].astype(str)))
            asr = [clean_text(m.get(u, "")) for u in out.uid]
            out["asr_text"] = [t if t else g for t, g in zip(asr, out.text)]
        elif verbose:
            print(f"[{split}] no ASR transcripts at {p}")
    if verbose:
        print(f"[{split}] {len(out)} utterances, {out.dlg.nunique()} dialogues, "
              f"{int((~out.file_ok).sum())} without an audio file")
    return out


def build_dialog_index(df: pd.DataFrame) -> Tuple[List[List[int]], np.ndarray]:
    """
    prev_rows[i]: rows earlier in the same dialogue (oldest -> newest)   [causal]
    spk_idx[i]  : 0-based speaker number by FIRST APPEARANCE in the dialogue.
    First-appearance numbering restricted to turns up to t is identical to what the live
    diarizer (Speaker 1, Speaker 2, ...) produces, so training matches deployment.
    """
    prev_rows: List[List[int]] = [[] for _ in range(len(df))]
    spk_idx = np.zeros(len(df), dtype=np.int64)
    for _, g in df.groupby("dlg", sort=False):
        rows = list(g.index)
        seen: Dict[str, int] = {}
        for pos, r in enumerate(rows):
            name = df.at[r, "speaker"]
            if name not in seen:
                seen[name] = len(seen)
            spk_idx[r] = min(seen[name], MAX_SPEAKERS - 1)
            prev_rows[r] = rows[:pos]
    return prev_rows, spk_idx


# --------------------------------------------------------------------------- #
# Causal, speaker-aware context builder
# --------------------------------------------------------------------------- #
class ContextBuilder:
    """
    ids = [CLS] (hist_1 [SEP]) (hist_2 [SEP]) ... [SEP] (cur) [SEP]
    every line is "Speaker n:" + utterance tokens. The newest history lines are kept first when
    the length budget is tight. The current utterance is never truncated away.
    """

    def __init__(self, tokenizer, max_len: int = 256, max_history: int = 12, max_utt_tokens: int = 64):
        self.tok = tokenizer
        self.max_len = max_len
        self.max_history = max_history
        self.max_utt_tokens = max_utt_tokens
        self.cls = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.tag_ids = [
            tokenizer(f"Speaker {n}:", add_special_tokens=False)["input_ids"]
            for n in range(1, MAX_SPEAKERS + 1)
        ]
        if any(len(t) == 0 for t in self.tag_ids):
            raise ValueError("Tokenizer produced no tokens for 'Speaker n:' - wrong or broken tokenizer?")

    def encode_text(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"][: self.max_utt_tokens]

    def build(self, hist: Sequence[Tuple[int, List[int]]], cur: Tuple[int, List[int]]):
        """hist: oldest->newest [(speaker_idx, token_ids)]; cur: (speaker_idx, token_ids).
        Returns (input_ids, target_start, target_end) with the target span = current line."""
        cur_line = self.tag_ids[cur[0]] + cur[1]
        budget = self.max_len - 3 - len(cur_line)
        kept: List[List[int]] = []
        used = 0
        for spk, ids in reversed(list(hist)[-self.max_history:]):
            line = self.tag_ids[spk] + ids
            if used + len(line) + 1 > budget:
                break
            kept.append(line)
            used += len(line) + 1
        kept.reverse()
        ids: List[int] = [self.cls]
        for line in kept:
            ids += line + [self.sep]
        ids.append(self.sep)
        target_start = len(ids)
        ids += cur_line
        target_end = len(ids)
        ids.append(self.sep)
        return ids, target_start, target_end


class DialogDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, builder: ContextBuilder, audio_path: Optional[str] = None,
                 audio_ok_path: Optional[str] = None, train: bool = False, speaker_aug: float = 0.5,
                 text_mode: str = "gold", asr_prob: float = 0.0):
        """text_mode: 'gold' = human transcripts, 'asr' = Whisper transcripts, 'mix' = each sample uses
        Whisper text with probability asr_prob (training on realistic ASR noise)."""
        self.df = df
        self.builder = builder
        self.train = train
        self.speaker_aug = speaker_aug
        self.text_mode = text_mode
        self.asr_prob = asr_prob
        self.prev_rows, self.spk_idx = build_dialog_index(df)
        self.has_asr = bool((df["asr_text"] != "").any()) if "asr_text" in df else False
        if text_mode in ("asr", "mix") and not self.has_asr:
            raise ValueError("text_mode needs Whisper transcripts: run transcribe_meld.py and pass --asr_dir")
        self.utt_ids = [builder.encode_text(t) for t in df.text]
        self.utt_ids_asr = [builder.encode_text(t) for t in df.asr_text] if self.has_asr else None
        self.n_spk = df.groupby("dlg")["speaker"].transform("nunique").to_numpy()
        self.y = df.emotion_id.to_numpy()
        self.ys = df.sentiment_id.to_numpy()
        self.audio_path = audio_path
        self._feats = None
        if audio_ok_path is not None:
            ok = np.load(audio_ok_path).astype(bool)
            assert len(ok) == len(df), "audio_ok length does not match CSV; re-run extract_audio_features.py"
            self.audio_ok = ok
        else:
            self.audio_ok = np.zeros(len(df), dtype=bool)

    def __len__(self):
        return len(self.df)

    def _feat(self, i):
        if self._feats is None:
            self._feats = np.load(self.audio_path, mmap_mode="r")
        return np.asarray(self._feats[i], dtype=np.float32)

    def __getitem__(self, i):
        spk = self.spk_idx
        perm = None
        if self.train and self.speaker_aug > 0 and random.random() < self.speaker_aug:
            n = int(min(self.n_spk[i], MAX_SPEAKERS))
            p = list(range(n))
            random.shuffle(p)
            perm = p
        remap = (lambda s: perm[s] if (perm is not None and s < len(perm)) else s)
        use_asr = self.text_mode == "asr" or (self.text_mode == "mix" and random.random() < self.asr_prob)
        utt = self.utt_ids_asr if use_asr else self.utt_ids
        hist = [(remap(int(spk[r])), utt[r]) for r in self.prev_rows[i][-self.builder.max_history:]]
        ids, ts, te = self.builder.build(hist, (remap(int(spk[i])), utt[i]))
        item = {
            "input_ids": ids, "target_start": ts, "target_end": te,
            "emotion": int(self.y[i]), "sentiment": int(self.ys[i]), "idx": i,
            "audio_ok": bool(self.audio_ok[i]),
        }
        if self.audio_path is not None and self.audio_ok[i]:
            item["audio"] = self._feat(i)
        return item


def make_collate(pad_id: int, audio_shape: Optional[Tuple[int, int]]):
    def collate(batch):
        B = len(batch)
        T = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        tmask = torch.zeros((B, T), dtype=torch.bool)
        for k, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[k, :n] = torch.tensor(b["input_ids"])
            attn[k, :n] = 1
            tmask[k, b["target_start"]:b["target_end"]] = True
        out = {
            "input_ids": input_ids, "attention_mask": attn, "target_mask": tmask,
            "emotion": torch.tensor([b["emotion"] for b in batch]),
            "sentiment": torch.tensor([b["sentiment"] for b in batch]),
            "idx": torch.tensor([b["idx"] for b in batch]),
            "audio_ok": torch.tensor([b["audio_ok"] and ("audio" in b) for b in batch]),
        }
        if audio_shape is not None:
            L, F = audio_shape
            audio = torch.zeros((B, L, F), dtype=torch.float32)
            for k, b in enumerate(batch):
                if "audio" in b:
                    audio[k] = torch.from_numpy(b["audio"])
            out["audio"] = audio
        return out
    return collate


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultimodalERC(nn.Module):
    def __init__(self, text_model: Optional[str] = None, text_config=None,
                 audio_shape: Optional[Tuple[int, int]] = None,
                 n_emotions: int = len(EMOTIONS), n_sent: int = len(SENTIMENTS),
                 hidden: int = 512, dropout: float = 0.2,
                 mod_drop_audio: float = 0.25, mod_drop_text: float = 0.05,
                 load_pretrained: bool = True, grad_checkpointing: bool = False):
        super().__init__()
        from transformers import AutoModel

        if load_pretrained:
            try:
                self.encoder = AutoModel.from_pretrained(text_model, add_pooling_layer=False)
            except TypeError:
                self.encoder = AutoModel.from_pretrained(text_model)
        else:
            try:
                self.encoder = AutoModel.from_config(text_config, add_pooling_layer=False)
            except TypeError:
                self.encoder = AutoModel.from_config(text_config)
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size
        self.mod_drop_audio = mod_drop_audio
        self.mod_drop_text = mod_drop_text
        self.use_audio = audio_shape is not None

        self.text_proj = nn.Sequential(nn.LayerNorm(2 * H), nn.Dropout(dropout), nn.Linear(2 * H, hidden), nn.GELU())
        self.text_head = nn.Linear(hidden, n_emotions)
        self.missing_text = nn.Parameter(torch.zeros(hidden))

        if self.use_audio:
            L, F = audio_shape
            self.register_buffer("a_mean", torch.zeros(L, F))
            self.register_buffer("a_std", torch.ones(L, F))
            self.layer_logits = nn.Parameter(torch.zeros(L))
            self.audio_proj = nn.Sequential(nn.LayerNorm(F), nn.Dropout(dropout), nn.Linear(F, hidden), nn.GELU())
            self.audio_head = nn.Linear(hidden, n_emotions)
            self.missing_audio = nn.Parameter(torch.zeros(hidden))

        fusion_in = 2 * hidden if self.use_audio else hidden
        self.fusion = nn.Sequential(nn.LayerNorm(fusion_in), nn.Dropout(dropout),
                                    nn.Linear(fusion_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.emo_head = nn.Linear(hidden, n_emotions)
        self.sent_head = nn.Linear(hidden, n_sent)

    def head_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return [(n, p) for n, p in self.named_parameters() if id(p) not in enc]

    def forward(self, input_ids, attention_mask, target_mask, audio=None, audio_ok=None,
                force_no_audio: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = h[:, 0]
        m = target_mask.unsqueeze(-1).to(h.dtype)
        tgt = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        t = self.text_proj(torch.cat([cls, tgt], dim=-1))
        out = {"text": self.text_head(t), "audio": None}
        B = t.size(0)

        if not self.use_audio:
            z = t
        else:
            if audio is None:
                a = self.missing_audio.to(t.dtype).unsqueeze(0).expand(B, -1)
                missing = torch.ones(B, dtype=torch.bool, device=t.device)
            else:
                x = (audio.float() - self.a_mean) / self.a_std
                w = torch.softmax(self.layer_logits, dim=0)
                a = self.audio_proj((x * w.view(1, -1, 1)).sum(1))
                out["audio"] = self.audio_head(a)
                missing = ~audio_ok.bool()
                if force_no_audio:
                    missing = torch.ones_like(missing)
                if self.training and self.mod_drop_audio > 0:
                    missing = missing | (torch.rand(B, device=a.device) < self.mod_drop_audio)
                a = torch.where(missing.unsqueeze(1),
                                self.missing_audio.to(a.dtype).unsqueeze(0).expand_as(a), a)
            t_in = t
            if self.training and self.mod_drop_text > 0:
                drop_t = (torch.rand(B, device=t.device) < self.mod_drop_text) & ~missing
                t_in = torch.where(drop_t.unsqueeze(1),
                                   self.missing_text.to(t.dtype).unsqueeze(0).expand_as(t), t)
            z = torch.cat([t_in, a.to(t_in.dtype)], dim=-1)

        f = self.fusion(z)
        out["emo"] = self.emo_head(f)
        out["sent"] = self.sent_head(f)
        out["layer_weights"] = torch.softmax(self.layer_logits, 0).detach() if self.use_audio else None
        return out


def build_param_groups(model: MultimodalERC, lr_enc: float, lr_head: float,
                       weight_decay: float, layer_decay: float):
    """AdamW groups: layer-wise LR decay for the text encoder, higher LR for the fresh heads."""
    n_layers = model.encoder.config.num_hidden_layers
    layer_re = re.compile(r"\.layer\.(\d+)\.")
    groups: Dict[Tuple[float, bool], List[nn.Parameter]] = {}

    def add(lr, decay, p):
        groups.setdefault((lr, decay), []).append(p)

    for name, p in model.encoder.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim == 1 or name.endswith(".bias")
        m = layer_re.search(name)
        if m:
            depth = n_layers - 1 - int(m.group(1))
        elif "embeddings" in name:
            depth = n_layers
        else:
            depth = 0
        add(lr_enc * (layer_decay ** depth), not no_decay, p)
    for name, p in model.head_parameters():
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        add(lr_head, not no_decay, p)
    return [{"params": ps, "lr": lr, "weight_decay": weight_decay if dec else 0.0}
            for (lr, dec), ps in groups.items()]


# --------------------------------------------------------------------------- #
# Metrics / calibration
# --------------------------------------------------------------------------- #
def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str] = EMOTIONS) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    ids = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "macro_recall": float(np.mean(r)),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in ids},
    }


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-scalar temperature scaling that minimises NLL on held-out data."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


# --------------------------------------------------------------------------- #
# Audio features (frozen speech SSL model -> per-layer mean/std pooled statistics)
# --------------------------------------------------------------------------- #
def load_audio_file(path: str, max_seconds: float = 15.0) -> Optional[np.ndarray]:
    """Mono float32 @16 kHz, centre-cropped to max_seconds. None if unreadable."""
    wav = None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), SAMPLE_RATE)
            wav = resample_poly(wav, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    except Exception:
        try:
            import librosa
            wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            return None
    return prepare_waveform(wav, max_seconds)


def prepare_waveform(wav: np.ndarray, max_seconds: float = 15.0) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    max_n = int(max_seconds * SAMPLE_RATE)
    if len(wav) > max_n:
        s = (len(wav) - max_n) // 2
        wav = wav[s:s + max_n]
    if len(wav) < 1600:  # the SSL conv stack needs a minimum length
        wav = np.pad(wav, (0, 1600 - len(wav)))
    return wav


class AudioFeatureExtractor:
    """
    Frozen HF speech model (default: microsoft/wavlm-large). For every transformer layer we keep
    the time-mean and time-std of the hidden states -> array [num_layers + 1, 2 * hidden].
    The same class is used offline (training) and online (live inference) so features match.
    """

    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = False):
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.n_states = self.model.config.num_hidden_layers + 1
        self.dim = 2 * self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        x = self.fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            hs = self.model(x, output_hidden_states=True).hidden_states
        hs = torch.stack(hs, 0).squeeze(1).float()          # [L, T, D]
        feats = torch.cat([hs.mean(1), hs.std(1, unbiased=False)], dim=-1)  # [L, 2D]
        return torch.nan_to_num(feats).cpu().numpy()


# --------------------------------------------------------------------------- #
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
