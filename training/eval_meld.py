"""
Evaluate the deployed emotion model (backend/model_bundle) on a MELD split, the same way
the app runs it: one dialogue at a time, causally (each turn sees only earlier turns).

Reports weighted/macro F1, per-class F1, accuracy and calibration (ECE) for:
  * fused     - the model's real output (words + voice + context)
  * text_head - the auxiliary words-only head
  * audio_head- the auxiliary voice-only head
plus a confusion matrix. Use the numbers for the report; the ablations show how much the
voice actually adds.

Examples (from the repo root, audio_sentiment env):
  python training/eval_meld.py --data_dir /path/to/MELD.Raw                      # gold text
  python training/eval_meld.py --data_dir ... --asr_dir /path/to/asr_csvs         # Whisper text (as trained)
  python training/eval_meld.py --data_dir ... --text live_asr --limit 300         # our live ASR, end-to-end
  python training/eval_meld.py --data_dir ... --no_audio                          # text-only ablation

Expects MELD's test_sent_emo.csv and a test_wav/ folder of diaX_uttY.wav files
(the same layout the training code uses).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "model_bundle"))

from common2 import (EMOTIONS, build_dialog_index, compute_metrics,  # noqa: E402
                     expected_calibration_error, load_audio_file, load_split)
from infer2 import EmotionContextEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--ckpt_dir", default=str(ROOT / "backend/model_bundle/runs/v2_rl_audio"))
    ap.add_argument("--text", default="auto", choices=["auto", "gold", "asr", "live_asr"],
                    help="auto = Whisper CSV if --asr_dir has one, else gold")
    ap.add_argument("--asr_dir", default=None)
    ap.add_argument("--whisper_model", default="large-v3-turbo")
    ap.add_argument("--no_audio", action="store_true", help="ablation: ignore the voice")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N utterances")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "eval"))
    args = ap.parse_args()

    df = load_split(args.data_dir, args.split, asr_dir=args.asr_dir)
    if args.limit:
        keep_dlgs = df.dlg.unique()
        rows, n = [], 0
        for d in keep_dlgs:  # whole dialogues only, so context stays intact
            g = df[df.dlg == d]
            rows.append(g)
            n += len(g)
            if n >= args.limit:
                break
        import pandas as pd
        df = pd.concat(rows).reset_index(drop=True)
    prev_rows, _ = build_dialog_index(df)

    text_mode = args.text
    if text_mode == "auto":
        text_mode = "asr" if (df["asr_text"] != "").any() else "gold"
    asr = None
    if text_mode == "live_asr":
        import torch
        from app.asr import ASR
        asr = ASR(args.whisper_model, "cuda" if torch.cuda.is_available() else "cpu")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = EmotionContextEngine(args.ckpt_dir, device=device,
                                  audio_fp16=device == "cuda", text_fp16=device == "cuda")

    y, fused, text_p, audio_p, times = [], [], [], [], []
    missing_audio = 0
    cur_dlg = None
    t_start = time.perf_counter()
    for i, row in df.iterrows():
        if row.dlg != cur_dlg:
            engine.reset()
            if asr:
                asr.reset()
            cur_dlg = row.dlg
        wav = None if args.no_audio or not row.file_ok else load_audio_file(row.wav_path)
        if wav is None and not args.no_audio:
            missing_audio += 1
        t = time.perf_counter()
        if text_mode == "gold":
            text = row.text
        elif text_mode == "asr":
            text = row.asr_text or row.text
        else:
            text = asr.transcribe(wav)[0] if wav is not None else row.text
        r = engine.update(row.speaker, text, audio=wav, commit=True)
        times.append(time.perf_counter() - t)
        y.append(int(row.emotion_id))
        fused.append([r["probs"][e] for e in EMOTIONS])
        text_p.append([r["text_probs"][e] for e in EMOTIONS])
        if r.get("audio_probs"):
            audio_p.append([r["audio_probs"][e] for e in EMOTIONS])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}  ({(time.perf_counter() - t_start) / (i + 1) * 1000:.0f} ms/utt)", flush=True)

    y = np.array(y)
    report = {"split": args.split, "n": len(y), "text": text_mode, "audio": not args.no_audio,
              "missing_audio": missing_audio, "ms_per_utterance": round(1000 * float(np.mean(times)), 1)}
    for name, probs in (("fused", fused), ("text_head", text_p), ("audio_head", audio_p)):
        if len(probs) != len(y):
            continue
        probs = np.array(probs)
        m = compute_metrics(y, probs.argmax(1))
        m["ece"] = expected_calibration_error(probs, y)
        report[name] = m

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, np.array(fused).argmax(1), labels=list(range(len(EMOTIONS))))
    report["confusion_matrix"] = {"labels": EMOTIONS, "rows_true_cols_pred": cm.tolist()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{text_mode}{'_noaudio' if args.no_audio else ''}"
    (out / f"metrics_{tag}.json").write_text(json.dumps(report, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
        ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
        for a in range(len(EMOTIONS)):
            for b in range(len(EMOTIONS)):
                ax.text(b, a, f"{norm[a, b]:.2f}", ha="center", va="center",
                        color="white" if norm[a, b] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"MELD {args.split} ({text_mode} text) - weighted F1 {report['fused']['weighted_f1']:.3f}")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{tag}.png", dpi=150)
    except Exception as exc:
        print("confusion plot skipped:", exc)

    print(json.dumps({k: (v if not isinstance(v, dict) or k == "confusion_matrix" else
                          {kk: vv for kk, vv in v.items() if kk != "per_class"})
                      for k, v in report.items() if k != "confusion_matrix"}, indent=2))
    for name in ("fused", "text_head", "audio_head"):
        if name in report:
            print(name, {e: round(report[name]["per_class"][e]["f1"], 3) for e in EMOTIONS})
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
