"""
Convert an openai-whisper checkpoint (~/.cache/whisper/<name>.pt) into a faster-whisper
(CTranslate2) model folder, without downloading the weights again.

Only the small tokenizer/config files are fetched from Hugging Face (or pass --hf_files
with a folder that already has them). Useful on slow or offline networks.

    python backend/tools/convert_whisper_ct2.py --name large-v3-turbo
    -> backend/models/faster-whisper-large-v3-turbo/   (set WHISPER_MODEL to this folder)
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import torch

HF_REPOS = {"large-v3-turbo": "openai/whisper-large-v3-turbo", "large-v3": "openai/whisper-large-v3",
            "medium": "openai/whisper-medium", "small": "openai/whisper-small"}
BACKEND = Path(__file__).resolve().parent.parent


def rename(k: str) -> str:
    k = k.replace("blocks", "layers")
    k = k.replace("mlp.0", "fc1").replace("mlp.2", "fc2").replace("mlp_ln", "final_layer_norm")
    # cross-attention first: its names contain the self-attention ones as substrings
    for a, b in ((".cross_attn.query", ".encoder_attn.q_proj"), (".cross_attn.key", ".encoder_attn.k_proj"),
                 (".cross_attn.value", ".encoder_attn.v_proj"), (".cross_attn.out", ".encoder_attn.out_proj"),
                 (".cross_attn_ln", ".encoder_attn_layer_norm"),
                 (".attn.query", ".self_attn.q_proj"), (".attn.key", ".self_attn.k_proj"),
                 (".attn.value", ".self_attn.v_proj"), (".attn.out", ".self_attn.out_proj"),
                 (".attn_ln", ".self_attn_layer_norm")):
        k = k.replace(a, b)
    k = k.replace("encoder.ln_post", "encoder.layer_norm").replace("decoder.ln.", "decoder.layer_norm.")
    k = k.replace("token_embedding", "embed_tokens")
    k = k.replace("encoder.positional_embedding", "encoder.embed_positions.weight")
    k = k.replace("decoder.positional_embedding", "decoder.embed_positions.weight")
    return "model." + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="large-v3-turbo")
    ap.add_argument("--checkpoint", default=None, help="default: ~/.cache/whisper/<name>.pt")
    ap.add_argument("--hf_files", default=None, help="folder with config.json + tokenizer files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quantization", default="float16")
    args = ap.parse_args()

    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from ctranslate2.converters import TransformersConverter

    ckpt_path = Path(args.checkpoint or Path.home() / ".cache" / "whisper" / f"{args.name}.pt")
    out = Path(args.out or BACKEND / "models" / f"faster-whisper-{args.name}")
    hf_dir = Path(args.hf_files) if args.hf_files else None
    if hf_dir is None:
        from huggingface_hub import snapshot_download
        hf_dir = Path(snapshot_download(HF_REPOS[args.name], allow_patterns=["*.json", "*.txt"]))

    print("loading", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {rename(k): v for k, v in ckpt["model_state_dict"].items()}
    config = WhisperConfig.from_pretrained(hf_dir)
    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "proj_out.weight"]  # tied to embed_tokens
    if missing or unexpected:
        raise SystemExit(f"weight mapping failed\nmissing: {missing[:10]}\nunexpected: {unexpected[:10]}")
    model.tie_weights()

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        for f in hf_dir.iterdir():
            if f.suffix in (".json", ".txt") and f.name != "config.json":
                shutil.copy(f, tmp)
        if out.exists():
            shutil.rmtree(out)
        TransformersConverter(tmp, copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
            str(out), quantization=args.quantization)
    print("saved", out)


if __name__ == "__main__":
    main()
