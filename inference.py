#!/usr/bin/env python3
"""
Run inference with the trained NER model from train2.py.

Usage:
    python inference.py "Released on DVD and Blu-ray on March 15, 2003"
    echo "some text" | python inference.py
    python inference.py --checkpoint ner_bert/checkpoint-1752 "some text"
    python inference.py --model google-bert/bert-base-uncased "some text"
"""

import sys
import argparse
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer

from model import DEFAULT_MODEL, KNOWN_MODELS, MEDIUM_LABELS, BIO_LABELS, MyNetwork


def find_latest_checkpoint(base_dir="ner_bert"):
    checkpoints = sorted(
        Path(base_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}/")
    return checkpoints[-1]


def load_model(checkpoint_path, bert_model_name, precision=None):
    checkpoint_path = Path(checkpoint_path)
    model = MyNetwork(bert_model_name, num_token_labels=3, num_medium_classes=len(MEDIUM_LABELS))

    # Trainer saves model.safetensors or pytorch_model.bin
    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    model.load_state_dict(state_dict)
    model.eval()

    if precision == "int8":
        import torch.nn as nn
        # Exclude attention projections and RoPE-adjacent layers from INT8 quantization.
        # ModernBERT's RoPE and GeGLU gating are precision-sensitive; quantizing them
        # corrupts positional information and compounds gating errors. FFN layers are
        # robust to INT8 and account for most of the parameter count anyway.
        attn_keywords = {"attn", "query", "key", "value", "Wqkv", "out_proj", "rope"}
        ffn_modules = {
            name
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
            and not any(kw in name for kw in attn_keywords)
        }
        if ffn_modules:
            model = torch.quantization.quantize_dynamic(
                model, ffn_modules, dtype=torch.qint8
            )
            print(f"Precision: dynamic INT8 on {len(ffn_modules)} FFN Linear layers (attention excluded)", file=sys.stderr)
        else:
            print("Precision: no eligible layers found, skipping INT8", file=sys.stderr)
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
        print("Precision: BF16", file=sys.stderr)

    return model


def extract_spans(text, token_logits, medium_logits, offsets, attention_mask):
    """Decode BIO tags into spans with character offsets and media types."""
    bio_preds = token_logits.argmax(dim=-1).squeeze(0).tolist()
    medium_probs = torch.sigmoid(medium_logits).squeeze(0)
    attn = attention_mask.squeeze(0).tolist()

    spans = []
    current_span = None

    for i, (bio, (char_start, char_end)) in enumerate(zip(bio_preds, offsets)):
        if not attn[i] or (char_start == 0 and char_end == 0):
            if current_span:
                spans.append(current_span)
                current_span = None
            continue

        if bio == 1:  # B
            if current_span:
                spans.append(current_span)
            mediums = {
                label: bool(medium_probs[i][j].item() > 0.5)
                for j, label in enumerate(MEDIUM_LABELS)
            }
            current_span = {
                "char_start": char_start,
                "char_end": char_end,
                "medium_logits_idx": i,
                "mediums": mediums,
            }
        elif bio == 2 and current_span:  # I — extend current span
            current_span["char_end"] = char_end
        else:  # O — close span
            if current_span:
                spans.append(current_span)
                current_span = None

    if current_span:
        spans.append(current_span)

    return spans


def resolve_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def run(text, bert_model_name, checkpoint_path=None, device=None, precision=None):
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    if device is None:
        device = resolve_device("auto")

    print(f"Loading model from {checkpoint_path}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = load_model(checkpoint_path, bert_model_name, precision=precision)
    model.to(device)

    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    warmup = 10
    for x in range(0, warmup):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


    t0 = time.perf_counter()

    iters = 10
    for x in range(0, iters):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    average_ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"Average time: {average_ms:.1f} ms", file=sys.stderr)

    spans = extract_spans(
        text, outputs["token_logits"], outputs["medium_logits"], offsets, attention_mask
    )
    return spans


def main():
    parser = argparse.ArgumentParser(description="Run date-extraction NER inference")
    parser.add_argument("file", nargs="?", help="Path to input file (reads stdin if omitted)")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to checkpoint dir (default: latest)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Base transformer model name (default: {DEFAULT_MODEL}). Known: {', '.join(KNOWN_MODELS)}"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device to run on: auto, cpu, cuda, cuda:0, mps, etc. (default: auto)"
    )
    parser.add_argument(
        "--precision", choices=["int8", "bf16"], default=None,
        help="Reduce model precision: int8 (dynamic INT8, FFN layers only, CPU) or bf16 (BF16, all layers)"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.file:
        text = Path(args.file).read_text()
    else:
        text = sys.stdin.read()
    spans = run(text, args.model, checkpoint_path=args.checkpoint, device=device,
                precision=args.precision)

    if not spans:
        print("No date spans found.")
        return

    for sp in spans:
        span_text = text[sp["char_start"]:sp["char_end"]]
        active_mediums = [m for m, v in sp["mediums"].items() if v]
        medium_str = ", ".join(active_mediums) if active_mediums else "(none)"
        print(f'[{sp["char_start"]}:{sp["char_end"]}] "{span_text}"  media: {medium_str}')


if __name__ == "__main__":
    main()
