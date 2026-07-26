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
import onnx
from pathlib import Path
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import DEFAULT_MODEL, KNOWN_MODELS, MEDIUM_LABELS, BIO_LABELS, MyNetwork


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

    return model


def strip_metadata(graph):
    for node in graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
        for attr in node.attribute:
            if attr.g.ByteSize() > 0:
                strip_metadata(attr.g)  # subgraphs (If/Loop bodies)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        del vi.metadata_props[:]
        vi.doc_string = ""
    graph.doc_string = ""


def main():
    parser = argparse.ArgumentParser(description="Run date-extraction NER inference")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to checkpoint dir (default: latest)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Base transformer model name (default: {DEFAULT_MODEL}). Known: {', '.join(KNOWN_MODELS)}"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.model)
    ids = torch.ones(2, 512, dtype=torch.long)
    mask = torch.ones(2, 512, dtype=torch.long)

    batch_dim = torch.export.dynamic_shapes.Dim("batch")
    tokens_dim = torch.export.dynamic_shapes.Dim("tokens", max=512)
    program = torch.onnx.export(model, (ids, mask), dynamo=True, dynamic_shapes=({0: batch_dim, 1: tokens_dim},{0: batch_dim, 1: tokens_dim }), output_names=("bios", "mediums"), verbose=False)
    proto = program.model_proto
    strip_metadata(proto.graph)
    proto.doc_string = ""
    del proto.metadata_props[:]

    onnx.save(proto, "model.onnx")

    quantize_dynamic("model.onnx", "model.int8.onnx", weight_type=QuantType.QUInt8)

if __name__ == "__main__":
    main()
