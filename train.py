#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from pathlib import Path
import json
import random

from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from model import DEFAULT_MODEL, KNOWN_MODELS, MEDIUM_LABELS, MyNetwork


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _char_to_bio_labels(self, text, spans, offsets):
        # offsets: list of (start_char, end_char) for each token position
        labels = ["O"] * len(offsets)
        mediums = [[False, False]] * len(offsets)

        # Build quick lookup of which span a char is in (simple approach)
        # You can improve this if spans overlap (decide a rule).
        span_ranges = []
        for sp in spans:
            if not any(sp["medium"][m] for m in ["home", "theatrical"]):
                continue
            this_mediums = []
            for medium in MEDIUM_LABELS:
                this_mediums.append(sp["medium"][medium])
            span_ranges.append((sp["date_start"], sp["date_end"], this_mediums))

        for i, (tok_s, tok_e) in enumerate(offsets):
            if tok_s is None or tok_e is None:
                continue

            # Find first span that overlaps token
            matching = None
            for s, e, m in span_ranges:
                if tok_e > s and tok_s < e:  # overlap
                    matching = (s, e, m)
                    break


            if matching is None:
                labels[i] = 0
                mediums[i] = [False, False]
                continue

            s, e, m = matching
            mediums[i] = m

            # Decide B vs I by whether previous token also overlaps same span type/range
            # Heuristic: if token start is at/near span start => B else I.
            if abs(tok_s - s) <= 1:
                labels[i] = 1
            else:
                labels[i] = 2

        return [[l, m] for l, m in zip(labels, mediums)]

    def __getitem__(self, idx):
        text_path = self.data[idx]
        with open(text_path) as f:
            text = f.read()
        with open(str(text_path) + ".json") as f:
            label_input = json.load(f)
        #label_input: [{
        #    date_start: idx,
        #    date_end: idx,
        #    medium: {
        #            home: true,
        #            theatrical: true,
        #    }
        #}]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True
        )

        offsets = enc["offset_mapping"]  # length == seq_len
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = self._char_to_bio_labels(text, label_input, offsets)

        # Ignore special tokens in loss:
        # For BERT, offsets for [CLS]/[SEP] are (0,0) or (None,None) depending on tokenizer.
        # We'll set -100 where attention_mask==0 or where offsets are (0,0).
        labels_out = []
        mediums_out = []
        for i, (mask, (s, e)) in enumerate(zip(attention_mask, offsets)):
            if mask == 0:
                labels_out.append(-100)
                mediums_out.append([False, False])
            elif s == e == 0:
                labels_out.append(-100)
                mediums_out.append([False, False])
            else:
                labels_out.append(labels[i][0])
                mediums_out.append(labels[i][1])

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels_out, dtype=torch.long),
            "medium_labels": torch.tensor(mediums_out, dtype=torch.float32),
        }
        return item


class BalancedTrainer(Trainer):
    """Trainer that samples evenly across sub-datasets of different sizes."""

    def __init__(self, *args, dataset_sizes, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_sizes = dataset_sizes

    def _get_train_sampler(self, train_dataset=None):
        weights = []
        for size in self._dataset_sizes:
            w = 1.0 / size
            weights.extend([w] * size)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def find_batch_size(model, max_length=512):
    """Empirically find the largest batch size that fits in GPU memory."""
    if not torch.cuda.is_available():
        print("No GPU found; defaulting to batch size 8")
        return 8

    device = torch.device("cuda")
    model = model.to(device)

    # Pre-allocate optimizer state with a warm-up step and keep it alive so the
    # binary search measures forward-pass memory with optimizer tensors resident,
    # matching the memory layout when resuming from a checkpoint.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    _ids = torch.ones(1, max_length, dtype=torch.long, device=device)
    _mask = torch.ones(1, max_length, dtype=torch.long, device=device)
    _lbls = torch.zeros(1, max_length, dtype=torch.long, device=device)
    _mlbls = torch.zeros(1, max_length, len(MEDIUM_LABELS), dtype=torch.float32, device=device)
    _out = model(input_ids=_ids, attention_mask=_mask, labels=_lbls, medium_labels=_mlbls)
    _out["loss"].backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    del _ids, _mask, _lbls, _mlbls, _out
    torch.cuda.empty_cache()

    def try_batch(bs):
        try:
            torch.cuda.empty_cache()
            ids = torch.ones(bs, max_length, dtype=torch.long, device=device)
            mask = torch.ones(bs, max_length, dtype=torch.long, device=device)
            lbls = torch.zeros(bs, max_length, dtype=torch.long, device=device)
            mlbls = torch.zeros(bs, max_length, len(MEDIUM_LABELS), dtype=torch.float32, device=device)
            out = model(input_ids=ids, attention_mask=mask, labels=lbls, medium_labels=mlbls)
            out["loss"].backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return False
            raise

    # Double until OOM to find upper bound
    bs = 1
    while bs <= 128 and try_batch(bs):
        bs *= 2

    # Binary search between last good (bs//2) and first bad (bs)
    lo, hi = bs // 2, bs
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if try_batch(mid):
            lo = mid
        else:
            hi = mid

    del optimizer
    # Back off 25% to leave headroom for Trainer overhead
    result = max(1, int(lo * 0.5))
    torch.cuda.empty_cache()
    print(f"Batch size: {result} (max that fit in one pass: {lo})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train date-extraction NER model")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Base transformer model name (default: {DEFAULT_MODEL}). Known: {', '.join(KNOWN_MODELS)}"
    )
    parser.add_argument(
        "--resume", default=None, metavar="CHECKPOINT",
        help="Path to a checkpoint directory (or parent dir containing checkpoints) to resume training from"
    )
    parser.add_argument(
        "--output-dir", default="./ner_bert", metavar="DIR",
        help="Directory to save checkpoints (default: ./ner_bert)"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing preprocessed training data"
    )
    args = parser.parse_args()

    bert_model_name = args.model
    resume_from = args.resume

    if resume_from:
        checkpoint_dir = Path(resume_from)
        # If given a parent output dir, resolve to the latest checkpoint inside it
        if not (checkpoint_dir / "trainer_state.json").exists():
            candidates = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            if not candidates:
                raise ValueError(f"No checkpoints found in {resume_from}")
            checkpoint_dir = candidates[-1]
            resume_from = str(checkpoint_dir)
        print(f"Resuming from checkpoint: {resume_from}")

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    model = MyNetwork(bert_model_name, 3, len(MEDIUM_LABELS))
    batch_size = find_batch_size(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="combined_f1",
        greater_is_better=True,
    )

    no_dates = []
    theatrical_only = []
    non_theatrical = []

    for p in sorted(Path(args.data_dir).iterdir()):
        if p.suffix == ".json":
            continue
        json_path = Path(str(p) + ".json")
        if not json_path.exists():
            continue

        with open(json_path) as f:
            spans = json.load(f)

        if not spans:
            no_dates.append(p)
            continue

        mediums = {k for sp in spans for k, v in sp["medium"].items() if v}
        if mediums - {"theatrical"}:
            non_theatrical.append(p)
        else:
            theatrical_only.append(p)

    # Stratified split: each category is split independently before oversampling
    random.seed(42)
    random.shuffle(no_dates)
    random.shuffle(theatrical_only)
    random.shuffle(non_theatrical)

    def split_20(lst):
        n = max(1, int(0.2 * len(lst)))
        return lst[:n], lst[n:]

    val_no_dates, train_no_dates = split_20(no_dates)
    val_theatrical, train_theatrical = split_20(theatrical_only)
    val_non_theatrical, train_non_theatrical = split_20(non_theatrical)

    train_datasets = {
        "no_dates": NERDataset(train_no_dates, tokenizer),
        "theatrical": NERDataset(train_theatrical, tokenizer),
        "non_theatrical": NERDataset(train_non_theatrical, tokenizer),
    }
    train_dataset = ConcatDataset(list(train_datasets.values()))

    eval_dataset = ConcatDataset([
        NERDataset(val_non_theatrical, tokenizer),
        NERDataset(val_theatrical, tokenizer),
        NERDataset(val_no_dates, tokenizer),
    ])

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        token_logits, medium_logits = predictions   # (N, seq, 3), (N, seq, 2)
        bio_labels, medium_labels = label_ids       # (N, seq), (N, seq, 2)

        # BIO span detection F1
        preds = token_logits.argmax(-1).flatten()
        flat_bio = bio_labels.flatten()
        mask = flat_bio != -100
        preds, flat_bio = preds[mask], flat_bio[mask]
        tp = ((preds != 0) & (flat_bio != 0)).sum()
        fp = ((preds != 0) & (flat_bio == 0)).sum()
        fn = ((preds == 0) & (flat_bio != 0)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Medium classification F1 at true B positions
        b_mask = (bio_labels == 1).flatten()
        medium_preds = (medium_logits.reshape(-1, 2) > 0)  # logit > 0 == sigmoid > 0.5
        medium_gt = (medium_labels.reshape(-1, 2) > 0.5)
        medium_preds, medium_gt = medium_preds[b_mask], medium_gt[b_mask]

        metrics = {"f1": float(f1), "precision": float(precision), "recall": float(recall)}
        medium_f1s = []
        for i, name in enumerate(["home", "theatrical"]):
            tp_m = (medium_preds[:, i] & medium_gt[:, i]).sum()
            fp_m = (medium_preds[:, i] & ~medium_gt[:, i]).sum()
            fn_m = (~medium_preds[:, i] & medium_gt[:, i]).sum()
            p = tp_m / (tp_m + fp_m + 1e-8)
            r = tp_m / (tp_m + fn_m + 1e-8)
            mf1 = float(2 * p * r / (p + r + 1e-8))
            medium_f1s.append(mf1)
            metrics[f"{name}_f1"] = mf1
            metrics[f"{name}_precision"] = float(p)
            metrics[f"{name}_recall"] = float(r)
        metrics["combined_f1"] = (f1 + sum(medium_f1s) / len(medium_f1s)) / 2
        return metrics

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        dataset_sizes=[len(ds) for ds in train_datasets.values()],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train(resume_from_checkpoint=resume_from)


if __name__ == "__main__":
    main()
