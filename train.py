"""Train SleepNet on Sleep-EDF and save a checkpoint for the API.

    ./fetch_data.sh                          # download recordings into model_data/sleep-edf/
    pip install -r requirements-dev.txt      # adds MNE, used only to read the EDF files
    python train.py                          # real data: writes model_data/sleepnet.pt

    python train.py --synthetic 12 --epochs 8 --out /tmp/demo.pt
                                             # made-up EEG: a pipeline smoke test, NOT a model

The recordings are split BY SUBJECT (60/20/20) so that both nights of a person land in the
same split. Splitting by night would leak a person's EEG signature into the test set and
inflate the scores. The test subjects are scored once, with the epoch picked on validation.
"""
import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sleepnet.data import CHANNEL, IGNORE, list_recordings, load_recording, synthetic_night
from sleepnet.metrics import summarize
from sleepnet.model import SEQ_LEN, SleepNet, predict_proba, save_checkpoint
from sleepnet.preprocessing import CLIP_UV, STAGES, normalize


@dataclass
class Recording:
    name: str
    subject: str
    epochs: np.ndarray  # (n, 3000) float16 microvolts: half the RAM of float32
    labels: np.ndarray  # (n,) int8, IGNORE where the stage is unknown


def load_dataset(args):
    """All recordings as a list of Recording, from real EDF files or synthetic nights."""
    if args.synthetic:
        print(f"generating {args.synthetic} SYNTHETIC nights (not real EEG)")
        recordings = []
        for i in range(args.synthetic):
            epochs, labels = synthetic_night(args.synthetic_epochs, seed=args.seed + i)
            recordings.append(Recording(f"synthetic{i:02d}", f"synthetic{i // 2:02d}",
                                        epochs.astype(np.float16), labels))
        return recordings

    found = list_recordings(args.data)
    if not found:
        raise SystemExit(f"No *-PSG.edf files found under {args.data}. Run ./fetch_data.sh first.")
    if args.max_subjects:
        keep = sorted({f["subject"] for f in found})[:args.max_subjects]
        found = [f for f in found if f["subject"] in keep]
    recordings = []
    for i, item in enumerate(found, 1):
        epochs, labels = load_recording(item["psg"], item["hypnogram"], channel=args.channel)
        if len(epochs) >= SEQ_LEN:
            recordings.append(Recording(item["name"], item["subject"],
                                        epochs.astype(np.float16), labels))
        print(f"  [{i}/{len(found)}] {item['name']}: {len(epochs)} epochs", flush=True)
    return recordings


def split_by_subject(recordings, seed, val_frac=0.2, test_frac=0.2):
    """(train, val, test) lists of recordings, with every subject in exactly one split."""
    subjects = sorted({r.subject for r in recordings})
    random.Random(seed).shuffle(subjects)
    n_test = max(1, round(len(subjects) * test_frac))
    n_val = max(1, round(len(subjects) * val_frac))
    if len(subjects) < 3:
        raise SystemExit("Need recordings from at least 3 different subjects.")
    where = {s: "test" for s in subjects[:n_test]}
    where.update({s: "val" for s in subjects[n_test:n_test + n_val]})
    parts = {"train": [], "val": [], "test": []}
    for rec in recordings:
        parts[where.get(rec.subject, "train")].append(rec)
    return parts["train"], parts["val"], parts["test"]


def signal_scale(recordings):
    """One number (microvolts) that all inputs are divided by; stored in the checkpoint."""
    sample = np.concatenate([r.epochs[::10] for r in recordings]).astype(np.float32)
    centred = sample - sample.mean(axis=1, keepdims=True)
    return float(np.clip(centred, -CLIP_UV, CLIP_UV).std())


class Windows(Dataset):
    """Windows of SEQ_LEN consecutive epochs, 50% overlapping, with light augmentation."""

    def __init__(self, recordings, scale, augment):
        self.recordings, self.scale, self.augment = recordings, scale, augment
        self.index = [(i, start) for i, rec in enumerate(recordings)
                      for start in range(0, len(rec.epochs) - SEQ_LEN + 1, SEQ_LEN // 2)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        rec_i, start = self.index[item]
        rec = self.recordings[rec_i]
        x = normalize(rec.epochs[start:start + SEQ_LEN].astype(np.float32), self.scale)
        if self.augment:  # amplitude varies between people and electrode contacts
            x = x * np.random.uniform(0.8, 1.25) + np.random.normal(0, 0.05, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(rec.labels[start:start + SEQ_LEN].astype(np.int64))


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler):
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        loss = loss_fn(model(x).reshape(-1, len(STAGES)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        count += 1
    return total / count


def evaluate(model, recordings, scale):
    """Score whole nights with predict_proba, the same function the API uses."""
    y_true, y_pred = [], []
    for rec in recordings:
        probs = predict_proba(model, normalize(rec.epochs.astype(np.float32), scale))
        known = rec.labels != IGNORE
        y_true.append(rec.labels[known])
        y_pred.append(probs.argmax(1)[known])
    return summarize(np.concatenate(y_true), np.concatenate(y_pred), STAGES)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="model_data/sleep-edf", help="folder of *-PSG.edf files")
    parser.add_argument("--out", default="model_data/sleepnet.pt")
    parser.add_argument("--channel", default=CHANNEL,
                        help="EDF channel to read (run `python -c \"import mne; print(mne.io."
                             "read_raw_edf('<file>-PSG.edf').ch_names)\"` to list them)")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="use only the first N subjects (quick experiments)")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N_NIGHTS",
                        help="train on N made-up nights instead of real data (smoke test only)")
    parser.add_argument("--synthetic-epochs", type=int, default=240,
                        help="epochs per synthetic night (240 = 2 hours)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    recordings = load_dataset(args)
    train, val, test = split_by_subject(recordings, args.seed)
    sizes = {k: {"recordings": len(v), "subjects": len({r.subject for r in v}),
                 "epochs": sum(len(r.epochs) for r in v)}
             for k, v in (("train", train), ("val", val), ("test", test))}
    print("split:", json.dumps(sizes))

    scale = signal_scale(train)
    counts = np.bincount(np.concatenate([r.labels[r.labels != IGNORE] for r in train]),
                         minlength=len(STAGES)).astype(np.float64)
    print("train stage share:", {s: round(float(c / counts.sum()), 3) for s, c in zip(STAGES, counts)})
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights / weights.mean(), ignore_index=IGNORE,
                                  label_smoothing=0.05)

    loader = DataLoader(Windows(train, scale, augment=True), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    model = SleepNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader), pct_start=0.2)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(loader)} steps/epoch, input scale {scale:.1f} uV")

    best_f1, best_state, best_epoch, started = -1.0, None, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, loss_fn, optimizer, scheduler)
        val_scores = evaluate(model, val, scale)
        if val_scores["macro_f1"] > best_f1:
            best_f1, best_state, best_epoch = val_scores["macro_f1"], copy.deepcopy(model.state_dict()), epoch
        print(f"epoch {epoch:2d}/{args.epochs}  loss {loss:.3f}  |  val acc {val_scores['accuracy']:.3f}"
              f"  macro-F1 {val_scores['macro_f1']:.3f}  kappa {val_scores['kappa']:.3f}"
              f"  |  {time.time() - started:.0f}s", flush=True)

    model.load_state_dict(best_state)
    results = evaluate(model, test, scale)  # the test subjects are touched exactly once
    print(f"\nTEST (best epoch {best_epoch}): accuracy {results['accuracy']:.3f}  "
          f"macro-F1 {results['macro_f1']:.3f}  kappa {results['kappa']:.3f}")
    for stage, row in results["per_stage"].items():
        print(f"  {stage:4s} precision {row['precision']:.2f}  recall {row['recall']:.2f}"
              f"  f1 {row['f1']:.2f}  n={row['support']}")

    source = "synthetic" if args.synthetic else "sleep-edf"
    results.update({"trained_on": source, "best_epoch": best_epoch, "split": sizes,
                    "best_val_macro_f1": best_f1, "stages": list(STAGES), "seed": args.seed,
                    "train_seconds": round(time.time() - started)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, model, scale,
                    metrics={k: results[k] for k in ("accuracy", "macro_f1", "kappa", "epochs")},
                    extra={"trained_on": source, "best_epoch": best_epoch})
    (out.parent / f"{out.stem}.metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.stem}.metrics.json")


if __name__ == "__main__":
    main()
