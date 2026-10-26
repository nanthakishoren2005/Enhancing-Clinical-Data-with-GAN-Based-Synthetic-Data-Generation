"""
Repeated-split evaluation of the Baseline and GAN-Augmented ResNet18 classifiers.

For each seed in --seeds:
  * Rebuilds an independent train/val/test split (600/100/100 per class) from
    data/original using that seed. Seed 42 reuses the reference data/original
    splits untouched.
  * Retrains a Baseline and an Augmented classifier with the SAME protocol as
    the verified runs (ResNet18, pretrained, Adam lr=3e-4 wd=1e-4,
    ReduceLROnPlateau, max 15 epochs, early stopping patience 5 on val loss),
    differing ONLY in the seed.
  * Evaluates each model on THAT seed's untouched real test split and appends
    accuracy + macro F1.

Prints mean and standard deviation across seeds for both models.
Only ADDS new results; never replaces the verified 92.75 / 88.50 / 92.50 numbers.

Usage:
    python scripts/evaluate_repeated.py --seeds 42,43,44
    python scripts/evaluate_repeated.py --seeds 42,43 --epochs 2 --patience 1 --limit-train 40  # smoke test
"""
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint
from src.data.dataset import LiverLesionDataset, get_transforms, CLASS_TO_IDX
from src.cnn.model import build_classifier
from src.evaluation.metrics import evaluate_model_on_dataset

DEFAULT_EPOCHS = 15
DEFAULT_PATIENCE = 5
N_TRAIN_PER_CLASS = 600
N_VAL_PER_CLASS = 100
N_TEST_PER_CLASS = 100


def build_seed_split(seed, data_dir, work_root):
    """
    Return the dataset root to use for a given seed.
    Seed 42 -> the reference data/original splits (never modified).
    Other seeds -> a freshly shuffled 600/100/100 split under a temp directory,
                   copied from data/original (originals are only read).
    """
    data_dir = Path(data_dir)
    if seed == 42:
        return data_dir

    # Pool all available images per class from the reference splits
    class_files = {}
    for cname in CLASS_TO_IDX:
        files = []
        for split in ["train", "val", "test"]:
            split_dir = data_dir / split / cname
            if split_dir.is_dir():
                files.extend(sorted(split_dir.iterdir()))
        class_files[cname] = [f for f in files if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    rng = random.Random(seed)
    split_root = Path(work_root) / f"repeated_split_seed_{seed}"
    if split_root.exists():
        shutil.rmtree(split_root)

    for cname, files in class_files.items():
        rng.shuffle(files)
        selected = files[:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS + N_TEST_PER_CLASS]
        splits = {
            "train": selected[:N_TRAIN_PER_CLASS],
            "val": selected[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS],
            "test": selected[N_TRAIN_PER_CLASS + N_VAL_PER_CLASS:],
        }
        for split_name, files in splits.items():
            target_dir = split_root / split_name / cname
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, target_dir / f.name)

    print(f"  [split] Built seed-{seed} split at {split_root} "
          f"({N_TRAIN_PER_CLASS}/train, {N_VAL_PER_CLASS}/val, {N_TEST_PER_CLASS}/test per class)")
    return split_root


def train_and_evaluate_protocol(seed, data_root, use_synthetic, synthetic_dir, epochs, patience,
                                batch_size, lr, ckpt_path, limit_train=None):
    """
    One full train + evaluate run following the verified-run protocol.
    Returns (metrics_dict, history_summary).
    """
    config = load_config()
    device = get_device(config)
    set_seed(seed)

    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    train_dataset = LiverLesionDataset(
        root_dir=data_root, split="train",
        synthetic_dir=(synthetic_dir if use_synthetic else None),
        transform=get_transforms("train", img_size, mean, std),
    )
    if limit_train is not None and len(train_dataset) > limit_train:
        # Deterministic prefix cap (dataset is ordered by class)
        train_dataset.samples = train_dataset.samples[:limit_train]

    val_dataset = LiverLesionDataset(
        root_dir=data_root, split="val", synthetic_dir=None,
        transform=get_transforms("val", img_size, mean, std),
    )
    test_dataset = LiverLesionDataset(
        root_dir=data_root, split="test", synthetic_dir=None,
        transform=get_transforms("test", img_size, mean, std),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training data in {data_root / 'train'}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation data in {data_root / 'val'}")
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test data in {data_root / 'test'}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_classifier(
        architecture="resnet18", num_classes=len(CLASS_TO_IDX),
        pretrained=True, freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["cnn"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_train = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            total_train += imgs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_loss += criterion(model(imgs), labels).item() * imgs.size(0)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                state={
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "architecture": "resnet18", "seed": seed,
                    "use_synthetic": use_synthetic, "best_val_loss": best_val_loss,
                },
                filepath=ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate the best checkpoint on the untouched test split
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    eval_model = build_classifier(
        architecture="resnet18", num_classes=len(CLASS_TO_IDX),
        pretrained=False, freeze_backbone=False,
    ).to(device)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    metrics, _, _, _ = evaluate_model_on_dataset(eval_model, test_loader, device)
    return metrics, {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


import random  # noqa: E402  (used in build_seed_split)


def main():
    parser = argparse.ArgumentParser(description="Repeated-split evaluation of baseline vs augmented classifiers.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds (e.g. 42,43,44)")
    parser.add_argument("--out-csv", default="outputs/v2/repeated_eval.csv", help="Output CSV path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max epochs per training run")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--data-dir", default="data/original", help="Reference dataset root")
    parser.add_argument("--synthetic-dir", default="data/synthetic", help="Synthetic data root (train subfolder used)")
    parser.add_argument("--limit-train", type=int, default=None, help="Cap training images (smoke-test aid)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    synthetic_dir = Path(args.synthetic_dir)
    work_root = Path(tempfile.gettempdir()) / "opencode_repeated_splits"

    rows = []
    per_model = {"baseline": [], "augmented": []}

    print(f"=== Repeated-split evaluation ===")
    print(f"Seeds: {seeds} | Max epochs: {args.epochs} | Patience: {args.patience} | Device: CPU")

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        data_root = build_seed_split(seed, data_dir, work_root)

        for use_synthetic, model_name in [(False, "baseline"), (True, "augmented")]:
            tag = "augmented (+synthetic)" if use_synthetic else "baseline (real only)"
            print(f"  [{tag}] training ResNet18, seed={seed}...")
            ckpt_path = out_csv.parent / f"repeated_{model_name}_seed_{seed}.pt"
            t0 = time.time()
            metrics, summary = train_and_evaluate_protocol(
                seed=seed,
                data_root=data_root,
                use_synthetic=use_synthetic,
                synthetic_dir=synthetic_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                ckpt_path=ckpt_path,
                limit_train=args.limit_train,
            )
            elapsed = time.time() - t0

            row = {
                "seed": seed,
                "model": model_name,
                "epochs": args.epochs,
                "best_epoch": summary["best_epoch"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "seconds": round(elapsed, 2),
            }
            rows.append(row)
            per_model[model_name].append(metrics["accuracy"])
            print(f"  [{tag}] seed={seed} -> Accuracy: {metrics['accuracy']*100:.2f}%  "
                  f"Macro F1: {metrics['macro_f1']*100:.2f}%  ({elapsed:.1f}s, best epoch {summary['best_epoch']})")

    # Write CSV (append mode so repeated runs only add rows)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "model", "epochs", "best_epoch",
                                               "accuracy", "macro_f1", "seconds"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # Summary: mean / std across seeds
    print("\n" + "=" * 55)
    print("REPEATED-SPLIT SUMMARY (mean +/- std across seeds)")
    print("=" * 55)
    for model_name in ["baseline", "augmented"]:
        accs = [r["accuracy"] for r in rows if r["model"] == model_name]
        f1s = [r["macro_f1"] for r in rows if r["model"] == model_name]
        print(f"{model_name:<12} accuracy: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%  |  "
              f"macro F1: {np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}%  (n={len(accs)})")
    print("=" * 55)
    print(f"Results appended to {out_csv}")
    print("NOTE: This only ADDS results. Verified numbers (92.75 / 88.50 / 92.50) are unchanged.")


import csv  # noqa: E402


if __name__ == "__main__":
    main()
