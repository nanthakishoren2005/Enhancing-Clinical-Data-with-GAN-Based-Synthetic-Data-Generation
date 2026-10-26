"""
Task 5: Filtered re-test (the upgrade step).

Step 1: Filter existing synthetic images with a classifier (threshold 0.90)
        into outputs/v2/filtered_synthetic/ (per-class folders).
        Source images are never deleted or moved.
Step 2: Train ONLY new augmented models with the verified-run protocol
        (ResNet18, 15 epochs max, patience 5):
          * Full-data arm:  real train (600/class) + filtered synthetics (100 kept/class)
          * Low-data arm:   data_low (100/class) + filtered synthetics (100 kept/class)
        Baselines are NOT retrained; comparison uses the existing verified baselines
        (full 92.75%, low-data 83.00%).

Writes outputs/v2/filtered_comparison.csv and prints it.
Only ADDS a new experiment; never changes Experiments 1, 2, or 3.

Usage:
    python scripts/run_filtered_experiment.py
    python scripts/run_filtered_experiment.py --threshold 0.90 --keep-per-class 100 --epochs 2  # smoke test
"""
import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.filter_synthetic import filter_synthetic  # Task 1
from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.data.dataset import LiverLesionDataset, get_transforms, CLASS_TO_IDX, IDX_TO_CLASS
from src.cnn.model import build_classifier
from src.evaluation.metrics import evaluate_model_on_dataset

OUTPUTS_V2 = get_project_root() / "outputs" / "v2"
FILTERED_SYNTHETIC_DIR = OUTPUTS_V2 / "filtered_synthetic"
FILTERED_TRAIN_DIR = OUTPUTS_V2 / "filtered_train_100perclass"
FULL_CKPT = OUTPUTS_V2 / "filtered_augmented_full_best.pt"
LOW_CKPT = OUTPUTS_V2 / "filtered_augmented_low_best.pt"
COMPARISON_CSV = OUTPUTS_V2 / "filtered_comparison.csv"

# Verified baseline numbers (reference only, never retrained here)
VERIFIED_BASELINE_FULL_ACC = 92.75
VERIFIED_BASELINE_LOW_ACC = 83.00
DATA_LOW_DIR = get_project_root() / "data_low"
TEST_DIR = get_project_root() / "data/original"


def cap_to_per_class(src_dir, dst_dir, keep_per_class, seed=42):
    """
    Copy up to `keep_per_class` kept images per class from src_dir/<class> to
    dst_dir/train/<class> (layout expected by LiverLesionDataset).
    Deterministic subsample when more images are kept than requested.
    Returns dict of kept counts per class actually copied.
    """
    rng = random.Random(seed)
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for class_name in CLASS_TO_IDX:
        src_class = Path(src_dir) / class_name
        dst_class = dst_dir / "train" / class_name
        images = []
        if src_class.is_dir():
            images = [f for f in sorted(src_class.iterdir())
                      if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        n_available = len(images)
        n_take = min(keep_per_class, n_available)
        if n_take < n_available:
            images = rng.sample(images, n_take)
        dst_class.mkdir(parents=True, exist_ok=True)
        for f in images:
            shutil.copy2(f, dst_class / f.name)
        copied[class_name] = n_take
        print(f"  [subset] {class_name}: {n_available} kept available -> {n_take} copied to training set")
    return copied


def train_protocol(seed, train_root, use_synthetic, synthetic_dir, epochs, patience,
                   batch_size, lr, ckpt_path, weight_decay):
    """One filtered-augmented training run (verified-run protocol). Returns test metrics."""
    device = get_device()
    config = load_config()
    set_seed(seed)

    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    train_dataset = LiverLesionDataset(
        root_dir=train_root, split="train",
        synthetic_dir=(synthetic_dir if use_synthetic else None),
        transform=get_transforms("train", img_size, mean, std),
    )
    val_dataset = LiverLesionDataset(
        root_dir=train_root, split="val", synthetic_dir=None,
        transform=get_transforms("val", img_size, mean, std),
    )
    test_dataset = LiverLesionDataset(
        root_dir=TEST_DIR, split="test", synthetic_dir=None,
        transform=get_transforms("test", img_size, mean, std),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training data in {train_root / 'train'}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation data in {train_root / 'val'}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_classifier(
        architecture="resnet18", num_classes=len(CLASS_TO_IDX),
        pretrained=True, freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
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

    # Evaluate the best checkpoint on the untouched 400-image test set
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    eval_model = build_classifier(
        architecture="resnet18", num_classes=len(CLASS_TO_IDX),
        pretrained=False, freeze_backbone=False,
    ).to(device)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    metrics, _, _, _ = evaluate_model_on_dataset(eval_model, test_loader, device)
    return metrics, best_epoch


def evaluate_existing_baseline():
    """Evaluate the existing baseline_best.pt checkpoint on the untouched test set."""
    config = load_config()
    device = get_device(config)
    ckpt_path = get_project_root() / "checkpoints" / "baseline_best.pt"
    ckpt = load_checkpoint(ckpt_path, device=device)
    model = build_classifier(
        architecture=ckpt.get("architecture", "resnet18"),
        num_classes=len(CLASS_TO_IDX), pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_dataset = LiverLesionDataset(
        root_dir=TEST_DIR, split="test", synthetic_dir=None,
        transform=get_transforms("test", config["dataset"]["image_size"],
                                 tuple(config["dataset"]["mean"]), tuple(config["dataset"]["std"])),
    )
    test_loader = DataLoader(test_dataset, batch_size=config["cnn"]["batch_size"], shuffle=False, num_workers=0)
    metrics, _, _, _ = evaluate_model_on_dataset(model, test_loader, device)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Filtered re-test experiment (Task 5 upgrade step).")
    parser.add_argument("--threshold", type=float, default=0.90, help="Filter confidence threshold")
    parser.add_argument("--keep-per-class", type=int, default=100,
                        help="Max kept synthetic images per class for training")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    args = parser.parse_args()

    OUTPUTS_V2.mkdir(parents=True, exist_ok=True)
    config = load_config()
    device = get_device(config)
    weight_decay = config["cnn"]["weight_decay"]

    print("=" * 65)
    print("TASK 5 - FILTERED RE-TEST (upgrade step)")
    print("=" * 65)

    # ---------------- Step 1: filter synthetics ----------------
    print(f"\n[Step 1] Filtering synthetic images (threshold={args.threshold})...")
    filter_synthetic(
        synthetic_dir=get_project_root() / "data/synthetic/train",
        checkpoint=get_project_root() / "checkpoints" / "baseline_best.pt",
        out_dir=FILTERED_SYNTHETIC_DIR,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    # ---------------- Step 2: build 100-kept-per-class training subset ----------------
    print(f"\n[Step 2] Capping kept synthetics to {args.keep_per_class} per class for training...")
    kept_counts = cap_to_per_class(FILTERED_SYNTHETIC_DIR, FILTERED_TRAIN_DIR,
                                   args.keep_per_class, seed=config["project"]["random_seed"])
    kept_total = sum(kept_counts.values())
    print(f"  Kept per class: {kept_counts} | Total kept: {kept_total}")

    # ---------------- Full-data arm ----------------
    print(f"\n[Step 3] Full-data augmented arm: real train + {args.keep_per_class} filtered/class")
    full_metrics, full_best_epoch = train_protocol(
        seed=config["project"]["random_seed"],
        train_root=get_project_root() / "data/original",
        use_synthetic=True,
        synthetic_dir=FILTERED_TRAIN_DIR,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_path=FULL_CKPT,
        weight_decay=weight_decay,
    )
    print(f"  [FULL] filtered-augmented -> Accuracy: {full_metrics['accuracy']*100:.2f}%  "
          f"Macro F1: {full_metrics['macro_f1']*100:.2f}%  (best epoch {full_best_epoch})")

    # ---------------- Low-data arm ----------------
    low_blocked = not DATA_LOW_DIR.is_dir()
    if low_blocked:
        print(f"\n[Step 4] Low-data arm BLOCKED: {DATA_LOW_DIR} not present on this machine.")
        print("         Pending the Kaggle results zip containing data_low/ (100 real images/class).")
        print("         Running full-data arm only, as specified.")
    else:
        print(f"\n[Step 4] Low-data augmented arm: data_low (100/class) + {args.keep_per_class} filtered/class")
        low_metrics, low_best_epoch = train_protocol(
            seed=config["project"]["random_seed"],
            train_root=DATA_LOW_DIR,
            use_synthetic=True,
            synthetic_dir=FILTERED_TRAIN_DIR,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            ckpt_path=LOW_CKPT,
            weight_decay=weight_decay,
        )
        print(f"  [LOW] filtered-augmented -> Accuracy: {low_metrics['accuracy']*100:.2f}%  "
              f"Macro F1: {low_metrics['macro_f1']*100:.2f}%  (best epoch {low_best_epoch})")

    # ---------------- Baseline reference (existing checkpoint, no retraining) ----------------
    print(f"\n[Step 5] Evaluating existing baseline_best.pt on untouched test set (reference only)...")
    baseline_eval = evaluate_existing_baseline()
    print(f"  [BASELINE] checkpoint eval -> Accuracy: {baseline_eval['accuracy']*100:.2f}%  "
          f"Macro F1: {baseline_eval['macro_f1']*100:.2f}%")
    print(f"  Verified baseline reference: full {VERIFIED_BASELINE_FULL_ACC}%, low-data {VERIFIED_BASELINE_LOW_ACC}%")

    # ---------------- Write comparison CSV ----------------
    rows = []
    full_row = {
        "regime": "full_data",
        "baseline_accuracy_verified": VERIFIED_BASELINE_FULL_ACC,
        "baseline_macro_f1_checkpoint_eval": round(baseline_eval["macro_f1"], 4),
        "baseline_accuracy_checkpoint_eval": round(baseline_eval["accuracy"], 4),
        "filtered_augmented_accuracy": round(full_metrics["accuracy"], 4),
        "filtered_augmented_macro_f1": round(full_metrics["macro_f1"], 4),
        "delta_accuracy_vs_verified": round(full_metrics["accuracy"] * 100 - VERIFIED_BASELINE_FULL_ACC, 4),
        "kept_normal": kept_counts.get("normal"),
        "kept_covid": kept_counts.get("covid"),
        "kept_lung_opacity": kept_counts.get("lung_opacity"),
        "kept_viral_pneumonia": kept_counts.get("viral_pneumonia"),
        "kept_total": kept_total,
        "note": "Full-data arm: real train + filtered synthetics. Baseline NOT retrained.",
    }
    rows.append(full_row)

    if low_blocked:
        rows.append({
            "regime": "low_data",
            "baseline_accuracy_verified": VERIFIED_BASELINE_LOW_ACC,
            "baseline_macro_f1_checkpoint_eval": None,
            "baseline_accuracy_checkpoint_eval": None,
            "filtered_augmented_accuracy": None,
            "filtered_augmented_macro_f1": None,
            "delta_accuracy_vs_verified": None,
            "kept_normal": kept_counts.get("normal"),
            "kept_covid": kept_counts.get("covid"),
            "kept_lung_opacity": kept_counts.get("lung_opacity"),
            "kept_viral_pneumonia": kept_counts.get("viral_pneumonia"),
            "kept_total": kept_total,
            "note": "BLOCKED: data_low/ not present on this machine; pending Kaggle results zip.",
        })
    else:
        rows.append({
            "regime": "low_data",
            "baseline_accuracy_verified": VERIFIED_BASELINE_LOW_ACC,
            "baseline_macro_f1_checkpoint_eval": round(low_metrics["macro_f1"], 4),
            "baseline_accuracy_checkpoint_eval": round(low_metrics["accuracy"], 4),
            "filtered_augmented_accuracy": round(low_metrics["accuracy"], 4),
            "filtered_augmented_macro_f1": round(low_metrics["macro_f1"], 4),
            "delta_accuracy_vs_verified": round(low_metrics["accuracy"] * 100 - VERIFIED_BASELINE_LOW_ACC, 4),
            "kept_normal": kept_counts.get("normal"),
            "kept_covid": kept_counts.get("covid"),
            "kept_lung_opacity": kept_counts.get("lung_opacity"),
            "kept_viral_pneumonia": kept_counts.get("viral_pneumonia"),
            "kept_total": kept_total,
            "note": "Low-data arm: data_low + filtered synthetics.",
        })

    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- Print comparison ----------------
    print("\n" + "=" * 65)
    print("FILTERED COMPARISON (baseline vs filtered-augmented)")
    print("=" * 65)
    for r in rows:
        print(f"Regime: {r['regime']}")
        print(f"  Baseline verified acc : {r['baseline_accuracy_verified']}%")
        print(f"  Baseline ckpt eval    : acc={r['baseline_accuracy_checkpoint_eval']} "
              f"macro F1={r['baseline_macro_f1_checkpoint_eval']}")
        if r['filtered_augmented_accuracy'] is not None:
            print(f"  Filtered-augmented    : acc={r['filtered_augmented_accuracy']*100:.2f}% "
                  f"macro F1={r['filtered_augmented_macro_f1']*100:.2f}%  "
                  f"delta vs verified={r['delta_accuracy_vs_verified']:+.2f}%")
        else:
            print(f"  Filtered-augmented    : BLOCKED")
        print(f"  Kept per class        : normal={r['kept_normal']} covid={r['kept_covid']} "
              f"lung_opacity={r['kept_lung_opacity']} viral_pneumonia={r['kept_viral_pneumonia']} "
              f"(total={r['kept_total']})")
        print(f"  Note: {r['note']}")
        print("-" * 65)
    print(f"Comparison written to {COMPARISON_CSV}")
    print("[OK] Task 5 complete. Experiments 1, 2, 3 are unchanged.")


if __name__ == "__main__":
    main()
