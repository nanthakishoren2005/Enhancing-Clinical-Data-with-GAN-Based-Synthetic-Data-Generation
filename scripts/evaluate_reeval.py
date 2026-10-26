"""
Re-evaluate filtered-augmented checkpoint(s) on the untouched 400-image test set.
Writes results to outputs/v2/filtered_reeval.csv (new file, never overwrites existing).

Usage:
    python scripts/evaluate_reeval.py
"""
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_project_root, get_device
from src.data.dataset import LiverLesionDataset, get_transforms, CLASS_TO_IDX
from src.cnn.model import build_classifier
from src.evaluation.metrics import evaluate_model_on_dataset
from src.utils.checkpoint import load_checkpoint


def evaluate_checkpoint(ckpt_path, test_loader, device):
    """Load a checkpoint and evaluate on the test loader. Returns (metrics, arch)."""
    ckpt = load_checkpoint(ckpt_path, device=device)
    arch = ckpt.get("architecture", "resnet18")
    model = build_classifier(
        architecture=arch, num_classes=len(CLASS_TO_IDX),
        pretrained=False, freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics, _, _, _ = evaluate_model_on_dataset(model, test_loader, device)
    return metrics, arch


def main():
    config = load_config()
    root = get_project_root()
    device = get_device(config)
    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    # Untouched 400-image test set (seed 42 split)
    test_dataset = LiverLesionDataset(
        root_dir=root / "data" / "original",
        split="test",
        synthetic_dir=None,
        transform=get_transforms("test", img_size, mean, std),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["cnn"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    print(f"Test set: {len(test_dataset)} images (seed 42 split, untouched)")

    results = []

    # 1. Verified baseline (reference)
    baseline_ckpt = root / "checkpoints" / "baseline_best.pt"
    if baseline_ckpt.exists():
        metrics, arch = evaluate_checkpoint(baseline_ckpt, test_loader, device)
        results.append({
            "model": "baseline",
            "checkpoint": str(baseline_ckpt),
            "architecture": arch,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "f1_normal": metrics["per_class"]["normal"]["f1_score"],
            "f1_covid": metrics["per_class"]["covid"]["f1_score"],
            "f1_lung_opacity": metrics["per_class"]["lung_opacity"]["f1_score"],
            "f1_viral_pneumonia": metrics["per_class"]["viral_pneumonia"]["f1_score"],
            "note": "Verified baseline reference (92.75% acc / 92.77% macro F1)",
        })
        print(f"  baseline: acc={metrics['accuracy']*100:.2f}% macro_f1={metrics['macro_f1']*100:.2f}%")

    # 2. Filtered-augmented checkpoint (Task 5 clean run)
    filtered_ckpt = root / "outputs" / "v2" / "filtered_augmented_full_best.pt"
    if filtered_ckpt.exists():
        metrics, arch = evaluate_checkpoint(filtered_ckpt, test_loader, device)
        results.append({
            "model": "filtered_augmented",
            "checkpoint": str(filtered_ckpt),
            "architecture": arch,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "f1_normal": metrics["per_class"]["normal"]["f1_score"],
            "f1_covid": metrics["per_class"]["covid"]["f1_score"],
            "f1_lung_opacity": metrics["per_class"]["lung_opacity"]["f1_score"],
            "f1_viral_pneumonia": metrics["per_class"]["viral_pneumonia"]["f1_score"],
            "note": "Task 5 filtered-augmented clean run (trained on 187 filtered synthetics)",
        })
        print(f"  filtered_augmented: acc={metrics['accuracy']*100:.2f}% macro_f1={metrics['macro_f1']*100:.2f}%")

    # Write results to new CSV (never overwrite existing)
    out_csv = root / "outputs" / "v2" / "filtered_reeval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "checkpoint", "architecture",
            "accuracy", "macro_f1",
            "f1_normal", "f1_covid", "f1_lung_opacity", "f1_viral_pneumonia",
            "note",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {out_csv}")


if __name__ == "__main__":
    main()
