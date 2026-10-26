"""
Classifier-guided synthetic image filtering.

Loads a trained CNN classifier checkpoint and runs it over every synthetic
image in --synthetic-dir (organised in per-class folders). An image is KEPT
only if:

  * the predicted class equals its folder class, AND
  * the softmax confidence for that class is >= --threshold (default 0.90)

Kept images are copied to --out-dir preserving the per-class folder structure.
SOURCE IMAGES ARE NEVER DELETED OR MOVED.

Usage:
    python scripts/filter_synthetic.py ^
        --synthetic-dir data/synthetic/train ^
        --checkpoint checkpoints/baseline_best.pt ^
        --out-dir outputs/v2/filtered_synthetic ^
        --threshold 0.90
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure `src` imports work when run as a standalone script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import load_checkpoint
from src.cnn.model import build_classifier
from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, VALID_EXTENSIONS, get_transforms


class SyntheticImageFolder(torch.utils.data.Dataset):
    """ImageFolder-style dataset over a per-class folder of synthetic images."""

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.class_to_idx = dict(class_to_idx) if class_to_idx else dict(CLASS_TO_IDX)
        self.samples = []  # (path, class_idx, class_name)
        for class_name in sorted(self.class_to_idx):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for f in sorted(class_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((f, self.class_to_idx[class_name], class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, class_name = self.samples[idx]
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label_idx, class_name, str(path)


def load_classifier(checkpoint_path, device, num_classes=None):
    """Load a CNN classifier from a training checkpoint (architecture agnostic)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    ckpt = load_checkpoint(checkpoint_path, device=device)
    num_classes = num_classes or len(CLASS_TO_IDX)
    arch = ckpt.get("architecture", "resnet18")

    model = build_classifier(
        architecture=arch,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("architecture", arch)


@torch.no_grad()
def filter_synthetic(synthetic_dir, checkpoint, out_dir, threshold=0.90,
                     batch_size=64, device=None, seed=42):
    """
    Run classifier-guided filtering.

    Returns a dict with per-class and overall kept/dropped counts and the
    list of kept image source paths.
    """
    set_seed(seed)
    device = device or get_device()
    synthetic_dir = Path(synthetic_dir)
    out_dir = Path(out_dir)

    if not synthetic_dir.is_dir():
        raise NotADirectoryError(f"Synthetic directory not found: {synthetic_dir}")

    config = load_config()
    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    model, arch = load_classifier(checkpoint, device)

    transform = get_transforms("val", img_size, mean, std)
    dataset = SyntheticImageFolder(synthetic_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No synthetic images found in {synthetic_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    per_class = {cname: {"kept": 0, "dropped": 0, "total": 0}
                 for cname in CLASS_TO_IDX}
    kept_paths = []  # (source_path, class_name)
    overall_kept = 0
    overall_dropped = 0

    print(f"\n=== Classifier-guided synthetic filtering ===")
    print(f"Synthetic dir : {synthetic_dir}")
    print(f"Checkpoint    : {checkpoint} (arch={arch})")
    print(f"Threshold     : {threshold}")
    print(f"Device        : {device} | Images: {len(dataset)}")
    print(f"Out dir       : {out_dir}\n")

    for imgs, labels, class_names, paths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        confidences, predicted = torch.max(probs, dim=1)

        for i in range(imgs.size(0)):
            true_idx = int(labels[i].item())
            pred_idx = int(predicted[i].item())
            conf = float(confidences[i].item())
            class_name = IDX_TO_CLASS[true_idx]
            src_path = paths[i]

            per_class[class_name]["total"] += 1

            if pred_idx == true_idx and conf >= threshold:
                per_class[class_name]["kept"] += 1
                overall_kept += 1
                kept_paths.append((src_path, class_name))
            else:
                per_class[class_name]["dropped"] += 1
                overall_dropped += 1

    # Copy kept images preserving per-class folders (never delete/move sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = {cname: 0 for cname in CLASS_TO_IDX}
    for src_path, class_name in kept_paths:
        class_folder = out_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        dst_path = class_folder / Path(src_path).name
        # Avoid collisions: if a same-named file already exists, add a suffix
        if dst_path.exists():
            stem = Path(src_path).stem
            suffix = Path(src_path).suffix
            k = 1
            while dst_path.exists():
                dst_path = class_folder / f"{stem}_{k}{suffix}"
                k += 1
        shutil.copy2(src_path, dst_path)
        copied[class_name] += 1

    # Print report
    print("Per-class results (predicted == folder class and confidence >= threshold):")
    print(f"{'Class':<16} {'Total':>6} {'Kept':>6} {'Dropped':>8} {'Keep rate':>10}")
    for cname in CLASS_TO_IDX:
        stats = per_class[cname]
        rate = stats["kept"] / stats["total"] if stats["total"] else 0.0
        print(f"{cname:<16} {stats['total']:>6} {stats['kept']:>6} "
              f"{stats['dropped']:>8} {rate * 100:>9.2f}%")
    overall_rate = overall_kept / (overall_kept + overall_dropped) if (overall_kept + overall_dropped) else 0.0
    print("-" * 52)
    print(f"{'OVERALL':<16} {overall_kept + overall_dropped:>6} {overall_kept:>6} "
          f"{overall_dropped:>8} {overall_rate * 100:>9.2f}%")
    print(f"\nCopied {overall_kept} kept images to {out_dir} (per-class counts: {copied})")
    print("[OK] Filtering complete. Source images were only read, never deleted or moved.")

    return {
        "threshold": threshold,
        "checkpoint": str(checkpoint),
        "architecture": arch,
        "device": str(device),
        "total_images": overall_kept + overall_dropped,
        "kept": overall_kept,
        "dropped": overall_dropped,
        "keep_rate": overall_rate,
        "per_class": per_class,
        "copied": copied,
        "out_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter synthetic images with a classifier (keep if prediction matches folder "
                    "class and softmax confidence >= threshold).")
    parser.add_argument("--synthetic-dir", required=True,
                        help="Folder of generated images organised per class")
    parser.add_argument("--checkpoint", required=True,
                        help="Classifier checkpoint (.pt) to use for filtering")
    parser.add_argument("--out-dir", required=True,
                        help="Output folder (new) under outputs/v2/ receiving kept images per class")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Softmax confidence threshold (default: 0.90)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    filter_synthetic(
        synthetic_dir=args.synthetic_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
