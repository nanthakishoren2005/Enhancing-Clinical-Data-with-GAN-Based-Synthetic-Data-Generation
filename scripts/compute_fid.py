"""
Frechet Inception Distance (FID) between real and synthetic images.

Uses a pretrained InceptionV3 (torchvision) feature extractor.
- Images are resized to 299x299 and normalised with ImageNet statistics.
- Features are taken from the InceptionV3 pool layer (2048-D, before the FC head).
- FID = ||mu_r - mu_s||^2 + tr(sigma_r + sigma_s - 2*sqrt(sigma_r @ sigma_s))

If InceptionV3 pretrained weights cannot be downloaded, the script stops and
reports that clearly instead of inventing a score.

Outputs: one CSV row per class plus an overall row (default: outputs/v2/fid_scores.csv).

Usage:
    python scripts/compute_fid.py ^
        --real-dir data/original/test ^
        --synthetic-dir data/synthetic/train ^
        --out-csv outputs/v2/fid_scores.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_project_root, get_device
from src.data.dataset import CLASS_TO_IDX, VALID_EXTENSIONS

try:
    from scipy.linalg import sqrtm as _scipy_sqrtm

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a declared dependency
    _HAS_SCIPY = False


class ImageFolderDataset(torch.utils.data.Dataset):
    """Flat image list over per-class folders (class inferred from parent folder)."""

    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []  # (path, class_name_or_None)
        if self.root.is_dir():
            for f in sorted(self.root.rglob("*")):
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                    class_name = f.parent.name if f.parent.name in CLASS_TO_IDX else None
                    self.samples.append((f, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path, class_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_name, str(path)


def _matrix_sqrt_numpy(mat):
    """Symmetric matrix square root via eigendecomposition (fallback when scipy is absent)."""
    mat = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet Inception Distance between two Gaussian feature distributions."""
    mu1 = np.atleast_1d(mu1.astype(np.float64))
    mu2 = np.atleast_1d(mu2.astype(np.float64))
    sigma1 = np.atleast_2d(sigma1.astype(np.float64))
    sigma2 = np.atleast_2d(sigma2.astype(np.float64))

    diff = mu1 - mu2
    cov_mean = sigma1.dot(sigma2)

    if _HAS_SCIPY:
        cov_sqrtm = _scipy_sqrtm(cov_mean)
    else:
        cov_sqrtm = _matrix_sqrt_numpy(cov_mean)

    if np.iscomplexobj(cov_sqrtm):
        if not np.allclose(np.diagonal(cov_sqrtm).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in matrix square root")
        cov_sqrtm = cov_sqrtm.real

    # Add small diagonal stabilisation if the product is singular
    if not np.all(np.isfinite(cov_sqrtm)):
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrtm = _scipy_sqrtm((sigma1 + offset).dot(sigma2 + offset)) if _HAS_SCIPY else _matrix_sqrt_numpy(
            (sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(cov_sqrtm):
            cov_sqrtm = cov_sqrtm.real

    tr_cov = np.trace(sigma1 + sigma2 - 2.0 * cov_sqrtm)
    return float(diff.dot(diff) + tr_cov)


def load_inception_feature_extractor(device):
    """
    Load pretrained InceptionV3 and strip the classification head so the
    forward pass returns 2048-D pool features. Raises if weights cannot be
    downloaded/loaded.
    """
    import torchvision.models as models
    from torchvision.models import Inception_V3_Weights

    try:
        weights = Inception_V3_Weights.DEFAULT
        # aux_logits=True is required when loading pretrained InceptionV3 weights;
        # in eval mode the forward pass still returns a single tensor.
        model = models.inception_v3(weights=weights, aux_logits=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not download/load pretrained InceptionV3 weights. "
            f"Original error: {exc}\n"
            "FID requires ImageNet-pretrained InceptionV3 weights from torchvision. "
            "Check internet access (weights are downloaded from download.pytorch.org) "
            "or place the weights file manually, then re-run."
        ) from exc

    model.fc = nn.Identity()  # 2048-D features from the average-pooling layer
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, dataset, device, batch_size=64):
    """Run InceptionV3 over a dataset and return (N, 2048) feature array."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        features.append(feats.cpu().numpy())
    if not features:
        return np.zeros((0, 2048), dtype=np.float64)
    return np.vstack(features)


@torch.no_grad()
def compute_fid(real_dir, synthetic_dir, out_csv, batch_size=64, device=None):
    """
    Compute per-class and overall FID. Writes results to out_csv and prints them.
    """
    device = device or get_device()
    root = get_project_root()
    real_dir = Path(real_dir)
    synthetic_dir = Path(synthetic_dir)
    out_csv = Path(out_csv)

    if not real_dir.is_dir():
        raise NotADirectoryError(f"Real image directory not found: {real_dir}")
    if not synthetic_dir.is_dir():
        raise NotADirectoryError(f"Synthetic image directory not found: {synthetic_dir}")

    # InceptionV3 preprocessing: 299x299, ImageNet normalisation on [0,1] tensors
    from torchvision import transforms

    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n=== Frechet Inception Distance ===")
    print(f"Real dir    : {real_dir}")
    print(f"Synthetic dir: {synthetic_dir}")
    print(f"Device      : {device}")

    # --- Load InceptionV3 (stop and report clearly if weights unavailable) ---
    try:
        inception = load_inception_feature_extractor(device)
    except RuntimeError as exc:
        print("\n" + "!" * 70)
        print("FID ABORTED: pretrained InceptionV3 weights could not be loaded.")
        print(str(exc))
        print("No FID score was computed or invented.")
        print("!" * 70 + "\n")
        sys.exit(1)

    # --- Per-class FID -------------------------------------------------------
    results = []
    all_real_features = []
    all_synth_features = []

    for class_name in CLASS_TO_IDX:
        real_class_dir = real_dir / class_name
        synth_class_dir = synthetic_dir / class_name

        real_ds = ImageFolderDataset(real_class_dir, transform=inception_transform) if real_class_dir.is_dir() else ImageFolderDataset(Path("__nonexistent__"), transform=inception_transform)
        synth_ds = ImageFolderDataset(synth_class_dir, transform=inception_transform) if synth_class_dir.is_dir() else ImageFolderDataset(Path("__nonexistent__"), transform=inception_transform)

        n_real = len(real_ds)
        n_synth = len(synth_ds)

        row = {"class": class_name.capitalize(), "real": n_real, "synthetic": n_synth, "fid": None}
        if n_real == 0 or n_synth == 0:
            row["fid"] = "N/A"
            print(f"  {class_name:<16} real={n_real:5d} synth={n_synth:5d} -> FID: N/A (insufficient images)")
        else:
            real_feats = extract_features(inception, real_ds, device, batch_size)
            synth_feats = extract_features(inception, synth_ds, device, batch_size)
            mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
            mu_s, sigma_s = synth_feats.mean(axis=0), np.cov(synth_feats, rowvar=False)
            fid = _fid(mu_r, sigma_r, mu_s, sigma_s)
            row["fid"] = round(fid, 4)
            print(f"  {class_name:<16} real={n_real:5d} synth={n_synth:5d} -> FID: {fid:.4f}")
            all_real_features.append(real_feats)
            all_synth_features.append(synth_feats)

        results.append(row)

    # --- Overall FID (all classes pooled) ------------------------------------
    if all_real_features and all_synth_features:
        real_all = np.vstack(all_real_features)
        synth_all = np.vstack(all_synth_features)
        mu_r, sigma_r = real_all.mean(axis=0), np.cov(real_all, rowvar=False)
        mu_s, sigma_s = synth_all.mean(axis=0), np.cov(synth_all, rowvar=False)
        fid_overall = _fid(mu_r, sigma_r, mu_s, sigma_s)
        overall_row = {"class": "OVERALL", "real": len(real_all), "synthetic": len(synth_all), "fid": round(fid_overall, 4)}
        print(f"  {'OVERALL':<16} real={len(real_all):5d} synth={len(synth_all):5d} -> FID: {fid_overall:.4f}")
    else:
        fid_overall = None
        overall_row = {"class": "OVERALL", "real": 0, "synthetic": 0, "fid": "N/A"}
        print("  OVERALL: N/A (insufficient images)")

    results.append(overall_row)

    # --- Write CSV -----------------------------------------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "real", "synthetic", "fid"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] FID scores written to {out_csv}")
    return results, fid_overall


def main():
    parser = argparse.ArgumentParser(
        description="Compute Frechet Inception Distance between real and synthetic images "
                    "using pretrained InceptionV3 features.")
    parser.add_argument("--real-dir", required=True, help="Folder of real images (per-class subfolders)")
    parser.add_argument("--synthetic-dir", required=True, help="Folder of synthetic images (per-class subfolders)")
    parser.add_argument("--out-csv", default="outputs/v2/fid_scores.csv",
                        help="Output CSV path (default: outputs/v2/fid_scores.csv)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    compute_fid(
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
