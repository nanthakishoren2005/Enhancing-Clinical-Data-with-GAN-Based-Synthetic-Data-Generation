"""
Uncertainty Threshold Calibration Script.
Evaluates the untouched validation cohort (data/original/val) to determine empirical
thresholds for Out-of-Distribution (OOD) detection, predictive entropy, and classification margin.
STRICT ACADEMIC INTEGRITY: Only uses validation data (NOT synthetic, NOT held-out test data).
"""
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils.config import load_config, get_project_root
from src.cnn.predict import load_trained_model, preprocess_for_inference, robust_image_loader
from src.data.dataset import CLASS_TO_IDX


def compute_image_color_dispersion(pil_img: Image.Image) -> float:
    """
    Compute inter-channel color dispersion:
    Chest X-rays are projection radiographs with near-identical R, G, B channels.
    Natural color photos (animals, cars, landscapes, selfies) have high inter-channel divergence.
    """
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    # Mean absolute difference between R, G, B channels normalized by 255
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
    diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])
    diff_br = np.abs(arr[:, :, 2] - arr[:, :, 0])
    mean_color_disp = float(np.mean(diff_rg + diff_gb + diff_br) / 3.0)
    return mean_color_disp


def calibrate_uncertainty_thresholds(
    model_type: str = "augmented",
    output_path: Path = None,
) -> dict:
    root = get_project_root()
    val_dir = root / "data" / "original" / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at: {val_dir}")

    output_path = output_path or root / "configs" / "uncertainty_thresholds.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_type} model for uncertainty threshold calibration on validation data...")
    model = load_trained_model(model_type=model_type)
    model.eval()

    energies = []
    entropies = []
    margins = []
    color_dispersions = []
    max_probs = []

    total_images = 0
    with torch.no_grad():
        for cname in CLASS_TO_IDX:
            cdir = val_dir / cname
            for f in sorted(cdir.glob("*.*")):
                img = robust_image_loader(f)
                color_disp = compute_image_color_dispersion(img)
                color_dispersions.append(color_disp)

                tensor = preprocess_for_inference(img).to("cpu")
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

                # Free energy with temperature T=1.0: E(x) = -LogSumExp(f(x))
                energy = -1.0 * torch.logsumexp(logits, dim=1).item()
                energies.append(energy)

                # Normalized predictive entropy: H(p) / log(C)
                n_classes = len(probs)
                ent = -float(np.sum(probs * np.log(probs + 1e-12)) / np.log(n_classes))
                entropies.append(ent)

                # Top-1 vs Top-2 Margin
                sorted_p = np.sort(probs)
                margin = float(sorted_p[-1] - sorted_p[-2])
                margins.append(margin)

                max_probs.append(float(sorted_p[-1]))
                total_images += 1

    energies = np.array(energies)
    entropies = np.array(entropies)
    margins = np.array(margins)
    color_dispersions = np.array(color_dispersions)

    # Establish validation percentiles
    # Energy OOD: 98th percentile of in-distribution validation data (higher energy indicates unfamiliarity)
    energy_ood_cutoff = float(np.percentile(energies, 98))
    
    # Entropy Ambiguity: 85th percentile of in-distribution validation data
    entropy_ambiguity_cutoff = float(np.percentile(entropies, 85))
    
    # Margin Ambiguity: 15th percentile of validation margins (lower margin indicates uncertainty)
    margin_ambiguity_cutoff = float(np.percentile(margins, 15))

    # Color dispersion threshold: 99th percentile of in-distribution CXRs + safety buffer
    color_disp_max = float(max(12.0, np.percentile(color_dispersions, 99) * 2.5))

    thresholds = {
        "calibrated_on": "data/original/val",
        "sample_count": total_images,
        "classes": list(CLASS_TO_IDX.keys()),
        "model_type": model_type,
        "energy_score": {
            "mean": float(np.mean(energies)),
            "std": float(np.std(energies)),
            "ood_threshold": round(energy_ood_cutoff, 3), # Inputs with energy > this are flagged as OOD
        },
        "predictive_entropy": {
            "mean": float(np.mean(entropies)),
            "ambiguity_threshold": round(entropy_ambiguity_cutoff, 3), # Inputs with entropy > this are flagged as ambiguous
        },
        "top2_margin": {
            "mean": float(np.mean(margins)),
            "ambiguity_threshold": round(margin_ambiguity_cutoff, 3), # Inputs with margin < this are flagged as borderline
        },
        "color_dispersion": {
            "mean": float(np.mean(color_dispersions)),
            "non_radiograph_threshold": round(color_disp_max, 2), # Inputs with color divergence > this are flagged as non-CXR
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Calibrated uncertainty thresholds on {total_images} validation images successfully!")
    print(f"Saved to: {output_path}")
    print(json.dumps(thresholds, indent=2))
    return thresholds


if __name__ == "__main__":
    calibrate_uncertainty_thresholds()
