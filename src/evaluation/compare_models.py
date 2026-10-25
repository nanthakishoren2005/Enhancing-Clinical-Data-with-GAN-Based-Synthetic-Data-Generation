"""
Comprehensive evaluation and comparison module.
Evaluates both Model A (Baseline CNN) and Model B (GAN-Augmented CNN)
on the exact same untouched real test set.
Outputs comparative metrics, confusion matrices, and saved report files.
Usage:
    python -m src.evaluation.compare_models
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.data.dataset import ChestXrayDataset, get_transforms, CLASS_TO_IDX
from src.cnn.predict import load_trained_model
from src.evaluation.metrics import evaluate_model_on_dataset
from src.evaluation.plots import (
    plot_side_by_side_confusion_matrices,
    plot_metric_comparison,
    plot_per_class_comparison,
)


def run_model_comparison() -> Dict[str, Any]:
    """
    Evaluates baseline and augmented models on real test set and generates reports.
    """
    config = load_config()
    root = get_project_root()
    device = get_device(config)
    set_seed(config["project"]["random_seed"])

    checkpoints_dir = root / config["paths"]["checkpoints_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    reports_dir = root / config["paths"]["reports_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = checkpoints_dir / "baseline_best.pt"
    aug_ckpt = checkpoints_dir / "augmented_best.pt"

    # Pre-check checkpoints
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"Baseline model checkpoint missing at: {base_ckpt}\n"
            "Please train the baseline model first using: python -m src.cnn.train_baseline"
        )
    if not aug_ckpt.exists():
        raise FileNotFoundError(
            f"GAN-Augmented model checkpoint missing at: {aug_ckpt}\n"
            "Please train the augmented model first using: python -m src.cnn.train_augmented"
        )

    # Load Test Set (STRICTLY real untouched images)
    data_dir = root / config["dataset"]["original_dir"]
    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    test_dataset = ChestXrayDataset(
        root_dir=data_dir,
        split="test",
        synthetic_dir=None,  # NEVER include synthetic data in testing!
        transform=get_transforms("test", img_size, mean, std),
    )

    if len(test_dataset) == 0:
        raise RuntimeError(f"No test images found in {data_dir / 'test'}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["cnn"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
    )

    print(f"\n=======================================================")
    print(f" EVALUATING MODELS ON UNTOUCHED REAL TEST SET")
    print(f" Test Set Size: {len(test_dataset)} images across {len(CLASS_TO_IDX)} classes")
    print(f" Device: {device}")
    print(f"=======================================================")

    # 1. Evaluate Baseline CNN
    print("\n[1/2] Evaluating Model A: Baseline CNN...")
    model_baseline = load_trained_model(model_type="baseline", checkpoint_path=base_ckpt, device=device)
    base_metrics, y_true_b, y_pred_b, _ = evaluate_model_on_dataset(model_baseline, test_loader, device)

    # 2. Evaluate GAN-Augmented CNN
    print("[2/2] Evaluating Model B: GAN-Augmented CNN...")
    model_augmented = load_trained_model(model_type="augmented", checkpoint_path=aug_ckpt, device=device)
    aug_metrics, y_true_a, y_pred_a, _ = evaluate_model_on_dataset(model_augmented, test_loader, device)

    # Compile Comparison Table
    metrics_summary = [
        {"Metric": "Accuracy", "Baseline CNN": base_metrics["accuracy"], "GAN-Augmented CNN": aug_metrics["accuracy"]},
        {"Metric": "Macro Precision", "Baseline CNN": base_metrics["macro_precision"], "GAN-Augmented CNN": aug_metrics["macro_precision"]},
        {"Metric": "Macro Recall (Sensitivity)", "Baseline CNN": base_metrics["macro_recall"], "GAN-Augmented CNN": aug_metrics["macro_recall"]},
        {"Metric": "Macro F1-Score", "Baseline CNN": base_metrics["macro_f1"], "GAN-Augmented CNN": aug_metrics["macro_f1"]},
        {"Metric": "Macro Specificity", "Baseline CNN": base_metrics["macro_specificity"], "GAN-Augmented CNN": aug_metrics["macro_specificity"]},
        {"Metric": "Weighted F1-Score", "Baseline CNN": base_metrics["weighted_f1"], "GAN-Augmented CNN": aug_metrics["weighted_f1"]},
    ]

    df_summary = pd.DataFrame(metrics_summary)
    df_summary["Absolute Difference (+/-)"] = df_summary["GAN-Augmented CNN"] - df_summary["Baseline CNN"]
    df_summary["Relative Gain (%)"] = np.where(
        df_summary["Baseline CNN"] > 0,
        (df_summary["Absolute Difference (+/-)"] / df_summary["Baseline CNN"]) * 100,
        0.0,
    )

    # Save CSV reports
    summary_csv = reports_dir / "comparison_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    # Save detailed per-class CSV
    per_class_rows = []
    for cname in CLASS_TO_IDX:
        b_class = base_metrics["per_class"][cname]
        a_class = aug_metrics["per_class"][cname]
        per_class_rows.append({
            "Class": cname.capitalize(),
            "Support": b_class["support"],
            "Baseline Precision": b_class["precision"],
            "Augmented Precision": a_class["precision"],
            "Baseline Recall (Sensitivity)": b_class["recall"],
            "Augmented Recall (Sensitivity)": a_class["recall"],
            "Baseline Specificity": b_class["specificity"],
            "Augmented Specificity": a_class["specificity"],
            "Baseline F1": b_class["f1_score"],
            "Augmented F1": a_class["f1_score"],
            "F1 Difference": round(a_class["f1_score"] - b_class["f1_score"], 4),
        })
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_csv = reports_dir / "per_class_metrics.csv"
    df_per_class.to_csv(per_class_csv, index=False)

    # Save Full JSON report
    full_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "test_samples": len(test_dataset),
        "baseline_metrics": base_metrics,
        "augmented_metrics": aug_metrics,
        "summary": df_summary.to_dict(orient="records"),
        "per_class": per_class_rows,
    }
    with open(reports_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Generate Figures
    cm_b = np.array(base_metrics["confusion_matrix"])
    cm_a = np.array(aug_metrics["confusion_matrix"])
    plot_side_by_side_confusion_matrices(cm_b, cm_a, save_path=figures_dir / "confusion_matrix_comparison.png")
    plot_metric_comparison(base_metrics, aug_metrics, save_path=figures_dir / "metric_comparison_barchart.png")
    plot_per_class_comparison(
        base_metrics["per_class"],
        aug_metrics["per_class"],
        metric="f1_score",
        save_path=figures_dir / "per_class_f1_comparison.png",
    )

    print("\n" + "=" * 65)
    print(" EXPERIMENTAL COMPARISON RESULTS")
    print("=" * 65)
    print(df_summary.to_string(index=False))
    print("-" * 65)
    print(f"Per-Class Breakdown:\n{df_per_class.to_string(index=False)}")
    print("=" * 65)
    print(f"Outputs successfully saved to:")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Reports: {reports_dir}")

    return full_report


def main():
    run_model_comparison()


if __name__ == "__main__":
    main()
