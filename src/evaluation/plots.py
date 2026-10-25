"""
Visualization and plotting module for model evaluation comparisons,
confusion matrices, and metric breakdown charts.
"""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import CLASS_TO_IDX


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Path = None,
    cmap: str = "Blues",
) -> None:
    """Plot single confusion matrix heatmap with counts and percentages."""
    if classes is None:
        classes = list(CLASS_TO_IDX.keys())

    classes_display = [c.capitalize() for c in classes]
    plt.figure(figsize=(6, 5))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=classes_display,
        yticklabels=classes_display,
        cbar=True,
        linewidths=1,
        linecolor="#f0f0f0",
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_side_by_side_confusion_matrices(
    cm_baseline: np.ndarray,
    cm_augmented: np.ndarray,
    classes: List[str] = None,
    save_path: Path = None,
) -> None:
    """Plot Baseline vs GAN-Augmented confusion matrices side by side for direct comparison."""
    if classes is None:
        classes = list(CLASS_TO_IDX.keys())
    classes_display = [c.capitalize() for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, cm, title, cmap in zip(
        axes,
        [cm_baseline, cm_augmented],
        ["Model A: Baseline CNN", "Model B: GAN-Augmented CNN"],
        ["Blues", "Greens"],
    ):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cmap,
            xticklabels=classes_display,
            yticklabels=classes_display,
            cbar=False,
            ax=ax,
            linewidths=1,
            linecolor="#e0e0e0",
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

    plt.suptitle("Test Set Confusion Matrix Comparison (Untouched Test Slices)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_metric_comparison(
    baseline_metrics: Dict[str, Any],
    augmented_metrics: Dict[str, Any],
    save_path: Path = None,
) -> None:
    """Grouped bar chart comparing key macro performance indicators."""
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Precision (Macro)"),
        ("macro_recall", "Recall / Sens (Macro)"),
        ("macro_f1", "F1-Score (Macro)"),
        ("macro_specificity", "Specificity (Macro)"),
    ]

    labels = [m[1] for m in metrics_to_plot]
    base_vals = [baseline_metrics[m[0]] * 100 for m in metrics_to_plot]
    aug_vals = [augmented_metrics[m[0]] * 100 for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    rects1 = plt.bar(x - width/2, base_vals, width, label="Baseline CNN", color="#3498db", edgecolor="#2980b9")
    rects2 = plt.bar(x + width/2, aug_vals, width, label="GAN-Augmented CNN", color="#2ecc71", edgecolor="#27ae60")

    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Comparative Performance: Baseline vs. GAN-Augmented CNN", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Attach values above bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_per_class_comparison(
    baseline_per_class: Dict[str, Dict[str, float]],
    augmented_per_class: Dict[str, Dict[str, float]],
    metric: str = "f1_score",
    save_path: Path = None,
) -> None:
    """Grouped bar chart for per-class comparisons (F1, Precision, or Sensitivity)."""
    classes = list(baseline_per_class.keys())
    base_vals = [baseline_per_class[c][metric] * 100 for c in classes]
    aug_vals = [augmented_per_class[c][metric] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width/2, base_vals, width, label="Baseline CNN", color="#3498db")
    plt.bar(x + width/2, aug_vals, width, label="GAN-Augmented CNN", color="#2ecc71")

    plt.ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=11)
    plt.title(f"Per-Class {metric.replace('_', ' ').title()} Comparison", fontsize=12, fontweight="bold")
    plt.xticks(x, [c.capitalize() for c in classes], fontsize=11)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()
