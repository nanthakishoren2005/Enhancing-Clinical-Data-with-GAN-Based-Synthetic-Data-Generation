"""
Rigorous evaluation metrics module for multiclass chest X-ray classification.
Calculates Accuracy, Precision, Recall, F1, Confusion Matrix, and
per-class Sensitivity / Specificity using One-versus-Rest (OvR) formulation.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS


def calculate_ovr_sensitivity_specificity(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute One-versus-Rest (OvR) Sensitivity (True Positive Rate)
    and Specificity (True Negative Rate) for each chest X-ray class.
    
    Formula for class i:
        TP_i = cm[i, i]
        FN_i = sum(cm[i, :]) - TP_i
        FP_i = sum(cm[:, i]) - TP_i
        TN_i = total_samples - (TP_i + FN_i + FP_i)
        
        Sensitivity_i = TP_i / (TP_i + FN_i)
        Specificity_i = TN_i / (TN_i + FP_i)
    """
    total = np.sum(cm)
    num_classes = cm.shape[0]
    results = {}

    for i in range(num_classes):
        cname = IDX_TO_CLASS[i]
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total - (tp + fn + fp))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[cname] = {
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
        }

    return results


def calculate_multiclass_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Computes overall summary metrics, per-class breakdowns, and confusion matrix.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    labels = list(range(len(CLASS_TO_IDX)))

    # Overall metrics
    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec_macro = precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    ovr_stats = calculate_ovr_sensitivity_specificity(cm)

    # Per-class precision, recall, f1, support
    per_class_prec = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_rec = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

    per_class = {}
    for i, cname in IDX_TO_CLASS.items():
        support = int(np.sum(y_true_arr == i))
        per_class[cname] = {
            "precision": round(float(per_class_prec[i]), 4),
            "recall": round(float(per_class_rec[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "sensitivity": ovr_stats[cname]["sensitivity"],
            "specificity": ovr_stats[cname]["specificity"],
            "support": support,
        }

    macro_sens = np.mean([stats["sensitivity"] for stats in ovr_stats.values()])
    macro_spec = np.mean([stats["specificity"] for stats in ovr_stats.values()])

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "macro_sensitivity": round(float(macro_sens), 4),
        "macro_specificity": round(float(macro_spec), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "ovr_details": ovr_stats,
        "sample_count": len(y_true),
    }


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[int], List[int], np.ndarray]:
    """
    Run full forward inference over a DataLoader, extracting actual predictions and probabilities.
    Returns:
        metrics_dict, y_true, y_pred, y_probs
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_probs.append(probs)

    all_probs = np.vstack(y_probs) if y_probs else np.array([])
    metrics = calculate_multiclass_metrics(y_true, y_pred)
    return metrics, y_true, y_pred, all_probs
