"""
Unit tests for multiclass metrics and One-vs-Rest sensitivity & specificity calculations.
"""
import numpy as np
from src.evaluation.metrics import calculate_multiclass_metrics, calculate_ovr_sensitivity_specificity


def test_metrics_calculation_perfect():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 3]

    metrics = calculate_multiclass_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_sensitivity"] == 1.0
    assert metrics["macro_specificity"] == 1.0


def test_ovr_sensitivity_specificity():
    # 4x4 confusion matrix:
    cm = np.array([
        [8, 1, 1, 0],
        [1, 8, 1, 0],
        [1, 1, 8, 0],
        [0, 0, 1, 9],
    ])

    ovr = calculate_ovr_sensitivity_specificity(cm)
    assert ovr["normal"]["sensitivity"] == 0.8
    assert "covid" in ovr
    assert "lung_opacity" in ovr
    assert "viral_pneumonia" in ovr
