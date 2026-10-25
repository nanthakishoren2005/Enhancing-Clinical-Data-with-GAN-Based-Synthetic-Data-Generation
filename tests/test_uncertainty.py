"""
Unit tests for Uncertainty-Aware Prediction & Out-of-Distribution Detection.
Tests validation calibration thresholding, Free Energy scoring, Entropy, Top-2 Margin, and Non-radiograph detection.
"""
import pytest
from PIL import Image
import numpy as np
import torch
from pathlib import Path

from src.uncertainty.detector import UncertaintyDetector
from src.cnn.predict import analyze_chest_xray
from src.utils.config import get_project_root


@pytest.fixture
def detector():
    return UncertaintyDetector()


@pytest.fixture
def sample_real_cxr():
    root = get_project_root()
    test_p = root / "data" / "original" / "test" / "covid"
    files = list(test_p.glob("*.png")) + list(test_p.glob("*.jpg"))
    if files:
        return Image.open(files[0]).convert("RGB")
    arr = np.random.randint(60, 190, (256, 256), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


@pytest.fixture
def sample_color_photo():
    """Vibrant non-radiograph color image (high color dispersion)."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # Red channel
    arr[:, :, 1] = 50   # Green channel
    arr[:, :, 2] = 20   # Blue channel
    return Image.fromarray(arr)


def test_uncertainty_thresholds_loaded(detector):
    """Verify that thresholds were loaded from configs/uncertainty_thresholds.json."""
    assert detector.calibrated_on == "data/original/val"
    assert detector.energy_threshold is not None
    assert detector.entropy_threshold is not None
    assert detector.margin_threshold is not None
    assert detector.color_dispersion_threshold is not None


def test_free_energy_calculation(detector):
    """Verify Free Energy formula: E(x) = -T * logsumexp(logits / T)."""
    logits = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float32)
    energy = detector.compute_free_energy(logits, temperature=1.0)
    expected = -np.log(np.sum(np.exp(logits)))
    assert np.isclose(energy, expected, atol=1e-4)


def test_predictive_entropy_and_margin(detector):
    """Verify entropy and margin metrics."""
    # Definite prediction: [1.0, 0.0, 0.0, 0.0]
    sharp_probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ent_sharp = detector.compute_predictive_entropy(sharp_probs)
    assert np.isclose(ent_sharp, 0.0, atol=1e-4)
    margin_sharp = detector.compute_top2_margin(sharp_probs)
    assert np.isclose(margin_sharp, 1.0, atol=1e-4)

    # Completely uniform prediction: [0.25, 0.25, 0.25, 0.25]
    flat_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    ent_flat = detector.compute_predictive_entropy(flat_probs)
    assert np.isclose(ent_flat, 1.0, atol=1e-4)
    margin_flat = detector.compute_top2_margin(flat_probs)
    assert np.isclose(margin_flat, 0.0, atol=1e-4)


def test_in_distribution_real_cxr(detector, sample_real_cxr):
    """Verify that an authentic CXR with sharp logits evaluates as reliable."""
    # Strong authentic logits
    logits = np.array([3.5, -1.2, -2.0, -3.1], dtype=np.float32)
    res = detector.evaluate(sample_real_cxr, logits)

    assert res["is_out_of_distribution"] is False
    assert res["is_non_radiograph"] is False
    assert res["is_ambiguous"] is False
    assert res["reliability_level"] == "HIGH"
    assert "RELIABLE" in res["status_badge"]


def test_color_photo_detection(detector, sample_color_photo):
    """Verify that non-radiograph color photos trigger an OOD warning."""
    logits = np.array([1.5, 0.5, -0.5, -1.0], dtype=np.float32)
    res = detector.evaluate(sample_color_photo, logits)

    assert res["is_non_radiograph"] is True
    assert res["is_out_of_distribution"] is True
    assert "NON-RADIOGRAPH" in res["reliability_level"]
    assert "UNRELIABLE" in res["reliability_level"]
    assert "non-radiograph" in res["clinical_warning"].lower() or "color" in res["clinical_warning"].lower()


def test_ambiguous_prediction_flagging(detector, sample_real_cxr):
    """Verify that ambiguous predictions with low margin / high entropy trigger warnings."""
    # Competing logits between class 0 and 1 with in-distribution total energy
    ambiguous_logits = np.array([3.01, 3.00, -1.0, -1.0], dtype=np.float32)
    res = detector.evaluate(sample_real_cxr, ambiguous_logits)

    assert res["is_ambiguous"] is True
    assert res["reliability_level"] in ["LOW (AMBIGUOUS / BORDERLINE)", "AMBIGUOUS", "MEDIUM"]
    assert "borderline" in res["clinical_warning"].lower() or "ambiguous" in res["status_badge"].lower()


def test_full_pipeline_uncertainty_integration(sample_real_cxr, sample_color_photo):
    """Test analyze_chest_xray end-to-end with uncertainty assessment enabled."""
    res_real = analyze_chest_xray(sample_real_cxr, model_type="augmented", generate_xai=True, assess_uncertainty=True)
    assert "uncertainty" in res_real
    assert "gradcam" in res_real
    assert res_real["uncertainty"]["reliability_level"] in ["HIGH", "MEDIUM"]

    res_color = analyze_chest_xray(sample_color_photo, model_type="augmented", generate_xai=True, assess_uncertainty=True)
    assert "uncertainty" in res_color
    assert res_color["uncertainty"]["is_non_radiograph"] is True
    assert res_color["uncertainty"]["reliability_level"] == "UNRELIABLE (NON-RADIOGRAPH / OOD)"
