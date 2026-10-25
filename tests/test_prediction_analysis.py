"""
Unit tests for Medical Scan Analysis & Quantitative Risk Rate Calculation.
Tests Normal, COVID-19, Pulmonary Opacity, and Viral Pneumonia classification.
"""
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from src.cnn.predict import analyze_chest_xray, analyze_lung_scan


@pytest.fixture
def sample_scans_dir():
    return Path("data/sample_scans")


def test_normal_scan_classification(sample_scans_dir):
    """Test that a normal scan is classified as NORMAL with low risk rate."""
    img_path = sample_scans_dir / "sample_normal.png"
    if not img_path.exists():
        img_path = sample_scans_dir / "sample_normal.jpg"
    assert img_path.exists(), f"Sample normal scan must exist at {img_path}"
    
    res = analyze_lung_scan(img_path, model_type="augmented")
    assert res["disease_status"] == "NORMAL"
    assert res["is_diseased"] is False
    assert "Normal" in res["pathology_name"]
    assert 0.0 <= res["risk_rate"] <= 20.0
    assert "normal" in res["confidence_scores"]


def test_covid_classification(sample_scans_dir):
    """Test that a COVID-19 scan is classified as DISEASED with high risk rate."""
    img_path = sample_scans_dir / "sample_covid.png"
    if not img_path.exists():
        img_path = sample_scans_dir / "sample_covid.jpg"
    assert img_path.exists(), f"Sample covid scan must exist at {img_path}"
    
    res = analyze_lung_scan(img_path, model_type="augmented")
    assert res["disease_status"] == "DISEASED"
    assert res["is_diseased"] is True
    assert res["predicted_class"] == "covid"
    assert "COVID" in res["pathology_name"]
    assert res["risk_rate"] >= 65.0


def test_lung_opacity_classification(sample_scans_dir):
    """Test that a lung opacity scan is classified as DISEASED."""
    img_path = sample_scans_dir / "sample_lung_opacity.png"
    if not img_path.exists():
        img_path = sample_scans_dir / "sample_lung_opacity.jpg"
    assert img_path.exists(), f"Sample opacity scan must exist at {img_path}"
    
    res = analyze_lung_scan(img_path, model_type="augmented")
    assert res["disease_status"] == "DISEASED"
    assert res["is_diseased"] is True
    assert res["predicted_class"] == "lung_opacity"
    assert "Opacity" in res["pathology_name"]


def test_viral_pneumonia_classification(sample_scans_dir):
    """Test that a viral pneumonia scan is classified as DISEASED with high risk rate."""
    img_path = sample_scans_dir / "sample_viral_pneumonia.png"
    if not img_path.exists():
        img_path = sample_scans_dir / "sample_viral_pneumonia.jpg"
    assert img_path.exists(), f"Sample viral pneumonia scan must exist at {img_path}"
    
    res = analyze_lung_scan(img_path, model_type="augmented")
    assert res["disease_status"] == "DISEASED"
    assert res["is_diseased"] is True
    assert res["predicted_class"] == "viral_pneumonia"
    assert res["risk_rate"] >= 60.0


def test_radiomic_indicators_and_structure(sample_scans_dir):
    """Test that all radiomic indicators and clinical findings are returned."""
    img_path = sample_scans_dir / "sample_normal.png"
    if not img_path.exists():
        img_path = sample_scans_dir / "sample_normal.jpg"
    
    res = analyze_lung_scan(img_path, model_type="augmented")
    
    rad = res["radiomic_indicators"]
    assert "mean_attenuation" in rad
    assert "roi_heterogeneity" in rad
    assert "intensity_range" in rad
    assert "margin_definition" in rad
    assert len(res["clinical_findings"]) > 0
    assert len(res["recommendations"]) > 0


def test_random_color_object_rejected():
    """Verify that an everyday random color object is intercepted and not diagnosed as COVID or diseased."""
    # Simulated color object (e.g., photo of a red mug on a wooden desk)
    photo = np.zeros((256, 256, 3), dtype=np.uint8)
    photo[:, :] = [120, 80, 50]  # desk background
    photo[80:180, 80:180] = [220, 30, 30]  # red mug
    img = Image.fromarray(photo)

    res = analyze_lung_scan(img, model_type="augmented")
    assert res["is_valid_scan"] is False
    assert res["disease_status"] == "INVALID_SCAN"
    assert res["is_diseased"] is False
    assert res["risk_rate"] == 0.0
    assert "INVALID INPUT" in res["status_badge"]
    assert "Non-Medical" in res["pathology_name"] or "Invalid" in res["pathology_name"]


def test_blank_canvas_rejected():
    """Verify that a blank uniform image is intercepted and rejected."""
    blank = Image.fromarray(np.full((256, 256), 255, dtype=np.uint8))
    res = analyze_lung_scan(blank, model_type="augmented")
    assert res["is_valid_scan"] is False
    assert res["disease_status"] == "INVALID_SCAN"
    assert res["is_diseased"] is False
    assert res["risk_rate"] == 0.0
