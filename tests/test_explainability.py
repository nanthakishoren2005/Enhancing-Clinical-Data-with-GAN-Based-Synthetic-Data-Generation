"""
Unit tests for Grad-CAM++ Explainable AI module.
Validates gradient hooks, activation maps, bicubic interpolation, colorization, and overlay blending.
"""
import pytest
from PIL import Image
import numpy as np
import torch
from pathlib import Path

from src.explainability.gradcam import (
    GradCAMPlusPlus,
    generate_gradcampp_explanation,
    colorize_heatmap,
    blend_heatmap_overlay
)
from src.cnn.predict import load_trained_model
from src.utils.config import get_project_root


@pytest.fixture
def sample_cxr():
    root = get_project_root()
    test_p = root / "data" / "original" / "test" / "normal"
    files = list(test_p.glob("*.png")) + list(test_p.glob("*.jpg"))
    if files:
        return Image.open(files[0]).convert("RGB")
    # Fallback synthetic grayscale CXR representation
    arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


def test_gradcampp_core_computation(sample_cxr):
    """Verify that GradCAMPlusPlus computes valid spatial saliency maps on ResNet-18."""
    root = get_project_root()
    ckpt_path = root / "checkpoints" / "augmented_best.pt"
    if not ckpt_path.exists():
        ckpt_path = root / "checkpoints" / "baseline_best.pt"
    
    assert ckpt_path.exists(), "Model checkpoint must exist for Grad-CAM++ testing."
    model = load_trained_model(checkpoint_path=ckpt_path)
    model.eval()

    # Target last conv layer (auto-detect works for both ResNet-18 and CustomCNN)
    cam = GradCAMPlusPlus(model, target_layer=None)

    # Prepare input tensor
    arr = np.array(sample_cxr.resize((256, 256)), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

    saliency = cam(tensor, target_class=1)

    assert isinstance(saliency, np.ndarray)
    assert saliency.shape == (256, 256)
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0 + 1e-5
    # Saliency should not be entirely NaN or Inf
    assert not np.isnan(saliency).any()
    assert not np.isinf(saliency).any()


def test_colorize_and_overlay(sample_cxr):
    """Verify heatmap colorization and alpha blending."""
    fake_map = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    
    for cmap in ["inferno", "plasma", "jet", "viridis"]:
        colored = colorize_heatmap(fake_map, colormap_name=cmap)
        assert isinstance(colored, Image.Image)
        assert colored.size == (256, 256)
        assert colored.mode == "RGB"

    overlay = blend_heatmap_overlay(sample_cxr, fake_map, alpha=0.5, colormap_name="inferno")
    assert isinstance(overlay, Image.Image)
    assert overlay.size == sample_cxr.size
    assert overlay.mode == "RGB"


def test_generate_gradcampp_explanation_pipeline(sample_cxr):
    """Test full high-level Grad-CAM++ pipeline on baseline and augmented models."""
    root = get_project_root()
    ckpt_b = root / "checkpoints" / "baseline_best.pt"
    ckpt_a = root / "checkpoints" / "augmented_best.pt"

    for m_type, ckpt in [("baseline", ckpt_b), ("augmented", ckpt_a)]:
        if not ckpt.exists():
            continue
        expl = generate_gradcampp_explanation(
            image_input=sample_cxr,
            model_type=m_type,
            checkpoint_path=ckpt,
            target_class_idx=None,  # auto-predict
            colormap_name="inferno",
            overlay_alpha=0.45,
        )
        assert "saliency_map" in expl
        assert "heatmap_image" in expl
        assert "overlay_image" in expl
        assert "target_class_name" in expl
        assert "target_layer_name" in expl
        assert isinstance(expl["target_layer_name"], str)
        assert len(expl["target_layer_name"]) > 0
        assert expl["saliency_map"].ndim == 2
        assert expl["heatmap_image"].size == sample_cxr.size
