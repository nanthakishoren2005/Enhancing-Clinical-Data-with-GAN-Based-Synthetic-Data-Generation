"""
Explainable AI (XAI) Module: Grad-CAM++ for ResNet-18 Medical Image Interpretability.
Implements the Grad-CAM++ algorithm (Chattopadhay et al., 2018) for fine-grained,
pixel-accurate visual explanations of convolutional neural network predictions.
"""
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm

from src.utils.config import load_config, get_project_root, get_device
from src.cnn.predict import robust_image_loader, preprocess_for_inference, load_trained_model
from src.data.dataset import IDX_TO_CLASS, CLASS_TO_IDX


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Generalized Gradient-based Visual Explanations) implementation.
    Extracts forward feature activations and backward gradients from the target
    convolutional layer to synthesize weighted localization maps.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        # Auto-detect target layer if not provided: last convolutional layer
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer

        # Store a human-readable name for the target layer (architecture-agnostic)
        self.target_layer_name = self._get_target_layer_name()

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []

    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in ResNet-18 or CustomCNN."""
        if hasattr(self.model, "layer4"):
            # ResNet-18: Last conv layer in layer4
            last_block = self.model.layer4[-1]
            if hasattr(last_block, "conv2"):
                return last_block.conv2
            return last_block
        elif hasattr(self.model, "features"):
            # CustomCNN: Last Conv2d in features sequential
            for layer in reversed(self.model.features):
                if isinstance(layer, nn.Conv2d):
                    return layer
            return self.model.features[-1]
        else:
            # Fallback: scan for last Conv2d in entire model
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv
            raise RuntimeError("Could not automatically locate a target Conv2d layer in the model.")

    def _get_target_layer_name(self) -> str:
        """Return a human-readable name for the target layer (architecture-agnostic)."""
        if hasattr(self.model, "layer4"):
            return "model.layer4[-1].conv2"
        elif hasattr(self.model, "features"):
            return "model.features[-1].conv"
        else:
            return "model.last_conv2d"

    def _register_hooks(self):
        self._remove_hooks()
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM++ heatmap from input tensor.
        Returns:
            - normalized heatmap (2D float numpy array in [0, 1] matching tensor spatial resolution)
            - target class index
            - target class logit score
        """
        self._register_hooks()
        try:
            # Ensure tensor requires gradients
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad_(True)

            # Ensure model parameters allow backward gradient calculation
            with torch.enable_grad():
                logits = self.model(input_tensor)

                if target_class_idx is None:
                    target_class_idx = int(logits.argmax(dim=1).item())

                score = logits[0, target_class_idx]
                score_val = float(score.item())

                self.model.zero_grad()
                score.backward(retain_graph=False)

                if self.activations is None or self.gradients is None:
                    raise RuntimeError("Failed to capture feature activations or gradients during Grad-CAM++ execution.")

                # Activations and gradients: (1, C, H, W)
                feat = self.activations
                grad = self.gradients

                # Grad-CAM++ Weight Computation
                # alpha_ijk = (grad^2) / (2 * grad^2 + sum(feat * grad^3) + eps)
                grad_2 = grad.pow(2)
                grad_3 = grad.pow(3)

                sum_feat_grad3 = (feat * grad_3).sum(dim=(2, 3), keepdim=True)
                eps = 1e-7
                denom = 2.0 * grad_2 + sum_feat_grad3
                denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)

                alpha = grad_2 / denom
                weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)

                # Weighted forward activation maps
                cam = (weights * feat).sum(dim=1, keepdim=True)
                cam = F.relu(cam)

                # Bilinear/bicubic interpolation to match input tensor spatial resolution
                cam = F.interpolate(
                    cam,
                    size=(input_tensor.size(2), input_tensor.size(3)),
                    mode="bilinear",
                    align_corners=False
                )

                # Normalize to [0, 1]
                cam_min = cam.min()
                cam_max = cam.max()
                if (cam_max - cam_min) > eps:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = torch.zeros_like(cam)

                heatmap = cam.squeeze().detach().cpu().numpy()
                return heatmap, target_class_idx, score_val

        finally:
            self._remove_hooks()

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Convenience callable: returns 2D numpy heatmap directly."""
        heatmap, _, _ = self.generate_heatmap(input_tensor=input_tensor, target_class_idx=target_class)
        return heatmap


def colorize_heatmap(
    heatmap_2d: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    colormap_name: str = "inferno",
) -> Image.Image:
    """
    Interpolate 2D float heatmap [0, 1] to target image size and apply colormap.
    """
    import cv2

    if target_size is None:
        target_size = (heatmap_2d.shape[1], heatmap_2d.shape[0])

    w, h = target_size
    # Resize heatmap to match original image dimensions using bicubic interpolation
    resized_hm = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    resized_hm = np.clip(resized_hm, 0.0, 1.0)

    # Colormap via matplotlib
    cmap = cm.get_cmap(colormap_name) if hasattr(cm, "get_cmap") else cm.plasma
    colored = (cmap(resized_hm)[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(colored)


def blend_heatmap_overlay(
    original_img: Image.Image,
    heatmap_img: Union[Image.Image, np.ndarray],
    alpha: float = 0.45,
    colormap_name: str = "inferno",
) -> Image.Image:
    """
    Overlay colormapped Grad-CAM++ heatmap onto the original anatomical scan.
    """
    orig_rgb = original_img.convert("RGB")
    if isinstance(heatmap_img, np.ndarray):
        heatmap_img = colorize_heatmap(heatmap_img, target_size=orig_rgb.size, colormap_name=colormap_name)

    if orig_rgb.size != heatmap_img.size:
        heatmap_img = heatmap_img.resize(orig_rgb.size, Image.Resampling.BILINEAR)

    orig_arr = np.array(orig_rgb, dtype=np.float32)
    heat_arr = np.array(heatmap_img, dtype=np.float32)

    blended = (1.0 - alpha) * orig_arr + alpha * heat_arr
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def generate_gradcampp_explanation(
    image_input: Union[str, Path, Image.Image],
    model: Optional[nn.Module] = None,
    model_type: str = "augmented",
    checkpoint_path: Optional[Union[str, Path]] = None,
    target_class_idx: Optional[int] = None,
    colormap_name: str = "inferno",
    overlay_alpha: float = 0.45,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    High-level API: Generates complete Grad-CAM++ explanation package for any medical scan.
    Returns:
        - original_image: PIL Image
        - heatmap_image: PIL Image (colormapped)
        - overlay_image: PIL Image (original + heatmap blend)
        - target_class_name: str
        - target_class_idx: int
        - heatmap_raw: 2D numpy array [0, 1]
        - target_layer_name: str
        - method: 'Grad-CAM++'
    """
    config = load_config()
    device = device or get_device(config)

    # 1. Load and normalize image
    image = robust_image_loader(image_input)
    orig_size = image.size  # (width, height)

    # 2. Preprocess tensor
    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])
    tensor = preprocess_for_inference(image, img_size, mean, std).to(device)

    # 3. Model acquisition
    if model is None:
        model = load_trained_model(model_type=model_type, checkpoint_path=checkpoint_path, device=device)

    explainer = GradCAMPlusPlus(model=model, device=device)
    raw_heatmap, class_idx, score_val = explainer.generate_heatmap(
        input_tensor=tensor,
        target_class_idx=target_class_idx,
    )

    class_name = IDX_TO_CLASS.get(class_idx, f"Class_{class_idx}")

    # 4. Colorize and overlay
    heatmap_colored = colorize_heatmap(raw_heatmap, target_size=orig_size, colormap_name=colormap_name)
    overlay_img = blend_heatmap_overlay(image, heatmap_colored, alpha=overlay_alpha)

    return {
        "original_image": image,
        "heatmap_image": heatmap_colored,
        "overlay_image": overlay_img,
        "heatmap_raw": raw_heatmap,
        "saliency_map": raw_heatmap,
        "target_class_idx": class_idx,
        "target_class_name": class_name,
        "score_value": score_val,
        "target_layer_name": explainer.target_layer_name,
        "method": "Grad-CAM++",
        "colormap": colormap_name,
        "overlay_alpha": overlay_alpha,
    }
