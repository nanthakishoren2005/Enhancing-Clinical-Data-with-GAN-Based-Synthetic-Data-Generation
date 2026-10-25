"""
Uncertainty-Aware Prediction & Out-of-Distribution (OOD) Detection Module.
Detects ambiguous classifications, borderline predictions, and unfamiliar out-of-distribution inputs
using empirical thresholds calibrated strictly on in-distribution validation data (data/original/val).
"""
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.config import load_config, get_project_root, get_device
from src.cnn.predict import robust_image_loader, preprocess_for_inference, load_trained_model


class UncertaintyDetector:
    """
    Uncertainty-Aware Predictor combining:
    1. Input domain consistency (color dispersion and structural texture variance)
    2. Logit-based Free Energy score for Out-of-Distribution (OOD) detection
    3. Predictive Entropy and Top-2 Margin for ambiguous prediction detection
    """
    def __init__(self, thresholds_path: Optional[Path] = None):
        root = get_project_root()
        self.thresholds_path = thresholds_path or (root / "configs" / "uncertainty_thresholds.json")
        self.thresholds = self._load_thresholds()
        
        self.calibrated_on = self.thresholds.get("calibrated_on", "data/original/val")
        self.energy_threshold = self.thresholds.get("energy_score", {}).get("ood_threshold", -1.792)
        self.entropy_threshold = self.thresholds.get("predictive_entropy", {}).get("ambiguity_threshold", 0.531)
        self.margin_threshold = self.thresholds.get("top2_margin", {}).get("ambiguity_threshold", 0.488)
        self.color_dispersion_threshold = self.thresholds.get("color_dispersion", {}).get("non_radiograph_threshold", 12.0)

    def _load_thresholds(self) -> dict:
        if not self.thresholds_path.exists():
            # Fallback to calibrated defaults if file not yet written
            return {
                "calibrated_on": "data/original/val",
                "energy_score": {"ood_threshold": -1.792},
                "predictive_entropy": {"ambiguity_threshold": 0.531},
                "top2_margin": {"ambiguity_threshold": 0.488},
                "color_dispersion": {"non_radiograph_threshold": 12.0},
            }
        with open(self.thresholds_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def compute_color_dispersion(image: Image.Image) -> float:
        """Measure RGB inter-channel dispersion. Real radiographs have near-zero dispersion."""
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
        diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])
        diff_br = np.abs(arr[:, :, 2] - arr[:, :, 0])
        return float(np.mean(diff_rg + diff_gb + diff_br) / 3.0)

    @classmethod
    def compute_free_energy(cls, logits: Union[torch.Tensor, np.ndarray], temperature: float = 1.0) -> float:
        """Alias for compute_energy_score."""
        return cls.compute_energy_score(logits, temperature=temperature)

    @staticmethod
    def compute_energy_score(logits: Union[torch.Tensor, np.ndarray], temperature: float = 1.0) -> float:
        """
        Energy-based score (Liu et al., NeurIPS 2020).
        E(x; T) = -T * logsumexp(f(x) / T)
        Higher energy (less negative) indicates out-of-distribution input.
        """
        if isinstance(logits, np.ndarray):
            scaled = logits / temperature
            energy = -temperature * (np.max(scaled) + np.log(np.sum(np.exp(scaled - np.max(scaled)))))
            return float(energy)
        with torch.no_grad():
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1).item()
        return float(energy)

    @staticmethod
    def compute_predictive_entropy(probs: np.ndarray) -> float:
        """Normalized predictive entropy in [0, 1]."""
        n_classes = len(probs)
        return -float(np.sum(probs * np.log(probs + 1e-12)) / np.log(n_classes))

    @staticmethod
    def compute_top2_margin(probs: np.ndarray) -> float:
        """Margin between highest and second-highest class probability."""
        sorted_p = np.sort(probs)
        return float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) >= 2 else float(sorted_p[-1])

    @classmethod
    def compute_entropy_and_margin(cls, probs: np.ndarray) -> tuple:
        """Compute normalized predictive entropy and margin between top-1 and top-2 predictions."""
        return cls.compute_predictive_entropy(probs), cls.compute_top2_margin(probs)

    def verify_scan_authenticity(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Verify whether an uploaded image possesses genuine radiographic optical and structural properties.
        Detects:
        1. Natural color photographs (inter-channel dispersion and HSV saturation)
        2. Blank, flat, or solid color canvases (pixel variance)
        3. Non-medical graphics, screenshots, comics, and line art (bimodal clipping / low unique levels)
        4. Extreme aspect ratios (banners or non-thoracic dimensions)
        5. Total variation / unstructured noise (pure noise or artificial high-frequency grids)
        """
        image = robust_image_loader(image_input)

        # 1. Color dispersion and saturation check
        rgb_arr = np.array(image.convert("RGB"), dtype=np.float32)
        diff_rg = np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 1])
        diff_gb = np.abs(rgb_arr[:, :, 1] - rgb_arr[:, :, 2])
        diff_br = np.abs(rgb_arr[:, :, 2] - rgb_arr[:, :, 0])
        color_disp = float(np.mean(diff_rg + diff_gb + diff_br) / 3.0)

        # HSV saturation check
        hsv_arr = np.array(image.convert("HSV"), dtype=np.float32)
        sat_ratio = float(np.mean(hsv_arr[:, :, 1] / 255.0 > 0.15))

        color_thresh = self.color_dispersion_threshold
        is_color_photo = bool(color_disp > color_thresh and sat_ratio > 0.10)

        # 2. Canvas flatness / blank check
        gray_arr = np.array(image.convert("L"), dtype=np.float32)
        img_std = float(np.std(gray_arr))
        is_flat_or_blank = bool(img_std < 8.0)

        # 3. Graphic / document / screenshot check
        extreme_ratio = float(np.mean((gray_arr < 8) | (gray_arr > 248)))
        n_unique = len(np.unique(gray_arr.astype(np.uint8)))
        is_graphic_or_doc = bool((extreme_ratio > 0.75) or (n_unique < 25))

        # 4. Aspect ratio check
        w, h = image.size
        aspect = max(w / h, h / w) if min(w, h) > 0 else 1.0
        is_extreme_aspect = bool(aspect > 2.2)

        # 5. Total variation / noise check
        g_min, g_max = float(np.min(gray_arr)), float(np.max(gray_arr))
        if g_max - g_min > 1e-4:
            norm_g = (gray_arr - g_min) / (g_max - g_min)
            tv = float((np.mean(np.abs(norm_g[:, 1:] - norm_g[:, :-1])) + np.mean(np.abs(norm_g[1:, :] - norm_g[:-1, :]))) / 2.0)
        else:
            tv = 0.0
        is_noise = bool(tv > 0.09)

        reasons = []
        if is_color_photo:
            reasons.append("Natural Color Photograph (Non-Radiographic Input)")
        if is_flat_or_blank:
            reasons.append("Blank / Uniform Canvas (Lack of Anatomical Contrast)")
        if is_graphic_or_doc:
            reasons.append("Graphic / Text Document / Artificial Line Art")
        if is_extreme_aspect:
            reasons.append(f"Abnormal Non-Thoracic Aspect Ratio ({aspect:.2f}:1)")
        if is_noise:
            reasons.append("Unstructured High-Frequency Synthetic Noise")

        is_valid = not (is_color_photo or is_flat_or_blank or is_graphic_or_doc or is_extreme_aspect or is_noise)
        primary_reason = reasons[0] if reasons else "Valid Thoracic Radiograph"

        return {
            "is_valid_scan": is_valid,
            "is_color_photo": is_color_photo,
            "is_flat_or_blank": is_flat_or_blank,
            "is_graphic_or_doc": is_graphic_or_doc,
            "is_extreme_aspect": is_extreme_aspect,
            "is_noise": is_noise,
            "reasons": reasons,
            "primary_reason": primary_reason,
            "metrics": {
                "color_dispersion": round(color_disp, 2),
                "saturation_ratio": round(sat_ratio, 3),
                "pixel_std": round(img_std, 1),
                "extreme_pixel_ratio": round(extreme_ratio, 3),
                "unique_levels": n_unique,
                "aspect_ratio": round(aspect, 2),
                "total_variation": round(tv, 4),
            }
        }

    def evaluate(
        self,
        image_input: Union[str, Path, Image.Image],
        logits: Union[torch.Tensor, np.ndarray],
        probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full uncertainty profile for a prediction.
        Returns:
            - is_out_of_distribution: bool
            - is_non_radiograph: bool
            - is_valid_scan: bool
            - rejection_reason: Optional[str]
            - is_ambiguous: bool
            - is_unreliable: bool
            - reliability_level: 'HIGH' | 'AMBIGUOUS' | 'UNRELIABLE' | 'UNRELIABLE (NON-RADIOGRAPH / OOD)'
            - status_badge: str
            - status_color: str
            - clinical_warning: Optional[str]
            - detailed_metrics: dict
        """
        image = robust_image_loader(image_input)
        
        # Auto-compute probabilities if not provided
        if probs is None:
            if isinstance(logits, torch.Tensor):
                probs = F.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
            else:
                l_arr = np.array(logits, dtype=np.float32)
                exp = np.exp(l_arr - np.max(l_arr))
                probs = exp / np.sum(exp)

        # 1. Structural radiographic and authenticity check
        auth = self.verify_scan_authenticity(image)
        is_valid_scan = auth["is_valid_scan"]
        is_color_photo = auth["is_color_photo"]
        is_flat_or_blank = auth["is_flat_or_blank"]
        color_disp = auth["metrics"]["color_dispersion"]
        color_thresh = self.color_dispersion_threshold
        img_std = auth["metrics"]["pixel_std"]

        # 2. Energy-based OOD check
        energy = self.compute_energy_score(logits)
        energy_thresh = self.thresholds.get("energy_score", {}).get("ood_threshold", -1.792)
        is_energy_ood = energy > energy_thresh

        # Combined OOD flag: either fails radiographic authenticity or has abnormal free energy
        is_ood = bool((not is_valid_scan) or is_energy_ood)

        # 3. Ambiguity check (entropy & margin)
        n_classes = len(probs)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)) / np.log(n_classes))
        sorted_p = np.sort(probs)
        margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) >= 2 else 1.0

        entropy_thresh = self.thresholds.get("predictive_entropy", {}).get("ambiguity_threshold", 0.531)
        margin_thresh = self.thresholds.get("top2_margin", {}).get("ambiguity_threshold", 0.488)

        is_high_entropy = entropy > entropy_thresh
        is_low_margin = margin < margin_thresh
        is_ambiguous = bool(is_high_entropy or is_low_margin) and not is_ood

        # 4. Synthesize reliability status
        rejection_reason = None
        if not is_valid_scan:
            rejection_reason = auth["primary_reason"]
            reliability_level = "UNRELIABLE (NON-RADIOGRAPH / OOD)"
            status_badge = "UNFAMILIAR / OUT-OF-DISTRIBUTION INPUT"
            status_color = "#ef4444"  # Red
            if is_color_photo:
                warning_desc = (
                    "⚠️ NON-RADIOGRAPHIC INPUT: The uploaded image contains high inter-channel color divergence "
                    f"({color_disp:.1f} vs threshold {color_thresh:.1f}), characteristic of a natural photograph "
                    "rather than a monochrome chest projection radiograph. Deep CNN predictions on non-medical "
                    "images are scientifically invalid."
                )
            elif is_flat_or_blank:
                warning_desc = (
                    "⚠️ INVALID SCAN ARTIFACT: The input image has near-zero pixel variance (blank or solid color). "
                    "No anatomical pulmonary parenchyma could be detected."
                )
            else:
                warning_desc = (
                    f"⚠️ NON-SCAN OBJECT DETECTED: {rejection_reason}. "
                    "The image does not correspond to an authentic thoracic chest projection radiograph."
                )
            action_advice = "Input Rejected: Please upload an authentic, calibrated thoracic chest radiograph."

        elif is_energy_ood:
            rejection_reason = f"Out-of-Distribution Feature Free Energy ({energy:.2f} > {energy_thresh:.2f})"
            reliability_level = "UNRELIABLE (OOD)"
            status_badge = "UNFAMILIAR / OUT-OF-DISTRIBUTION INPUT"
            status_color = "#ef4444"
            warning_desc = (
                f"⚠️ OUT-OF-DISTRIBUTION IMAGE: The input produced an atypical free energy score ({energy:.2f} "
                f"vs validation cutoff {energy_thresh:.2f}). The model has not encountered similar feature "
                "representations during training. Prediction uncertain — further clinical review required."
            )
            action_advice = "Clinical Review Required: Verify the file is an authentic, calibrated chest radiograph."

        elif is_ambiguous:
            reliability_level = "AMBIGUOUS"
            status_badge = "AMBIGUOUS PREDICTION — Clinical Review Required"
            status_color = "#f59e0b"  # Amber
            warning_desc = (
                f"⚡ BORDERLINE CLASSIFICATION: Model output shows high predictive entropy ({entropy:.2f} > {entropy_thresh:.2f}) "
                f"and narrow probability separation (margin: {margin*100:.1f}% < {margin_thresh*100:.1f}%). "
                "The neural network cannot decisively distinguish between candidate pathologies."
            )
            action_advice = "Multidisciplinary radiologist review and supplementary diagnostic testing recommended."

        else:
            reliability_level = "HIGH"
            status_badge = "RELIABLE PREDICTION (In-Distribution)"
            status_color = "#10b981"  # Emerald Green
            warning_desc = (
                "✅ High classification confidence with low entropy and strong in-distribution feature alignment. "
                "Output is mathematically consistent with validation cohort behavior."
            )
            action_advice = "AI-assisted clinical interpretation with standard physician oversight."

        is_unreliable = is_ood or is_ambiguous

        metrics_dict = {
            "energy_score": round(energy, 3),
            "energy_threshold": energy_thresh,
            "predictive_entropy": round(entropy, 3),
            "entropy_threshold": entropy_thresh,
            "top2_margin": round(margin, 3),
            "margin_threshold": margin_thresh,
            "color_dispersion": round(color_disp, 2),
            "color_threshold": color_thresh,
            "pixel_std": round(img_std, 1),
            "is_valid_scan": is_valid_scan,
            "rejection_reason": rejection_reason,
        }

        return {
            "is_out_of_distribution": is_ood,
            "is_non_radiograph": bool(is_color_photo or not is_valid_scan),
            "is_valid_scan": is_valid_scan,
            "rejection_reason": rejection_reason,
            "scan_authenticity": auth,
            "is_ambiguous": is_ambiguous,
            "is_unreliable": is_unreliable,
            "reliability_level": reliability_level,
            "status_badge": status_badge,
            "status_color": status_color,
            "clinical_warning": warning_desc,
            "action_advice": action_advice,
            "metrics": metrics_dict,
            "detailed_metrics": metrics_dict,
        }


def assess_prediction_uncertainty(
    image_input: Union[str, Path, Image.Image],
    model: Optional[torch.nn.Module] = None,
    model_type: str = "augmented",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper to run forward pass and uncertainty evaluation.
    """
    config = load_config()
    device = device or get_device(config)

    image = robust_image_loader(image_input)
    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])
    tensor = preprocess_for_inference(image, img_size, mean, std).to(device)

    if model is None:
        model = load_trained_model(model_type=model_type, device=device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    detector = UncertaintyDetector()
    return detector.evaluate(image_input=image, logits=logits, probs=probs)
