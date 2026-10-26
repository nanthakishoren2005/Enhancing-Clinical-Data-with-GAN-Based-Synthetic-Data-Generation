"""
Page 7: Compare: Baseline model vs cDCGAN 256x256
Unified page: metric comparison, dual prediction with Grad-CAM++, explainable AI,
and generated samples side by side.
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import io
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns

from src.utils.config import load_config, get_project_root, get_device
from src.utils.disease_knowledge import get_disease_details
from src.utils.checkpoint import load_checkpoint
from src.cnn.predict import (
    analyze_liver_scan, load_trained_model, preprocess_for_inference,
    robust_image_loader,
)
from src.cnn.model import build_classifier
from src.explainability.gradcam import (
    GradCAMPlusPlus, generate_gradcampp_explanation,
    colorize_heatmap, blend_heatmap_overlay,
)
from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, get_transforms
from src.uncertainty.detector import UncertaintyDetector
from src.ui.styles import apply_theme

st.set_page_config(page_title="Compare: Baseline model vs cDCGAN 256x256", layout="wide")
apply_theme()

root = get_project_root()
config = load_config()
device = get_device(config)

# ── Paths ──────────────────────────────────────────────────────────────
ckpt_dir = root / config["paths"]["checkpoints_dir"]
base_ckpt = ckpt_dir / "baseline_best.pt"          # Baseline model checkpoint
ckpt_256 = root / "outputs" / "v2" / "res256" / "cnn256_best.pt"   # 256 model
sample_128 = root / "outputs" / "gan_samples" / "cgan_epoch_050.png"
sample_256 = root / "outputs" / "v2" / "res256" / "cgan256_epoch_050.png"

img_size_128 = config["dataset"]["image_size"]     # 128
img_size_256 = 256
mean = tuple(config["dataset"]["mean"])
std = tuple(config["dataset"]["std"])

# ── Verify required files exist ────────────────────────────────────────
missing = []
if not base_ckpt.exists():
    missing.append(str(base_ckpt))
if not ckpt_256.exists():
    missing.append(str(ckpt_256))
if not sample_256.exists():
    missing.append(str(sample_256))

try:
    from src.utils.disease_knowledge import get_disease_details as _gdd
    _ = _gdd("normal")
except Exception as e:
    missing.append(f"disease_knowledge module: {e}")

if missing:
    st.error("Required files missing — cannot load page:")
    for m in missing:
        st.markdown(f"- {m}")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="pred-title">Compare: Baseline model vs cDCGAN 256x256</div>', unsafe_allow_html=True)
st.markdown('<div class="pred-subtitle">Side-by-side evaluation of the Baseline model and the cDCGAN 256x256 classifier on the same chest X-ray upload, with Grad-CAM++ explainability and disease-level care information.</div>', unsafe_allow_html=True)

# Academic disclaimer
st.markdown("""
<div style="background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 0.9rem 1.2rem; border-radius: 4px; margin-bottom: 1.5rem; font-size: 0.88rem; color: #7f1d1d;">
    <strong>Warning MANDATORY CLINICAL & RESEARCH DISCLAIMER:</strong><br>
    This software is an <em>academic experimental prototype</em> evaluating deep CNN classifiers with GAN-augmented medical data.
    The computed classifications, Grad-CAM++ heatmaps, and risk estimates are AI-assisted algorithmic outputs and
    <strong>DO NOT</strong> replace certified radiological readings or clinical physician diagnosis.
</div>
""", unsafe_allow_html=True)

# ── Model info cards ───────────────────────────────────────────────────
st.markdown("### Model Identification")

c_left, c_right = st.columns(2)
with c_left:
    st.markdown(f"""
    <div class="med-card" style="border-left: 5px solid #0284c7; padding: 1rem;">
        <div style="font-weight: 700; color: #0284c7;">Baseline model</div>
        <div style="font-size: 0.85rem; color: var(--med-text-muted);">Checkpoint: baseline_best.pt</div>
        <div style="font-size: 0.85rem; color: var(--med-text-muted);">Input: Conventional augmentation (real images only)</div>
    </div>
    """, unsafe_allow_html=True)

with c_right:
    st.markdown(f"""
    <div class="med-card" style="border-left: 5px solid #10b981; padding: 1rem;">
        <div style="font-weight: 700; color: #10b981;">cDCGAN 256x256</div>
        <div style="font-size: 0.85rem; color: var(--med-text-muted);">Checkpoint: outputs/v2/res256/cnn256_best.pt</div>
        <div style="font-size: 0.85rem; color: var(--med-text-muted);">Input: 256x256 px | GAN synthetic augmentation (100/class)</div>
    </div>
    """, unsafe_allow_html=True)

# ── SECTION 1: Comparison about the disease ────────────────────────────
st.markdown("## About the Disease: Test-Set Performance Comparison")

# Baseline verified metrics
b_acc = 0.9275
b_f1 = 0.9277
b_sens = 0.9275
b_spec = 0.9758
b_prec = 0.9326

# cDCGAN 256x256 verified metrics
p1_acc = 0.9400
p1_f1 = 0.9405
p1_sens = 0.9400
p1_spec = 0.9800
p1_prec = 0.9417

p1_delta_acc = (p1_acc - b_acc) * 100
p1_delta_f1 = (p1_f1 - b_f1) * 100
p1_delta_sens = (p1_sens - b_sens) * 100
p1_delta_spec = (p1_spec - b_spec) * 100
p1_delta_prec = (p1_prec - b_prec) * 100

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Accuracy</div>
        <div class="med-metric-value">{p1_acc*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">{p1_delta_acc:+.2f}pp vs Baseline model ({b_acc*100:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Macro F1-Score</div>
        <div class="med-metric-value">{p1_f1:.4f}</div>
        <div class="med-metric-sub" style="color: #10b981;">{p1_delta_f1:+.4f} vs Baseline model ({b_f1:.4f})</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Sensitivity (Recall)</div>
        <div class="med-metric-value">{p1_sens*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">{p1_delta_sens:+.2f}pp vs Baseline model ({b_sens*100:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Macro Specificity</div>
        <div class="med-metric-value">{p1_spec*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">{p1_delta_spec:+.2f}pp vs Baseline model ({b_spec*100:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)
with m5:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Macro Precision</div>
        <div class="med-metric-value">{p1_prec*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">{p1_delta_prec:+.2f}pp vs Baseline model ({b_prec*100:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Per-class table
st.markdown("### Per-Class Metrics: Baseline model vs cDCGAN 256x256")

class_label_map = {
    "normal": "Normal Lung",
    "covid": "COVID-19",
    "lung_opacity": "Lung Opacity",
    "viral_pneumonia": "Viral Pneumonia",
}

# Baseline per-class from evaluation_report.json
b_per_class = {
    "normal": {"precision": 0.8276, "recall": 0.96, "f1_score": 0.8889, "specificity": 0.9333, "support": 100},
    "covid": {"precision": 0.9688, "recall": 0.93, "f1_score": 0.9490, "specificity": 0.9900, "support": 100},
    "lung_opacity": {"precision": 0.9540, "recall": 0.83, "f1_score": 0.8877, "specificity": 0.9867, "support": 100},
    "viral_pneumonia": {"precision": 0.9802, "recall": 0.99, "f1_score": 0.9851, "specificity": 0.9933, "support": 100},
}

# cDCGAN 256x256 per-class from phase1_summary.json
p1_per_class = {
    "normal": {"precision": 0.8857, "recall": 0.9300, "f1_score": 0.9073, "specificity": 0.9600, "support": 100},
    "covid": {"precision": 1.0000, "recall": 0.9400, "f1_score": 0.9691, "specificity": 1.0000, "support": 100},
    "lung_opacity": {"precision": 0.8911, "recall": 0.9000, "f1_score": 0.8955, "specificity": 0.9633, "support": 100},
    "viral_pneumonia": {"precision": 0.9900, "recall": 0.9900, "f1_score": 0.9900, "specificity": 0.9967, "support": 100},
}

rows = []
for cls in ["normal", "covid", "lung_opacity", "viral_pneumonia"]:
    b = b_per_class[cls]
    p = p1_per_class[cls]
    rows.append({
        "Class": class_label_map[cls],
        "Support": p["support"],
        "Baseline model Precision": f"{b['precision']:.4f}",
        "cDCGAN 256x256 Precision": f"{p['precision']:.4f}",
        "Baseline model Recall": f"{b['recall']:.4f}",
        "cDCGAN 256x256 Recall": f"{p['recall']:.4f}",
        "Baseline model F1": f"{b['f1_score']:.4f}",
        "cDCGAN 256x256 F1": f"{p['f1_score']:.4f}",
        "Baseline model Specificity": f"{b['specificity']:.4f}",
        "cDCGAN 256x256 Specificity": f"{p['specificity']:.4f}",
    })
df_pc = pd.DataFrame(rows)
st.dataframe(df_pc, width="stretch")

# Confusion matrices
st.markdown("### Confusion Matrices (Held-Out Test Set, 400 images)")
cm_labels = ["Normal", "COVID-19", "Lung Opacity", "Viral Pneumonia"]

cm_b = np.array([[96, 2, 1, 1], [3, 93, 3, 1], [16, 1, 83, 0], [1, 0, 0, 99]])
cm_p1 = np.array([[93, 0, 7, 0], [1, 94, 4, 1], [10, 3, 87, 0], [1, 0, 0, 99]])

fig_cm, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, cm, title, cmap in [
    (axes[0], cm_b, "Baseline model", "Blues"),
    (axes[1], cm_p1, "cDCGAN 256x256", "Greens"),
]:
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)"
    sns.heatmap(
        cm, annot=annot, fmt="", cmap=cmap,
        xticklabels=cm_labels, yticklabels=cm_labels,
        cbar=False, ax=ax, linewidths=1, linecolor="#e0e0e0",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)

plt.suptitle("Confusion Matrices: Baseline model vs cDCGAN 256x256 (Held-Out Test Set)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
st.pyplot(fig_cm, width="stretch")
plt.close(fig_cm)

# Key finding note
st.markdown("""
<div class="med-card" style="border-left: 5px solid #f59e0b; padding: 0.9rem 1.2rem; margin: 1rem 0;">
    <strong>Key Finding:</strong> Lung Opacity remains the weakest class (cDCGAN 256x256 F1 0.8955); Normal vs Lung Opacity is the main error pair
    (17 of the 24 cDCGAN 256x256 errors).
</div>
""", unsafe_allow_html=True)

# ── SECTION 2: Upload and predict ──────────────────────────────────────
st.markdown("---")
st.markdown("## Upload & Predict: Both Models on the Same X-Ray")

uploaded = st.file_uploader("Select a chest X-ray image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"], key="compare_upload")

# Load models lazily
@st.cache_resource
def load_baseline_model():
    return load_trained_model(model_type="baseline", checkpoint_path=base_ckpt, device=device)

@st.cache_resource
def load_cdcgan256_model():
    m = build_classifier(architecture="resnet18", num_classes=4, pretrained=False).to(device)
    ckpt = load_checkpoint(ckpt_256, device=device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

model_baseline = load_baseline_model()
model_256 = load_cdcgan256_model()

if uploaded is not None:
    try:
        img_pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.stop()

    st.image(img_pil, caption="Uploaded Chest X-Ray", width="stretch")

    # ── Run both models ──────────────────────────────────────────
    t0 = time.perf_counter()

    # Baseline model via analyze_liver_scan (same as Prediction page)
    try:
        analysis_128 = analyze_liver_scan(
            image_input=img_pil,
            model_type="baseline",
            checkpoint_path=base_ckpt,
            generate_xai=True,
            assess_uncertainty=True,
        )
    except Exception as e:
        st.error(f"Baseline model inference failed: {str(e)}")
        analysis_128 = None

    # cDCGAN 256x256 model — manual forward pass
    analysis_256 = None
    expl_256 = None
    try:
        tensor_256 = preprocess_for_inference(
            img_pil, image_size=img_size_256, mean=mean, std=std
        ).to(device)

        with torch.no_grad():
            logits_256 = model_256(tensor_256)
            probs_256 = F.softmax(logits_256, dim=1).squeeze(0).cpu().numpy()

        pred_idx_256 = int(probs_256.argmax())
        pred_class_256 = IDX_TO_CLASS[pred_idx_256]
        pred_name_256 = class_label_map[pred_class_256]
        conf_256 = float(probs_256[pred_idx_256]) * 100

        # Grad-CAM++ for cDCGAN 256x256 model
        explainer_256 = GradCAMPlusPlus(model=model_256, device=device)
        heatmap_256, class_idx_256, score_256 = explainer_256.generate_heatmap(
            tensor_256, target_class_idx=pred_idx_256
        )
        heatmap_colored_256 = colorize_heatmap(
            heatmap_256, target_size=img_pil.size, colormap_name="inferno"
        )
        expl_256 = {
            "overlay_image": blend_heatmap_overlay(img_pil, heatmap_colored_256, alpha=0.45),
            "heatmap_image": heatmap_colored_256,
            "target_class_name": pred_class_256,
            "score_value": score_256,
        }

        analysis_256 = {
            "predicted_class": pred_class_256,
            "pathology_name": pred_name_256,
            "confidence_scores": {c: float(probs_256[i]) for i, c in IDX_TO_CLASS.items()},
            "highest_confidence": float(probs_256[pred_idx_256]),
        }
    except Exception as e:
        st.error(f"cDCGAN 256x256 inference failed: {str(e)}")
        analysis_256 = None

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    st.markdown(f"**Inference latency:** {latency_ms:.1f} ms")

    # ── Display results side by side ──────────────────────────────
    col_m1, col_m2 = st.columns(2)

    # Baseline model result
    with col_m1:
        st.markdown("### Baseline model prediction")
        if analysis_128 and analysis_128.get("disease_status") != "INVALID_SCAN":
            st.markdown(f"""
            <div style="border-left: 5px solid #0284c7; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                <div style="font-weight: 700; font-size: 1.2rem; color: #0f172a;">{analysis_128['pathology_name']}</div>
                <div style="font-size: 0.9rem; color: var(--med-text-muted);">{analysis_128.get('status_badge', '')}</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0284c7; margin-top: 0.4rem;">Confidence: {analysis_128.get('highest_confidence', 0)*100:.1f}%</div>
                <div style="font-size: 0.85rem; color: var(--med-text-muted); margin-top: 0.3rem;">Risk: {analysis_128.get('risk_rate', 0):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Softmax Probabilities:**")
            conf_128 = analysis_128.get("confidence_scores", {})
            for cls in ["normal", "covid", "lung_opacity", "viral_pneumonia"]:
                val = conf_128.get(cls, 0.0) * 100
                st.markdown(f"- {class_label_map[cls]}: {val:.1f}%")

            expl_128 = analysis_128.get("gradcam")
            if expl_128 and "overlay_image" in expl_128:
                st.markdown("**Grad-CAM++ Heatmap:**")
                st.image(expl_128["overlay_image"], caption=f"Baseline model — {analysis_128['pathology_name']}", width="stretch")
                st.image(expl_128["heatmap_image"], caption="Grad-CAM++ Saliency (inferno)", width="stretch")

            unc_128 = analysis_128.get("uncertainty", {})
            if unc_128:
                st.markdown("**Uncertainty (Validation-Calibrated):**")
                st.markdown(f"- Reliability: {unc_128.get('reliability_level', 'N/A')}")
                st.markdown(f"- Status: {unc_128.get('status_badge', 'N/A')}")
                st.markdown(f"- Predictive Entropy: {unc_128.get('detailed_metrics', {}).get('predictive_entropy', 'N/A')}")
                st.markdown(f"- **Entropy threshold: 0.531** (reliable/unreliable boundary)")
        else:
            st.error("Baseline model returned an invalid scan result.")

    # cDCGAN 256x256 result
    with col_m2:
        st.markdown("### cDCGAN 256x256 prediction")
        if analysis_256:
            st.markdown(f"""
            <div style="border-left: 5px solid #10b981; padding: 1rem; background: #f0fdf4; border-radius: 8px;">
                <div style="font-weight: 700; font-size: 1.2rem; color: #0f172a;">{analysis_256['pathology_name']}</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #10b981; margin-top: 0.4rem;">Confidence: {conf_256:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Softmax Probabilities:**")
            conf_256_dict = analysis_256.get("confidence_scores", {})
            for cls in ["normal", "covid", "lung_opacity", "viral_pneumonia"]:
                val = conf_256_dict.get(cls, 0.0) * 100
                st.markdown(f"- {class_label_map[cls]}: {val:.1f}%")

            if expl_256:
                st.markdown("**Grad-CAM++ Heatmap:**")
                st.image(expl_256["overlay_image"], caption=f"cDCGAN 256x256 — {analysis_256['pathology_name']}", width="stretch")
                st.image(expl_256["heatmap_image"], caption="Grad-CAM++ Saliency (inferno)", width="stretch")

            st.markdown("**Uncertainty (Reference Only):**")
            ent_256 = -float(np.sum(probs_256 * np.log(probs_256 + 1e-12)) / np.log(len(probs_256)))
            st.markdown(f"- Predictive Entropy: {ent_256:.3f}")
            st.markdown(f"- **Label: Reference only — threshold not recalibrated for cDCGAN 256x256** (0.531 not applied)")
        else:
            st.error("cDCGAN 256x256 inference failed.")

    # Disagreement notice
    if analysis_128 and analysis_256 and analysis_128["predicted_class"] != analysis_256["predicted_class"]:
        st.markdown("""
        <div style="background: #fef2f2; border-left: 5px solid #ef4444; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>Warning: Model Disagreement</strong><br>
            The two models produced different predictions on this image. This is shown honestly — neither result is suppressed.
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 3: Explainable AI about the disease ──────────────────
    st.markdown("---")
    st.markdown("## Explainable AI About the Disease")

    # Where: Grad-CAM++ heatmaps
    st.markdown("### Where the Model Looked (Grad-CAM++ Heatmaps)")
    st.markdown("The Grad-CAM++ heatmaps shown above for each model indicate the spatial pulmonary zones that most influenced each model's decision. Darker/redder regions indicate higher neural activation driving the prediction.")

    # How sure: uncertainty blocks
    st.markdown("### How Sure: Prediction Uncertainty")

    uc1, uc2 = st.columns(2)
    with uc1:
        st.markdown("**Baseline model — Validation-Calibrated:**")
        if analysis_128 and analysis_128.get("uncertainty"):
            unc = analysis_128["uncertainty"]
            dm = unc.get("detailed_metrics", {})
            st.markdown(f"- Reliability: **{unc.get('reliability_level', 'N/A')}**")
            st.markdown(f"- Status: {unc.get('status_badge', 'N/A')}")
            st.markdown(f"- Predictive Entropy: {dm.get('predictive_entropy', 'N/A')}")
            st.markdown(f"- Threshold: **0.531** (reliable / unreliable boundary)")
            st.markdown(f"- Clinical Warning: {unc.get('clinical_warning', 'N/A')}")
        else:
            st.info("Uncertainty data not available for Baseline model.")

    with uc2:
        st.markdown("**cDCGAN 256x256 — Reference Only:**")
        if analysis_256:
            st.markdown(f"- Predictive Entropy: {ent_256:.3f}")
            st.markdown(f"- **Label: Reference only — threshold not recalibrated for cDCGAN 256x256**")
            st.markdown(f"- **0.531 NOT applied** to cDCGAN 256x256 predictions.")
        else:
            st.info("Uncertainty data not available for cDCGAN 256x256.")

    # What it means: Disease Details panel(s)
    st.markdown("### What It Means: Disease Details & Care Information")

    # Determine which disease(s) to show
    disease_keys = []
    if analysis_128 and analysis_128.get("predicted_class"):
        disease_keys.append(("Baseline model", analysis_128["predicted_class"]))
    if analysis_256 and analysis_256.get("predicted_class"):
        disease_keys.append(("cDCGAN 256x256", analysis_256["predicted_class"]))

    # Show one panel per unique disease, or one per model if they differ
    shown_diseases = set()
    for model_label, cls_key in disease_keys:
        if cls_key in shown_diseases:
            continue
        shown_diseases.add(cls_key)

        dd = get_disease_details(cls_key)
        if not dd:
            st.error(f"Disease knowledge not found for class '{cls_key}' — stopping.")
            st.stop()

        st.markdown(f"#### {dd.get('disease_name', cls_key)}")
        if len(disease_keys) > 1:
            st.markdown(f"*Predicted by: {model_label}*")

        st.markdown(f"**Overview:** {dd.get('overview', 'No overview available.')}")

        dd_col1, dd_col2 = st.columns(2)
        with dd_col1:
            st.markdown("**Key Symptoms**")
            for s in dd.get("key_symptoms", []):
                st.markdown(f"- {s}")
            st.markdown("**Causes & Risk Factors**")
            for c in dd.get("causes_risk_factors", []):
                st.markdown(f"- {c}")
        with dd_col2:
            st.markdown("**General Remedies & Treatment Options**")
            for rem in dd.get("remedies_treatment", []):
                st.markdown(f"- {rem}")
            st.markdown("**Prevention Tips**")
            for p in dd.get("prevention", []):
                st.markdown(f"- {p}")

        st.markdown("**When to Seek Urgent Medical Care (Red Flags):**")
        for u in dd.get("when_to_seek_urgent_care", []):
            st.markdown(f"- {u}")

        st.error(
            "**Important Medical Disclaimer:** "
            + dd.get("disclaimer", "This information is for educational purposes only and is not a substitute for professional medical advice.")
        )
        st.markdown("---")

# ── SECTION 4: Generated samples ───────────────────────────────────────
st.markdown("## Generated Samples")
st.markdown("The baseline is a classifier trained on real images only, so it has no generated samples of its own.")

sg1, sg2 = st.columns(2)
with sg1:
    st.markdown("**cDCGAN 128x128 Sample Grid (Epoch 50) — earlier generator, for reference**")
    if sample_128.exists():
        st.image(str(sample_128), caption="cDCGAN 128x128 Sample Grid (Epoch 50) — earlier generator, for reference", width="stretch")
    else:
        st.info(f"128 sample grid not found: {sample_128}")

with sg2:
    st.markdown("**cDCGAN 256x256 Sample Grid (Epoch 50)**")
    if sample_256.exists():
        st.image(str(sample_256), caption="cDCGAN 256x256 Sample Grid (Epoch 50)", width="stretch")
    else:
        st.info(f"256 sample grid not found: {sample_256}")

# ── SECTION 5: Honest note ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="med-card" style="border-left: 5px solid #f59e0b; padding: 0.9rem 1.2rem;">
    <strong>Phase 1 changed two things at once</strong> — input resolution (128 to 256) and synthetic augmentation (100 per class) — 
    so the gain cannot be credited to synthetic data alone. A 256 real-only control would be needed to separate the two effects.
</div>
""", unsafe_allow_html=True)

# ── SECTION 6: Future Work ─────────────────────────────────────────────
st.markdown("## Future Work")
st.markdown("""
**Planned next steps (future implementation):**
- StyleGAN2-ADA and conditional diffusion (DDPM) generators, to be compared on the same test set.
""")
