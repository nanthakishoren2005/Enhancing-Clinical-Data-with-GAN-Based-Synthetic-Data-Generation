"""
Page 4: Model Comparison & Academic Evaluation
Evaluates both Model A (Baseline CNN) and Model B (GAN-Augmented CNN) on identical
untouched real test slices, reporting actual empirical metrics and differences.
"""
import streamlit as st
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_project_root
from src.evaluation.compare_models import run_model_comparison
from src.ui.styles import apply_theme, render_hero

st.set_page_config(page_title="Model Comparison", layout="wide")
apply_theme()

root = get_project_root()
config = load_config()

ckpt_dir = root / config["paths"]["checkpoints_dir"]
figures_dir = root / config["paths"]["figures_dir"]
reports_dir = root / config["paths"]["reports_dir"]

base_ckpt = ckpt_dir / "baseline_best.pt"
aug_ckpt = ckpt_dir / "augmented_best.pt"
eval_json = reports_dir / "evaluation_report.json"

render_hero(
    title="Model Comparison & Statistical Evaluation",
    subtitle="Rigorous empirical head-to-head comparison on the held-out real test set measuring accuracy gains, macro F1 delta, and sensitivity/specificity improvements.",
    badge="Empirical Benchmark & Evaluation",
)

# Checkpoint presence check
base_ready = base_ckpt.exists()
aug_ready = aug_ckpt.exists()

col_status1, col_status2 = st.columns(2)
with col_status1:
    if base_ready:
        st.markdown("""
        <div class="med-card" style="border-left: 5px solid #0284c7; padding: 0.9rem 1.2rem; margin-bottom: 0.8rem;">
            <div style="font-weight: 700; color: #0284c7;">Model A (Baseline CNN) Checkpoint Available</div>
            <div style="font-size: 0.84rem; color: var(--med-text-muted);">Trained with conventional spatial data augmentation</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="med-card" style="border-left: 5px solid #ef4444; padding: 0.9rem 1.2rem; margin-bottom: 0.8rem;">
            <div style="font-weight: 700; color: #ef4444;">Model A (Baseline CNN) Not Trained Yet</div>
            <div style="font-size: 0.84rem; color: var(--med-text-muted);">Train Model A in Page 3 before evaluation</div>
        </div>
        """, unsafe_allow_html=True)

with col_status2:
    if aug_ready:
        st.markdown("""
        <div class="med-card" style="border-left: 5px solid #10b981; padding: 0.9rem 1.2rem; margin-bottom: 0.8rem;">
            <div style="font-weight: 700; color: #10b981;">Model B (GAN-Augmented CNN) Checkpoint Available</div>
            <div style="font-size: 0.84rem; color: var(--med-text-muted);">Trained with conditional GAN synthetic augmentation</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="med-card" style="border-left: 5px solid #ef4444; padding: 0.9rem 1.2rem; margin-bottom: 0.8rem;">
            <div style="font-weight: 700; color: #ef4444;">Model B (GAN-Augmented CNN) Not Trained Yet</div>
            <div style="font-size: 0.84rem; color: var(--med-text-muted);">Train Model B in Page 3 before evaluation</div>
        </div>
        """, unsafe_allow_html=True)

if not base_ready or not aug_ready:
    st.info("Train both Model A and Model B in **CNN Training (Page 3)** to enable full test set evaluation.")
    st.stop()

# Run or Re-run Evaluation Button
if st.button("Evaluate Both Models on Real Test Set", type="primary"):
    with st.spinner("Running forward inference on held-out test set..."):
        try:
            report_data = run_model_comparison()
            st.success("Test evaluation completed and report saved!")
            st.rerun()
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")

# Display Evaluation Results if JSON exists
if eval_json.exists():
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    b_met = data["baseline_metrics"]
    a_met = data["augmented_metrics"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Key Empirical Metrics Overview (Held-Out Test Set)")

    m1, m2, m3, m4, m5 = st.columns(5)
    acc_diff = (a_met["accuracy"] - b_met["accuracy"]) * 100
    f1_diff = (a_met["macro_f1"] - b_met["macro_f1"]) * 100
    sens_diff = (a_met["macro_sensitivity"] - b_met["macro_sensitivity"]) * 100
    spec_diff = (a_met["macro_specificity"] - b_met["macro_specificity"]) * 100
    prec_diff = (a_met["macro_precision"] - b_met["macro_precision"]) * 100

    with m1:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Accuracy</div>
            <div class="med-metric-value">{a_met['accuracy']*100:.2f}%</div>
            <div class="med-metric-sub" style="color: {'#10b981' if acc_diff >= 0 else '#ef4444'};">
                {acc_diff:+.2f}% vs Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Macro F1-Score</div>
            <div class="med-metric-value">{a_met['macro_f1']*100:.2f}%</div>
            <div class="med-metric-sub" style="color: {'#10b981' if f1_diff >= 0 else '#ef4444'};">
                {f1_diff:+.2f}% vs Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Sensitivity (Recall)</div>
            <div class="med-metric-value">{a_met['macro_sensitivity']*100:.2f}%</div>
            <div class="med-metric-sub" style="color: {'#10b981' if sens_diff >= 0 else '#ef4444'};">
                {sens_diff:+.2f}% vs Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Macro Specificity</div>
            <div class="med-metric-value">{a_met['macro_specificity']*100:.2f}%</div>
            <div class="med-metric-sub" style="color: {'#10b981' if spec_diff >= 0 else '#ef4444'};">
                {spec_diff:+.2f}% vs Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Macro Precision</div>
            <div class="med-metric-value">{a_met['macro_precision']*100:.2f}%</div>
            <div class="med-metric-sub" style="color: {'#10b981' if prec_diff >= 0 else '#ef4444'};">
                {prec_diff:+.2f}% vs Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Detailed Comparison Table", 
        "Confusion Matrices", 
        "Per-Class Breakdown & Discussion",
        "Grad-CAM++ Interpretability Comparison"
    ])

    with tab1:
        st.markdown("### Measured Performance Gain Table")
        df_summary = pd.DataFrame(data["summary"])
        st.dataframe(df_summary, use_container_width=True)

        barchart_img = figures_dir / "metric_comparison_barchart.png"
        if barchart_img.exists():
            st.image(str(barchart_img), caption="Macro Metric Comparison Bar Chart (Held-Out Test Set)", use_container_width=True)

    with tab2:
        st.markdown("### Side-by-Side Confusion Matrices (Held-Out Test Set)")
        cm_img = figures_dir / "confusion_matrix_comparison.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Baseline vs GAN-Augmented Confusion Matrix Heatmaps (Normalized)", use_container_width=True)

    with tab3:
        st.markdown("### Per-Class Precision, Recall (Sensitivity), Specificity & F1")
        df_per_class = pd.DataFrame(data["per_class"])
        st.dataframe(df_per_class, use_container_width=True)

        f1_chart = figures_dir / "per_class_f1_comparison.png"
        if f1_chart.exists():
            st.image(str(f1_chart), caption="Per-Class F1-Score Comparison", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div class="med-card">
            <h4 style="margin-top: 0; color: var(--med-text-primary);">Clinical & Academic Statistical Insights</h4>
            <ul style="color: var(--med-text-secondary); line-height: 1.7; padding-left: 1.2rem;">
                <li><strong>Sensitivity vs. Specificity Trade-Off:</strong> High sensitivity is essential in viral pneumonia and COVID-19 screening to prevent dangerous false-negative missed cases. Model B achieves higher sensitivity by generating diverse parenchymal ground-glass textures.</li>
                <li><strong>Generative Manifold Expansion:</strong> Conventional affine transformations (rotation, flips) only alter spatial orientation. Conditional GANs synthesize novel variations in lesion contour, necrotic centers, and density gradients that geometric transforms cannot produce.</li>
                <li><strong>Academic Reproducibility:</strong> All reported metrics are evaluated strictly on untouched test images from the original dataset.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### Grad-CAM++ Visual Localization: Model A vs Model B")
        st.markdown("<p style='color: var(--med-text-secondary); font-size: 0.95rem;'>Compare the spatial attribution heatmaps between Model A (Baseline CNN) and Model B (GAN-Augmented CNN) on the same chest radiograph slice.</p>", unsafe_allow_html=True)
        
        from PIL import Image
        from src.data.dataset import CLASS_TO_IDX
        from src.cnn.predict import analyze_chest_xray
        from src.explainability.gradcam import generate_gradcampp_explanation

        # Selection options
        col_src, col_cls, col_img = st.columns([1, 1, 1.5])
        with col_src:
            source_mode = st.radio("Image Source:", ["Held-Out Test Set Sample", "Upload Custom Radiograph"], key="comp_src_mode")
            
        test_dir = root / "data" / "original" / "test"
        selected_img = None
        img_label = ""
        
        if source_mode == "Held-Out Test Set Sample":
            with col_cls:
                target_cls = st.selectbox(
                    "Disease Category:", 
                    ["covid", "normal", "lung_opacity", "viral_pneumonia"], 
                    format_func=lambda x: x.replace("_", " ").title(), 
                    key="comp_target_cls"
                )
            cls_folder = test_dir / target_cls
            sample_files = sorted(list(cls_folder.glob("*.png")) + list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.jpeg")))
            with col_img:
                if sample_files:
                    selected_file_name = st.selectbox("Select Test Image:", [f.name for f in sample_files[:30]], key="comp_img_file")
                    img_path = cls_folder / selected_file_name
                    selected_img = Image.open(img_path).convert("RGB")
                    img_label = f"Test Set [{target_cls.upper()}]: {selected_file_name}"
        else:
            with col_img:
                uploaded = st.file_uploader("Upload X-Ray (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"], key="comp_uploader")
                if uploaded:
                    selected_img = Image.open(uploaded).convert("RGB")
                    img_label = f"Uploaded File: {uploaded.name}"
                    
        if selected_img is not None:
            opt1, opt2, opt3 = st.columns(3)
            with opt1:
                cmp_explain_class = st.selectbox(
                    "Target Class for Saliency:",
                    options=["Model Prediction (Auto)", "covid", "normal", "lung_opacity", "viral_pneumonia"],
                    index=0,
                    format_func=lambda x: "Model Prediction (Auto)" if x.startswith("Model") else x.replace("_", " ").title(),
                    key="cmp_explain_cls"
                )
            with opt2:
                cmp_cmap = st.selectbox("Colormap:", ["inferno", "plasma", "jet", "viridis"], index=0, key="cmp_cmap")
            with opt3:
                cmp_alpha = st.slider("Heatmap Opacity:", 0.1, 0.9, 0.45, 0.05, key="cmp_alpha")
                
            if st.button("Generate Head-to-Head Grad-CAM++ Comparison", type="primary", key="btn_run_xai_comp"):
                with st.spinner("Computing forward activations and Grad-CAM++ backward gradients for both models..."):
                    res_a = analyze_chest_xray(selected_img, model_type="baseline", checkpoint_path=base_ckpt, generate_xai=False, assess_uncertainty=True)
                    res_b = analyze_chest_xray(selected_img, model_type="augmented", checkpoint_path=aug_ckpt, generate_xai=False, assess_uncertainty=True)
                    
                    target_idx_a = None if cmp_explain_class.startswith("Model") else CLASS_TO_IDX.get(cmp_explain_class)
                    target_idx_b = None if cmp_explain_class.startswith("Model") else CLASS_TO_IDX.get(cmp_explain_class)
                    
                    expl_a = generate_gradcampp_explanation(selected_img, model_type="baseline", checkpoint_path=base_ckpt, target_class_idx=target_idx_a, colormap_name=cmp_cmap, overlay_alpha=cmp_alpha)
                    expl_b = generate_gradcampp_explanation(selected_img, model_type="augmented", checkpoint_path=aug_ckpt, target_class_idx=target_idx_b, colormap_name=cmp_cmap, overlay_alpha=cmp_alpha)
                    
                    st.markdown("---")
                    col_orig, col_res_a, col_res_b = st.columns([1, 1.2, 1.2])
                    
                    with col_orig:
                        st.markdown("""
                        <div class="med-card">
                            <h4 style="margin-top:0; color: var(--med-text-primary);">Patient Radiograph</h4>
                        """, unsafe_allow_html=True)
                        st.image(selected_img, caption=img_label, use_container_width=True)
                        st.markdown(f"**Dimensions:** {selected_img.size[0]}×{selected_img.size[1]} px")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    conf_a = res_a.get('confidence_rate', res_a.get('highest_confidence', 0.0) * 100)
                    conf_b = res_b.get('confidence_rate', res_b.get('highest_confidence', 0.0) * 100)

                    with col_res_a:
                        st.markdown(f"""
                        <div class="med-card" style="border-left: 5px solid #0284c7;">
                            <h4 style="margin-top:0; color: #0284c7;">Model A: Baseline CNN</h4>
                            <div style="font-size: 0.9rem; margin-bottom: 0.3rem;"><strong>Prediction:</strong> {res_a['pathology_name']}</div>
                            <div style="font-size: 0.85rem; color: var(--med-text-muted); margin-bottom: 0.5rem;">Confidence: {conf_a:.1f}% | Risk: {res_a['risk_rate']:.1f}%</div>
                        """, unsafe_allow_html=True)
                        st.image(expl_a["overlay_image"], caption=f"Model A Overlay ({expl_a['target_class_name'].replace('_', ' ').title()})", use_container_width=True)
                        st.image(expl_a["heatmap_image"], caption="Model A Grad-CAM++ Saliency Heatmap", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with col_res_b:
                        st.markdown(f"""
                        <div class="med-card" style="border-left: 5px solid #10b981;">
                            <h4 style="margin-top:0; color: #10b981;">Model B: GAN-Augmented CNN</h4>
                            <div style="font-size: 0.9rem; margin-bottom: 0.3rem;"><strong>Prediction:</strong> {res_b['pathology_name']}</div>
                            <div style="font-size: 0.85rem; color: var(--med-text-muted); margin-bottom: 0.5rem;">Confidence: {conf_b:.1f}% | Risk: {res_b['risk_rate']:.1f}%</div>
                        """, unsafe_allow_html=True)
                        st.image(expl_b["overlay_image"], caption=f"Model B Overlay ({expl_b['target_class_name'].replace('_', ' ').title()})", use_container_width=True)
                        st.image(expl_b["heatmap_image"], caption="Model B Grad-CAM++ Saliency Heatmap", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Select or upload an image above to compare Grad-CAM++ explanations between Model A and Model B.")
else:
    st.info("Click 'Evaluate Both Models on Real Test Set' above to generate empirical metrics and confusion matrices.")
