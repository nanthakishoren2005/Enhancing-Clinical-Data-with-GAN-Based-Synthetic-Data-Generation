"""
Main Streamlit Application - Home Dashboard
"GAN-Based Synthetic Medical Image Augmentation for Multiclass COVID-19 & Pulmonary Disease Classification"
"""
import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image

from src.utils.config import load_config, get_project_root
from src.utils.disease_knowledge import get_disease_details
from src.data.validate_dataset import validate_dataset
from src.cnn.predict import analyze_chest_xray
from src.ui.styles import apply_theme, render_hero, render_disclaimer

# Streamlit Page Setup
st.set_page_config(
    page_title="COVID-19 & Pulmonary Disease GAN-CNN",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Centralized Medical AI Design System
apply_theme()


def main():
    root = get_project_root()
    config = load_config()

    # Premium Hero Header
    render_hero(
        title="Pulmonary AI Diagnostics & Generative Augmentation",
        subtitle="Conditional DCGAN Synthetic Augmentation for Multiclass COVID-19, Normal, Pulmonary Opacity & Viral Pneumonia Deep CNN Classification with Calibrated Clinical Risk Assessment.",
        badge="Clinical Deep Learning Platform v2.4",
    )

    # Academic Disclaimer Alert
    render_disclaimer()

    # System Status & Pipeline Indicators
    st.markdown("### System Status & Model Pipelines")
    orig_dir = root / config["dataset"]["original_dir"]
    synth_dir = root / config["dataset"]["synthetic_dir"]
    ckpt_dir = root / config["paths"]["checkpoints_dir"]

    cgan_ready = (ckpt_dir / "cgan_checkpoint.pt").exists()
    base_ready = (ckpt_dir / "baseline_best.pt").exists()
    aug_ready = (ckpt_dir / "augmented_best.pt").exists()

    val_res = validate_dataset(orig_dir, synth_dir)
    total_real = val_res.get("total_images", 0)
    total_synth = val_res.get("synthetic_train", {}).get("total", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Real Scan Cohort</div>
            <div class="med-metric-value">{total_real}</div>
            <div class="med-metric-sub" style="color: {'#10b981' if total_real > 0 else '#ef4444'};">
                {'4 Classes Balanced' if total_real > 0 else 'Missing Files'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Conditional DCGAN</div>
            <div class="med-metric-value">{'Trained' if cgan_ready else 'Standby'}</div>
            <div class="med-metric-sub" style="color: {'#10b981' if cgan_ready else '#f59e0b'};">
                {f'{total_synth} Synthetics Active' if total_synth > 0 else 'Ready for Training'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">Baseline CNN (Model A)</div>
            <div class="med-metric-value">{'Ready' if base_ready else 'Untrained'}</div>
            <div class="med-metric-sub" style="color: {'#10b981' if base_ready else '#64748b'};">
                {'Checkpoint Available' if base_ready else 'Awaiting Training'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">GAN-Augmented (Model B)</div>
            <div class="med-metric-value">{'Ready' if aug_ready else 'Untrained'}</div>
            <div class="med-metric-sub" style="color: {'#10b981' if aug_ready else '#64748b'};">
                {'Optimal Checkpoint' if aug_ready else 'Awaiting Training'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture Overview & Workflow
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("""
        <div class="med-card">
            <h3 style="margin-top: 0; color: var(--med-text-primary);">Controlled Scientific Methodology</h3>
            <p style="color: var(--med-text-secondary); line-height: 1.6;">
                This experimental platform systematically measures the diagnostic impact of Generative Adversarial Network (GAN) augmentation on deep convolutional classifiers:
            </p>
            <ul style="color: var(--med-text-secondary); line-height: 1.7; padding-left: 1.2rem;">
                <li><strong>Model A (Baseline CNN):</strong> Trained solely on genuine patient radiographs with classical spatial transforms.</li>
                <li><strong>Model B (GAN-Augmented CNN):</strong> Augmented with novel, class-conditioned synthetic radiographs synthesized by our <strong>Conditional DCGAN</strong>.</li>
                <li><strong>Strict Leakage Prevention:</strong> Both models are rigorously validated against the exact same untouched held-out real test set (400 patient scans).</li>
            </ul>
            <div style="margin-top: 1rem; padding: 0.8rem; background: var(--med-bg-card-subtle); border-radius: 8px; border: 1px solid var(--med-border);">
                <div style="font-weight: 700; font-size: 0.85rem; color: var(--med-text-primary); margin-bottom: 0.4rem;">TARGET PULMONARY PATHOLOGIES:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <span class="med-chip" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);">Normal Lung</span>
                    <span class="med-chip" style="background: rgba(139, 92, 246, 0.15); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.3);">COVID-19</span>
                    <span class="med-chip" style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3);">Pulmonary Opacity</span>
                    <span class="med-chip" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);">Viral Pneumonia</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="med-card">
            <h3 style="margin-top: 0; color: var(--med-text-primary);">Guided System Walkthrough</h3>
            <p style="color: var(--med-text-secondary); line-height: 1.6; margin-bottom: 0.8rem;">
                Follow the clinical evaluation workflow step-by-step:
            </p>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="background: var(--med-primary-light); color: var(--med-primary); font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: 0.85rem;">1</span>
                    <span style="color: var(--med-text-secondary); font-size: 0.92rem;"><strong>Dataset Explorer:</strong> Inspect balances and patient partitions.</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="background: var(--med-primary-light); color: var(--med-primary); font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: 0.85rem;">2</span>
                    <span style="color: var(--med-text-secondary); font-size: 0.92rem;"><strong>GAN Generation:</strong> Synthesize novel class-conditioned chest X-ray patches.</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="background: var(--med-primary-light); color: var(--med-primary); font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: 0.85rem;">3</span>
                    <span style="color: var(--med-text-secondary); font-size: 0.92rem;"><strong>CNN Training:</strong> Train Baseline vs GAN-Augmented CNNs.</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="background: var(--med-primary-light); color: var(--med-primary); font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: 0.85rem;">4</span>
                    <span style="color: var(--med-text-secondary); font-size: 0.92rem;"><strong>Model Comparison:</strong> Statistically evaluate accuracy and F1 gains.</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="background: var(--med-primary-light); color: var(--med-primary); font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: 0.85rem;">5</span>
                    <span style="color: var(--med-text-secondary); font-size: 0.92rem;"><strong>Prediction & XAI:</strong> Full inference, Grad-CAM++, and OOD gate.</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="background: var(--med-primary-light); color: var(--med-primary); font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: 0.85rem;">6</span>
                    <span style="color: var(--med-text-secondary); font-size: 0.92rem;"><strong>Reports:</strong> Download CSV metric tables and PDF artifacts.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # Instant Scan Upload & Risk Rate Assessment Widget
    # =========================================================================
    st.markdown("---")
    st.markdown("###  Instant Clinical Scan Analyzer & Risk Rate Predictor")
    st.markdown("<p style='color: var(--med-text-secondary); font-size: 0.95rem; margin-top: -0.5rem;'>Upload any patient radiograph or click a curated sample case below to receive automated <strong>Normal vs. Diseased</strong> classification, specific pathology, and a calibrated <strong>Risk Rate (%)</strong> with Scan Authenticity verification.</p>", unsafe_allow_html=True)

    col_u1, col_u2 = st.columns([1.1, 1.3])
    
    # Session state for home scan
    if "home_img" not in st.session_state:
        st.session_state["home_img"] = None
    if "home_img_label" not in st.session_state:
        st.session_state["home_img_label"] = ""

    sample_scans_dir = root / "data" / "sample_scans"

    with col_u1:
        st.markdown("**Option 1: Upload Patient Radiograph**")
        home_upload = st.file_uploader("Select Scan Image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"], key="home_uploader")
        if home_upload is not None:
            st.session_state["home_img"] = Image.open(home_upload).convert("RGB")
            st.session_state["home_img_label"] = f"Uploaded Scan: {home_upload.name}"

        st.markdown("<div style='margin-top: 1rem;'><strong>Option 2: 1-Click Curated Clinical Samples</strong></div>", unsafe_allow_html=True)
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        
        with s_col1:
            if st.button(" Normal", key="home_btn_norm", use_container_width=True):
                p = sample_scans_dir / "sample_normal.png"
                if not p.exists():
                    p = sample_scans_dir / "sample_normal.jpg"
                if p.exists():
                    st.session_state["home_img"] = Image.open(p).convert("RGB")
                    st.session_state["home_img_label"] = "Sample Case: Normal Lung Fields (Healthy)"
        with s_col2:
            if st.button(" COVID-19", key="home_btn_cov", use_container_width=True):
                p = sample_scans_dir / "sample_covid.png"
                if not p.exists():
                    p = sample_scans_dir / "sample_covid.jpg"
                if p.exists():
                    st.session_state["home_img"] = Image.open(p).convert("RGB")
                    st.session_state["home_img_label"] = "Sample Case: COVID-19 (SARS-CoV-2 Infection)"
        with s_col3:
            if st.button(" Opacity", key="home_btn_op", use_container_width=True):
                p = sample_scans_dir / "sample_lung_opacity.png"
                if not p.exists():
                    p = sample_scans_dir / "sample_lung_opacity.jpg"
                if p.exists():
                    st.session_state["home_img"] = Image.open(p).convert("RGB")
                    st.session_state["home_img_label"] = "Sample Case: Pulmonary Opacity"
        with s_col4:
            if st.button(" Pneumonia", key="home_btn_vp", use_container_width=True):
                p = sample_scans_dir / "sample_viral_pneumonia.png"
                if not p.exists():
                    p = sample_scans_dir / "sample_viral_pneumonia.jpg"
                if p.exists():
                    st.session_state["home_img"] = Image.open(p).convert("RGB")
                    st.session_state["home_img_label"] = "Sample Case: Viral Pneumonia"

    with col_u2:
        uploaded_home_img = st.session_state.get("home_img")
        if uploaded_home_img is not None:
            model_type_home = "augmented" if aug_ready else "baseline"
            with st.spinner("Analyzing radiograph with deep neural network & scan validation..."):
                try:
                    res_home = analyze_chest_xray(uploaded_home_img, model_type=model_type_home)
                    is_inv = res_home.get("disease_status") == "INVALID_SCAN"
                    
                    banner_bg = "rgba(239, 68, 68, 0.12)" if is_inv else ("rgba(16, 185, 129, 0.12)" if res_home['disease_status'] == 'NORMAL' else ("rgba(239, 68, 68, 0.12)" if res_home['risk_rate'] >= 70.0 else "rgba(245, 158, 11, 0.12)"))
                    border_c = "#ef4444" if is_inv else res_home['risk_color']
                    risk_txt = "0.0%" if is_inv else f"{res_home['risk_rate']:.1f}%"

                    st.markdown(f"""
                    <div style="background-color: {banner_bg}; border-left: 6px solid {border_c}; 
                                padding: 1.25rem 1.5rem; border-radius: 12px; margin-bottom: 1.2rem;
                                border-top: 1px solid var(--med-border); border-right: 1px solid var(--med-border); border-bottom: 1px solid var(--med-border);">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <span class="med-chip" style="background: var(--med-bg-card); color: {border_c}; border: 1px solid {border_c};">
                                    {'Stop' if is_inv else 'Active'} {res_home['status_badge']}
                                </span>
                                <h2 style="margin: 0.35rem 0 0.15rem 0; color: var(--med-text-primary); font-size: 1.6rem;">{res_home['pathology_name']}</h2>
                                <span style="font-size: 0.88rem; color: var(--med-text-muted);">Category: {res_home['pathology_category']}</span>
                            </div>
                            <div style="text-align: right; min-width: 130px;">
                                <div style="font-size: 0.78rem; font-weight: 700; color: var(--med-text-muted); text-transform: uppercase;">ESTIMATED RISK</div>
                                <div style="font-size: 2.2rem; font-weight: 900; color: {border_c}; line-height: 1;">
                                    {risk_txt}
                                </div>
                                <div style="font-size: 0.85rem; font-weight: 700; color: {border_c}; margin-top: 0.2rem;">
                                    {res_home['risk_level']}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c_img, c_detail = st.columns([1, 1.2])
                    with c_img:
                        st.image(uploaded_home_img, caption=st.session_state.get("home_img_label", "Scanned ROI"), use_container_width=True)
                    with c_detail:
                        st.markdown("**Diagnostic Impression:**")
                        st.write(res_home["diagnostic_summary"])
                        st.markdown(f"**Clinical Urgency:** `{res_home['urgency']}`")
                        st.markdown(f"**Evaluated by:** `{res_home['model_used']}`")
                        # Brief disease summary (new add-on feature)
                        dd_home = res_home.get("disease_details", {})
                        if dd_home and res_home.get("is_valid_scan", True):
                            st.markdown(
                                f"**About {dd_home.get('disease_name', res_home['pathology_name'])}:** "
                                f"{dd_home.get('overview', '')[:200]}..."
                            )
                        st.page_link("pages/5_Prediction.py", label="Open Comprehensive Clinical Diagnostic & XAI Dashboard")

                except Exception as e:
                    st.error(f"Inference error: {str(e)}")
        else:
            st.info("Upload a patient chest radiograph or click one of the 4 clinical samples on the left to see instant classification and risk rate estimation.")


if __name__ == "__main__":
    main()
