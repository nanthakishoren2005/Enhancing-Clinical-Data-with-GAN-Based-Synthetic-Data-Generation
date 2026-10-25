"""
Page 3: CNN Training
Controls for training Model A (Baseline CNN) and Model B (GAN-Augmented CNN).
Displays live training curves, validation metrics, early stopping status, and checkpoint saving.
"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_project_root
from src.cnn.trainer import run_cnn_training
from src.ui.styles import apply_theme, render_hero

st.set_page_config(page_title="CNN Training", layout="wide")
apply_theme()

root = get_project_root()
config = load_config()
ckpt_dir = root / config["paths"]["checkpoints_dir"]
figures_dir = root / config["paths"]["figures_dir"]

base_ckpt = ckpt_dir / "baseline_best.pt"
aug_ckpt = ckpt_dir / "augmented_best.pt"

render_hero(
    title="CNN Classifier Training Pipelines",
    subtitle="Train and evaluate Model A (Baseline CNN) and Model B (GAN-Augmented CNN) under identical hyperparameter controls and patient separation.",
    badge="Deep Learning Experimentation",
)

col1, col2 = st.columns(2)

# ==============================================================================
# Model A: Baseline CNN
# ==============================================================================
with col1:
    st.markdown("""
    <div class="med-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <h3 style="margin: 0; color: #0284c7;">Model A: Baseline CNN</h3>
            <span class="med-chip" style="background: rgba(2, 132, 199, 0.15); color: #0284c7;">Conventional Aug</span>
        </div>
        <p style="color: var(--med-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
            Trained exclusively on authentic clinical radiographs using standard geometric data augmentation (mild rotations, translations, and horizontal flips).
        </p>
    """, unsafe_allow_html=True)

    b_arch = st.selectbox("Architecture (Baseline)", ["resnet18", "custom_cnn"], index=0, key="base_arch")
    b_epochs = st.number_input("Epochs (Baseline)", min_value=1, max_value=100, value=config["cnn"]["epochs"], key="base_ep")
    b_batch = st.selectbox("Batch Size (Baseline)", [8, 16, 32], index=1, key="base_bs")
    b_lr = st.number_input("Learning Rate (Baseline)", min_value=1e-5, max_value=1e-2, value=float(config["cnn"]["learning_rate"]), format="%.4f", key="base_lr")

    if base_ckpt.exists():
        st.markdown(f"<div class='med-chip' style='background: rgba(16, 185, 129, 0.15); color: #10b981; margin: 0.8rem 0;'>Checkpoint Active: {base_ckpt.name}</div>", unsafe_allow_html=True)
        curve_img = figures_dir / "baseline_learning_curve.png"
        if curve_img.exists():
            st.image(str(curve_img), caption="Baseline Training & Validation Curves", use_container_width=True)
    else:
        st.warning("Not trained yet")

    if st.button("Train Model A (Baseline CNN)", type="secondary", use_container_width=True):
        prog_bar = st.progress(0)
        status = st.empty()

        def base_cb(epoch, total, t_loss, t_acc, v_loss, v_acc):
            prog_bar.progress(epoch / total)
            status.write(f"Epoch {epoch}/{total} | Train Acc: {t_acc*100:.1f}% | Val Acc: {v_acc*100:.1f}%")

        try:
            with st.spinner("Training Baseline CNN..."):
                res = run_cnn_training(
                    use_synthetic=False,
                    experiment_name="baseline",
                    epochs=int(b_epochs),
                    batch_size=int(b_batch),
                    lr=float(b_lr),
                    architecture=b_arch,
                    progress_callback=base_cb,
                )
            st.success(f"Baseline Training Complete! Best Val Acc: {res['best_val_acc']*100:.2f}%")
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# Model B: GAN-Augmented CNN
# ==============================================================================
with col2:
    st.markdown("""
    <div class="med-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <h3 style="margin: 0; color: #10b981;">Model B: GAN-Augmented CNN</h3>
            <span class="med-chip" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">Generative Aug</span>
        </div>
        <p style="color: var(--med-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
            Trained on identical genuine radiographs PLUS class-conditioned synthetic chest X-ray patches synthesized by our Conditional DCGAN.
        </p>
    """, unsafe_allow_html=True)

    a_arch = st.selectbox("Architecture (Augmented)", ["resnet18", "custom_cnn"], index=0, key="aug_arch")
    a_epochs = st.number_input("Epochs (Augmented)", min_value=1, max_value=100, value=config["cnn"]["epochs"], key="aug_ep")
    a_batch = st.selectbox("Batch Size (Augmented)", [8, 16, 32], index=1, key="aug_bs")
    a_lr = st.number_input("Learning Rate (Augmented)", min_value=1e-5, max_value=1e-2, value=float(config["cnn"]["learning_rate"]), format="%.4f", key="aug_lr")

    if aug_ckpt.exists():
        st.markdown(f"<div class='med-chip' style='background: rgba(16, 185, 129, 0.15); color: #10b981; margin: 0.8rem 0;'>Checkpoint Active: {aug_ckpt.name}</div>", unsafe_allow_html=True)
        curve_img = figures_dir / "augmented_learning_curve.png"
        if curve_img.exists():
            st.image(str(curve_img), caption="GAN-Augmented Training & Validation Curves", use_container_width=True)
    else:
        st.warning("Not trained yet")

    if st.button("Train Model B (GAN-Augmented CNN)", type="primary", use_container_width=True):
        prog_bar_a = st.progress(0)
        status_a = st.empty()

        def aug_cb(epoch, total, t_loss, t_acc, v_loss, v_acc):
            prog_bar_a.progress(epoch / total)
            status_a.write(f"Epoch {epoch}/{total} | Train Acc: {t_acc*100:.1f}% | Val Acc: {v_acc*100:.1f}%")

        try:
            with st.spinner("Training GAN-Augmented CNN..."):
                res = run_cnn_training(
                    use_synthetic=True,
                    experiment_name="augmented",
                    epochs=int(a_epochs),
                    batch_size=int(a_batch),
                    lr=float(a_lr),
                    architecture=a_arch,
                    progress_callback=aug_cb,
                )
            st.success(f"GAN-Augmented Training Complete! Best Val Acc: {res['best_val_acc']*100:.2f}%")
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
