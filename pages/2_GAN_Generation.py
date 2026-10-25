"""
Page 2: GAN Generation & Training (Display-Only)
Displays completed Conditional DCGAN training results.
Training was performed on Kaggle GPU (50 epochs).
"""
import streamlit as st
from pathlib import Path
from PIL import Image

from src.utils.config import load_config, get_project_root
from src.data.dataset import CLASS_TO_IDX
from src.ui.styles import apply_theme, render_hero

st.set_page_config(page_title="GAN Training & Generation", layout="wide")
apply_theme()

root = get_project_root()
config = load_config()

ckpt_path = root / config["paths"]["checkpoints_dir"] / "cgan_checkpoint.pt"
samples_dir = root / config["paths"]["samples_dir"]
figures_dir = root / config["paths"]["figures_dir"]
synth_train_dir = root / config["dataset"]["synthetic_dir"] / "train"

render_hero(
    title="Conditional DCGAN Synthesis & Augmentation",
    subtitle="Display-only view of the completed Conditional DCGAN training results (trained on Kaggle GPU, 50 epochs).",
    badge="Generative Deep Learning Pipeline",
)

# Note about Kaggle GPU training
st.markdown("""
<div style="background: rgba(2, 132, 199, 0.08); border-left: 5px solid #0284c7; padding: 0.9rem 1.25rem; border-radius: 8px; margin-bottom: 1.4rem; font-size: 0.88rem; color: var(--med-text-secondary); border-top: 1px solid var(--med-border); border-right: 1px solid var(--med-border); border-bottom: 1px solid var(--med-border);">
    <strong style="color: #0284c7;">Note:</strong> Training was performed on Kaggle GPU. This page is display-only to protect the trained checkpoint and experiment results.
</div>
""", unsafe_allow_html=True)

# Training Summary Card
st.markdown("### Training Summary")
gan_cfg = config["gan"]
dataset_cfg = config["dataset"]
ckpt_active = ckpt_path.exists()

st.markdown(f"""
<div class="med-card">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
        <div style="background: var(--med-bg-card-subtle); padding: 0.8rem; border-radius: 8px; border: 1px solid var(--med-border);">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--med-text-muted); margin-bottom: 0.2rem;">GAN Epochs</div>
            <div style="font-size: 1.1rem; color: var(--med-text-primary);">{gan_cfg.get('epochs', 'N/A')}</div>
        </div>
        <div style="background: var(--med-bg-card-subtle); padding: 0.8rem; border-radius: 8px; border: 1px solid var(--med-border);">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--med-text-muted); margin-bottom: 0.2rem;">Batch Size</div>
            <div style="font-size: 1.1rem; color: var(--med-text-primary);">{gan_cfg.get('batch_size', 'N/A')}</div>
        </div>
        <div style="background: var(--med-bg-card-subtle); padding: 0.8rem; border-radius: 8px; border: 1px solid var(--med-border);">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--med-text-muted); margin-bottom: 0.2rem;">Generator LR</div>
            <div style="font-size: 1.1rem; color: var(--med-text-primary);">{gan_cfg.get('lr_g', 'N/A')}</div>
        </div>
        <div style="background: var(--med-bg-card-subtle); padding: 0.8rem; border-radius: 8px; border: 1px solid var(--med-border);">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--med-text-muted); margin-bottom: 0.2rem;">Discriminator LR</div>
            <div style="font-size: 1.1rem; color: var(--med-text-primary);">{gan_cfg.get('lr_d', 'N/A')}</div>
        </div>
        <div style="background: var(--med-bg-card-subtle); padding: 0.8rem; border-radius: 8px; border: 1px solid var(--med-border);">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--med-text-muted); margin-bottom: 0.2rem;">Latent Dim</div>
            <div style="font-size: 1.1rem; color: var(--med-text-primary);">{gan_cfg.get('latent_dim', 'N/A')}</div>
        </div>
        <div style="background: var(--med-bg-card-subtle); padding: 0.8rem; border-radius: 8px; border: 1px solid var(--med-border);">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--med-text-muted); margin-bottom: 0.2rem;">Image Size</div>
            <div style="font-size: 1.1rem; color: var(--med-text-primary);">{dataset_cfg.get('image_size', 'N/A')} x {dataset_cfg.get('image_size', 'N/A')}</div>
        </div>
    </div>
    <div style="margin-top: 1rem; padding: 0.8rem; background: {'rgba(16, 185, 129, 0.15)' if ckpt_active else 'rgba(239, 68, 68, 0.15)'}; border: 1px solid {'rgba(16, 185, 129, 0.35)' if ckpt_active else 'rgba(239, 68, 68, 0.35)'}; border-radius: 8px;">
        <div style="font-weight: 700; color: {'#10b981' if ckpt_active else '#ef4444'};">
            {'Trained Checkpoint Active: cgan_checkpoint.pt' if ckpt_active else 'Checkpoint Missing: cgan_checkpoint.pt not found'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Loss graph
st.markdown("### Generator vs Discriminator Loss")
loss_plot = figures_dir / "gan_loss_curve.png"
if loss_plot.exists():
    st.image(str(loss_plot), caption="BCEWithLogitsLoss Equilibrium (Generator vs Discriminator)", use_container_width=True)
else:
    st.info("No trained GAN loss curve found.")

# Synthetic Dataset Generated section
st.markdown("### Synthetic Dataset Generated")
synth_counts = {}
if synth_train_dir.exists():
    for class_name in CLASS_TO_IDX:
        c_dir = synth_train_dir / class_name
        count = 0
        if c_dir.exists():
            count = sum(1 for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"])
        synth_counts[class_name] = count
else:
    for class_name in CLASS_TO_IDX:
        synth_counts[class_name] = 0

total_synth = sum(synth_counts.values())

cols = st.columns(len(CLASS_TO_IDX))
for idx, cname in enumerate(CLASS_TO_IDX):
    with cols[idx]:
        st.markdown(f"""
        <div class="med-metric-card">
            <div class="med-metric-label">{cname.replace('_', ' ').title()}</div>
            <div class="med-metric-value">{synth_counts[cname]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div class="med-metric-card" style="margin-top: 0.5rem;">
    <div class="med-metric-label">Total Synthetic Images</div>
    <div class="med-metric-value">{total_synth}</div>
</div>
""", unsafe_allow_html=True)

# Real vs Synthetic Visual Comparison
st.markdown("### Real vs Synthetic Visual Comparison")
selected_class = st.selectbox(
    "Select Chest X-ray Class to Compare:",
    list(CLASS_TO_IDX.keys()),
    format_func=lambda x: "COVID-19" if x.lower() == "covid" else x.replace("_", " ").title()
)

col_real, col_synth = st.columns(2)
with col_real:
    st.markdown(f"""
    <div class="med-card">
        <h4 style="margin-top:0; color: var(--med-text-primary);">Real Clinical Sample ({selected_class.capitalize()})</h4>
    """, unsafe_allow_html=True)
    real_folder = root / config["dataset"]["original_dir"] / "train" / selected_class
    if real_folder.exists():
        r_files = [f for f in real_folder.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        if r_files:
            st.image(str(r_files[0]), caption=f"Real: {r_files[0].name}", use_container_width=True)
        else:
            st.info("No real training images found.")
    else:
        st.info("Real folder missing.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_synth:
    st.markdown(f"""
    <div class="med-card">
        <h4 style="margin-top:0; color: var(--med-text-primary);">cDCGAN Synthesized ({selected_class.capitalize()})</h4>
    """, unsafe_allow_html=True)
    synth_folder = synth_train_dir / selected_class
    if synth_folder.exists():
        s_files = [f for f in synth_folder.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        if s_files:
            st.image(str(s_files[0]), caption=f"Synthetic: {s_files[0].name}", use_container_width=True)
        else:
            st.info("No synthetic images generated yet.")
    else:
        st.info("Synthetic folder not yet created.")
    st.markdown("</div>", unsafe_allow_html=True)

# Visual Sample Evolution Across Epochs
st.markdown("---")
st.markdown("### Visual Sample Evolution Across Epochs")
if samples_dir.exists():
    grids = sorted([f for f in samples_dir.iterdir() if f.suffix == ".png"])
    if grids:
        selected_grid = st.selectbox("Select Sample Checkpoint Grid:", grids, index=len(grids) - 1, format_func=lambda x: x.name)
        st.image(str(selected_grid), caption=f"GAN Grid at {selected_grid.name}", use_container_width=True)
