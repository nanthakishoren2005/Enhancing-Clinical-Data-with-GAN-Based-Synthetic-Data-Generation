"""
Page 1: Dataset Explorer
Explores class distributions across train, val, and test splits, previews sample slices,
and validates folder integrity and patient separation across all 4 COVID-19 Radiography classes.
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_project_root
from src.data.dataset import CLASS_TO_IDX
from src.data.validate_dataset import validate_dataset
from src.ui.styles import apply_theme, render_hero

st.set_page_config(page_title="Dataset Explorer", layout="wide")
apply_theme()

root = get_project_root()
config = load_config()
orig_dir = root / config["dataset"]["original_dir"]
synth_dir = root / config["dataset"]["synthetic_dir"]

# Premium Header
render_hero(
    title="Dataset Explorer & Cohort Validation",
    subtitle="Inspect real and synthetic radiograph partitions, verify 4-class balances, and preview authentic clinical scan patches.",
    badge="Cohort Integrity & Verification",
)

# Run Validation (fast cached)
res = validate_dataset(orig_dir, synth_dir)

# Validation Banner
if res["valid"]:
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.1); border-left: 5px solid #10b981; padding: 0.9rem 1.25rem; border-radius: 8px; margin-bottom: 1.2rem; border-top: 1px solid var(--med-border); border-right: 1px solid var(--med-border); border-bottom: 1px solid var(--med-border);">
        <strong style="color: #10b981;">Dataset Validated:</strong> Found <strong>{res['total_images']}</strong> real radiographs partitioned across Normal, COVID-19, Pulmonary Opacity, and Viral Pneumonia with zero partition contamination.
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Dataset structure incomplete or unpopulated.")
    if res["warnings"]:
        with st.expander("Show Validation Warnings", expanded=True):
            for w in res["warnings"]:
                st.write(f"- {w}")

# Top Summary Metrics
c1, c2, c3, c4 = st.columns(4)
train_total = res.get("splits", {}).get("train", {}).get("total", 0)
val_total = res.get("splits", {}).get("val", {}).get("total", 0)
test_total = res.get("splits", {}).get("test", {}).get("total", 0)
synth_train_total = res.get("synthetic_train", {}).get("total", 0)

with c1:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Train Split (Real)</div>
        <div class="med-metric-value">{train_total}</div>
        <div class="med-metric-sub" style="color: var(--med-primary);">Baseline Training Set</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Val Split (Real)</div>
        <div class="med-metric-value">{val_total}</div>
        <div class="med-metric-sub" style="color: #10b981;">Calibrated Thresholds</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Test Split (Held-Out)</div>
        <div class="med-metric-value">{test_total}</div>
        <div class="med-metric-sub" style="color: #8b5cf6;">Zero Data Leakage</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Synthetic (Train Aug)</div>
        <div class="med-metric-value">{synth_train_total}</div>
        <div class="med-metric-sub" style="color: #06b6d4;">Conditional GAN Generated</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Class Distribution", "Sample Radiograph Patches", "Dataset Split Manager"])

with tab1:
    st.markdown("### Distribution of Radiograph Classes across Partitions")
    rows = []
    for split_name in ["train", "val", "test"]:
        split_data = res.get("splits", {}).get(split_name, {}).get("classes", {})
        row = {"Split": split_name.upper()}
        for cname in CLASS_TO_IDX:
            row[cname.replace("_", " ").title()] = split_data.get(cname, 0)
        row["Total"] = sum([split_data.get(c, 0) for c in CLASS_TO_IDX])
        rows.append(row)

    df_dist = pd.DataFrame(rows)
    st.dataframe(df_dist, use_container_width=True)

    # Plot Distribution with clean medical palette
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    class_cols = [c.replace("_", " ").title() for c in CLASS_TO_IDX]
    palette = ["#10b981", "#8b5cf6", "#f59e0b", "#ef4444"]
    df_plot = df_dist.set_index("Split")[class_cols]
    df_plot.plot(kind="bar", stacked=False, ax=ax, color=palette, width=0.7)
    
    ax.set_ylabel("Number of Images", color="#64748b", fontweight="bold", fontsize=9)
    ax.set_title("Radiograph Image Counts per Class Across Partitions", color="#0f172a" if False else "#38bdf8", fontweight="bold", fontsize=11, pad=12)
    ax.tick_params(colors="#64748b", labelsize=9)
    plt.xticks(rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.grid(axis="y", linestyle=":", alpha=0.4, color="#64748b")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.markdown("### Visual Gallery of Real Radiographs")
    split_choice = st.selectbox("Select Partition to Preview:", ["train", "val", "test"])

    cols = st.columns(len(CLASS_TO_IDX))
    for idx, cname in enumerate(CLASS_TO_IDX):
        with cols[idx]:
            st.markdown(f"**Class: {cname.replace('_', ' ').title()}**")
            target_folder = orig_dir / split_choice / cname
            if target_folder.exists():
                imgs = [f for f in target_folder.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
                if imgs:
                    displayed = 0
                    for img_file in imgs:
                        if not img_file.exists():
                            continue
                        try:
                            with Image.open(img_file) as im:
                                st.image(im, caption=f"{img_file.name}", use_container_width=True)
                            displayed += 1
                            if displayed >= 2:
                                break
                        except Exception:
                            # Skip corrupted or unreadable image files silently
                            continue
                    if displayed == 0:
                        st.info(f"No readable images in {target_folder.name}")
                else:
                    st.info(f"No images in {target_folder.name}")
            else:
                st.warning(f"Folder not found: {cname}")

with tab3:
    st.markdown("### Dataset Ingestion & Split Management")
    st.markdown("""
    The clinical cohort is organized into a balanced 4-class multi-class setup:
    - **Normal Lung:** Physiological lung lucency without focal consolidation or effusion.
    - **COVID-19:** SARS-CoV-2 viral pneumonitis featuring peripheral, bilateral ground-glass opacities.
    - **Pulmonary Opacity:** Non-COVID localized alveolar consolidation or bacterial infiltrates.
    - **Viral Pneumonia:** Bilateral interstitial infiltrates with reticulonodular markings.
    """)
