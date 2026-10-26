"""
Page 6: Phase 1 - 256x256 Experiment
Mirrors the Model Comparison page layout: metric cards, tabs (no Grad-CAM),
confusion matrix heatmaps, per-class comparison table + chart, generated samples.
Display-only page; no live inference or Grad-CAM.
"""
import streamlit as st
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import load_config, get_project_root
from src.ui.styles import apply_theme, render_hero

st.set_page_config(page_title="Phase 1: 256x256 Experiment", layout="wide")
apply_theme()

root = get_project_root()
config = load_config()

phase1_dir = root / "outputs" / "v2" / "res256"
phase1_json = phase1_dir / "phase1_summary.json"
sample_img = phase1_dir / "cgan256_epoch_050.png"

render_hero(
    title="Phase 1: 256x256 Experiment",
    subtitle="256x256 Conditional DCGAN synthetic augmentation experiment with ResNet-18 classifier on 256x256 inputs.",
    badge="Phase 1 — Higher Resolution Experiment",
)

# Verify required files exist
if not phase1_json.exists():
    st.error(f"Phase 1 results not found: {phase1_json}")
    st.stop()

if not sample_img.exists():
    st.error(f"Sample grid not found: {sample_img}")
    st.stop()

# Load Phase 1 data
with open(phase1_json, "r", encoding="utf-8") as f:
    phase1_data = json.load(f)

# Load baseline data from the same source the Model Comparison page uses
eval_json_path = root / config["paths"]["reports_dir"] / "evaluation_report.json"
baseline_available = eval_json_path.exists()
if baseline_available:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    b_met = eval_data["baseline_metrics"]
    b_per_class = b_met["per_class"]
    b_cm = np.array(b_met["confusion_matrix"])
else:
    b_met = None
    b_per_class = None
    b_cm = None

p1_metrics = phase1_data["results"]["test_metrics"]
p1_per_class = p1_metrics["per_class"]
p1_cm = np.array(p1_metrics["confusion_matrix"])

# Headline comparison values
p1_acc = p1_metrics["accuracy"]
p1_f1 = p1_metrics["macro_f1"]
p1_sens = p1_metrics["macro_sensitivity"]
p1_spec = p1_metrics["macro_specificity"]
p1_prec = p1_metrics["macro_precision"]

b_acc = b_met["accuracy"] if b_met else None
b_f1 = b_met["macro_f1"] if b_met else None
b_sens = b_met["macro_sensitivity"] if b_met else None
b_spec = b_met["macro_specificity"] if b_met else None
b_prec = b_met["macro_precision"] if b_met else None

# Deltas (all positive for Phase 1)
acc_delta = (p1_acc - b_acc) * 100 if b_acc is not None else None
f1_delta = (p1_f1 - b_f1) * 100 if b_f1 is not None else None
sens_delta = (p1_sens - b_sens) * 100 if b_sens is not None else None
spec_delta = (p1_spec - b_spec) * 100 if b_spec is not None else None
prec_delta = (p1_prec - b_prec) * 100 if b_prec is not None else None

# Honest caveat note (required wording)
st.markdown("""
<div class="med-card" style="border-left: 5px solid #f59e0b; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem;">
    <h4 style="margin-top: 0; color: var(--med-text-primary);">Important Caveat</h4>
    <p style="color: var(--med-text-secondary); line-height: 1.6;">
        Phase 1 changed two things at once — input resolution (128 to 256) and synthetic augmentation (100 per class) — 
        so the gain cannot be credited to synthetic data alone. A 256 real-only control would be needed to separate the two effects.
    </p>
</div>
""", unsafe_allow_html=True)

# Key Empirical Metrics Overview (Held-Out Test Set) — 5 cards mirroring Model Comparison
st.markdown("### Key Empirical Metrics Overview (Held-Out Test Set)")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Accuracy</div>
        <div class="med-metric-value">{p1_acc*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">
            {acc_delta:+.2f}% vs Baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Macro F1-Score</div>
        <div class="med-metric-value">{p1_f1*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">
            {f1_delta:+.2f}% vs Baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Sensitivity (Recall)</div>
        <div class="med-metric-value">{p1_sens*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">
            {sens_delta:+.2f}% vs Baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Macro Specificity</div>
        <div class="med-metric-value">{p1_spec*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">
            {spec_delta:+.2f}% vs Baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="med-metric-card">
        <div class="med-metric-label">Macro Precision</div>
        <div class="med-metric-value">{p1_prec*100:.2f}%</div>
        <div class="med-metric-sub" style="color: #10b981;">
            {prec_delta:+.2f}% vs Baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs — same names/order as Model Comparison page, EXCEPT no Grad-CAM tab
tab1, tab2, tab3 = st.tabs([
    "Detailed Comparison Table",
    "Confusion Matrices",
    "Per-Class Breakdown & Discussion",
])

with tab1:
    st.markdown("### Measured Performance Gain Table")
    table_rows = [
        {
            "Metric": "Accuracy",
            "Baseline CNN": f"{b_acc*100:.2f}%" if b_acc is not None else "N/A",
            "Phase 1 CNN": f"{p1_acc*100:.2f}%",
            "Absolute Difference (+/-)": f"{acc_delta:+.2f}%" if acc_delta is not None else "N/A",
            "Relative Gain (%)": f"{(acc_delta/b_acc*100):+.2f}%" if (acc_delta is not None and b_acc) else "N/A",
        },
        {
            "Metric": "Macro Precision",
            "Baseline CNN": f"{b_prec*100:.2f}%" if b_prec is not None else "N/A",
            "Phase 1 CNN": f"{p1_prec*100:.2f}%",
            "Absolute Difference (+/-)": f"{prec_delta:+.2f}%" if prec_delta is not None else "N/A",
            "Relative Gain (%)": f"{(prec_delta/b_prec*100):+.2f}%" if (prec_delta is not None and b_prec) else "N/A",
        },
        {
            "Metric": "Macro Recall (Sensitivity)",
            "Baseline CNN": f"{b_sens*100:.2f}%" if b_sens is not None else "N/A",
            "Phase 1 CNN": f"{p1_sens*100:.2f}%",
            "Absolute Difference (+/-)": f"{sens_delta:+.2f}%" if sens_delta is not None else "N/A",
            "Relative Gain (%)": f"{(sens_delta/b_sens*100):+.2f}%" if (sens_delta is not None and b_sens) else "N/A",
        },
        {
            "Metric": "Macro F1-Score",
            "Baseline CNN": f"{b_f1*100:.2f}%" if b_f1 is not None else "N/A",
            "Phase 1 CNN": f"{p1_f1*100:.2f}%",
            "Absolute Difference (+/-)": f"{f1_delta:+.2f}%" if f1_delta is not None else "N/A",
            "Relative Gain (%)": f"{(f1_delta/b_f1*100):+.2f}%" if (f1_delta is not None and b_f1) else "N/A",
        },
        {
            "Metric": "Macro Specificity",
            "Baseline CNN": f"{b_spec*100:.2f}%" if b_spec is not None else "N/A",
            "Phase 1 CNN": f"{p1_spec*100:.2f}%",
            "Absolute Difference (+/-)": f"{spec_delta:+.2f}%" if spec_delta is not None else "N/A",
            "Relative Gain (%)": f"{(spec_delta/b_spec*100):+.2f}%" if (spec_delta is not None and b_spec) else "N/A",
        },
    ]
    df_summary = pd.DataFrame(table_rows)
    st.dataframe(df_summary, use_container_width=True)

with tab2:
    st.markdown("### Side-by-Side Confusion Matrices (Held-Out Test Set)")
    class_label_map = {
        "normal": "Normal",
        "covid": "COVID-19",
        "lung_opacity": "Lung Opacity",
        "viral_pneumonia": "Viral Pneumonia",
    }
    labels_display = [class_label_map[c] for c in ["normal", "covid", "lung_opacity", "viral_pneumonia"]]

    fig_cm, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Phase 1
    cm_p1_norm = p1_cm.astype("float") / p1_cm.sum(axis=1)[:, np.newaxis]
    cm_p1_norm = np.nan_to_num(cm_p1_norm)
    annot_p1 = np.empty_like(p1_cm).astype(str)
    for i in range(p1_cm.shape[0]):
        for j in range(p1_cm.shape[1]):
            annot_p1[i, j] = f"{int(p1_cm[i, j])}\n({cm_p1_norm[i, j]*100:.1f}%)"
    sns.heatmap(
        p1_cm,
        annot=annot_p1,
        fmt="",
        cmap="Greens",
        xticklabels=labels_display,
        yticklabels=labels_display,
        cbar=False,
        ax=axes[0],
        linewidths=1,
        linecolor="#e0e0e0",
    )
    axes[0].set_title("Phase 1 (256x256, 100 synthetics/class)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Class", fontsize=10)
    axes[0].set_ylabel("True Class", fontsize=10)

    # Right: Baseline
    if b_cm is not None:
        cm_b_norm = b_cm.astype("float") / b_cm.sum(axis=1)[:, np.newaxis]
        cm_b_norm = np.nan_to_num(cm_b_norm)
        annot_b = np.empty_like(b_cm).astype(str)
        for i in range(b_cm.shape[0]):
            for j in range(b_cm.shape[1]):
                annot_b[i, j] = f"{int(b_cm[i, j])}\n({cm_b_norm[i, j]*100:.1f}%)"
        sns.heatmap(
            b_cm,
            annot=annot_b,
            fmt="",
            cmap="Blues",
            xticklabels=labels_display,
            yticklabels=labels_display,
            cbar=False,
            ax=axes[1],
            linewidths=1,
            linecolor="#e0e0e0",
        )
        axes[1].set_title("Baseline CNN (128x128)", fontsize=12, fontweight="bold", pad=10)
        axes[1].set_xlabel("Predicted Class", fontsize=10)
        axes[1].set_ylabel("True Class", fontsize=10)

    plt.suptitle("Confusion Matrix Comparison: Phase 1 vs Baseline (Held-Out Test Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

with tab3:
    st.markdown("### Per-Class Precision, Recall (Sensitivity), Specificity & F1")

    class_labels = ["normal", "covid", "lung_opacity", "viral_pneumonia"]
    class_display_names = [class_label_map[c] for c in class_labels]

    # Side-by-side table
    rows = []
    for cname in class_labels:
        b = b_per_class[cname] if b_per_class else None
        p = p1_per_class[cname]
        rows.append({
            "Class": class_label_map[cname],
            "Support": p["support"],
            "Baseline Precision": f"{b['precision']:.4f}" if b else "N/A",
            "Phase 1 Precision": f"{p['precision']:.4f}",
            "Baseline Recall": f"{b['recall']:.4f}" if b else "N/A",
            "Phase 1 Recall": f"{p['recall']:.4f}",
            "Baseline Specificity": f"{b['specificity']:.4f}" if b else "N/A",
            "Phase 1 Specificity": f"{p['specificity']:.4f}",
            "Baseline F1": f"{b['f1_score']:.4f}" if b else "N/A",
            "Phase 1 F1": f"{p['f1_score']:.4f}",
        })
    df_per_class = pd.DataFrame(rows)
    st.dataframe(df_per_class, use_container_width=True)

    # Per-Class F1-Score Comparison chart (same style as Model Comparison page)
    st.markdown("### Per-Class F1-Score Comparison")
    classes_order = ["normal", "covid", "lung_opacity", "viral_pneumonia"]
    labels_display_f1 = [class_label_map[c] for c in classes_order]
    base_f1_vals = [b_per_class[c]["f1_score"] * 100 if b_per_class else 0 for c in classes_order]
    p1_f1_vals = [p1_per_class[c]["f1_score"] * 100 for c in classes_order]

    fig_f1, ax_f1 = plt.subplots(figsize=(7.5, 3.2))
    y_pos = np.arange(len(classes_order))
    bar_width = 0.35

    rects1 = ax_f1.barh(y_pos - bar_width/2, base_f1_vals, bar_width, label="Baseline CNN (128x128)", color="#0284c7")
    rects2 = ax_f1.barh(y_pos + bar_width/2, p1_f1_vals, bar_width, label="Phase 1 CNN (256x256)", color="#10b981")

    ax_f1.set_xlabel("AI Predicted Probability (%)", fontsize=9, fontweight="bold")
    ax_f1.set_yticks(y_pos)
    ax_f1.set_yticklabels(labels_display_f1, fontsize=9, fontweight="bold")
    ax_f1.set_xlim(0, 105)
    ax_f1.legend(loc="lower right", framealpha=0.95)
    ax_f1.grid(axis="x", linestyle=":", alpha=0.6)
    ax_f1.spines["top"].set_visible(False)
    ax_f1.spines["right"].set_visible(False)

    for bar in rects1:
        w = bar.get_width()
        if w > 2.0:
            ax_f1.annotate(f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                           xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color="#0284c7")
    for bar in rects2:
        w = bar.get_width()
        if w > 2.0:
            ax_f1.annotate(f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                           xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color="#047857", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig_f1, use_container_width=True)
    plt.close(fig_f1)

# Generated Samples section
st.markdown("---")
st.markdown("### Generated Samples")
st.image(str(sample_img), caption="Conditional DCGAN 256x256 — Epoch 50 Sample Grid", use_container_width=True)

# Honest note (word-for-word, required)
st.markdown("---")
st.markdown("""
<div class="med-card" style="border-left: 5px solid #f59e0b; padding: 0.9rem 1.2rem; margin-top: 1rem;">
    <h4 style="margin-top: 0; color: var(--med-text-primary);">Important Caveat</h4>
    <p style="color: var(--med-text-secondary); line-height: 1.6;">
        Phase 1 changed two things at once — input resolution (128 to 256) and synthetic augmentation (100 per class) — 
        so the gain cannot be credited to synthetic data alone. A 256 real-only control would be needed to separate the two effects.
    </p>
</div>
""", unsafe_allow_html=True)
