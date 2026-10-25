# GAN-Based Synthetic Medical Image Augmentation for Increased CNN Performance in COVID-19 Chest X-ray Classification

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

## Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Experiments and Results](#experiments-and-results)
7. [System Requirements](#system-requirements)
8. [Installation & Setup](#installation--setup)
9. [Command-Line Execution Workflow](#command-line-execution-workflow)
10. [Streamlit Web Application Guide](#streamlit-web-application-guide)
11. [Project Directory Structure](#project-directory-structure)
12. [Limitations & Ethical Disclaimers](#limitations--ethical-disclaimers)
13. [Future Scope](#future-scope)
14. [Academic References](#academic-references)

---

## Abstract

Accurate classification of chest X-ray images is critical for early diagnosis and treatment planning of respiratory diseases. Deep learning models trained on medical image datasets frequently suffer from severe data scarcity and class imbalance due to high acquisition costs, privacy regulations, and rarity of specific pathologies. Traditional geometric data augmentation (e.g., flips, rotations) generates visually limited anatomical variance.

This project implements and experimentally validates an end-to-end framework applying a Conditional Deep Convolutional Generative Adversarial Network (cDCGAN) to synthesize class-conditioned chest X-ray patches belonging to four target classes: Normal, COVID-19, Lung Opacity, and Viral Pneumonia. The study compares:

- **Model A (Baseline CNN):** Trained exclusively on original real training images with conventional augmentation.
- **Model B (GAN-Augmented CNN):** Trained on the same original scans and conventional augmentation, supplemented with GAN-generated synthetic chest X-ray images.

Both models are evaluated on the exact same, untouched, held-out real test set to measure empirical gains in Accuracy, Macro Precision, Recall (Sensitivity), Specificity, and F1-Score without data leakage.

---

## Problem Statement

Small medical imaging datasets restrict deep convolutional neural networks from generalizing effectively, leading to:

1. **Severe Overfitting:** High training accuracy with poor validation and test generalization.
2. **Class Imbalance & Diagnostic Bias:** Models favor dominant classes while misclassifying critical minority classes.
3. **Geometric Limits of Traditional Augmentation:** Standard affine transforms cannot simulate diverse internal textures, opacity patterns, or radiographic variations.

---

## Project Objectives

1. **Develop a Class-Conditional DCGAN** capable of mapping latent vectors and class labels to 128 x 128 chest X-ray region-of-interest patches.
2. **Establish a Controlled Experimental Protocol** comparing a Baseline CNN (Model A) against a GAN-Augmented CNN (Model B).
3. **Enforce Strict Anti-Leakage Separation:** Ensure validation and test sets consist solely of real, untouched chest X-ray images, grouping patient slices together.
4. **Conduct Rigorous Statistical Evaluation:** Calculate multiclass Accuracy, Macro F1, Confusion Matrices, and One-vs-Rest (OvR) Sensitivity and Specificity.
5. **Provide an Interactive Streamlit Web Platform** featuring dataset inspection, live training controls, model comparison, chest X-ray prediction inference, and automated report exports.

---

## Dataset

The project uses the **COVID-19 Radiography Database** (Kaggle), a chest X-ray dataset with four classes:

| Class | Description |
|-------|-------------|
| Normal | Healthy lung fields |
| COVID-19 | SARS-CoV-2 associated pneumonia |
| Lung Opacity | Non-COVID localized alveolar consolidation or bacterial infiltrates |
| Viral Pneumonia | Bilateral interstitial infiltrates with reticulonodular markings |

**Splits (seed 42):**
- Train: 2400 images (600 per class)
- Validation: 400 images (100 per class)
- Test: 400 images (100 per class)
- Image size: 128 x 128 pixels

**Anti-Leakage Rules:**
1. **Patient-Level Grouping:** All images belonging to the same patient remain strictly inside the same partition.
2. **Purity of Evaluation Sets:** Validation and Test splits are populated exclusively with untouched real chest X-ray images.
3. **Synthetic Training Bound:** GAN-generated images are deposited solely in `data/synthetic/train/`.

---

## Methodology

### Conditional DCGAN Architecture

The cDCGAN conditions both the Generator ($G$) and Discriminator ($D$) on class labels $y \in \{0, 1, 2, 3\}$ representing Normal, COVID-19, Lung Opacity, and Viral Pneumonia.

- **Generator ($G$):** Receives a 100-dimensional Gaussian noise vector $z \sim \mathcal{N}(0, I)$ concatenated with a 32-dimensional class embedding. Projects via a dense layer to $512 \times 4 \times 4$ and upsamples through 5 transposed convolution blocks with Batch Normalization and ReLU activations, culminating in a Tanh layer producing $3 \times 128 \times 128$ normalized images in $[-1, 1]$.
- **Discriminator ($D$):** Takes an input image ($3 \times 128 \times 128$) concatenated with a spatially mapped class embedding ($1 \times 128 \times 128$), forming a 4-channel input. Employs 5 strided convolution blocks with LeakyReLU ($\alpha = 0.2$) and Batch Normalization to output a single authenticity logit.

- **Objective Function:** $\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z|y)))]$
- **Optimizer:** Adam ($\beta_1=0.5, \beta_2=0.999$, Learning Rate $= 0.0002$).
- **Stabilization:** One-sided label smoothing ($0.9$ for real targets) and batch normalization across intermediate layers.
- **Training:** 50 epochs on Kaggle GPU (T4), batch size 32. Training is complete; the final checkpoint exists at `checkpoints/cgan_checkpoint.pt`.

### CNN Classifier Architectures

Two modular backbones are supported:

1. **ResNet-18 Backbone (Transfer Learning):** Pretrained residual network featuring skip connections, fine-tuned with a customized 4-class classification head.
2. **CustomCNN (Lightweight):** 4 Conv-BN-ReLU-MaxPool feature blocks followed by Adaptive Average Pooling and Dropout ($p=0.4, 0.2$), optimized for rapid CPU inference.

---

## Experiments and Results

### Experiment Setup

Both Model A and Model B use ResNet-18 with identical hyperparameters:

- Max 15 epochs, early stopping patience 5
- Best checkpoint selected by validation loss
- Adam optimizer with learning rate 0.0003 and weight decay $10^{-4}$
- Cross-Entropy Loss for CNN; ReduceLROnPlateau scheduler

### Final Test Results (400 untouched real test images)

| Experiment | Synthetic Images per Class | Total Synthetic | Accuracy | Macro F1 |
|------------|---------------------------|-----------------|----------|----------|
| Model A: Baseline CNN | 0 | 0 | 92.75% | 0.9277 |
| Model B: Experiment 1 | 40 | 160 | 88.50% | 0.8833 |
| Model B: Experiment 2 | 100 | 400 | 92.50% | 0.9248 |

### Key Findings

- **Experiment 1 (40 synthetic/class):** Hurt accuracy compared to Baseline. Normal recall dropped to 0.69 due to Normal vs Lung Opacity confusion.
- **Experiment 2 (100 synthetic/class):** Fixed the Normal vs Lung Opacity confusion (Normal recall improved to 0.90) and matched the Baseline within 1 test image (92.50% vs 92.75%).
- **Conclusion:** GAN augmentation did not beat the Baseline. The larger synthetic set (100 per class) recovered performance to near-baseline levels but did not exceed it. Both experiments are reported honestly; no claim is made that GAN augmentation improved over the Baseline.

---

## System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.11 or 3.13.
- **Hardware:**
  - Minimum: Intel Core i5 / AMD Ryzen 5, 8 GB RAM (Runs on CPU).
  - Recommended: NVIDIA GPU with CUDA support (automatically detected when available).

---

## Installation & Setup

1. **Clone or Open Workspace:**
   ```
   cd "C:\Users\nanth\OneDrive\Desktop\final yr project final"
   ```

2. **Verify Python & Dependencies:**
   ```
   py -m pip install -r requirements.txt
   ```

3. **Verify Installation:**
   ```
   py -m pytest tests/ -v
   ```

---

## Command-Line Execution Workflow

```powershell
# 1. Validate dataset integrity and split distributions
py -m src.data.validate_dataset

# 2. Train the Conditional DCGAN (50 epochs, already complete)
py -m src.gan.train_gan --epochs 50 --batch-size 32

# 3. Generate synthetic chest X-ray patches
py -m src.gan.generate_images --num-per-class 40

# 4. Train Model A: Baseline CNN (Real training images only)
py -m src.cnn.train_baseline --epochs 15 --architecture resnet18

# 5. Train Model B: GAN-Augmented CNN (Real + Synthetic images)
py -m src.cnn.train_augmented --epochs 15 --architecture resnet18

# 6. Evaluate and Compare both models on untouched real test set
py -m src.evaluation.compare_models
```

*Note: The GAN training (step 2) is already complete. The checkpoint exists at `checkpoints/cgan_checkpoint.pt`.*

---

## Streamlit Web Application Guide

Launch the interactive graphical interface:

```
py -m streamlit run app.py
```

*(Or double-click `run_dashboard.bat`)*

### Web Dashboard Pages:

- **Home Dashboard (`app.py`):** System architecture, real-time pipeline status, academic notices, and an Instant Chest X-ray Upload & Risk Rate Predictor widget.
- **Page 1: Dataset Explorer (`1_Dataset_Explorer.py`):** Class distribution graphs, real image preview gallery, and dataset validation.
- **Page 2: GAN Generation (`2_GAN_Generation.py`):** Display-only page showing completed Conditional DCGAN training results, loss curve, synthetic dataset counts, real vs synthetic visual comparison, and sample evolution across epochs.
- **Page 3: CNN Training (`3_CNN_Training.py`):** Live side-by-side training for Baseline and Augmented CNNs with learning curves.
- **Page 4: Model Comparison (`4_Model_Comparison.py`):** Head-to-head metrics table, accuracy delta, side-by-side confusion matrices, and sensitivity/specificity breakdowns.
- **Page 5: Diagnosis & Risk Rate (`5_Prediction.py`):** Chest X-ray classification (Normal vs Diseased / Specific Pathology), calibrated quantitative Risk Rate (%), dynamic visual risk meter, Grad-CAM++ visual interpretability, predictive-entropy uncertainty detection, radiomic biomarker extraction, actionable clinical management recommendations, and clinical diagnostic summary report download.
- **Page 6: Reports (`6_Reports.py`):** Downloadable CSV tables, training history logs, and formatted experimental summaries.

---

## Project Directory Structure

```
final yr project final/
├── app.py                      # Streamlit Home Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── PROJECT_REPORT.md           # Formal college report template
├── COLLEGE_EVALUATION_GUIDE.md # Viva & demonstration walkthrough
├── run_dashboard.bat           # 1-click dashboard launcher
├── run_pipeline.bat            # 1-click CLI pipeline runner
├── run_tests.bat               # 1-click pytest runner
├── configs/
│   └── config.yaml             # Hyperparameters, paths, and hardware settings
├── data/
│   ├── original/               # Real chest X-ray slices (train, val, test)
│   └── synthetic/              # GAN-generated training patches
├── src/
│   ├── data/                   # Dataset loader, transforms, validator, generator
│   ├── gan/                    # Generator, Discriminator, Trainer, Synthesizer
│   ├── cnn/                    # CNN models, Baseline & Augmented trainers, Predictor
│   ├── evaluation/             # Metrics (Accuracy, F1, OvR Sens/Spec), Plots, Comparator
│   └── utils/                  # Config loader, reproducibility seed, checkpoint utils
├── pages/                      # Multi-page Streamlit interface
├── checkpoints/                # Saved PyTorch model checkpoints (.pt)
├── outputs/
│   ├── figures/                # Confusion matrices, loss graphs, comparison charts
│   ├── reports/                # Summary CSV tables and JSON reports
│   └── gan_samples/            # Periodic sample grids saved during GAN training
└── tests/                      # Automated pytest unit test suite
```

---

## Limitations & Ethical Disclaimers

1. **Educational & Research Prototype:** This software is built strictly for scientific evaluation and computer vision experimentation. It is **not** certified by the FDA, CE, or CDSCO as a medical diagnostic device.
2. **Uncalibrated Confidence Scores:** Output percentages are softmax outputs, not clinical Bayesian probabilities.
3. **Synthetic Image Quality:** The 128 x 128 synthetic chest X-ray images are soft and lack fine anatomical detail (ribs, heart border). They should not be presented as clinical-quality radiographs.
4. **Single Data Split:** Results are based on a single train/validation/test split with seed 42. Different splits may yield different results.
5. **Limited Test Set:** Results are measured on a 400-image held-out test set (100 per class). Larger independent test sets would provide more robust estimates.
6. **GAN Augmentation Finding:** In this project, GAN augmentation did not outperform the Baseline on the held-out test set. The larger synthetic set (100 per class) recovered near-baseline performance but did not exceed it.

---

## Future Scope

- Transition from 2D chest X-ray patches to volumetric 3D Conditional Diffusion Models.
- Integration of multi-phase or multi-view chest imaging for improved diagnostic context.
- Grad-CAM++ visual interpretability to verify that CNN focus aligns with radiologist-identified regions of interest.
- Larger independent test sets and cross-validation for more robust performance estimates.

---

## Academic References

1. COVID-19 Radiography Database. Kaggle dataset for chest X-ray classification (Normal, COVID-19, Lung Opacity, Viral Pneumonia).
2. Radford, A., Metz, L., & Chintala, S. "Unsupervised representation learning with deep convolutional generative adversarial networks." *International Conference on Learning Representations (ICLR)*, 2016.
3. Mirza, M., & Osindero, S. "Conditional Generative Adversarial Nets." *arXiv preprint arXiv:1411.1784*, 2014.
4. He, K., Zhang, X., Ren, S., & Sun, J. "Deep residual learning for image recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
5. Chattopadhay, A., et al. "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks." *IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2018.
