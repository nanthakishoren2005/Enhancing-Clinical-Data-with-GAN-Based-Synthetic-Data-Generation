"""
Train 256x256 Conditional DCGAN (Phase 1).
Trains cDCGAN on 256x256 chest X-ray images, generates 100 synthetic images per class,
trains a ResNet-18 classifier on 256x256 inputs, evaluates on the test set,
and computes FID vs the 128x128 cDCGAN.

Usage:
    python -m scripts.train_gan256
    python -m scripts.train_gan256 --epochs 50 --batch-size 16

All outputs written to outputs/v2/res256/
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.data.dataset import LiverLesionDataset, get_gan_transforms, CLASS_TO_IDX, get_transforms
from src.gan.cdcgan256 import ConditionalGenerator256, ConditionalDiscriminator256
from src.cnn.model import build_classifier
from src.cnn.trainer import run_cnn_training


def train_cgan256(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr_g: Optional[float] = None,
    lr_d: Optional[float] = None,
    resume: bool = False,
) -> Dict[str, List[float]]:
    """
    Train Conditional DCGAN 256x256 on original training images.
    Returns loss history dictionary.
    """
    # Load Phase 1 config
    config_path = project_root / "configs" / "gan256.yaml"
    config = load_config(str(config_path))
    root = get_project_root()
    device = get_device(config)

    # Output directory
    v2_dir = root / config["v2_res256"]["output_dir"]
    synth_dir = root / config["v2_res256"]["synthetic_dir"]
    v2_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    epochs = epochs or config["gan"]["epochs"]
    batch_size = batch_size or config["gan"]["batch_size"]
    lr_g = lr_g or config["gan"]["lr_g"]
    lr_d = lr_d or config["gan"]["lr_d"]
    beta1 = config["gan"]["beta1"]
    beta2 = config["gan"]["beta2"]
    latent_dim = config["gan"]["latent_dim"]
    embedding_dim = config["gan"]["embedding_dim"]
    num_classes = config["dataset"]["num_classes"]
    img_size = config["dataset"]["image_size"]
    save_interval = config["gan"]["save_sample_interval"]

    set_seed(config["project"]["random_seed"])

    # Checkpoint path
    ckpt_path = root / config["v2_res256"]["gan_checkpoint"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset & DataLoader (Train split only)
    data_dir = root / config["dataset"]["original_dir"]
    dataset = LiverLesionDataset(
        root_dir=data_dir,
        split="train",
        synthetic_dir=None,
        transform=get_gan_transforms(img_size),
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training images found in {data_dir / 'train'}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(dataset) > batch_size),
        num_workers=config["hardware"]["num_workers"],
    )

    # Initialize Networks (256x256 variant)
    netG = ConditionalGenerator256(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        feature_maps=config["gan"]["gen_feature_maps"],
    ).to(device)

    netD = ConditionalDiscriminator256(
        num_classes=num_classes,
        in_channels=3,
        feature_maps=config["gan"]["disc_feature_maps"],
        img_size=img_size,
    ).to(device)

    # Loss & Optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))

    start_epoch = 1
    history = {"g_loss": [], "d_loss": []}

    # Resume from checkpoint if requested
    if resume and ckpt_path.exists():
        print(f"Resuming GAN256 training from {ckpt_path}...")
        ckpt = load_checkpoint(ckpt_path, device=device)
        netG.load_state_dict(ckpt["generator_state_dict"])
        netD.load_state_dict(ckpt["discriminator_state_dict"])
        optimizerG.load_state_dict(ckpt["optimizerG_state_dict"])
        optimizerD.load_state_dict(ckpt["optimizerD_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f"Resumed at epoch {start_epoch}")

    # Fixed noise & labels for tracking sample progression
    fixed_samples_per_class = 4
    fixed_noise = torch.randn(fixed_samples_per_class * num_classes, latent_dim, device=device)
    fixed_labels = torch.tensor(
        [cls_idx for cls_idx in range(num_classes) for _ in range(fixed_samples_per_class)],
        dtype=torch.long,
        device=device,
    )

    print(f"\n--- Starting Conditional DCGAN 256x256 Training on {device} ---")
    print(f"Total training images: {len(dataset)} | Batch size: {batch_size} | Epochs: {start_epoch} -> {epochs}")
    print(f"Output directory: {v2_dir}")

    for epoch in range(start_epoch, epochs + 1):
        netD.train()
        netG.train()
        running_d_loss = 0.0
        running_g_loss = 0.0
        batches = 0

        for real_imgs, real_labels, _ in loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Label smoothing for real targets
            real_targets = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
            fake_targets = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

            # 1. Update Discriminator
            optimizerD.zero_grad()
            d_real_logits = netD(real_imgs, real_labels)
            d_loss_real = criterion(d_real_logits, real_targets)

            noise = torch.randn(b_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = netG(noise, fake_labels)

            d_fake_logits = netD(fake_imgs.detach(), fake_labels)
            d_loss_fake = criterion(d_fake_logits, fake_targets)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()

            # 2. Update Generator
            optimizerG.zero_grad()
            g_targets = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            d_fake_logits_for_g = netD(fake_imgs, fake_labels)
            g_loss = criterion(d_fake_logits_for_g, g_targets)

            g_loss.backward()
            optimizerG.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            batches += 1

        avg_d_loss = running_d_loss / max(1, batches)
        avg_g_loss = running_g_loss / max(1, batches)
        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)

        print(f"Epoch [{epoch:03d}/{epochs:03d}]  D Loss: {avg_d_loss:.4f}  G Loss: {avg_g_loss:.4f}")

        # Save sample grid periodically
        if epoch % save_interval == 0 or epoch == epochs:
            netG.eval()
            with torch.no_grad():
                gen_samples = netG(fixed_noise, fixed_labels)
                gen_samples = (gen_samples * 0.5) + 0.5
                sample_file = v2_dir / f"cgan256_epoch_{epoch:03d}.png"
                save_image(gen_samples, sample_file, nrow=fixed_samples_per_class, normalize=False)

        # Save checkpoint each epoch
        save_checkpoint(
            state={
                "epoch": epoch,
                "generator_state_dict": netG.state_dict(),
                "discriminator_state_dict": netD.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
                "history": history,
                "config": config["gan"],
                "img_size": img_size,
            },
            filepath=ckpt_path,
        )

    # Plot & save loss curve
    plot_loss_curve(history, v2_dir / "gan256_loss_curve.png")
    print(f"\n[OK] GAN256 Training Complete! Checkpoint saved to: {ckpt_path}")
    return history


def plot_loss_curve(history: Dict[str, List[float]], save_path: Path) -> None:
    """Generate and save GAN256 training loss curve."""
    plt.figure(figsize=(9, 5))
    epochs_range = range(1, len(history["d_loss"]) + 1)
    plt.plot(epochs_range, history["d_loss"], label="Discriminator Loss", color="#e74c3c", lw=2)
    plt.plot(epochs_range, history["g_loss"], label="Generator Loss", color="#3498db", lw=2)
    plt.title("Conditional DCGAN 256x256 Training Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (BCEWithLogits)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_cnn_history(history: Dict[str, Any], exp_name: str, save_path: Path) -> None:
    """Plot and save CNN training & validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = history["epochs"]
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#2980b9", lw=2, linestyle="--")
    ax1.set_title(f"{exp_name.title()} - Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#27ae60", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#8e44ad", lw=2, linestyle="--")
    ax2.set_title(f"{exp_name.title()} - Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_synthetic_256(num_per_class: int = 100, checkpoint_path: Optional[Path] = None) -> Dict[str, int]:
    """
    Generate 100 synthetic chest X-ray images per class using the trained 256x256 cDCGAN.
    Saves to outputs/v2/res256/synthetic/<class>/.
    """
    config_path = project_root / "configs" / "gan256.yaml"
    config = load_config(str(config_path))
    root = get_project_root()
    device = get_device(config)

    seed = config["project"]["random_seed"]
    set_seed(seed)

    if checkpoint_path is None:
        checkpoint_path = root / config["v2_res256"]["gan_checkpoint"]
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained GAN256 checkpoint not found at: {checkpoint_path}\n"
            "Please run training first: python -m scripts.train_gan256"
        )

    target_dir = root / config["v2_res256"]["synthetic_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    latent_dim = config["gan"]["latent_dim"]
    embedding_dim = config["gan"]["embedding_dim"]
    num_classes = config["dataset"]["num_classes"]
    img_size = config["dataset"]["image_size"]

    # Load Generator
    netG = ConditionalGenerator256(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        feature_maps=config["gan"]["gen_feature_maps"],
    ).to(device)

    ckpt = load_checkpoint(checkpoint_path, device=device)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()

    counts = {}
    print(f"\nGenerating {num_per_class} synthetic images per class (256x256) using {checkpoint_path.name}...")

    with torch.no_grad():
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = target_dir / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            labels = torch.full((num_per_class,), class_idx, dtype=torch.long, device=device)
            noise = torch.randn(num_per_class, latent_dim, device=device)

            synth_batch = netG(noise, labels)
            synth_batch = (synth_batch * 0.5) + 0.5
            synth_batch = torch.clamp(synth_batch, 0.0, 1.0)

            valid_count = 0
            for i in range(num_per_class):
                img_tensor = synth_batch[i]
                std_val = float(img_tensor.std())
                if std_val < 1e-4:
                    print(f"Warning: Low variance for synthetic sample {i} of class {class_name}")

                out_path = class_folder / f"syn256_{class_name}_{i+1:04d}.png"
                save_image(img_tensor, out_path)
                valid_count += 1

            counts[class_name] = valid_count
            print(f"  [OK] Saved {valid_count} synthetic 256x256 images for class '{class_name}' -> {class_folder}")

    print(f"\n[OK] Synthetic 256x256 generation finished! Total: {sum(counts.values())} images")
    return counts


def train_classifier_256(use_synthetic: bool = True) -> Dict:
    """
    Train ResNet-18 classifier on 256x256 inputs.
    Protocol identical to baseline: ResNet18, 15 epochs max, patience 5.
    """
    config_path = project_root / "configs" / "gan256.yaml"
    config = load_config(str(config_path))
    root = get_project_root()
    device = get_device(config)

    v2_dir = root / config["v2_res256"]["output_dir"]
    img_size = config["dataset"]["image_size"]  # 256
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    epochs = config["cnn"]["epochs"]
    batch_size = config["cnn"]["batch_size"]
    lr = config["cnn"]["learning_rate"]
    arch = config["cnn"]["architecture"]
    patience = config["cnn"]["early_stopping_patience"]

    set_seed(config["project"]["random_seed"])

    # Paths
    orig_dir = root / config["dataset"]["original_dir"]
    synth_dir_path = root / config["v2_res256"]["synthetic_dir"] if use_synthetic else None
    ckpt_path = root / config["v2_res256"]["classifier_checkpoint"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = LiverLesionDataset(
        root_dir=orig_dir,
        split="train",
        synthetic_dir=synth_dir_path,
        transform=get_transforms("train", img_size, mean, std),
    )

    val_dataset = LiverLesionDataset(
        root_dir=orig_dir,
        split="val",
        synthetic_dir=None,
        transform=get_transforms("val", img_size, mean, std),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training data found in {orig_dir / 'train'}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation data found in {orig_dir / 'val'}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config["hardware"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=config["hardware"]["num_workers"])

    # Model: ResNet18 for 256x256
    model = build_classifier(
        architecture=arch,
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["cnn"]["pretrained"],
        freeze_backbone=config["cnn"]["freeze_backbone"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["cnn"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "epochs": [], "training_time_seconds": 0.0,
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    exp_name = "cnn256_augmented" if use_synthetic else "cnn256_baseline"
    print(f"\n=======================================================")
    print(f" Starting Experiment: {exp_name.upper()} (256x256 inputs)")
    print(f" Backbone: {arch} | Device: {device}")
    print(f" Train Samples: {len(train_dataset)} (Synthetic: {use_synthetic})")
    print(f" Validation Samples: {len(val_dataset)} (Real untouched)")
    print(f"=======================================================")

    import time
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += imgs.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += imgs.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        scheduler.step(epoch_val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "architecture": arch,
                    "experiment_name": exp_name,
                    "use_synthetic": use_synthetic,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "config": config["cnn"],
                    "img_size": img_size,
                },
                filepath=ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    total_time = time.time() - start_time
    history["training_time_seconds"] = round(total_time, 2)

    # Save learning curve
    plot_cnn_history(history, exp_name, v2_dir / f"{exp_name}_learning_curve.png")

    print(f"\n[OK] CNN256 Training finished in {total_time:.2f}s! Best Val Acc: {best_val_acc * 100:.2f}% at epoch {best_epoch}")
    print(f"Saved best model checkpoint to: {ckpt_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(ckpt_path),
        "history": history,
        "img_size": img_size,
    }


def evaluate_256(model_checkpoint: Path, img_size: int = 256) -> Dict:
    """
    Evaluate the 256x256 classifier on the held-out test set (resized to 256x256).
    """
    config_path = project_root / "configs" / "gan256.yaml"
    config = load_config(str(config_path))
    root = get_project_root()
    device = get_device(config)

    from src.evaluation.metrics import evaluate_model_on_dataset

    test_dir = root / config["dataset"]["original_dir"] / "test"
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    test_dataset = LiverLesionDataset(
        root_dir=root / config["dataset"]["original_dir"],
        split="test",
        synthetic_dir=None,
        transform=get_transforms("test", img_size, mean, std),
    )

    test_loader = DataLoader(test_dataset, batch_size=config["cnn"]["batch_size"],
                             shuffle=False, num_workers=config["hardware"]["num_workers"])

    model = build_classifier(
        architecture=config["cnn"]["architecture"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=False,
    ).to(device)

    ckpt = load_checkpoint(model_checkpoint, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics, y_true, y_pred, y_probs = evaluate_model_on_dataset(model, test_loader, device)
    return metrics


def compute_fid_256_vs_128() -> Dict[str, float]:
    """
    Compute FID between 256x256 cDCGAN samples and 128x128 cDCGAN samples.
    Uses InceptionV3 features resized appropriately.
    """
    try:
        import torchvision.models as models
        import torch.nn.functional as F
        from torchvision import transforms as T
        from scipy import linalg
        import numpy as np
    except ImportError as e:
        print(f"Warning: Cannot compute FID — missing dependency: {e}")
        return {"fid_256_vs_128": None, "error": str(e)}

    config_path = project_root / "configs" / "gan256.yaml"
    config = load_config(str(config_path))
    root = get_project_root()
    device = get_device(config)

    # Load InceptionV3 for feature extraction
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
    inception = inception.to(device)
    inception.eval()
    # Remove final classification layer
    inception.fc = torch.nn.Identity()

    # Preprocessing for Inception (expects 299x299, normalized to [0,1] then ImageNet normalize)
    inception_preprocess = T.Compose([
        T.Resize((299, 299)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(images_tensor):
        """images_tensor: (B, 3, H, W) in [0, 1]"""
        features = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(images_tensor), batch_size):
                batch = images_tensor[i:i + batch_size].to(device)
                batch = inception_preprocess(batch)
                feat = inception(batch)
                features.append(feat.cpu().numpy())
        return np.vstack(features)

    def calculate_fid(features1, features2):
        mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            covmean = np.zeros_like(sigma1)
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(np.real(fid))

    # Collect synthetic 256x256 images
    synth_256_dir = root / config["v2_res256"]["synthetic_dir"]
    synth_128_dir = root / config["dataset"]["synthetic_dir"] / "train"

    def load_images_from_dir(img_dir, img_size):
        from PIL import Image
        images = []
        if not img_dir.exists():
            print(f"Warning: Directory not found: {img_dir}")
            return torch.zeros(0, 3, img_size, img_size)
        for class_folder in sorted(img_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            for img_path in sorted(class_folder.iterdir()):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize((img_size, img_size), Image.Resampling.BICUBIC)
                        img_tensor = T.ToTensor()(img)  # [0, 1]
                        images.append(img_tensor)
                    except Exception:
                        continue
        if not images:
            return torch.zeros(0, 3, img_size, img_size)
        return torch.stack(images)

    print("\n--- Computing FID: 256x256 cDCGAN vs 128x128 cDCGAN ---")
    imgs_256 = load_images_from_dir(synth_256_dir, 256)
    imgs_128 = load_images_from_dir(synth_128_dir, 128)

    if len(imgs_256) == 0 or len(imgs_128) == 0:
        print("Warning: Not enough images to compute FID.")
        return {"fid_256_vs_128": None}

    # Resize both to 299x299 for Inception feature extraction
    fid_score = calculate_fid(
        extract_features(imgs_256[:200]),
        extract_features(imgs_128[:200]),
    )
    print(f"[OK] FID (256x256 cDCGAN vs 128x128 cDCGAN): {fid_score:.4f}")
    return {"fid_256_vs_128": round(fid_score, 4)}


def main():
    parser = argparse.ArgumentParser(description="Train 256x256 Conditional DCGAN (Phase 1).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of GAN training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=None, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=None, help="Discriminator learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip-train", action="store_true", help="Skip GAN training, use existing checkpoint")
    parser.add_argument("--skip-generate", action="store_true", help="Skip synthetic generation")
    parser.add_argument("--skip-classifier", action="store_true", help="Skip classifier training")
    parser.add_argument("--skip-fid", action="store_true", help="Skip FID computation")
    args = parser.parse_args()

    config_path = project_root / "configs" / "gan256.yaml"
    config = load_config(str(config_path))
    root = get_project_root()
    v2_dir = root / config["v2_res256"]["output_dir"]
    v2_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Phase 1a: Train GAN256
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("PHASE 1A: Training 256x256 Conditional DCGAN")
        print("=" * 60)
        history = train_cgan256(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            resume=args.resume,
        )
        results["gan_history"] = history

    # Phase 1b: Generate synthetic images
    if not args.skip_generate:
        print("\n" + "=" * 60)
        print("PHASE 1B: Generating 100 synthetic images per class (256x256)")
        print("=" * 60)
        counts = generate_synthetic_256(num_per_class=100)
        results["synthetic_counts"] = counts

    # Phase 1c: Train classifier on 256x256
    if not args.skip_classifier:
        print("\n" + "=" * 60)
        print("PHASE 1C: Training ResNet-18 classifier on 256x256 inputs")
        print("=" * 60)
        cnn_results = train_classifier_256(use_synthetic=True)
        results["cnn_training"] = cnn_results

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("PHASE 1D: Evaluating 256x256 classifier on test set")
        print("=" * 60)
        ckpt_path = root / config["v2_res256"]["classifier_checkpoint"]
        img_size = config["dataset"]["image_size"]
        test_metrics = evaluate_256(ckpt_path, img_size=img_size)
        results["test_metrics"] = test_metrics
        print(f"\nTest Results (256x256 inputs):")
        print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}% (Baseline 128x128: 92.75%)")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f} (Baseline 128x128: 0.9277)")

    # Phase 1e: FID computation
    if not args.skip_fid:
        print("\n" + "=" * 60)
        print("PHASE 1E: Computing FID (256x256 cDCGAN vs 128x128 cDCGAN)")
        print("=" * 60)
        fid_results = compute_fid_256_vs_128()
        results["fid"] = fid_results

    # Save summary report
    report = {
        "phase": "1",
        "description": "256x256 Conditional DCGAN",
        "baseline_accuracy_128": 0.9275,
        "baseline_macro_f1_128": 0.9277,
        "results": results,
    }
    report_path = v2_dir / "phase1_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OK] Phase 1 summary saved to: {report_path}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"All outputs in: {v2_dir}")


if __name__ == "__main__":
    main()
