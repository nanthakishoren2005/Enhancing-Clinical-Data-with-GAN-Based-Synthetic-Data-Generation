"""
Generic training and validation engine for CNN Chest X-ray classifiers.
Ensures rigorous experimental control between Baseline and GAN-Augmented pipelines.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint
from src.data.dataset import ChestXrayDataset, get_transforms
from src.cnn.model import build_classifier


def run_cnn_training(
    use_synthetic: bool = False,
    experiment_name: str = "baseline",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    architecture: Optional[str] = None,
    patience: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Train a CNN classifier (Baseline or GAN-Augmented).
    - Validation set strictly consists of untouched real chest X-ray images.
    - Early stopping monitored on validation loss.
    - Saves best checkpoint and training history.
    """
    config = load_config()
    root = get_project_root()
    device = get_device(config)

    # Hyperparameters
    epochs = epochs or config["cnn"]["epochs"]
    batch_size = batch_size or config["cnn"]["batch_size"]
    lr = lr or config["cnn"]["learning_rate"]
    arch = architecture or config["cnn"]["architecture"]
    patience = patience or config["cnn"]["early_stopping_patience"]
    weight_decay = config["cnn"]["weight_decay"]
    img_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    set_seed(config["project"]["random_seed"])

    # File paths
    orig_dir = root / config["dataset"]["original_dir"]
    synth_dir = root / config["dataset"]["synthetic_dir"] if use_synthetic else None
    checkpoints_dir = root / config["paths"]["checkpoints_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / f"{experiment_name}_best.pt"

    # Datasets
    # 1. Training set: Real only (if baseline) OR Real + Synthetic (if augmented)
    train_dataset = ChestXrayDataset(
        root_dir=orig_dir,
        split="train",
        synthetic_dir=synth_dir,
        transform=get_transforms("train", img_size, mean, std),
    )

    # 2. Validation set: ALWAYS real untouched images only
    val_dataset = ChestXrayDataset(
        root_dir=orig_dir,
        split="val",
        synthetic_dir=None,
        transform=get_transforms("val", img_size, mean, std),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training data found in {orig_dir / 'train'}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation data found in {orig_dir / 'val'}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
    )

    # Initialize Model
    model = build_classifier(
        architecture=arch,
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["cnn"]["pretrained"],
        freeze_backbone=config["cnn"]["freeze_backbone"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": [],
        "training_time_seconds": 0.0,
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n=======================================================")
    print(f" Starting Experiment: {experiment_name.upper()}")
    print(f" Backbone: {arch} | Device: {device}")
    print(f" Train Samples: {len(train_dataset)} (Synthetic: {use_synthetic})")
    print(f" Validation Samples: {len(val_dataset)} (Real untouched)")
    print(f"=======================================================")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
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

        # --- Validation Phase ---
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

        # Update History
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

        if progress_callback:
            progress_callback(epoch, epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)

        # Checkpoint Saving & Early Stopping
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
                    "experiment_name": experiment_name,
                    "use_synthetic": use_synthetic,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "config": config["cnn"],
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

    # Plot & save learning curves
    plot_curves_path = figures_dir / f"{experiment_name}_learning_curve.png"
    plot_cnn_history(history, experiment_name, plot_curves_path)

    print(f"\n[OK] Training finished in {total_time:.2f}s! Best Val Acc: {best_val_acc * 100:.2f}% at epoch {best_epoch}")
    print(f"Saved best model checkpoint to: {ckpt_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }


def plot_cnn_history(history: Dict[str, Any], exp_name: str, save_path: Path) -> None:
    """Plot and save training & validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = history["epochs"]
    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#2980b9", lw=2, linestyle="--")
    ax1.set_title(f"{exp_name.title()} - Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy curve
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
