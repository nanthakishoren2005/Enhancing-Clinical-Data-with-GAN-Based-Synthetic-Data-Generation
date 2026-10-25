"""
Conditional DCGAN training script for Chest X-ray Image Generation.
Supports configurable epochs, checkpoint saving, sample grid generation,
loss history plotting, and resuming interrupted runs.
Usage:
    python -m src.gan.train_gan
    python -m src.gan.train_gan --epochs 20 --batch-size 16
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.data.dataset import ChestXrayDataset, get_gan_transforms, CLASS_TO_IDX
from src.gan.generator import ConditionalGenerator
from src.gan.discriminator import ConditionalDiscriminator


def train_cgan(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr_g: Optional[float] = None,
    lr_d: Optional[float] = None,
    resume: bool = False,
    progress_callback: Optional[callable] = None,
) -> Dict[str, List[float]]:
    """
    Train Conditional DCGAN on original training images.
    Returns loss history dictionary.
    """
    config = load_config()
    root = get_project_root()
    device = get_device(config)

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

    # Output paths
    checkpoints_dir = root / config["paths"]["checkpoints_dir"]
    samples_dir = root / config["paths"]["samples_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / "cgan_checkpoint.pt"

    # Dataset & DataLoader (Train split only, no data leakage!)
    data_dir = root / config["dataset"]["original_dir"]
    dataset = ChestXrayDataset(
        root_dir=data_dir,
        split="train",
        synthetic_dir=None,
        transform=get_gan_transforms(img_size),
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training images found in {data_dir / 'train'}. "
            "Please run `python -m src.data.generate_simulated_data` or import real dataset first."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(dataset) > batch_size),
        num_workers=config["hardware"]["num_workers"],
    )

    # Initialize Networks
    netG = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        feature_maps=config["gan"]["gen_feature_maps"],
    ).to(device)

    netD = ConditionalDiscriminator(
        num_classes=num_classes,
        in_channels=3,
        feature_maps=config["gan"]["disc_feature_maps"],
        img_size=img_size,
    ).to(device)

    # Loss & Optimizers (BCEWithLogits for stability)
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))

    start_epoch = 1
    history = {"g_loss": [], "d_loss": []}

    # Resume from checkpoint if requested and exists
    if resume and ckpt_path.exists():
        print(f"Resuming GAN training from {ckpt_path}...")
        ckpt = load_checkpoint(ckpt_path, device=device)
        netG.load_state_dict(ckpt["generator_state_dict"])
        netD.load_state_dict(ckpt["discriminator_state_dict"])
        optimizerG.load_state_dict(ckpt["optimizerG_state_dict"])
        optimizerD.load_state_dict(ckpt["optimizerD_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f"Resumed at epoch {start_epoch}")

    # Fixed noise & labels for tracking visual sample progression across epochs
    fixed_samples_per_class = 4
    fixed_noise = torch.randn(fixed_samples_per_class * num_classes, latent_dim, device=device)
    fixed_labels = torch.tensor(
        [cls_idx for cls_idx in range(num_classes) for _ in range(fixed_samples_per_class)],
        dtype=torch.long,
        device=device,
    )

    print(f"\n--- Starting Conditional DCGAN Training on {device} ---")
    print(f"Total training images: {len(dataset)} | Batch size: {batch_size} | Epochs: {start_epoch} -> {epochs}")

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

            # Label smoothing for real targets (0.9 instead of 1.0)
            real_targets = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
            fake_targets = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

            # -----------------------------------------------------------------
            # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # -----------------------------------------------------------------
            optimizerD.zero_grad()

            # Real batch forward
            d_real_logits = netD(real_imgs, real_labels)
            d_loss_real = criterion(d_real_logits, real_targets)

            # Fake batch forward
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = netG(noise, fake_labels)

            d_fake_logits = netD(fake_imgs.detach(), fake_labels)
            d_loss_fake = criterion(d_fake_logits, fake_targets)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()

            # -----------------------------------------------------------------
            # 2. Update Generator: maximize log(D(G(z)))
            # -----------------------------------------------------------------
            optimizerG.zero_grad()
            # Generator wants discriminator to believe fakes are real
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

        if progress_callback:
            progress_callback(epoch, epochs, avg_d_loss, avg_g_loss)

        # Periodically save sample grid
        if epoch % save_interval == 0 or epoch == epochs:
            netG.eval()
            with torch.no_grad():
                gen_samples = netG(fixed_noise, fixed_labels)
                # Rescale from [-1, 1] to [0, 1]
                gen_samples = (gen_samples * 0.5) + 0.5
                sample_file = samples_dir / f"cgan_epoch_{epoch:03d}.png"
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
            },
            filepath=ckpt_path,
        )

    # Plot & save loss curve
    plot_loss_curve(history, figures_dir / "gan_loss_curve.png")
    print(f"\n[OK] GAN Training Complete! Checkpoint saved to: {ckpt_path}")
    return history


def plot_loss_curve(history: Dict[str, List[float]], save_path: Path) -> None:
    """Generate and save GAN training loss progression curve."""
    plt.figure(figsize=(9, 5))
    epochs = range(1, len(history["d_loss"]) + 1)
    plt.plot(epochs, history["d_loss"], label="Discriminator Loss", color="#e74c3c", lw=2)
    plt.plot(epochs, history["g_loss"], label="Generator Loss", color="#3498db", lw=2)
    plt.title("Conditional DCGAN Training Loss Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (BCEWithLogits)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Conditional DCGAN on chest X-ray images.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=None, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=None, help="Discriminator learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    train_cgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
