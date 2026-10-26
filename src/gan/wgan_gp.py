"""
Conditional WGAN-GP (Wasserstein GAN with Gradient Penalty) trainer.

Reuses the existing ConditionalGenerator / ConditionalDiscriminator designs
from src.gan.generator and src.gan.discriminator (those files are NOT edited).
The critic returns unnormalised scalar scores (no sigmoid), trained with the
WGAN-GP objective:

    critic loss   = -mean(D(real)) + mean(D(fake)) + lambda_gp * gradient_penalty
    generator loss= -mean(D(fake))

Gradient penalty (Gulrajani et al., 2017) enforces the 1-Lipschitz constraint
on interpolations between real and fake images.

Usage:
    python -m src.gan.wgan_gp --epochs 2 --batch-size 16 --limit 40   # smoke test
    python scripts/train_wgan_gp.py                                    # full run
"""
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint
from src.data.dataset import LiverLesionDataset, get_gan_transforms
from src.gan.generator import ConditionalGenerator, weights_init  # noqa: F401 (reused design)
from src.gan.discriminator import ConditionalDiscriminator


def gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    lambda_gp: float,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty on random interpolations between real and fake batches.
    """
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_images + (1.0 - epsilon) * fake_images
    interpolated.requires_grad_(True)

    scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norms = gradients.norm(2, dim=1)
    return ((grad_norms - 1.0) ** 2).mean() * lambda_gp


def train_wgan_gp(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr_g: Optional[float] = None,
    lr_d: Optional[float] = None,
    n_critic: Optional[int] = None,
    lambda_gp: Optional[float] = None,
    latent_dim: Optional[int] = None,
    resume: bool = False,
    limit: Optional[int] = None,
    config_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, List[float]]:
    """
    Train a Conditional WGAN-GP on original training images only.

    Returns loss history dict. Saves:
      * checkpoints/wgan_gp_checkpoint.pt            (every epoch)
      * outputs/v2/wgan_gp_loss_log.csv              (per-epoch loss log)
      * outputs/v2/wgan_gp_samples/wgan_gp_epoch_*.png (sample grids)
    """
    config = load_config(config_path)
    root = get_project_root()
    device = get_device(config)

    # --- Hyperparameters (CLI overrides config) ------------------------------
    epochs = epochs or config["gan"]["epochs"]
    batch_size = batch_size or config["gan"]["batch_size"]
    lr_g = lr_g or config["gan"]["lr_g"]
    lr_d = lr_d or config["gan"]["lr_d"]
    beta1 = config["gan"]["beta1"]
    beta2 = config["gan"]["beta2"]
    n_critic = n_critic if n_critic is not None else config["gan"]["n_critic"]
    lambda_gp = lambda_gp if lambda_gp is not None else config["gan"]["lambda_gp"]
    latent_dim = latent_dim or config["gan"]["latent_dim"]
    embedding_dim = config["gan"]["embedding_dim"]
    num_classes = config["dataset"]["num_classes"]
    img_size = config["dataset"]["image_size"]
    save_interval = config["gan"]["save_sample_interval"]

    set_seed(config["project"]["random_seed"])

    # --- Output paths --------------------------------------------------------
    checkpoints_dir = root / config["paths"]["checkpoints_dir"]
    outputs_v2 = root / "outputs" / "v2"
    samples_dir = outputs_v2 / "wgan_gp_samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    outputs_v2.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / "wgan_gp_checkpoint.pt"
    loss_log_path = outputs_v2 / "wgan_gp_loss_log.csv"

    # --- Dataset (train split only, no data leakage) -------------------------
    data_dir = root / config["dataset"]["original_dir"]
    dataset = LiverLesionDataset(
        root_dir=data_dir,
        split="train",
        synthetic_dir=None,
        transform=get_gan_transforms(img_size),
    )

    if limit is not None and len(dataset) > limit:
        # Deterministic subset: keep class balance by capping per class
        per_class_limit = max(1, limit // num_classes)
        class_indices: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
        for idx, (_, label_idx, _) in enumerate(dataset.samples):
            class_indices[label_idx].append(idx)
        subset_indices = []
        for i in range(num_classes):
            subset_indices.extend(class_indices[i][:per_class_limit])
        dataset = Subset(dataset, sorted(subset_indices))

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training images found in {data_dir / 'train'}."
        )

    drop_last = len(dataset) > batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=config["hardware"]["num_workers"],
    )

    # --- Networks (reuses existing cDCGAN generator/discriminator designs) --
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

    # WGAN-GP standard: Adam with beta1=0.0
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))

    start_epoch = 1
    history = {"d_loss": [], "g_loss": [], "gp": [], "wasserstein": []}

    # Resume support
    if resume and ckpt_path.exists():
        print(f"Resuming WGAN-GP training from {ckpt_path}...")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["generator_state_dict"])
        netD.load_state_dict(ckpt["critic_state_dict"])
        optimizerG.load_state_dict(ckpt["optimizerG_state_dict"])
        optimizerD.load_state_dict(ckpt["optimizerD_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f"Resumed at epoch {start_epoch}")

    # Fixed noise & labels for sample tracking
    fixed_samples_per_class = 4
    fixed_noise = torch.randn(fixed_samples_per_class * num_classes, latent_dim, device=device)
    fixed_labels = torch.tensor(
        [cls_idx for cls_idx in range(num_classes) for _ in range(fixed_samples_per_class)],
        dtype=torch.long,
        device=device,
    )

    print(f"\n--- Starting Conditional WGAN-GP Training on {device} ---")
    print(f"Training images: {len(dataset)} | Batch size: {batch_size} | Epochs: {start_epoch} -> {epochs}")
    print(f"n_critic: {n_critic} | lambda_gp: {lambda_gp} | lr_g: {lr_g} | lr_d: {lr_d}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Loss log:   {loss_log_path}")

    # Initialise CSV loss log
    log_file_exists = loss_log_path.exists()
    log_file = open(loss_log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if not log_file_exists:
        log_writer.writerow(["epoch", "d_loss", "g_loss", "gradient_penalty", "wasserstein_distance"])

    try:
        for epoch in range(start_epoch, epochs + 1):
            netG.train()
            netD.train()
            running_d = 0.0
            running_g = 0.0
            running_gp = 0.0
            running_w = 0.0
            batches = 0

            for real_imgs, real_labels, _ in loader:
                b_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                real_labels = real_labels.to(device)

                # ---------------------------------------------------------
                # 1. Update the critic n_critic times
                # ---------------------------------------------------------
                for _ in range(n_critic):
                    optimizerD.zero_grad()

                    d_real = netD(real_imgs, real_labels)

                    noise = torch.randn(b_size, latent_dim, device=device)
                    fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
                    fake_imgs = netG(noise, fake_labels).detach()
                    d_fake = netD(fake_imgs, fake_labels)

                    gp = gradient_penalty(netD, real_imgs, fake_imgs, fake_labels, device, lambda_gp)
                    d_loss = (-d_real.mean() + d_fake.mean() + gp)
                    d_loss.backward()
                    optimizerD.step()

                # ---------------------------------------------------------
                # 2. Update the generator once
                # ---------------------------------------------------------
                optimizerG.zero_grad()
                noise = torch.randn(b_size, latent_dim, device=device)
                gen_labels = torch.randint(0, num_classes, (b_size,), device=device)
                fake_imgs = netG(noise, gen_labels)
                d_fake_for_g = netD(fake_imgs, gen_labels)
                g_loss = -d_fake_for_g.mean()
                g_loss.backward()
                optimizerG.step()

                with torch.no_grad():
                    wasserstein = d_real.mean() - d_fake.mean()

                running_d += d_loss.item()
                running_g += g_loss.item()
                running_gp += gp.item()
                running_w += wasserstein.item()
                batches += 1

            avg_d = running_d / max(1, batches)
            avg_g = running_g / max(1, batches)
            avg_gp = running_gp / max(1, batches)
            avg_w = running_w / max(1, batches)

            history["d_loss"].append(avg_d)
            history["g_loss"].append(avg_g)
            history["gp"].append(avg_gp)
            history["wasserstein"].append(avg_w)

            log_writer.writerow([epoch, round(avg_d, 6), round(avg_g, 6),
                                 round(avg_gp, 6), round(avg_w, 6)])
            log_file.flush()

            print(f"Epoch [{epoch:03d}/{epochs:03d}]  D loss: {avg_d:.4f}  "
                  f"G loss: {avg_g:.4f}  GP: {avg_gp:.4f}  W: {avg_w:.4f}")

            if progress_callback:
                progress_callback(epoch, epochs, avg_d, avg_g)

            # Periodic sample grid
            if epoch % save_interval == 0 or epoch == epochs:
                netG.eval()
                with torch.no_grad():
                    gen_samples = netG(fixed_noise, fixed_labels)
                    gen_samples = (gen_samples * 0.5) + 0.5
                    gen_samples = torch.clamp(gen_samples, 0.0, 1.0)
                    save_image(gen_samples,
                               samples_dir / f"wgan_gp_epoch_{epoch:03d}.png",
                               nrow=fixed_samples_per_class, normalize=False)

            # Checkpoint every epoch
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "generator_state_dict": netG.state_dict(),
                    "critic_state_dict": netD.state_dict(),
                    "optimizerG_state_dict": optimizerG.state_dict(),
                    "optimizerD_state_dict": optimizerD.state_dict(),
                    "history": history,
                    "config": {
                        "gan": config["gan"],
                        "latent_dim": latent_dim,
                        "n_critic": n_critic,
                        "lambda_gp": lambda_gp,
                    },
                },
                filepath=ckpt_path,
            )

        # --- Loss curve ------------------------------------------------------
        _plot_loss_curve(history, outputs_v2 / "wgan_gp_loss_curve.png")
        print(f"\n[OK] WGAN-GP Training Complete!")
        print(f"Checkpoint saved to: {ckpt_path}")
        print(f"Loss log saved to:   {loss_log_path}")
        return history
    finally:
        log_file.close()


def _plot_loss_curve(history: Dict[str, List[float]], save_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    epochs = range(1, len(history["d_loss"]) + 1)
    plt.plot(epochs, history["d_loss"], label="Critic Loss", color="#e74c3c", lw=2)
    plt.plot(epochs, history["g_loss"], label="Generator Loss", color="#3498db", lw=2)
    plt.title("Conditional WGAN-GP Training Loss Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Conditional WGAN-GP on CT images.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to WGAN-GP config YAML (default: configs/wgan_gp.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=None, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=None, help="Critic learning rate")
    parser.add_argument("--n-critic", type=int, default=None, help="Critic updates per generator update")
    parser.add_argument("--lambda-gp", type=float, default=None, help="Gradient penalty coefficient")
    parser.add_argument("--latent-dim", type=int, default=None, help="Latent vector dimension")
    parser.add_argument("--resume", action="store_true", help="Resume from wgan_gp_checkpoint.pt")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit training set size (deterministic, class-balanced subset). Smoke-test aid.")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = "configs/wgan_gp.yaml"

    train_wgan_gp(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        latent_dim=args.latent_dim,
        resume=args.resume,
        limit=args.limit,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
