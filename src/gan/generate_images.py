"""
Synthetic image generation module using trained Conditional DCGAN.
Synthesizes targeted chest X-ray patches and saves them into data/synthetic/train/<class>.
Usage:
    python -m src.gan.generate_images --num-per-class 40
"""
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from torchvision.utils import save_image

from src.utils.config import load_config, get_project_root, get_device
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import load_checkpoint
from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS
from src.gan.generator import ConditionalGenerator


def generate_synthetic_dataset(
    num_per_class: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    target_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Generate synthetic chest X-ray images for all configured classes using the trained Conditional DCGAN.
    """
    config = load_config()
    root = get_project_root()
    device = get_device(config)

    num_per_class = num_per_class or config["gan"]["num_synthetic_per_class"]
    seed = seed if seed is not None else config["project"]["random_seed"]
    set_seed(seed)

    if checkpoint_path is None:
        checkpoint_path = root / config["paths"]["checkpoints_dir"] / "cgan_checkpoint.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained GAN checkpoint not found at: {checkpoint_path}\n"
            "GAN training is required before synthetic images can be generated.\n"
            "Please run: python -m src.gan.train_gan"
        )

    if target_dir is None:
        target_dir = root / config["dataset"]["synthetic_dir"] / "train"
    else:
        target_dir = Path(target_dir)

    # Initialize and load Generator
    latent_dim = config["gan"]["latent_dim"]
    embedding_dim = config["gan"]["embedding_dim"]
    num_classes = config["dataset"]["num_classes"]

    netG = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        feature_maps=config["gan"]["gen_feature_maps"],
    ).to(device)

    ckpt = load_checkpoint(checkpoint_path, device=device)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()

    counts = {}
    print(f"Generating {num_per_class} synthetic images per class using {checkpoint_path.name}...")

    with torch.no_grad():
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = target_dir / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            labels = torch.full((num_per_class,), class_idx, dtype=torch.long, device=device)
            noise = torch.randn(num_per_class, latent_dim, device=device)

            # Generate batch
            synth_batch = netG(noise, labels)
            # Rescale from [-1, 1] to [0, 1]
            synth_batch = (synth_batch * 0.5) + 0.5
            synth_batch = torch.clamp(synth_batch, 0.0, 1.0)

            valid_count = 0
            for i in range(num_per_class):
                img_tensor = synth_batch[i]
                # Basic sanity check: ensure image is not totally blank / collapsed
                std_val = float(img_tensor.std())
                if std_val < 1e-4:
                    print(f"Warning: Low variance detected for synthetic sample {i} of class {class_name}")

                out_path = class_folder / f"syn_{class_name}_{i+1:04d}.png"
                save_image(img_tensor, out_path)
                valid_count += 1

            counts[class_name] = valid_count
            print(f"  [OK] Saved {valid_count} synthetic images for class '{class_name}' -> {class_folder}")

    print("\n[OK] Synthetic generation finished successfully!")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic chest X-ray images using trained GAN.")
    parser.add_argument("--num-per-class", type=int, default=None, help="Target number of synthetic images per class")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to GAN checkpoint")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    generate_synthetic_dataset(num_per_class=args.num_per_class, checkpoint_path=ckpt)


if __name__ == "__main__":
    main()
