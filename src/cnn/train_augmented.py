"""
Train Model B: GAN-Augmented CNN Classifier.
Trains on original real chest X-ray images + GAN-generated synthetic training images.
Evaluates model selection against the exact same untouched real validation set.
Usage:
    python -m src.cnn.train_augmented
    python -m src.cnn.train_augmented --epochs 15 --batch-size 16 --architecture resnet18
"""
import argparse
from pathlib import Path
from src.utils.config import load_config, get_project_root
from src.data.dataset import CLASS_TO_IDX
from src.cnn.trainer import run_cnn_training


def check_synthetic_data_availability() -> int:
    """Verify that synthetic data folder contains generated images."""
    config = load_config()
    root = get_project_root()
    synth_dir = root / config["dataset"]["synthetic_dir"] / "train"

    if not synth_dir.exists():
        raise FileNotFoundError(
            f"Synthetic data directory not found at: {synth_dir}\n"
            "GAN training and synthetic generation must be completed before running GAN-augmented training.\n"
            "Please run:\n"
            "  1. python -m src.gan.train_gan\n"
            "  2. python -m src.gan.generate_images"
        )

    total_synth = 0
    for cname in CLASS_TO_IDX:
        cdir = synth_dir / cname
        if cdir.exists():
            total_synth += len([f for f in cdir.iterdir() if f.is_file()])

    if total_synth == 0:
        raise RuntimeError(
            f"No synthetic images found in {synth_dir}.\n"
            "Please generate synthetic images first using: python -m src.gan.generate_images"
        )

    return total_synth


def main():
    parser = argparse.ArgumentParser(description="Train GAN-Augmented CNN Classifier.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--architecture", type=str, default=None, help="CNN architecture: resnet18 or custom_cnn")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    args = parser.parse_args()

    # Pre-flight check
    total_synth = check_synthetic_data_availability()
    print(f"[OK] Found {total_synth} synthetic images available for GAN augmentation.")

    results = run_cnn_training(
        use_synthetic=True,
        experiment_name="augmented",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        architecture=args.architecture,
        patience=args.patience,
    )
    return results


if __name__ == "__main__":
    main()
