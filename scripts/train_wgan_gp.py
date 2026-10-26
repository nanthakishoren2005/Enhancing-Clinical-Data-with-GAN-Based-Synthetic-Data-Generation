"""
Entry point for training the Conditional WGAN-GP variant.
Delegates to src.gan.wgan_gp.train_wgan_gp.

Usage:
    python scripts/train_wgan_gp.py --epochs 2 --batch-size 16 --limit 40   # smoke test
    python scripts/train_wgan_gp.py                                          # full run (50 epochs)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gan.wgan_gp import train_wgan_gp


def main():
    parser = argparse.ArgumentParser(description="Train Conditional WGAN-GP (new files only; cDCGAN files untouched).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to WGAN-GP config YAML (default: configs/wgan_gp.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=None, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=None, help="Critic learning rate")
    parser.add_argument("--n-critic", type=int, default=None, help="Critic updates per generator update")
    parser.add_argument("--lambda-gp", type=float, default=None, help="Gradient penalty coefficient")
    parser.add_argument("--latent-dim", type=int, default=None, help="Latent vector dimension")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/wgan_gp_checkpoint.pt")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit training set size (deterministic, class-balanced subset). Smoke-test aid.")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = "configs/wgan_gp.yaml"

    train_wgan_gp(
        config_path=config_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        latent_dim=args.latent_dim,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
