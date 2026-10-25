"""
Train Model A: Baseline CNN Classifier.
Trains on original real chest X-ray images only with standard data augmentation.
Usage:
    python -m src.cnn.train_baseline
    python -m src.cnn.train_baseline --epochs 15 --batch-size 16 --architecture resnet18
"""
import argparse
from src.cnn.trainer import run_cnn_training


def main():
    parser = argparse.ArgumentParser(description="Train Baseline CNN Classifier (Original data + standard augmentation).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--architecture", type=str, default=None, help="CNN architecture: resnet18 or custom_cnn")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    args = parser.parse_args()

    results = run_cnn_training(
        use_synthetic=False,
        experiment_name="baseline",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        architecture=args.architecture,
        patience=args.patience,
    )
    return results


if __name__ == "__main__":
    main()
