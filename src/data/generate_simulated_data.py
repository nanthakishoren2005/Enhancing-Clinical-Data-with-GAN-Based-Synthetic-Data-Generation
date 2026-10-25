"""
Synthetic Benchmark Chest X-ray Patch Generator for Software Smoke Testing and Academic Demonstration.
IMPORTANT DISCLAIMER:
These images are mathematically generated synthetic chest X-ray-like region-of-interest (ROI) patches
created strictly for software verification, unit testing, and workflow demonstration.
They are NOT clinical patient scans and MUST NOT be presented or used as clinical diagnostic data.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from src.utils.config import load_config, get_project_root
from src.utils.reproducibility import set_seed
from src.data.dataset import CLASS_TO_IDX


def generate_cxr_parenchyma(size: int = 128, base_val: float = 150.0, noise_std: float = 6.0) -> np.ndarray:
    """Simulate healthy background lung parenchyma with detector quantum noise."""
    parenchyma = np.full((size, size), base_val, dtype=np.float32)
    noise = np.random.normal(0, noise_std, (size, size))
    # Add subtle bronchovascular texture variations
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    subtle_grad = 8.0 * np.sin(xx * 3.0) * np.cos(yy * 3.0)
    return parenchyma + noise + subtle_grad


def add_ground_glass_opacity(img: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Ground-glass opacity: diffuse hazy increase in lung attenuation (COVID-19-like),
    with poorly defined borders preserving bronchial visibility.
    """
    center_x = size // 2 + np.random.randint(-15, 15)
    center_y = size // 2 + np.random.randint(-15, 15)
    radius_x = np.random.randint(30, 50)
    radius_y = int(radius_x * np.random.uniform(0.7, 1.3))

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt(((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2)
    mask = dist <= 1.0

    # Soft, hazy attenuation increase (~20-40 units above background) with falloff
    opacity = np.random.uniform(20.0, 40.0) * np.exp(-(dist ** 2) / 0.8)
    internal_noise = np.random.normal(0, 4.0, (size, size))
    img[mask] += opacity[mask] + internal_noise[mask]
    return img


def add_focal_consolidation(img: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Focal consolidation: localized dense alveolar filling (Lung Opacity-like),
    well-circumscribed hyperdense region with sharp margins.
    """
    center_x = size // 2 + np.random.randint(-10, 10)
    center_y = size // 2 + np.random.randint(-10, 10)
    radius = np.random.randint(22, 36)

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = dist <= radius

    # Dense consolidation (~60-90 units above background)
    consolidation = np.random.uniform(60.0, 90.0) * np.exp(-((dist / radius) ** 2) * 2.0)
    internal_noise = np.random.normal(0, 5.0, (size, size))
    img[mask] += consolidation[mask] + internal_noise[mask]
    return img


def add_interstitial_markings(img: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Interstitial markings: reticular/reticulonodular linear densities (Viral Pneumonia-like),
    distributed across both lung fields with moderate attenuation.
    """
    y, x = np.ogrid[:size, :size]
    # Reticular network of fine linear opacities
    lines = np.zeros((size, size), dtype=np.float32)
    for _ in range(np.random.randint(8, 14)):
        x0, y0 = np.random.randint(0, size, 2)
        x1, y1 = np.random.randint(0, size, 2)
        xx = np.linspace(x0, x1, size).astype(int)
        yy = np.linspace(y0, y1, size).astype(int)
        xx = np.clip(xx, 0, size - 1)
        yy = np.clip(yy, 0, size - 1)
        lines[yy, xx] += np.random.uniform(25.0, 45.0)

    img += lines + np.random.normal(0, 3.0, (size, size))
    return img


def generate_single_sample(class_name: str, size: int = 128) -> Image.Image:
    """Generate a single simulated 128x128 chest X-ray patch for the specified class."""
    base = generate_cxr_parenchyma(size=size)
    if class_name == "normal":
        patch = base
    elif class_name == "covid":
        patch = add_ground_glass_opacity(base, size=size)
    elif class_name == "lung_opacity":
        patch = add_focal_consolidation(base, size=size)
    elif class_name == "viral_pneumonia":
        patch = add_interstitial_markings(base, size=size)
    else:
        raise ValueError(f"Unknown class name: {class_name}")

    # Clip to standard 8-bit range [0, 255]
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    # Convert to 3-channel RGB image
    rgb_patch = np.stack([patch, patch, patch], axis=-1)
    return Image.fromarray(rgb_patch)


def create_simulated_dataset(
    target_dir: Path,
    num_train_per_class: int = 35,
    num_val_per_class: int = 10,
    num_test_per_class: int = 15,
    image_size: int = 128,
    seed: int = 42,
) -> None:
    """
    Generate synthetic chest X-ray benchmark dataset partitioned into train, val, and test splits.
    Includes patient-tagged filenames to simulate clinical case isolation.
    """
    set_seed(seed)
    splits = {
        "train": num_train_per_class,
        "val": num_val_per_class,
        "test": num_test_per_class,
    }

    print(f"Generating benchmark simulated chest X-ray dataset into {target_dir}...")
    for split_name, count in splits.items():
        for class_name in CLASS_TO_IDX:
            split_dir = target_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Generate samples organized across simulated patients
            for idx in range(count):
                patient_num = (idx // 3) + 1  # 3 slices per patient to test patient grouping
                filename = f"sim_pat_{split_name}_{patient_num:03d}_{class_name}_slice_{idx:02d}.png"
                img = generate_single_sample(class_name, size=image_size)
                img.save(split_dir / filename)

    print("Simulated benchmark chest X-ray dataset generated successfully!")
    print(f"  Train: {num_train_per_class} per class ({num_train_per_class * 3} total)")
    print(f"  Val:   {num_val_per_class} per class ({num_val_per_class * 3} total)")
    print(f"  Test:  {num_test_per_class} per class ({num_test_per_class * 3} total)")


def main():
    parser = argparse.ArgumentParser(description="Generate simulated chest X-ray benchmark patches for testing.")
    parser.add_argument("--train-count", type=int, default=35, help="Train samples per class")
    parser.add_argument("--val-count", type=int, default=10, help="Val samples per class")
    parser.add_argument("--test-count", type=int, default=15, help="Test samples per class")
    parser.add_argument("--size", type=int, default=128, help="Image size")
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()
    target = root / config["dataset"]["original_dir"]

    create_simulated_dataset(
        target_dir=target,
        num_train_per_class=args.train_count,
        num_val_per_class=args.val_count,
        num_test_per_class=args.test_count,
        image_size=args.size,
        seed=config["project"]["random_seed"],
    )


if __name__ == "__main__":
    main()
