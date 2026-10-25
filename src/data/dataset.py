"""
PyTorch Dataset and transforms for Chest X-ray Classification.
Supports loading original chest X-ray slices and GAN-generated synthetic samples.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Standard class mapping for COVID-19 & Pulmonary Disease Radiography dataset
CLASS_TO_IDX = {
    "normal": 0,
    "covid": 1,
    "lung_opacity": 2,
    "viral_pneumonia": 3,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def get_transforms(
    split: str = "train",
    image_size: int = 128,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Build transformation pipeline for CNN training or evaluation.
    Medically conservative data augmentation applied only during training:
    - Random horizontal flip (pathology can present in varied orientations)
    - Mild random rotation (-15 to +15 deg)
    - Mild affine translation (up to 5%)
    Validation and test splits undergo strictly deterministic resize & normalize.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_gan_transforms(image_size: int = 128) -> transforms.Compose:
    """
    Transform pipeline for GAN training. Normalizes pixel values to [-1, 1]
    matching standard Tanh activation output from DCGAN Generator.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


class ChestXrayDataset(Dataset):
    """
    Dataset class for loading chest X-ray images across classes.
    Optionally includes GAN-generated synthetic samples for training splits.
    """
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        synthetic_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.transform = transform
        self.samples: List[Tuple[Path, int, str]] = []  # (path, label_idx, source: 'original'|'synthetic')
        self._load_samples()

    def _load_samples(self) -> None:
        split_dir = self.root_dir / self.split
        if split_dir.exists():
            for class_name, class_idx in CLASS_TO_IDX.items():
                class_folder = split_dir / class_name
                if class_folder.exists():
                    for file_path in sorted(class_folder.iterdir()):
                        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                            self.samples.append((file_path, class_idx, "original"))

        # For the training split, optionally include synthetic data
        if self.split == "train" and self.synthetic_dir is not None:
            synth_train_dir = self.synthetic_dir / "train"
            if synth_train_dir.exists():
                for class_name, class_idx in CLASS_TO_IDX.items():
                    synth_class_folder = synth_train_dir / class_name
                    if synth_class_folder.exists():
                        for file_path in sorted(synth_class_folder.iterdir()):
                            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                                self.samples.append((file_path, class_idx, "synthetic"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, source = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label, source
        except Exception as e:
            raise IOError(f"Error loading image at {path}: {str(e)}")

    def get_class_counts(self) -> dict:
        """Return counts per class and source breakdown."""
        counts = {c: {"original": 0, "synthetic": 0, "total": 0} for c in CLASS_TO_IDX}
        for _, label_idx, source in self.samples:
            cname = IDX_TO_CLASS[label_idx]
            counts[cname][source] += 1
            counts[cname]["total"] += 1
        return counts


# Backward-compatible alias for code referencing the legacy liver-lesion name.
LiverLesionDataset = ChestXrayDataset
