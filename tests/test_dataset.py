"""
Unit tests for Dataset loading, split validation, and transforms for 4-class COVID-19 Radiography.
"""
from pathlib import Path
import torch
from src.utils.config import load_config, get_project_root
from src.data.dataset import ChestXrayDataset, get_transforms, get_gan_transforms, CLASS_TO_IDX


def test_class_mapping():
    assert len(CLASS_TO_IDX) == 4
    assert CLASS_TO_IDX["normal"] == 0
    assert CLASS_TO_IDX["covid"] == 1
    assert CLASS_TO_IDX["lung_opacity"] == 2
    assert CLASS_TO_IDX["viral_pneumonia"] == 3


def test_dataset_transforms():
    train_tf = get_transforms("train", image_size=128)
    val_tf = get_transforms("val", image_size=128)
    gan_tf = get_gan_transforms(image_size=128)

    assert train_tf is not None
    assert val_tf is not None
    assert gan_tf is not None


def test_dataset_loading_if_present():
    config = load_config()
    root = get_project_root()
    orig_dir = root / config["dataset"]["original_dir"]

    if (orig_dir / "train").exists():
        dataset = ChestXrayDataset(root_dir=orig_dir, split="train", transform=get_transforms("train", 128))
        if len(dataset) > 0:
            img, label, source = dataset[0]
            assert isinstance(img, torch.Tensor)
            assert img.shape == (3, 128, 128)
            assert label in [0, 1, 2, 3]
            assert source in ["original", "synthetic"]
