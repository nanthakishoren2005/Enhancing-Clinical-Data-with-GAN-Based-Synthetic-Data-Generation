"""
Dataset preparation module.
Splits images organized by class into train, val, and test splits.
Prevents patient-level data leakage by grouping images belonging to the same patient
or lesion into the same split when patient ID identifiers are present.
"""
import argparse
import shutil
import re
from pathlib import Path
from typing import Dict, List
import numpy as np

from src.utils.config import load_config, get_project_root
from src.utils.reproducibility import set_seed
from src.data.dataset import CLASS_TO_IDX, VALID_EXTENSIONS


def extract_patient_id(filename: str) -> str:
    """
    Extract patient or case ID if formatted in common medical formats,
    e.g. 'patient_001_slice_02.png', 'case12_01.jpg', 'P03_lesion.png'.
    Falls back to individual filename if no pattern is detected.
    """
    match = re.match(r"^(patient[_-]?\d+|case[_-]?\d+|p\d+|subj[_-]?\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return filename  # fallback: image treated individually


def split_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """
    Organize class folders into train/val/test splits without patient-level leakage.
    """
    set_seed(seed)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    splits = ["train", "val", "test"]
    summary = {s: {c: 0 for c in CLASS_TO_IDX} for s in splits}

    # Ensure target split folders exist
    for split in splits:
        for class_name in CLASS_TO_IDX:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_TO_IDX:
        class_folder = source_dir / class_name
        if not class_folder.exists():
            print(f"Warning: Class folder {class_name} not found in {source_dir}")
            continue

        files = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        if not files:
            print(f"Warning: No valid images found in {class_folder}")
            continue

        # Group by patient ID to eliminate data leakage
        patient_groups: Dict[str, List[Path]] = {}
        for f in files:
            pid = extract_patient_id(f.name)
            patient_groups.setdefault(pid, []).append(f)

        patients = list(patient_groups.keys())
        np.random.shuffle(patients)

        n_patients = len(patients)
        n_train = max(1, int(round(n_patients * train_ratio)))
        n_val = max(1, int(round(n_patients * val_ratio))) if n_patients >= 3 else 0
        # Remaining go to test
        train_pids = set(patients[:n_train])
        val_pids = set(patients[n_train:n_train + n_val])
        test_pids = set(patients[n_train + n_val:])

        # If very few patients, ensure at least 1 file in test
        if not test_pids and len(patients) >= 2:
            test_pids.add(patients[-1])
            if patients[-1] in val_pids:
                val_pids.remove(patients[-1])
            elif patients[-1] in train_pids and len(train_pids) > 1:
                train_pids.remove(patients[-1])

        # Copy files into target splits
        for pid, pfiles in patient_groups.items():
            if pid in train_pids:
                target_split = "train"
            elif pid in val_pids:
                target_split = "val"
            else:
                target_split = "test"

            for file_path in pfiles:
                dest = target_dir / target_split / class_name / file_path.name
                shutil.copy2(file_path, dest)
                summary[target_split][class_name] += 1

    return summary


def main():
    parser = argparse.ArgumentParser(description="Split raw class-organized dataset into train/val/test splits.")
    parser.add_argument("--source", type=str, required=True, help="Path to raw folder containing class subdirectories")
    parser.add_argument("--target", type=str, default=None, help="Target original data folder (defaults to config)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()
    target_path = Path(args.target) if args.target else root / config["dataset"]["original_dir"]
    source_path = Path(args.source)

    print(f"Splitting dataset from {source_path} to {target_path}...")
    summary = split_dataset(
        source_dir=source_path,
        target_dir=target_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=config["project"]["random_seed"],
    )

    print("Splitting complete! Image counts per split:")
    for split, counts in summary.items():
        print(f"  {split.upper()}: {counts}")


if __name__ == "__main__":
    main()
