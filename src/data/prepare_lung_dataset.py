"""
Prepare Lung X-Ray dataset from 'lunds dataset' into data/original/ split.
"""
import shutil
from pathlib import Path
import numpy as np

from src.utils.config import get_project_root
from src.utils.reproducibility import set_seed

def main():
    root = get_project_root()
    set_seed(42)

    src_root = root / "lunds dataset" / "Lung X-Ray Image" / "Lung X-Ray Image"
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found at {src_root}")

    dest_orig = root / "data" / "original"
    dest_synth = root / "data" / "synthetic"
    dest_samples = root / "data" / "sample_scans"

    # Reset data/original
    if dest_orig.exists():
        shutil.rmtree(dest_orig)
    if dest_synth.exists():
        shutil.rmtree(dest_synth)
    dest_samples.mkdir(parents=True, exist_ok=True)

    class_mapping = {
        "Normal": "normal",
        "Lung_Opacity": "lung_opacity",
        "Viral Pneumonia": "viral_pneumonia",
    }

    splits = ["train", "val", "test"]
    for s in splits:
        for c in class_mapping.values():
            (dest_orig / s / c).mkdir(parents=True, exist_ok=True)
            (dest_synth / s / c).mkdir(parents=True, exist_ok=True)

    # Balanced split targets per class: 400 train, 80 val, 120 test
    n_train = 400
    n_val = 80
    n_test = 120
    total_needed = n_train + n_val + n_test

    summary = {s: {c: 0 for c in class_mapping.values()} for s in splits}

    for src_folder_name, target_class in class_mapping.items():
        folder = src_root / src_folder_name
        files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        
        # Shuffle reproducibly
        np.random.seed(42)
        indices = np.random.permutation(len(files))
        shuffled = [files[i] for i in indices]

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:total_needed]

        # Copy train
        for idx, f in enumerate(train_files):
            dest = dest_orig / "train" / target_class / f"{target_class}_{idx:04d}.jpg"
            shutil.copy2(f, dest)
            summary["train"][target_class] += 1

        # Copy val
        for idx, f in enumerate(val_files):
            dest = dest_orig / "val" / target_class / f"{target_class}_{idx:04d}.jpg"
            shutil.copy2(f, dest)
            summary["val"][target_class] += 1

        # Copy test
        for idx, f in enumerate(test_files):
            dest = dest_orig / "test" / target_class / f"{target_class}_{idx:04d}.jpg"
            shutil.copy2(f, dest)
            summary["test"][target_class] += 1

        # Save representative sample scan for quick testing
        shutil.copy2(test_files[0], dest_samples / f"sample_{target_class}.jpg")

    # Extra normal sample
    norm_sample_2 = shuffled[total_needed + 1] if len(shuffled) > total_needed + 1 else test_files[1]
    shutil.copy2(norm_sample_2, dest_samples / "sample_normal_2.jpg")

    print("=====================================================================")
    print("Lung X-Ray Dataset successfully partitioned:")
    for s in splits:
        total_s = sum(summary[s].values())
        print(f"  {s.upper():5s}: {summary[s]} (Total: {total_s})")
    print("=====================================================================")

if __name__ == "__main__":
    main()
