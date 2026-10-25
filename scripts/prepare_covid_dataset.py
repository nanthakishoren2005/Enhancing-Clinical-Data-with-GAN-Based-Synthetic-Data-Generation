"""
Script to prepare balanced train/val/test splits from archive/COVID-19_Radiography_Dataset.
Partitions 600 train, 100 val, 100 test per class across:
- normal
- covid
- lung_opacity
- viral_pneumonia
"""
import shutil
import random
from pathlib import Path
from PIL import Image

def main():
    random.seed(42)
    base_src = Path("archive/COVID-19_Radiography_Dataset")
    base_dst = Path("data/original")
    sample_dst = Path("data/sample_scans")
    sample_dst.mkdir(parents=True, exist_ok=True)

    class_source_dirs = {
        "normal": base_src / "Normal" / "images",
        "covid": base_src / "COVID" / "images",
        "lung_opacity": base_src / "Lung_Opacity" / "images",
        "viral_pneumonia": base_src / "Viral Pneumonia" / "images",
    }

    # Clean existing data/original to avoid mixing
    for split in ["train", "val", "test"]:
        split_dir = base_dst / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    print("Extracting and partitioning dataset from archive/COVID-19_Radiography_Dataset...")
    
    n_train = 600
    n_val = 100
    n_test = 100
    n_total = n_train + n_val + n_test

    for cname, sdir in class_source_dirs.items():
        if not sdir.exists():
            raise FileNotFoundError(f"Source folder not found: {sdir}")
        
        all_files = sorted([f for f in sdir.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        print(f"Found {len(all_files)} total images for class '{cname}'")
        
        # Shuffle deterministically
        random.shuffle(all_files)
        selected = all_files[:n_total]

        train_files = selected[:n_train]
        val_files = selected[n_train:n_train + n_val]
        test_files = selected[n_train + n_val:n_total]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, files in splits.items():
            target_dir = base_dst / split_name / cname
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, target_dir / f.name)

        # Copy sample scans for instant testing
        sample_img = test_files[0]
        sample_file_dst = sample_dst / f"sample_{cname}.png"
        shutil.copy2(sample_img, sample_file_dst)
        print(f"  -> {cname}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test. Sample: {sample_file_dst.name}")

    print("\nDataset partitioning complete!")
    for s in ["train", "val", "test"]:
        total_s = sum(len(list((base_dst / s / c).glob("*.*"))) for c in class_source_dirs)
        print(f"Total in {s.upper()}: {total_s} images")

if __name__ == "__main__":
    main()
