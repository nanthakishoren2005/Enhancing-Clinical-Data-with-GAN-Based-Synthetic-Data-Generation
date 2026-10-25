"""
Dataset validation module.
Verifies file integrity, directory hierarchy, class distributions, and warns on missing or corrupted files.
Includes fast directory scanning and in-memory caching to eliminate UI page load freezes.
Can be executed as: python -m src.data.validate_dataset
"""
import sys
import os
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List

from src.utils.config import load_config, get_project_root
from src.data.dataset import CLASS_TO_IDX, VALID_EXTENSIONS

# In-memory validation cache to prevent repeated disk walks on every Streamlit page rerun
_VALIDATION_CACHE: Dict[str, Any] = {}


def clear_validation_cache():
    global _VALIDATION_CACHE
    _VALIDATION_CACHE.clear()


def validate_dataset(
    original_dir: Path,
    synthetic_dir: Path = None,
    deep_verify: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Validates dataset structure, counts, file validity, and flags any corrupted images.
    deep_verify=False performs ultra-fast directory scans (milliseconds).
    deep_verify=True opens and verifies individual image headers.
    """
    cache_key = f"{str(original_dir)}_{str(synthetic_dir)}_{deep_verify}"
    if use_cache and cache_key in _VALIDATION_CACHE:
        return _VALIDATION_CACHE[cache_key]

    results = {
        "valid": True,
        "splits": {},
        "corrupted_files": [],
        "warnings": [],
        "total_images": 0,
    }

    splits = ["train", "val", "test"]
    for split in splits:
        split_path = Path(original_dir) / split
        split_info = {
            "exists": split_path.exists(),
            "classes": {},
            "total": 0,
        }

        if not split_path.exists():
            results["warnings"].append(f"Split directory missing: {split_path}")
            results["valid"] = False
            results["splits"][split] = split_info
            continue

        for class_name in CLASS_TO_IDX:
            class_path = split_path / class_name
            class_count = 0
            if not class_path.exists():
                results["warnings"].append(f"Missing class folder: {class_path}")
                split_info["classes"][class_name] = 0
                continue

            # Fast scandir traversal
            try:
                with os.scandir(str(class_path)) as entries:
                    for entry in entries:
                        if entry.is_file():
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in VALID_EXTENSIONS:
                                if deep_verify:
                                    try:
                                        with Image.open(entry.path) as img:
                                            img.verify()
                                        class_count += 1
                                    except Exception as e:
                                        results["corrupted_files"].append((entry.path, str(e)))
                                        results["valid"] = False
                                else:
                                    class_count += 1
                            else:
                                results["warnings"].append(
                                    f"Unsupported file format ignored: {entry.name} in {class_path}"
                                )
            except Exception as e:
                results["warnings"].append(f"Error accessing folder {class_path}: {e}")

            split_info["classes"][class_name] = class_count
            split_info["total"] += class_count

            if class_count == 0:
                results["warnings"].append(f"Class folder is empty: {class_path}")

        results["splits"][split] = split_info
        results["total_images"] += split_info["total"]

    # Check synthetic images if path provided
    if synthetic_dir:
        synth_dir_path = Path(synthetic_dir)
        synth_train_dir = synth_dir_path / "train"
        if synth_train_dir.exists():
            synth_info = {"classes": {}, "total": 0}
            for class_name in CLASS_TO_IDX:
                c_dir = synth_train_dir / class_name
                count = 0
                if c_dir.exists():
                    try:
                        with os.scandir(str(c_dir)) as entries:
                            count = sum(1 for e in entries if e.is_file() and os.path.splitext(e.name)[1].lower() in VALID_EXTENSIONS)
                    except Exception:
                        pass
                synth_info["classes"][class_name] = count
                synth_info["total"] += count
            results["synthetic_train"] = synth_info

    if results["total_images"] == 0:
        results["valid"] = False
        results["warnings"].append("No real images found across any dataset splits.")

    if use_cache:
        _VALIDATION_CACHE[cache_key] = results

    return results


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print a clean CLI report of validation results."""
    print("=" * 65)
    print(" DATASET VALIDATION REPORT")
    print("=" * 65)
    print(f"Overall Status: {'[OK] VALID' if results['valid'] else '[!] ATTENTION NEEDED'}")
    print(f"Total Real Images: {results['total_images']}")
    print("-" * 65)

    for split, info in results.get("splits", {}).items():
        print(f"\nSplit: {split.upper()} (Total: {info['total']})")
        for cls_name, count in info.get("classes", {}).items():
            print(f"  - {cls_name:15s}: {count:5d} images")

    if "synthetic_train" in results:
        print(f"\nSynthetic Split: TRAIN (Total: {results['synthetic_train']['total']})")
        for cls_name, count in results['synthetic_train'].get("classes", {}).items():
            print(f"  - {cls_name:15s}: {count:5d} synthetic images")

    if results["corrupted_files"]:
        print("\n[!] Corrupted files detected:")
        for path, err in results["corrupted_files"]:
            print(f"  - {path}: {err}")

    if results["warnings"]:
        print("\n[!] Warnings & Recommendations:")
        for w in results["warnings"]:
            print(f"  * {w}")
    print("=" * 65)


def main():
    config = load_config()
    root = get_project_root()
    orig_dir = root / config["dataset"]["original_dir"]
    synth_dir = root / config["dataset"]["synthetic_dir"]

    results = validate_dataset(orig_dir, synth_dir, deep_verify=True)
    print_validation_report(results)
    if not results["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
