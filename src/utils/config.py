"""
Config loader and path resolution utilities.
"""
from pathlib import Path
import yaml
import torch


def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    # This file is located at <project_root>/src/utils/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """Load configuration YAML file with fallback to default path."""
    root = get_project_root()
    if config_path is None:
        target = root / "configs" / "config.yaml"
    else:
        target = Path(config_path)
        if not target.is_absolute():
            target = root / target

    if not target.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device(config: dict = None) -> torch.device:
    """Determine compute device based on config and system availability."""
    if config is not None:
        pref = config.get("hardware", {}).get("device", "auto").lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif pref == "cpu":
            return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
