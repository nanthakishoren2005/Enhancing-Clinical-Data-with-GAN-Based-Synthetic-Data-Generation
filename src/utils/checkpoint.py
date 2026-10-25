"""
Model checkpoint saving and loading utilities.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = False,
    best_filepath: Optional[Path] = None,
) -> None:
    """Save training checkpoint to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(filepath))
    if is_best and best_filepath is not None:
        best_filepath = Path(best_filepath)
        best_filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(best_filepath))


def load_checkpoint(
    filepath: Path,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load model checkpoint safely."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)
    return checkpoint
