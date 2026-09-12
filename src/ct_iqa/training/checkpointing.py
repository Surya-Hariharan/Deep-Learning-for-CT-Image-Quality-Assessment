"""Checkpoint saving/loading for CT-IQA training runs.

Extracted from `ct_iqa.training.trainer` with no change in behavior.
Checkpoints are written under an experiment's own directory
(`experiments/<NNN_name>/checkpoint/`), never under `weights/` --
`weights/pretrained/` is reserved for externally-sourced pretrained weights,
never experiment-generated output (see docs/decisions/README.md).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

BEST_CHECKPOINT_NAME = "best.pt"


def best_checkpoint_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / BEST_CHECKPOINT_NAME


def save_checkpoint(model: nn.Module, checkpoint_dir: str | Path) -> Path:
    """Save `model`'s state dict as `<checkpoint_dir>/best.pt`, creating the directory if needed."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = best_checkpoint_path(checkpoint_dir)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(
    model: nn.Module, checkpoint_dir: str | Path, map_location: str = "cpu"
) -> Path:
    """Load a previously-saved `<checkpoint_dir>/best.pt` checkpoint into `model` in place."""
    path = best_checkpoint_path(checkpoint_dir)
    model.load_state_dict(torch.load(path, map_location=map_location))
    return path
