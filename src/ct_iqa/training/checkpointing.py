"""Checkpoint saving/loading for CT-IQA training runs.

Checkpoints are written under an experiment's own directory
(`experiments/<NNN_name>/checkpoint/`), never under `weights/` --
`weights/pretrained/` is reserved for externally-sourced pretrained weights,
never experiment-generated output (see docs/internal/decisions/README.md).

Checkpoint format (updated 2026-09-13, ahead of experiment 001's first real
training run): a dict with `model_state_dict` plus whatever of
`optimizer_state_dict`/`epoch`/`val_loss`/`config`/`seed`/`checkpoint_type`
the caller supplied -- enough to reproduce or resume the run, not just the
bare weights. `load_checkpoint` still only mutates the given `model` in
place (matching its pre-existing call sites in `ct_iqa.training.trainer`
and the evaluation notebook); use `load_checkpoint_metadata` to read the
rest of the payload without constructing a model.

Filename (updated 2026-09-13, ahead of experiment 003's final-training run,
which has no validation split and therefore no "best" checkpoint to
select): defaults to `best.pt` everywhere, unchanged for experiments
001/002's existing usage, but every function accepts an explicit
`filename` to save/load a differently-named checkpoint in the same
directory (e.g. `final.pt` for a final-epoch-only checkpoint).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

BEST_CHECKPOINT_NAME = "best.pt"


def best_checkpoint_path(checkpoint_dir: str | Path, filename: str = BEST_CHECKPOINT_NAME) -> Path:
    return Path(checkpoint_dir) / filename


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    filename: str = BEST_CHECKPOINT_NAME,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    val_loss: float | None = None,
    config=None,
    seed: int | None = None,
    checkpoint_type: str | None = None,
) -> Path:
    """Save a checkpoint to `<checkpoint_dir>/<filename>`, creating the directory if needed.

    Always includes `model_state_dict`. Any of `optimizer`, `epoch`,
    `val_loss`, `config` (an `ExperimentConfig`, or anything with `.to_dict()`,
    or a plain dict), `seed`, and `checkpoint_type` (e.g. `"best_val_loss"`
    or `"final_epoch"`, a free-form label -- not validated) that are
    provided are also recorded, so the checkpoint alone is enough to know
    what produced it and to resume training if needed.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = best_checkpoint_path(checkpoint_dir, filename=filename)

    payload: dict = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else config
    if seed is not None:
        payload["seed"] = seed
    if checkpoint_type is not None:
        payload["checkpoint_type"] = checkpoint_type

    torch.save(payload, path)
    return path


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    map_location: str = "cpu",
    filename: str = BEST_CHECKPOINT_NAME,
) -> Path:
    """Load a previously-saved `<checkpoint_dir>/<filename>` checkpoint's model weights into `model` in place.

    Accepts both the current dict payload (`{"model_state_dict": ..., ...}`)
    and a bare state dict (pre-2026-09-13 checkpoints), so older checkpoints
    remain loadable.
    """
    path = best_checkpoint_path(checkpoint_dir, filename=filename)
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return path


def load_checkpoint_metadata(
    checkpoint_dir: str | Path, map_location: str = "cpu", filename: str = BEST_CHECKPOINT_NAME
) -> dict:
    """Load everything in a checkpoint EXCEPT the model weights themselves:
    `epoch`, `val_loss`, `config`, `seed`, `checkpoint_type`,
    `optimizer_state_dict` -- whichever were present when it was saved.
    Returns `{}` for a bare-state-dict (pre-2026-09-13) checkpoint, which
    carries no such metadata.
    """
    path = best_checkpoint_path(checkpoint_dir, filename=filename)
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return {}
    return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
