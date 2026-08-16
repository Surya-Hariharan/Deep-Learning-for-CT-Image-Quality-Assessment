"""Checkpoint save/load.

Behavior migrated from the author's `utils/common.py:save_ckpts` /
`get_pred_model`'s load path. Checkpoint dict shape (`{"net", "optimizer"}`)
is unchanged so checkpoints trained with the author's original scripts
remain loadable.
"""

from __future__ import annotations

import os

import torch
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(model: Module,
                    optim: Optimizer,
                    save_dir: str,
                    filename: str) -> str:
    ''' Save `{"net": model.state_dict(), "optimizer": optim.state_dict()}`. '''
    state_dict = {
        "net": model.state_dict(),
        "optimizer": optim.state_dict(),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state_dict, path)
    return path


def load_checkpoint(model: Module, path: str, map_location=None) -> Module:
    ''' Load a checkpoint's `"net"` state dict into `model` in-place. '''
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict["net"])
    return model
