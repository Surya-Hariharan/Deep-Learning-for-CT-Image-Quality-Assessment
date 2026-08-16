"""NGP-Net configuration.

Replaces the author's ``utils/config.py``, which parsed CLI arguments and
seeded RNGs as import-time side effects (``args = parser.parse_args()`` ran
whenever any module imported it). Importing this module does nothing;
``NGPNetConfig`` fields carry the same defaults the author's argparse setup
used, and ``build_arg_parser`` / ``config_from_args`` provide the CLI entry
point for scripts (see scripts/train.py, scripts/evaluate.py).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class NGPNetConfig:
    ''' NGP-Net model, training and evaluation configuration.

    Defaults are inherited unchanged from the author's `utils/config.py`.
    '''

    # --- runtime ---
    pred_model: str = "NGPnet"
    device: str = "cuda"  # resolved to "cpu" if CUDA is unavailable; see resolve_device()
    seed: int = 35202
    num_workers: int = 8

    # --- paths ---
    data_root: str = "data/external/png"
    json_name: str = "NGP3T.json"
    num_folds: int = 5
    fold: int = 0
    model_save_dir: str = "checkpoints/runs"
    trained_model: str = "best_model.pth"
    log_save_dir: str = "outputs/logs"
    pred_save_dir: str = "outputs/predictions"
    cross_validate: bool = False

    # --- training ---
    batch_size: int = 4
    max_epochs: int = 200
    val_freq: int = 2
    optim_name: str = "AdamW"
    optim_lr: float = 1e-3
    reg_weight: float = 1e-4
    lrschedule: str = "WarmUpCosine"
    warmup_steps: int = 10

    # --- preprocessing (CT windowing) ---
    s_min: float = -1200
    s_max: float = 600
    r_min: float = 0
    r_max: float = 1

    # --- network ---
    in_size: int = 64
    feat_dim: int = 8
    depths: Tuple[int, int, int, int] = (2, 2, 2, 2)

    # --- loss / metric weighting ---
    w_out: float = 0.6
    expand: float = 1.2
    w0_sim: float = 2.0
    w1_seg: float = 1.0
    w2_reg: float = 2.0
    w3_smooth: float = 1.0

    def resolve_device(self):
        ''' Resolve `self.device` against actual CUDA availability. '''
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device


def build_arg_parser() -> argparse.ArgumentParser:
    ''' CLI parser mirroring `NGPNetConfig`'s fields, for use by scripts/. '''
    defaults = NGPNetConfig()
    parser = argparse.ArgumentParser(description="Lung Nodule Growth Prediction")

    parser.add_argument("--pred-model", type=str, default=defaults.pred_model)
    parser.add_argument("--device", type=str, default=defaults.device,
                        choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)

    parser.add_argument("--data-root", type=str, default=defaults.data_root,
                        help="root of the PNG dataset")
    parser.add_argument("--json-name", type=str, default=defaults.json_name)
    parser.add_argument("--num-folds", type=int, default=defaults.num_folds)
    parser.add_argument("--fold", type=int, default=defaults.fold)

    parser.add_argument("--model-save-dir", type=str, default=defaults.model_save_dir)
    parser.add_argument("--trained-model", type=str, default=defaults.trained_model)
    parser.add_argument("--log-save-dir", type=str, default=defaults.log_save_dir)
    parser.add_argument("--pred-save-dir", type=str, default=defaults.pred_save_dir)
    parser.add_argument("--cross-validate", action="store_true", default=defaults.cross_validate,
                        help="run k-fold cross-validation instead of a single train/val split")

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--val-freq", type=int, default=defaults.val_freq)
    parser.add_argument("--optim-name", type=str, default=defaults.optim_name)
    parser.add_argument("--optim-lr", type=float, default=defaults.optim_lr)
    parser.add_argument("--reg-weight", type=float, default=defaults.reg_weight)
    parser.add_argument("--lrschedule", type=str, default=defaults.lrschedule)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)

    parser.add_argument("--s-min", type=float, default=defaults.s_min)
    parser.add_argument("--s-max", type=float, default=defaults.s_max)
    parser.add_argument("--r-min", type=float, default=defaults.r_min)
    parser.add_argument("--r-max", type=float, default=defaults.r_max)

    parser.add_argument("--in-size", type=int, default=defaults.in_size)
    parser.add_argument("--feat-dim", type=int, default=defaults.feat_dim)

    parser.add_argument("--w-out", type=float, default=defaults.w_out)
    parser.add_argument("--expand", type=float, default=defaults.expand)
    parser.add_argument("--w0-sim", type=float, default=defaults.w0_sim)
    parser.add_argument("--w1-seg", type=float, default=defaults.w1_seg)
    parser.add_argument("--w2-reg", type=float, default=defaults.w2_reg)
    parser.add_argument("--w3-smooth", type=float, default=defaults.w3_smooth)
    return parser


def config_from_args(argv=None) -> NGPNetConfig:
    ''' Parse CLI args (or `argv`) into an `NGPNetConfig`. '''
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    kwargs = vars(ns)
    kwargs["depths"] = NGPNetConfig().depths  # fixed architecture, not a CLI knob
    return NGPNetConfig(**kwargs)
