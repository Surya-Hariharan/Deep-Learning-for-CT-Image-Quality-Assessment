"""Reproducibility configuration for the Ohashi-ResNet50 / LDCT-IQAC baseline experiment.

All Ohashi replication hyperparameters (Part 9 of the task) are collected
here with their paper-specified defaults. Values NOT specified by the
paper (e.g. `dropout_p`) have no default and must be set explicitly by the
caller, so that every experiment run records a deliberate choice rather
than a silently invented one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    # --- reproducibility ---
    seed: int = 42

    # --- Ohashi ResNet50 training hyperparameters (paper-specified) ---
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "mse"
    image_size: int = 224

    # --- NOT specified by the paper: must be set explicitly per experiment ---
    # No default is provided on purpose. The paper states a Dropout layer
    # precedes the final Dense layer but never gives its probability; this
    # project will not silently invent one on the caller's behalf. Set this
    # explicitly (e.g. 0.5, a common default elsewhere in the literature)
    # and record that choice as an assumption in the experiment writeup.
    dropout_p: float | None = None

    # --- dataset paths ---
    train_image_dir: str = "data/training/image"
    train_json_path: str = "data/training/train.json"
    test_image_dir: str = "data/testing/images"
    test_json_path: str = "data/testing/test.json"

    # --- validation split (created from the training set only; the
    #     official test set is never used for model selection) ---
    val_fraction: float = 0.1

    # --- RadImageNet weights ---
    radimagenet_weights_path: str | None = None  # None => backbone stays randomly initialized.

    # --- checkpointing ---
    checkpoint_dir: str = "checkpoints/ohashi_resnet50"

    # --- device ---
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(**json.load(f))
