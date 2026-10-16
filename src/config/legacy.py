"""Reproducibility configuration for the Ohashi-ResNet50 / LDCT-IQAC baseline experiment.

`ExperimentConfig` is the single authoritative configuration object every
notebook and every `ct_iqa.training`/`ct_iqa.data` function consumes. It can
be built two ways, which must never be allowed to silently disagree:

    1. `ExperimentConfig(...)`      -- explicit keyword arguments / defaults.
    2. `ExperimentConfig.from_yaml_files(...)` -- merged from `configs/*.yaml`.

All Ohashi replication hyperparameters (Part 9 of the task) are collected
here with their paper-specified defaults, and also mirrored in
`configs/training.yaml`/`configs/model.yaml`/`configs/preprocessing.yaml`.
Values NOT specified by the paper (e.g. `dropout_p`) have no default and
must be set explicitly by the caller, so that every experiment run records a
deliberate choice rather than a silently invented one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


def _yaml_default(yaml_path: str, key: str, fallback):
    """Read a single default value out of a `configs/*.yaml` file at import time.

    This keeps `ExperimentConfig`'s dataclass defaults from silently drifting
    away from `configs/*.yaml` (the actual audited on-disk paths/sizes) --
    the YAML file is the source of truth whenever it's present, and
    `fallback` only applies when the file is missing (e.g. `ExperimentConfig`
    used standalone, outside a checkout of this repo's `configs/` directory).
    """
    path = Path(yaml_path)
    if not path.is_file():
        return fallback
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get(key, fallback)


# Dataset paths (canonical, post-migration -- see
# docs/internal/repository_architecture_audit.md and docs/internal/decisions/README.md).
# These previously pointed at `data/training/` and `data/testing/`, which
# stopped existing when commit 730bb2c renamed the on-disk directories to
# `data/train/`/`data/test/`; they now point at the current canonical
# locations under `data/raw/ldct_iqac/` and `data/labels/ldct_iqac/`.
#
# Sourced from `configs/dataset.yaml` at import time (with the literal below
# as a fallback only if that file isn't present) so this default can never
# silently disagree with the checked-in YAML -- see
# docs/internal/repository_architecture_audit.md, "Multiple Sources of Truth".
_DEFAULT_TRAIN_IMAGE_DIR = _yaml_default(
    "configs/dataset.yaml", "train_image_dir", "data/raw/ldct_iqac/train/image"
)
_DEFAULT_TRAIN_JSON_PATH = _yaml_default(
    "configs/dataset.yaml", "train_json_path", "data/labels/ldct_iqac/train.json"
)
_DEFAULT_TEST_IMAGE_DIR = _yaml_default(
    "configs/dataset.yaml", "test_image_dir", "data/raw/ldct_iqac/test/images"
)
_DEFAULT_TEST_JSON_PATH = _yaml_default(
    "configs/dataset.yaml", "test_json_path", "data/labels/ldct_iqac/test.json"
)

# Shared input resolution, sourced the same way from `configs/preprocessing.yaml`
# -- see `ct_iqa.data.ldct_iqac.INPUT_SIZE` and `ct_iqa.models.ohashi_resnet50.INPUT_SIZE`,
# which both import this instead of hardcoding their own copy of `224`.
DEFAULT_IMAGE_SIZE = _yaml_default("configs/preprocessing.yaml", "image_size", 224)

# Model architectures this project actually implements. `from_yaml_files()`
# validates `configs/model.yaml`'s `architecture` field against this set
# instead of silently ignoring it.
SUPPORTED_ARCHITECTURES = frozenset({"ohashi_resnet50"})

_CONFIG_FIELD_NAMES = frozenset(
    {
        "seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "optimizer",
        "loss",
        "image_size",
        "dropout_p",
        "in_channels",
        "train_image_dir",
        "train_json_path",
        "test_image_dir",
        "test_json_path",
        "val_fraction",
        "radimagenet_weights_path",
        "experiment_dir",
        "device",
        "architecture",
        "selection_metric",
    }
)


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
    image_size: int = DEFAULT_IMAGE_SIZE

    # --- model selection criterion for `ct_iqa.training.trainer.train`'s
    #     best-checkpoint saving. IMPLEMENTATION DECISION (the paper doesn't
    #     specify a selection rule): default to PLCC, the primary IQA
    #     evaluation metric, rather than the training loss -- see
    #     docs/replication/deviations.md. "val_loss" is kept for
    #     backward-compatible/ablation use.
    selection_metric: str = "plcc"

    # --- NOT specified by the paper: must be set explicitly per experiment ---
    # No default is provided on purpose. The paper states a Dropout layer
    # precedes the final Dense layer but never gives its probability; this
    # project will not silently invent one on the caller's behalf. Set this
    # explicitly (e.g. 0.5, a common default elsewhere in the literature)
    # and record that choice as an assumption in the experiment writeup
    # (see docs/replication/deviations.md).
    dropout_p: float | None = None

    # LDCT-IQAC images are single-channel float TIFFs; see
    # ct_iqa.models.ohashi_resnet50 for the 3-channel replication convention.
    in_channels: int = 1

    # --- dataset paths (LDCT-IQAC -- our dataset, NOT the paper's own
    #     dataset; see docs/replication/dataset_adaptation.md) ---
    train_image_dir: str = _DEFAULT_TRAIN_IMAGE_DIR
    train_json_path: str = _DEFAULT_TRAIN_JSON_PATH
    test_image_dir: str = _DEFAULT_TEST_IMAGE_DIR
    test_json_path: str = _DEFAULT_TEST_JSON_PATH

    # --- documentation-only identity fields from configs/dataset.yaml
    #     ("name") and configs/model.yaml ("architecture"). Consumed by
    #     `from_yaml_files()` as a guardrail against config/code drift (it
    #     validates `architecture` against the model this repo actually
    #     implements) rather than silently discarded -- see
    #     docs/internal/repository_architecture_audit.md, "Misleading Configuration
    #     Keys". `None` when built via `ExperimentConfig(...)` directly.
    dataset_name: str | None = None
    architecture: str | None = None

    # --- validation split (created from the training set only; the
    #     official test set is never used for model selection) ---
    val_fraction: float = 0.1

    # --- RadImageNet weights ---
    radimagenet_weights_path: str | None = None  # None => backbone stays randomly initialized.

    # --- experiment identity ---
    # Experiment-generated checkpoints live under the experiment's own
    # directory (`experiments/<NNN_name>/checkpoint/`), never under
    # `weights/` -- `weights/pretrained/` is reserved for externally-sourced
    # pretrained weights only (see docs/internal/decisions/README.md).
    experiment_dir: str = "experiments/001_resnet50_baseline"

    # --- device ---
    device: str = "cpu"

    @property
    def checkpoint_dir(self) -> str:
        """Where this run's checkpoints are written: `<experiment_dir>/checkpoint`."""
        return str(Path(self.experiment_dir) / "checkpoint")

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

    @classmethod
    def from_yaml_files(
        cls,
        dataset: str | Path = "configs/dataset.yaml",
        preprocessing: str | Path = "configs/preprocessing.yaml",
        model: str | Path = "configs/model.yaml",
        training: str | Path = "configs/training.yaml",
        **overrides,
    ) -> "ExperimentConfig":
        """Build an `ExperimentConfig` from the four `configs/*.yaml` files.

        This is the ONE authoritative path from on-disk configuration to a
        runnable `ExperimentConfig` -- notebooks should prefer this over
        re-typing hyperparameters, so that YAML config, `ExperimentConfig`
        defaults, and notebook code cannot silently drift apart the way the
        pre-migration dataset paths did (see
        docs/internal/repository_architecture_audit.md, "Import/Dependency Problems").

        `dataset.yaml`'s `name` field is consumed into `dataset_name` and
        `model.yaml`'s `architecture` field is consumed into `architecture`
        and validated against `SUPPORTED_ARCHITECTURES` (raising
        `ValueError` on a mismatch) -- these previously looked like config
        knobs but were silently dropped by this method, which gave the
        false impression that editing them would change behavior; see
        docs/internal/repository_architecture_audit.md, "Misleading Configuration
        Keys". Any other key not in `_CONFIG_FIELD_NAMES` is still ignored,
        so the YAML files may still carry additional documentation-only
        fields. `**overrides` take precedence over anything loaded from
        YAML.
        """
        merged: dict = {}
        for yaml_path in (dataset, preprocessing, model, training):
            yaml_path = Path(yaml_path)
            if not yaml_path.is_file():
                continue
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            merged.update({k: v for k, v in data.items() if k in _CONFIG_FIELD_NAMES})
            if "name" in data:
                merged["dataset_name"] = data["name"]
        merged.update(overrides)

        architecture = merged.get("architecture")
        if architecture is not None and architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"configs/model.yaml declares architecture={architecture!r}, but this "
                f"project only implements {sorted(SUPPORTED_ARCHITECTURES)!r}. Update "
                f"configs/model.yaml or ct_iqa.config.SUPPORTED_ARCHITECTURES."
            )

        return cls(**merged)
