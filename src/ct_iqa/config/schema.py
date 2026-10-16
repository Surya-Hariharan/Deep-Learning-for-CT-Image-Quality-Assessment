"""Typed, validated configuration for the study-v2 experiment framework.

The legacy `ct_iqa.config.ExperimentConfig` (flat dataclass) is unchanged and still used by
the historical notebooks/tools. New code uses `StudyConfig` from this module. Unknown keys
are rejected (see `ct_iqa.config.loader`), and scientific choices (`bn_eps`, `dropout_p`)
have no hidden defaults: they must appear in the YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

MODEL_NAMES = ("resnet50_regression", "resnet50_gaussian")
LOSS_NAMES = ("mse", "mse_ranking", "gaussian_nll")
OPTIMIZER_NAMES = ("adam",)
SELECTION_METRICS = ("val_plcc", "val_srocc", "val_loss", "final_epoch")


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration."""


@dataclass
class ModelConfig:
    name: str
    dropout_p: float
    bn_eps: float
    in_channels: int = 1
    pretrained_weights: str | None = None
    pretrained_sha256: str | None = None


@dataclass
class RankingConfig:
    lambda_rank: float = 0.0          # 0 reduces the objective exactly to MSE
    margin: float = 0.05              # in the NORMALIZED [0, 1] score space
    min_score_diff: float = 0.25      # raw [0, 4] units; pairs closer than this are ignored


@dataclass
class LossConfig:
    name: str
    ranking: RankingConfig = field(default_factory=RankingConfig)


@dataclass
class OptimConfig:
    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class DataConfig:
    train_image_dir: str
    train_json_path: str
    test_image_dir: str
    test_json_path: str
    split_manifest: str
    image_size: int = 224


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    seed: int
    device: str = "cpu"
    selection_metric: str = "val_plcc"


@dataclass
class CalibrationConfig:
    method: str = "split_conformal"   # the single predeclared primary method
    target_coverage: float = 0.90     # 1 - alpha, fixed before any calibration result is viewed


@dataclass
class RunConfig:
    experiment_id: str
    output_root: str = "experiments/study_v2"


@dataclass
class StudyConfig:
    model: ModelConfig
    loss: LossConfig
    optim: OptimConfig
    data: DataConfig
    train: TrainConfig
    run: RunConfig
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        m, l, o, t = self.model, self.loss, self.optim, self.train
        if m.name not in MODEL_NAMES:
            raise ConfigError(f"model.name {m.name!r} not in {MODEL_NAMES}")
        if not 0.0 <= m.dropout_p < 1.0:
            raise ConfigError("model.dropout_p must be in [0, 1)")
        if not 0.0 < m.bn_eps < 1e-2:
            raise ConfigError("model.bn_eps must be in (0, 0.01)")
        if m.in_channels not in (1, 3):
            raise ConfigError("model.in_channels must be 1 or 3")
        if (m.pretrained_weights is None) != (m.pretrained_sha256 is None):
            raise ConfigError("model.pretrained_weights and model.pretrained_sha256 must be set together")
        if l.name not in LOSS_NAMES:
            raise ConfigError(f"loss.name {l.name!r} not in {LOSS_NAMES}")
        if (l.name == "gaussian_nll") != (m.name == "resnet50_gaussian"):
            raise ConfigError("gaussian_nll requires model resnet50_gaussian, and vice versa")
        r = l.ranking
        if r.lambda_rank < 0 or r.margin < 0 or r.min_score_diff < 0:
            raise ConfigError("loss.ranking values must be non-negative")
        if l.name == "mse_ranking" and r.lambda_rank == 0:
            raise ConfigError("mse_ranking with lambda_rank=0 is plain MSE; use loss.name='mse'")
        if o.name not in OPTIMIZER_NAMES:
            raise ConfigError(f"optim.name {o.name!r} not in {OPTIMIZER_NAMES}")
        if o.learning_rate <= 0:
            raise ConfigError("optim.learning_rate must be > 0")
        if t.epochs < 1 or t.batch_size < 1:
            raise ConfigError("train.epochs and train.batch_size must be >= 1")
        if t.selection_metric not in SELECTION_METRICS:
            raise ConfigError(f"train.selection_metric {t.selection_metric!r} not in {SELECTION_METRICS}")
        if not 0.0 < self.calibration.target_coverage < 1.0:
            raise ConfigError("calibration.target_coverage must be in (0, 1)")
        if self.calibration.method != "split_conformal":
            raise ConfigError("calibration.method must be 'split_conformal' (predeclared primary method)")


def field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}
