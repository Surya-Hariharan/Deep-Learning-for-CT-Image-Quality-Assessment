"""Strict YAML loader for `StudyConfig`: base config + experiment overlay, unknown keys rejected."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ct_iqa.config.schema import (
    CalibrationConfig, ConfigError, DataConfig, LossConfig, ModelConfig, OptimConfig,
    RankingConfig, RunConfig, StudyConfig, TrainConfig, field_names,
)


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, data: dict[str, Any], where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    unknown = set(data) - field_names(cls)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as e:  # missing required field
        raise ConfigError(f"{where}: {e}") from e


def study_config_from_dict(raw: dict[str, Any]) -> StudyConfig:
    unknown = set(raw) - field_names(StudyConfig)
    if unknown:
        raise ConfigError(f"top level: unknown key(s) {sorted(unknown)}")
    loss_raw = dict(raw.get("loss", {}))
    ranking = _build(RankingConfig, loss_raw.pop("ranking", {}), "loss.ranking")
    loss = _build(LossConfig, {**loss_raw, "ranking": ranking}, "loss")
    cfg = StudyConfig(
        model=_build(ModelConfig, raw.get("model", {}), "model"),
        loss=loss,
        optim=_build(OptimConfig, raw.get("optim", {}), "optim"),
        data=_build(DataConfig, raw.get("data", {}), "data"),
        train=_build(TrainConfig, raw.get("train", {}), "train"),
        run=_build(RunConfig, raw.get("run", {}), "run"),
        calibration=_build(CalibrationConfig, raw.get("calibration", {}), "calibration"),
    )
    cfg.validate()
    return cfg


def load_study_config(experiment_yaml: str | Path, base_yaml: str | Path = "configs/base.yaml") -> StudyConfig:
    merged: dict = {}
    for p in (base_yaml, experiment_yaml):
        p = Path(p)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            merged = deep_merge(merged, yaml.safe_load(f) or {})
    return study_config_from_dict(merged)
