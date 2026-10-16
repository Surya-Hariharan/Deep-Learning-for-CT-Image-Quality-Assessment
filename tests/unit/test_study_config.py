"""Unit tests for the strict study-v2 configuration loader (synthetic YAML in tmp_path)."""

from __future__ import annotations

import copy

import pytest
import yaml

from ct_iqa.config.loader import deep_merge, load_study_config, study_config_from_dict
from ct_iqa.config.schema import ConfigError

VALID = {
    "model": {"name": "resnet50_regression", "dropout_p": 0.5, "bn_eps": 1.001e-5},
    "loss": {"name": "mse"},
    "data": {"train_image_dir": "a", "train_json_path": "b", "test_image_dir": "c",
             "test_json_path": "d", "split_manifest": "m.json"},
    "train": {"epochs": 2, "batch_size": 4, "seed": 1},
    "run": {"experiment_id": "X"},
}


def test_valid_config_loads():
    cfg = study_config_from_dict(copy.deepcopy(VALID))
    assert cfg.calibration.method == "split_conformal" and cfg.calibration.target_coverage == 0.90


@pytest.mark.parametrize("path,value", [
    (("model", "bn_eps"), 0.5), (("model", "dropout_p"), 1.0), (("model", "name"), "vgg16"),
    (("train", "selection_metric"), "test_plcc"), (("loss", "name"), "gaussian_nll"),
    (("calibration", "method"), "scale_recalibration"), (("calibration", "target_coverage"), 1.0),
])
def test_invalid_values_rejected(path, value):
    raw = copy.deepcopy(VALID)
    raw.setdefault(path[0], {})[path[1]] = value
    with pytest.raises(ConfigError):
        study_config_from_dict(raw)


def test_unknown_and_missing_keys_rejected():
    raw = copy.deepcopy(VALID); raw["model"]["surprise"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        study_config_from_dict(raw)
    raw = copy.deepcopy(VALID); del raw["model"]["bn_eps"]
    with pytest.raises(ConfigError):
        study_config_from_dict(raw)
    raw = copy.deepcopy(VALID); raw["extra_section"] = {}
    with pytest.raises(ConfigError, match="unknown"):
        study_config_from_dict(raw)


def test_mse_ranking_with_zero_lambda_rejected():
    raw = copy.deepcopy(VALID); raw["loss"] = {"name": "mse_ranking", "ranking": {"lambda_rank": 0.0}}
    with pytest.raises(ConfigError):
        study_config_from_dict(raw)


def test_base_plus_overlay_merge(tmp_path):
    base = {k: v for k, v in VALID.items() if k != "run"}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"run": {"experiment_id": "E"}, "train": {"epochs": 7}}))
    cfg = load_study_config(tmp_path / "exp.yaml", tmp_path / "base.yaml")
    assert cfg.run.experiment_id == "E" and cfg.train.epochs == 7 and cfg.train.batch_size == 4
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
