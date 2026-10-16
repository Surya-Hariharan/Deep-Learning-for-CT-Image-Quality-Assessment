"""Tests for `ct_iqa.config.ExperimentConfig`, in particular the
`checkpoint_dir` property (experiments/<name>/checkpoint/, never
weights/checkpoints/) and `from_yaml_files` (the single authoritative
configuration source -- see docs/internal/repository_architecture_audit.md,
"Experiment Checkpoint Decision" / "Configuration").
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ct_iqa.config import ExperimentConfig


def test_checkpoint_dir_is_derived_from_experiment_dir():
    config = ExperimentConfig(experiment_dir="experiments/001_resnet50_baseline")
    assert config.checkpoint_dir == str(Path("experiments/001_resnet50_baseline") / "checkpoint")


def test_default_dataset_paths_use_canonical_locations():
    """Regression test: these previously pointed at data/training/, data/testing/,
    which no longer exist after the data/ directory rename (see
    docs/internal/repository_architecture_audit.md, "Current Live Bug")."""
    config = ExperimentConfig()
    assert "data/training" not in config.train_image_dir
    assert "data/testing" not in config.test_image_dir
    assert config.train_image_dir == "data/raw/ldct_iqac/train/image"
    assert config.train_json_path == "data/labels/ldct_iqac/train.json"
    assert config.test_image_dir == "data/raw/ldct_iqac/test/images"
    assert config.test_json_path == "data/labels/ldct_iqac/test.json"


def test_save_and_load_roundtrip(tmp_path):
    config = ExperimentConfig(dropout_p=0.3, seed=7)
    path = tmp_path / "config.json"
    config.save(path)

    loaded = ExperimentConfig.load(path)
    assert loaded.dropout_p == 0.3
    assert loaded.seed == 7
    assert loaded.checkpoint_dir == config.checkpoint_dir


def test_from_yaml_files_merges_all_four_files(tmp_path):
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump({"name": "ldct_iqac", "train_image_dir": "d/train", "val_fraction": 0.2})
    )
    (tmp_path / "preprocessing.yaml").write_text(yaml.safe_dump({"image_size": 128}))
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"dropout_p": 0.4, "in_channels": 1}))
    (tmp_path / "training.yaml").write_text(yaml.safe_dump({"seed": 99, "epochs": 5}))

    config = ExperimentConfig.from_yaml_files(
        dataset=tmp_path / "dataset.yaml",
        preprocessing=tmp_path / "preprocessing.yaml",
        model=tmp_path / "model.yaml",
        training=tmp_path / "training.yaml",
    )

    assert config.train_image_dir == "d/train"
    assert config.val_fraction == 0.2
    assert config.image_size == 128
    assert config.dropout_p == 0.4
    assert config.seed == 99
    assert config.epochs == 5


def test_from_yaml_files_consumes_dataset_name_and_architecture(tmp_path):
    """dataset.yaml's `name` and model.yaml's `architecture` are no longer
    silently dropped -- they're consumed into `dataset_name`/`architecture`
    (see docs/internal/repository_architecture_audit.md, "Misleading Configuration
    Keys")."""
    (tmp_path / "dataset.yaml").write_text(yaml.safe_dump({"name": "ldct_iqac"}))
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"architecture": "ohashi_resnet50"}))
    for fn in ("preprocessing.yaml", "training.yaml"):
        (tmp_path / fn).write_text(yaml.safe_dump({}))

    config = ExperimentConfig.from_yaml_files(
        dataset=tmp_path / "dataset.yaml",
        preprocessing=tmp_path / "preprocessing.yaml",
        model=tmp_path / "model.yaml",
        training=tmp_path / "training.yaml",
    )
    assert config.dataset_name == "ldct_iqac"
    assert config.architecture == "ohashi_resnet50"
    assert not hasattr(config, "name")


def test_from_yaml_files_rejects_unsupported_architecture(tmp_path):
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"architecture": "not_a_real_model"}))
    for fn in ("dataset.yaml", "preprocessing.yaml", "training.yaml"):
        (tmp_path / fn).write_text(yaml.safe_dump({}))

    with pytest.raises(ValueError, match="not_a_real_model"):
        ExperimentConfig.from_yaml_files(
            dataset=tmp_path / "dataset.yaml",
            preprocessing=tmp_path / "preprocessing.yaml",
            model=tmp_path / "model.yaml",
            training=tmp_path / "training.yaml",
        )


def test_from_yaml_files_overrides_take_precedence(tmp_path):
    (tmp_path / "training.yaml").write_text(yaml.safe_dump({"seed": 1}))
    for fn in ("dataset.yaml", "preprocessing.yaml", "model.yaml"):
        (tmp_path / fn).write_text(yaml.safe_dump({}))

    config = ExperimentConfig.from_yaml_files(
        dataset=tmp_path / "dataset.yaml",
        preprocessing=tmp_path / "preprocessing.yaml",
        model=tmp_path / "model.yaml",
        training=tmp_path / "training.yaml",
        seed=2,
    )
    assert config.seed == 2


def test_from_yaml_files_reads_real_configs_directory():
    """The actual configs/*.yaml checked into this repository must load cleanly."""
    config = ExperimentConfig.from_yaml_files()
    assert config.experiment_dir == "experiments/001_resnet50_baseline"
    assert config.dropout_p == 0.5
    assert config.dataset_name == "ldct_iqac"
    assert config.architecture == "ohashi_resnet50"
    assert config.selection_metric == "plcc"
