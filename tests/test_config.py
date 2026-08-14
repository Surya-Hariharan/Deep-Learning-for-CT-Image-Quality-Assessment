"""Tests for ct_iqa.config.loader: configs load, Ohashi values are present
and correct, training defaults match the specification."""

from __future__ import annotations

from ct_iqa.config.loader import (
    dataset_config,
    ldctiqac_config,
    load_config,
    ohashi_degradation_config,
    paths_config,
    random_seed,
    resnet50_config,
)


def test_paths_config_loads():
    cfg = paths_config()
    assert "random_seed" in cfg
    assert "data" in cfg


def test_random_seed_is_an_int():
    assert isinstance(random_seed(), int)


def test_dataset_config_loads_for_every_configured_dataset():
    for name in ("png", "lidc_idri", "luna16", "lndb", "deeplesion", "cq500", "ldctiqa"):
        cfg = dataset_config(name)
        assert cfg["name"] == name
        assert cfg.get("role")


def test_ldctiqac_config_convenience_accessor():
    assert ldctiqac_config() == dataset_config("ldctiqa")


def test_load_config_is_cached():
    """Configs are read-only for the process lifetime -- the loader caches them."""
    a = load_config("configs/paths.yaml")
    b = load_config("configs/paths.yaml")
    assert a is b


# ------------------------------------------------- Ohashi degradation values

def test_active_degradation_config_uses_ldctiqac():
    cfg = ohashi_degradation_config(active=True)
    assert cfg["references"]["source"] == "ldctiqa"
    assert cfg["references"]["n_total"] == 1000


def test_original_degradation_config_documents_ohashi_baseline():
    cfg = ohashi_degradation_config(active=False)
    assert cfg["references"]["n_total"] == 105
    assert cfg["references"]["sources"] == {"deeplesion": 90, "cq500": 15}


def test_noise_sigma_grid_matches_ohashi_spec():
    cfg = ohashi_degradation_config()
    assert cfg["noise"]["sigma"] == [1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 14, 21, 30, 50]


def test_blur_sigma_grid_matches_ohashi_spec():
    cfg = ohashi_degradation_config()
    assert cfg["blur"]["sigma"] == [0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.4, 2.1, 3.0, 5.0]


def test_degradation_split_ratios_are_60_20_20():
    cfg = ohashi_degradation_config()
    assert cfg["splits"]["ratios"] == {"train": 0.6, "val": 0.2, "test": 0.2}


def test_degradation_dataset_total_is_derived_correctly():
    cfg = ohashi_degradation_config()
    assert cfg["counts"]["dataset_total"] == 169000
    assert cfg["splits"]["image_counts"] == {"train": 101400, "val": 33800, "test": 33800}


def test_synthetic_target_is_vif_not_expert_score():
    cfg = ohashi_degradation_config()
    assert cfg["labels"]["synthetic_target"] == "vif"


# ------------------------------------------------------------ training defaults

def test_training_defaults_match_ohashi_spec():
    cfg = resnet50_config()
    training = cfg["training"]
    assert training["optimizer"] == "adam"
    assert training["batch_size"] == 64
    assert training["epochs"] == 30
    assert training["loss"] == "mse"
    assert training["learning_rate_grid"] == [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
    assert training["learning_rate_reported_best_resnet50"] == 1.0e-3


def test_model_architecture_matches_ohashi_spec():
    cfg = resnet50_config()
    model = cfg["model"]
    assert model["backbone"] == "resnet50"
    assert model["pretrained_weights"] == "radimagenet"
    assert model["input_shape"] == [224, 224, 3]
    assert model["crop_strategy"] == "central_crop"


def test_dropout_and_freeze_backbone_are_now_resolved():
    """Resolved 2026-08-14 (docs/research_decisions.md decisions S-01, A-22):
    freeze_backbone is OHASHI-SPECIFIED (fine-tuned, confirmed from paper
    text); dropout_rate is a documented PROJECT-ADAPTATION baseline (0.5,
    sourced from RadImageNet's own transfer-learning recipe), not a guess
    and not claimed to be Ohashi-specified."""
    cfg = resnet50_config()
    assert cfg["model"]["freeze_backbone"] is False
    assert cfg["model"]["dropout_rate"] == 0.5


def test_three_stage_evaluation_structure_is_present():
    cfg = resnet50_config()
    evaluation = cfg["evaluation"]
    assert "stage_1_synthetic" in evaluation
    assert "stage_2_subjective" in evaluation
    assert "stage_3_real_image" in evaluation
    assert evaluation["stage_1_synthetic"]["compares"] == "predicted_score vs vif_score"
    assert evaluation["stage_2_subjective"]["compares"] == "predicted_score vs expert_score"
