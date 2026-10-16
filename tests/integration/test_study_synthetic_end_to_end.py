"""SYNTHETIC end-to-end tests of the study-v2 pipeline (Experiments A, B, C).

Everything here is fabricated: 48 random 72x72 float TIFFs with random scores in [0, 4], a 64x64 crop, 2 epochs,
no pretrained weights, a temporary directory. These tests verify that the code paths run, produce the documented
artifacts, are reproducible, and never touch the official test set. They say NOTHING about model quality.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ct_iqa.config.loader import study_config_from_dict
from ct_iqa.data.splits import build_manifest, save_manifest
from ct_iqa.experiment import run as run_module
from ct_iqa.experiment.run import final_test_evaluation, run_experiment
from ct_iqa.training import engine

N, SRC, CROP = 48, 72, 64


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_ldct")
    rng = np.random.default_rng(0)
    img_dir = root / "train"; img_dir.mkdir()
    labels = {}
    for i in range(N):
        arr = rng.random((SRC, SRC), dtype=np.float32)
        Image.fromarray(arr, mode="F").save(img_dir / f"{i:04d}.tif")
        labels[f"{i:04d}.tif"] = float(rng.integers(0, 21)) / 5.0
    (root / "train.json").write_text(json.dumps(labels))
    save_manifest(build_manifest(labels, 8, 8, seed=3, rationale="synthetic test"), root / "split.json")
    return root


def _cfg(root: Path, model: str, loss: str, experiment_id: str, seed: int = 1, ranking=None) -> "StudyConfig":
    raw = {
        "model": {"name": model, "dropout_p": 0.5, "bn_eps": 1.001e-5},
        "loss": {"name": loss, **({"ranking": ranking} if ranking else {})},
        "data": {"train_image_dir": str(root / "train"), "train_json_path": str(root / "train.json"),
                 "test_image_dir": str(root / "no_test"), "test_json_path": str(root / "no_test.json"),
                 "split_manifest": str(root / "split.json"), "image_size": CROP},
        "train": {"epochs": 2, "batch_size": 8, "seed": seed, "device": "cpu"},
        "run": {"experiment_id": experiment_id, "output_root": str(root / "out")},
    }
    return study_config_from_dict(raw)


def test_experiment_a_artifacts_and_provenance(synthetic_root):
    run_dir = run_experiment(_cfg(synthetic_root, "resnet50_regression", "mse", "A_syn"))
    for rel in ("config.snapshot.json", "history.json", "metrics_validation.json", "predictions_validation.csv",
                "checkpoints/best.pt", "run_manifest.json"):
        assert (run_dir / rel).is_file(), rel
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert m["status"] == "completed" and m["official_test_set_used"] is False
    assert m["epoch_indexing"] == "1-based" and 1 <= m["best_epoch_1based"] <= 2
    assert m["split_manifest_sha256"] and set(m["git"]) == {"commit", "branch", "dirty"}
    assert set(m["artifact_sha256"]) >= {"checkpoints/best.pt", "history.json"}
    val = json.loads((run_dir / "metrics_validation.json").read_text())
    assert val["n"] == 8 and val["scale"] == "raw [0, 4]" and np.isfinite(val["mae"])
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    assert ck["epoch_indexing"] == "1-based" and ck["bn_eps"] == 1.001e-5
    assert not (run_dir / "calibration.json").exists()                  # calibration is Experiment C only


def test_distinct_runs_never_share_a_directory_and_same_seed_reproduces_training_loss(synthetic_root):
    a = run_experiment(_cfg(synthetic_root, "resnet50_regression", "mse", "A_repro", seed=5))
    b = run_experiment(_cfg(synthetic_root, "resnet50_regression", "mse", "A_repro", seed=6))
    c = run_experiment(_cfg(synthetic_root, "resnet50_regression", "mse", "A_repro2", seed=5))
    assert len({a, b, c}) == 3
    ha = json.loads((a / "history.json").read_text())["epochs"]
    hc = json.loads((c / "history.json").read_text())["epochs"]
    assert [e["train_loss"] for e in ha] == pytest.approx([e["train_loss"] for e in hc], rel=1e-5)


def test_experiment_b_runs_and_records_valid_pairs(synthetic_root):
    cfg = _cfg(synthetic_root, "resnet50_regression", "mse_ranking", "B_syn",
               ranking={"lambda_rank": 0.5, "margin": 0.05, "min_score_diff": 0.25})
    run_dir = run_experiment(cfg)
    hist = json.loads((run_dir / "history.json").read_text())
    parts = hist["epochs"][0]["train_loss_parts"]
    assert parts["n_valid_pairs"] > 0 and "rank" in parts and "mse" in parts


def test_experiment_c_calibration_uses_calibration_partition_and_reports_both_intervals(synthetic_root):
    run_dir = run_experiment(_cfg(synthetic_root, "resnet50_gaussian", "gaussian_nll", "C_syn"))
    cal = json.loads((run_dir / "calibration.json").read_text())
    assert cal["method"] == "split_conformal" and cal["n_calibration"] == 8
    assert cal["rank_k"] == 9 and cal["unbounded"] is True       # n=8: ceil(9 * 0.9) = 9 > 8 -> unbounded interval
    assert cal["fitted_on"] == "calibration partition" and "validation" in cal["evaluated_on"]
    assert {"unclipped", "clipped_to_0_4", "nll_score_scale", "sigma_error_spearman"} <= set(cal)
    assert np.isfinite(cal["nll_score_scale"])
    csv_header = (run_dir / "predictions_validation.csv").read_text().splitlines()[0]
    assert "log_var_normalized" in csv_header


def test_final_test_evaluation_is_guarded(synthetic_root, tmp_path):
    run_dir = run_experiment(_cfg(synthetic_root, "resnet50_regression", "mse", "A_guard"))
    with pytest.raises(PermissionError, match="retrospective"):
        final_test_evaluation(run_dir, acknowledge_retrospective=False, marker=tmp_path / "FROZEN")
    with pytest.raises(PermissionError, match="not frozen"):
        final_test_evaluation(run_dir, acknowledge_retrospective=True, marker=tmp_path / "FROZEN")
    assert not (run_dir / "test_retrospective").exists()


def test_training_and_calibration_code_never_references_the_test_set():
    for fn in (run_module.run_experiment, engine.fit, engine.evaluate_loader):
        src = inspect.getsource(fn)
        assert "test_image_dir" not in src and "test_json_path" not in src and "build_test" not in src, fn.__name__
    import ct_iqa.data.loader as loader
    study_src = inspect.getsource(loader.build_study_datasets) + inspect.getsource(loader.build_study_loaders)
    assert "test" not in study_src.lower().replace("validation", "")
