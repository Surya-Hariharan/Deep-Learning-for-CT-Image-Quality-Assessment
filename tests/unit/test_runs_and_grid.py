"""Unit tests for the B grid declaration, strict overrides, run-artifact checks and the new report helpers.

SYNTHETIC: no images, no training, no model. Run directories are fabricated JSON/CSV files in tmp_path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from ct_iqa.config.loader import load_study_config
from ct_iqa.config.schema import ConfigError
from ct_iqa.data.sampling import ordered_pair_mask
from ct_iqa.experiment import grid, report
from ct_iqa.experiment.metadata import sha256_path
from ct_iqa.experiment.runs import RunArtifactError, check_compatible, run_summary, runs_table

REPO = Path(__file__).resolve().parents[2]


def test_grid_constants_match_the_declared_grid_in_the_experiment_doc():
    text = (REPO / grid.GRID_DOC).read_text(encoding="utf-8")
    section = text[text.index("## 7."):text.index("## 8.")]
    assert re.search(r"lambda_rank in \{0\.03, 0\.1, 0\.3, 1\.0\}", section)
    assert "`margin = 0.05`" in section and "`min_score_diff = 0.25`" in section
    assert re.search(r"margin in \{0\.02, 0\.1\}", section) and re.search(r"min_score_diff in \{0\.5\}", section)
    assert "within 5% of A" in section
    assert grid.STAGE1_LAMBDAS == (0.03, 0.1, 0.3, 1.0)
    assert (grid.STAGE1_MARGIN, grid.STAGE1_MIN_SCORE_DIFF) == (0.05, 0.25)
    assert grid.STAGE2_MARGINS == (0.02, 0.1) and grid.STAGE2_MIN_SCORE_DIFFS == (0.5,)
    assert grid.RMSE_TOLERANCE == 0.05


def test_stage_points():
    s1 = grid.b_grid_stage1()
    assert [p["lambda_rank"] for p in s1] == [0.03, 0.1, 0.3, 1.0]
    s2 = grid.b_grid_stage2(0.3)
    assert s2 == [{"lambda_rank": 0.3, "margin": 0.02, "min_score_diff": 0.25},
                  {"lambda_rank": 0.3, "margin": 0.1, "min_score_diff": 0.25},
                  {"lambda_rank": 0.3, "margin": 0.05, "min_score_diff": 0.5}]
    with pytest.raises(ValueError, match="not a stage-1 grid value"):
        grid.b_grid_stage2(0.5)


def test_apply_overrides_is_strict_and_does_not_mutate():
    cfg = load_study_config(REPO / "configs/experiments/b_ranking_aware.yaml", REPO / "configs/base.yaml")
    new = grid.apply_overrides(cfg, grid.grid_overrides(grid.b_grid_stage1()[0], seed=43, device="cpu"))
    assert new.loss.ranking.lambda_rank == 0.03 and new.train.seed == 43
    assert cfg.loss.ranking.lambda_rank == 0.1 and cfg.train.seed == 42             # original untouched
    assert new.model == cfg.model and new.data == cfg.data and new.optim == cfg.optim
    with pytest.raises(ConfigError, match="unknown key"):
        grid.apply_overrides(cfg, {"train": {"learning_rate": 1.0}})


def test_declared_deltas_are_off_the_label_grid_so_float32_cannot_flip_a_pair():
    # LDCT-IQAC labels lie on a 0.2 grid; differences k*0.2 computed in float32 may land a hair below k*0.2.
    y = torch.tensor([round(0.2 * k, 1) for k in range(21)], dtype=torch.float32)
    exact = np.subtract.outer(np.arange(21) * 0.2, np.arange(21) * 0.2)
    for delta in (grid.STAGE1_MIN_SCORE_DIFF, *grid.STAGE2_MIN_SCORE_DIFFS):
        assert abs(delta / 0.2 - round(delta / 0.2)) > 0.1                          # not a grid multiple
        assert int(ordered_pair_mask(y, delta).sum()) == int((exact >= delta - 1e-9).sum())


def test_pair_mask_boundary_direction_ties_and_empty():
    y = torch.tensor([1.0, 1.25, 1.0, 3.0])
    m = ordered_pair_mask(y, 0.25)
    assert m[1, 0] and m[1, 2] and not m[0, 1]                 # difference exactly delta is a pair, higher first
    assert not m[0, 2] and not m[2, 0] and not m.diagonal().any()   # ties and the diagonal are excluded
    assert int(ordered_pair_mask(torch.tensor([2.0, 2.0, 2.0]), 0.25).sum()) == 0


def _fake_run(root: Path, exp: str, run_id: str, status: str = "completed", files=None, loss="mse",
              ranking=None, seed=42, metrics=None, labels=(1.0, 2.0, 3.0), preds=(1.1, 2.2, 2.7), names=None):
    d = root / exp / "runs" / run_id
    (d / "checkpoints").mkdir(parents=True)
    names = names or [f"{i}.tif" for i in range(len(labels))]
    metrics = metrics or {"plcc": 0.9, "srocc": 0.8, "krocc": 0.7, "mse": 0.1, "mae": 0.2, "rmse": 0.3}
    (d / "predictions_validation.csv").write_text(
        "filename,label_0_4,prediction_0_4,prediction_normalized\n"
        + "".join(f"{n},{y},{p},{p / 4}\n" for n, y, p in zip(names, labels, preds)))
    cfg = {"data": {"image_size": 224}, "loss": {"name": loss, "ranking": ranking or {}}}
    (d / "config.snapshot.json").write_text(json.dumps(cfg))
    (d / "history.json").write_text(json.dumps({"best_epoch": 2, "epochs": []}))
    (d / "metrics_validation.json").write_text(json.dumps(metrics))
    artifacts = ["config.snapshot.json", "history.json", "metrics_validation.json", "predictions_validation.csv"]
    manifest = {"status": status, "loss": loss, "seed": seed, "split_manifest_sha256": "abc", "bn_eps": 1.001e-5,
                "selection_rule": "val_plcc", "best_epoch_1based": 2, "validation_metrics": metrics,
                "environment": {"device_requested": "cpu"},
                "artifact_sha256": {a: sha256_path(d / a) for a in artifacts}}
    if status == "failed":
        manifest["error"] = {"type": "FloatingPointError", "message": "non-finite"}
    (d / "run_manifest.json").write_text(json.dumps(manifest))
    return d


def test_run_summary_rejects_failed_started_and_tampered_runs(tmp_path):
    ok = _fake_run(tmp_path, "A", "r1")
    assert run_summary(ok)["metrics"]["plcc"] == 0.9
    with pytest.raises(RunArtifactError, match="'failed'.*FloatingPointError"):
        run_summary(_fake_run(tmp_path, "A", "r2", status="failed"))
    with pytest.raises(RunArtifactError, match="'started'"):
        run_summary(_fake_run(tmp_path, "A", "r3", status="started"))
    (ok / "metrics_validation.json").write_text(json.dumps({"plcc": 0.99}))
    with pytest.raises(RunArtifactError, match="SHA-256 differs"):
        run_summary(ok)


def test_check_compatible(tmp_path):
    a = _fake_run(tmp_path, "A", "a1")
    b = _fake_run(tmp_path, "B", "b1", loss="mse_ranking")
    assert set(check_compatible({"A": a, "B": b})) == {"A", "B"}
    c = _fake_run(tmp_path, "C", "c1", names=["0.tif", "2.tif", "1.tif"])
    with pytest.raises(RunArtifactError, match="validation filenames differs"):
        check_compatible({"A": a, "C": c})
    f = _fake_run(tmp_path, "C", "c2", status="failed")
    with pytest.raises(RunArtifactError, match="C: c2: status is 'failed'"):
        check_compatible({"A": a, "C": f})


def test_runs_table_lists_every_status(tmp_path):
    _fake_run(tmp_path, grid.B_EXPERIMENT_ID, "x1", loss="mse_ranking",
              ranking={"lambda_rank": 0.1, "margin": 0.05, "min_score_diff": 0.25})
    _fake_run(tmp_path, grid.B_EXPERIMENT_ID, "x2", status="failed", loss="mse_ranking",
              ranking={"lambda_rank": 0.3, "margin": 0.05, "min_score_diff": 0.25})
    rows = runs_table(grid.B_EXPERIMENT_ID, tmp_path)
    assert [(r["run_id"], r["status"], r["lambda_rank"]) for r in rows] == [("x1", "completed", 0.1), ("x2", "failed", 0.3)]
    assert rows[1]["error"] == "FloatingPointError"
    pending = grid.pending_points(grid.b_grid_stage1(), seed=42, output_root=tmp_path)
    assert [p["lambda_rank"] for p in pending] == [0.03, 0.3, 1.0]               # only completed points are done


def _complete_stage1(tmp_path, rmses, corrs):
    for i, (p, rmse, c) in enumerate(zip(grid.b_grid_stage1(), rmses, corrs)):
        _fake_run(tmp_path, grid.B_EXPERIMENT_ID, f"b{i}", loss="mse_ranking", ranking=p,
                  metrics={"plcc": c, "srocc": c, "krocc": c, "mse": rmse ** 2, "mae": rmse, "rmse": rmse})


def test_select_stage1_requires_complete_unambiguous_stage(tmp_path):
    a = _fake_run(tmp_path, "A_reference_resnet50", "a1")
    with pytest.raises(ValueError, match="stage 1 is incomplete"):
        grid.select_stage1(a, seed=42, output_root=tmp_path)
    _complete_stage1(tmp_path, [0.30] * 4, [0.8] * 4)
    _fake_run(tmp_path, grid.B_EXPERIMENT_ID, "b9", loss="mse_ranking", ranking=grid.b_grid_stage1()[0])
    with pytest.raises(ValueError, match="ambiguous"):
        grid.select_stage1(a, seed=42, output_root=tmp_path)


def test_select_stage1_applies_rmse_guard_then_highest_mean_correlation(tmp_path):
    a = _fake_run(tmp_path, "A_reference_resnet50", "a1")                         # rmse 0.3, mean corr 0.8
    # lambda 0.03: corr 0.85 ok; 0.1: corr 0.95 but RMSE 0.32 > 0.315 (fails guard); 0.3: 0.9 ok; 1.0: 0.7 ok
    _complete_stage1(tmp_path, [0.30, 0.32, 0.31, 0.30], [0.85, 0.95, 0.90, 0.70])
    rec = grid.select_stage1(a, seed=42, output_root=tmp_path)
    assert rec["selected"]["lambda_rank"] == 0.3 and rec["selected"]["run_id"] == "b2"
    assert [c["rmse_within_5pct_of_A"] for c in rec["candidates"]] == [True, False, True, True]
    assert "bootstrap_vs_A" in rec and isinstance(rec["improvement_demonstrated"], bool)
    assert "validation" in rec["partition"] and "selection-biased" in rec["partition"]
    path = grid.write_selection_record(rec, tmp_path)
    assert json.loads(path.read_text())["stage"] == 1
    with pytest.raises(FileExistsError):
        grid.write_selection_record(rec, tmp_path)


def test_select_stage1_no_eligible_point(tmp_path):
    a = _fake_run(tmp_path, "A_reference_resnet50", "a1")
    _complete_stage1(tmp_path, [0.5] * 4, [0.9] * 4)
    rec = grid.select_stage1(a, seed=42, output_root=tmp_path)
    assert rec["selected"] is None and "A remains" in rec["outcome"]


def test_config_frame_side_by_side():
    a = load_study_config(REPO / "configs/experiments/a_reference.yaml", REPO / "configs/base.yaml").to_dict()
    b = load_study_config(REPO / "configs/experiments/b_ranking_aware.yaml", REPO / "configs/base.yaml").to_dict()
    side = report.config_frame(A=a, B=b)
    differs = side.index[side["A"].astype(str) != side["B"].astype(str)].tolist()
    assert set(differs) == {"loss.name", "loss.ranking.lambda_rank", "run.experiment_id"}
    assert side.loc["model.bn_eps", "A"] == 1.001e-5


def test_report_tables():
    y = np.array([0.0, 0.4, 1.0, 2.2, 3.0, 4.0])
    p = y + np.array([0.1, -0.1, 0.2, 0.0, -0.3, -0.2])
    t = report.error_by_score_bin(y, p)
    assert t["n"].tolist() == [2, 1, 1, 2] and t["n"].sum() == len(y)          # 4.0 lands in the closed last bin
    assert t.loc["[0, 1)", "MAE"] == pytest.approx(0.1)
    b = report.binned_sigma_calibration(np.arange(1, 11, dtype=float), np.ones(10), n_bins=5)
    assert b["n"].tolist() == [2] * 5
    h = report.clamp_hits([-10.0, -3.0, 2.0, 2.0])
    assert h["n"] == 4 and h["at lower bound -10"] == 1 and h["at upper bound 2"] == 2
    s = report.seed_aggregate([{"plcc": 0.8, "srocc": 0.7, "krocc": 0.6, "mse": 0.1, "mae": 0.2, "rmse": 0.3},
                               {"plcc": 0.9, "srocc": 0.7, "krocc": 0.6, "mse": 0.1, "mae": 0.2, "rmse": 0.3}])
    assert s.loc["plcc", "mean"] == pytest.approx(0.85) and s.loc["srocc", "SD across seeds"] == 0
    assert (s["n seeds"] == 2).all()


def test_report_figures():
    import matplotlib
    matplotlib.use("Agg")
    hist = {"best_epoch": 2, "epochs": [
        {"epoch": e, "train_loss": 1.0 / e, "val_loss": 1.2 / e, "val_plcc": 0.5, "val_srocc": 0.5, "val_krocc": 0.4,
         "val_mse": 0.1, "val_mae": 0.2, "val_rmse": 0.3, "selected": e == 2,
         "train_loss_parts": {"mse": 0.5 / e, "rank": 0.1, "n_valid_pairs": 30.0, "n_possible_pairs": 40.0},
         "val_loss_parts": {"mse": 0.6 / e, "rank": 0.1, "n_valid_pairs": 20.0, "n_possible_pairs": 40.0}}
        for e in (1, 2, 3)]}
    assert len(report.plot_loss_parts(hist, "t").axes) == 2
    assert len(report.plot_residuals([1, 2, 3], [1.1, 1.9, 3.2]).axes) == 2
    assert len(report.plot_sigma_vs_error([0.1, 0.2], [0.05, 0.3]).axes) == 1
