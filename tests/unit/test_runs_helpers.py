"""Unit tests for run-directory helpers (SYNTHETIC directories in tmp_path)."""

from __future__ import annotations

import json

from ct_iqa.experiment.runs import completed_runs, latest_completed_run, list_runs, read_predictions


def _mk(root, exp, run_id, status):
    d = root / exp / "runs" / run_id
    d.mkdir(parents=True)
    (d / "run_manifest.json").write_text(json.dumps({"status": status}))
    return d


def test_listing_filters_incomplete_and_orders_by_run_id(tmp_path):
    a = _mk(tmp_path, "E", "E_20260101T000000Z_s1_aaaaaa", "completed")
    _mk(tmp_path, "E", "E_20260102T000000Z_s1_bbbbbb", "started")
    c = _mk(tmp_path, "E", "E_20260103T000000Z_s1_cccccc", "completed")
    assert [p.name for p in list_runs("E", tmp_path)][0] == a.name
    assert completed_runs("E", tmp_path) == [a, c]
    assert latest_completed_run("E", tmp_path) == c
    assert latest_completed_run("missing", tmp_path) is None


def test_read_predictions(tmp_path):
    (tmp_path / "predictions_validation.csv").write_text(
        "filename,label_0_4,prediction_0_4,prediction_normalized\na.tif,1.0,1.5,0.375\nb.tif,2.0,2.5,0.625\n")
    cols = read_predictions(tmp_path)
    assert cols["filename"] == ["a.tif", "b.tif"] and cols["prediction_0_4"] == [1.5, 2.5]


def test_select_run_is_explicit_and_never_guesses(tmp_path):
    import pytest

    from ct_iqa.experiment.runs import select_run

    assert select_run("E", output_root=tmp_path) is None                     # nothing run yet
    a = _mk(tmp_path, "E", "E_20260101T000000Z_s1_aaaaaa", "completed")
    _mk(tmp_path, "E", "E_20260102T000000Z_s1_bbbbbb", "started")           # incomplete runs are ignored
    assert select_run("E", output_root=tmp_path) == a                        # exactly one completed run
    c = _mk(tmp_path, "E", "E_20260103T000000Z_s2_cccccc", "completed")
    with pytest.raises(ValueError, match="set RUN_ID"):                      # two completed runs: ambiguous
        select_run("E", output_root=tmp_path)
    assert select_run("E", c.name, tmp_path) == c and select_run("E", a.name, tmp_path) == a
    with pytest.raises(ValueError, match="not a completed run"):             # unknown or incomplete id
        select_run("E", "E_20260102T000000Z_s1_bbbbbb", tmp_path)
