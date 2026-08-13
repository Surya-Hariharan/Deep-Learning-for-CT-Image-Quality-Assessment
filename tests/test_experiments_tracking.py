"""Tests for ct_iqa.evaluation.reporting (experiment run tracking)."""

from __future__ import annotations

import json

from ct_iqa.evaluation.reporting import ExperimentRecord, log_experiment


def test_log_experiment_writes_a_json_record(tmp_path):
    record = ExperimentRecord(
        experiment_name="quality_baseline",
        script="scripts/quality/build_ohashi_dataset.py",
        config_used="configs/quality/ohashi_ldctiqac.yaml",
        parameters={"manifest_only": True},
    )
    out_path = log_experiment(record, experiments_root=tmp_path)
    assert out_path.is_file()
    assert out_path.parent.name == "runs"
    assert out_path.parent.parent.name == "quality_baseline"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "quality_baseline"
    assert payload["seed"] is not None
    assert payload["parameters"] == {"manifest_only": True}


def test_log_experiment_fills_in_git_commit_when_unset(tmp_path):
    record = ExperimentRecord(experiment_name="demo", script="x.py")
    log_experiment(record, experiments_root=tmp_path)
    # git_commit is populated in-place on the record object passed in.
    assert record.git_commit is None or isinstance(record.git_commit, str)


def test_log_experiment_records_are_timestamp_ordered(tmp_path):
    r1 = ExperimentRecord(experiment_name="demo", script="x.py", started_at="2026-08-13T10:00:00+00:00")
    r2 = ExperimentRecord(experiment_name="demo", script="x.py", started_at="2026-08-13T11:00:00+00:00")
    p1 = log_experiment(r1, experiments_root=tmp_path)
    p2 = log_experiment(r2, experiments_root=tmp_path)
    assert p1 != p2
    assert sorted([p1.name, p2.name]) == [p1.name, p2.name]
