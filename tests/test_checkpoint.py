"""Tests for reference-level checkpoint/resume support
(``ct_iqa.data.checkpoint``)."""

from __future__ import annotations

import json

import pytest

from ct_iqa.data.checkpoint import (
    CheckpointManager,
    atomic_append_line,
    atomic_write_text,
)


def _records(ref_id: str, n: int = 3) -> list[dict]:
    return [
        {
            "reference_id": ref_id, "source_filename": f"{ref_id}.png", "expert_score": "2.0",
            "degraded_id": f"{ref_id}_cond{i}", "degradation_type": "noise", "noise_sigma": "1.0",
            "blur_sigma": "", "vif_score": "0.5", "diagnostic_status": "ok",
            "max_channel_gain": "1.0", "covariance_condition_number": "10.0", "split": "train",
        }
        for i in range(n)
    ]


def _manager(tmp_path, total_references=2) -> CheckpointManager:
    return CheckpointManager(
        tmp_path / "checkpoint",
        config_hash_value="abc123",
        code_commit_hash_value="deadbeef",
        seed=42,
        total_references=total_references,
    )


def test_fresh_checkpoint_has_nothing_complete(tmp_path):
    mgr = _manager(tmp_path)
    assert mgr.load_completed() == {}
    assert mgr.is_reference_complete("REF_0000", expected_n_records=3) is False


def test_write_shard_and_mark_complete_round_trips(tmp_path):
    mgr = _manager(tmp_path)
    records = _records("REF_0000")
    mgr.write_shard_and_mark_complete("REF_0000", records)

    assert mgr.is_reference_complete("REF_0000", expected_n_records=3) is True
    read_back = mgr.read_shard("REF_0000")
    assert len(read_back) == 3
    assert read_back[0]["degraded_id"] == "REF_0000_cond0"


def test_resume_sees_prior_completion_via_new_manager_instance(tmp_path):
    mgr1 = _manager(tmp_path)
    mgr1.write_shard_and_mark_complete("REF_0000", _records("REF_0000"))

    mgr2 = _manager(tmp_path)
    assert mgr2.is_reference_complete("REF_0000", expected_n_records=3) is True
    assert "REF_0000" in mgr2.load_completed()


def test_wrong_record_count_is_not_complete(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write_shard_and_mark_complete("REF_0000", _records("REF_0000", n=3))
    assert mgr.is_reference_complete("REF_0000", expected_n_records=169) is False


def test_missing_shard_file_is_not_complete(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write_shard_and_mark_complete("REF_0000", _records("REF_0000"))
    (mgr.shard_dir / "REF_0000.csv").unlink()
    assert mgr.is_reference_complete("REF_0000", expected_n_records=3) is False


def test_corrupted_shard_content_is_not_complete(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write_shard_and_mark_complete("REF_0000", _records("REF_0000"))
    shard_path = mgr.shard_dir / "REF_0000.csv"
    shard_path.write_text(shard_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    assert mgr.is_reference_complete("REF_0000", expected_n_records=3) is False


def test_truncated_final_log_line_is_skipped_not_fatal(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write_shard_and_mark_complete("REF_0000", _records("REF_0000"))
    with mgr.completed_log_path.open("a", encoding="utf-8") as fh:
        fh.write('{"reference_id": "REF_0001", "n_reco')  # truncated, no newline

    mgr2 = _manager(tmp_path)
    completed = mgr2.load_completed()
    assert "REF_0000" in completed
    assert "REF_0001" not in completed


def test_progress_manifest_starts_fresh_then_resumes(tmp_path):
    mgr = _manager(tmp_path)
    progress = mgr.start_or_resume()
    assert progress.status == "in_progress"
    assert progress.n_references_completed == 0

    progress.n_references_completed = 1
    mgr.checkpoint_progress(progress)

    mgr2 = _manager(tmp_path)
    resumed = mgr2.start_or_resume()
    assert resumed.n_references_completed == 1
    assert resumed.updated_at != ""


def test_progress_manifest_rejects_config_hash_mismatch(tmp_path):
    mgr = _manager(tmp_path)
    mgr.start_or_resume()

    mismatched = CheckpointManager(
        tmp_path / "checkpoint", config_hash_value="different-hash",
        code_commit_hash_value="deadbeef", seed=42, total_references=2,
    )
    with pytest.raises(ValueError, match="config_hash mismatch"):
        mismatched.start_or_resume()


def test_progress_manifest_rejects_seed_mismatch(tmp_path):
    mgr = _manager(tmp_path)
    mgr.start_or_resume()

    mismatched = CheckpointManager(
        tmp_path / "checkpoint", config_hash_value="abc123",
        code_commit_hash_value="deadbeef", seed=999, total_references=2,
    )
    with pytest.raises(ValueError, match="seed mismatch"):
        mismatched.start_or_resume()


def test_iter_all_shard_records_streams_across_references(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write_shard_and_mark_complete("REF_0000", _records("REF_0000", n=2))
    mgr.write_shard_and_mark_complete("REF_0001", _records("REF_0001", n=3))

    all_records = list(mgr.iter_all_shard_records())
    assert len(all_records) == 5
    ref_ids = {r["reference_id"] for r in all_records}
    assert ref_ids == {"REF_0000", "REF_0001"}


def test_atomic_write_text_leaves_no_tmp_file_behind(tmp_path):
    path = tmp_path / "out" / "file.json"
    atomic_write_text(path, json.dumps({"a": 1}))
    assert path.exists()
    leftovers = [p for p in path.parent.iterdir() if p.name != "file.json"]
    assert leftovers == []


def test_atomic_append_line_appends_multiple_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    atomic_append_line(path, json.dumps({"a": 1}))
    atomic_append_line(path, json.dumps({"a": 2}))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1
    assert json.loads(lines[1])["a"] == 2
