"""Tests for scripts/verify_git_safety.py and the .gitignore dataset guard.

These import the script's pure functions directly (no subprocess) for the
suffix/size logic, and separately shell out to `git check-ignore` to prove the
.gitignore patterns actually work against this repository's real config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ct_iqa.utils.paths import project_root  # noqa: E402
import verify_git_safety as vgs  # noqa: E402


# ------------------------------------------------------------- pure logic

def test_common_ct_formats_are_flagged_suspicious():
    tracked = ["data/raw/ldctiqa/LDCTiqa_png/0000.png", "notes/scan.dcm", "archive.nii.gz", "backup.zip"]
    suspicious = vgs.find_suspicious(tracked)
    assert set(suspicious) == set(tracked)


def test_checkpoints_are_flagged_suspicious():
    tracked = ["model.ckpt", "weights.pth", "weights.h5", "export.onnx"]
    assert set(vgs.find_suspicious(tracked)) == set(tracked)


def test_source_and_config_files_are_not_flagged():
    tracked = ["src/ct_iqa/degradation/generator.py", "configs/paths.yaml", "README.md", "requirements.txt"]
    assert vgs.find_suspicious(tracked) == []


def test_docs_prefix_is_allow_listed():
    """Small documentation images are not a leak; only bulk dataset dirs are."""
    tracked = ["docs/architecture_diagram.png"]
    assert vgs.find_suspicious(tracked) == []


def test_generated_outputs_are_flagged():
    tracked = ["experiments/quality_baseline/runs/predictions.npy", "checkpoints/best.pt"]
    assert set(vgs.find_suspicious(tracked)) == set(tracked)


def test_no_tracked_files_is_safe():
    assert vgs.find_suspicious([]) == []


def test_find_large_files_flags_over_threshold(tmp_path):
    big = tmp_path / "big.csv"
    big.write_bytes(b"0" * (6 * 1000 * 1000))  # decimal MB, matching find_large_files' /1e6
    small = tmp_path / "small.csv"
    small.write_bytes(b"0" * 100)

    import verify_git_safety as vgs_module

    original_root = vgs_module.project_root
    vgs_module.project_root = lambda: tmp_path  # type: ignore[assignment]
    try:
        large = vgs_module.find_large_files(["big.csv", "small.csv"], threshold_mb=5.0)
    finally:
        vgs_module.project_root = original_root
    assert large == [("big.csv", pytest.approx(6.0, abs=0.01))]


# ------------------------------------------------------- real .gitignore check

def _git_check_ignore(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=project_root(), check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/ldctiqa/LDCTiqa_png/0000.png",
        "data/interim/ldctiqa/whatever.png",
        "data/processed/quality_iqa/degraded/noise/x.png",
        "experiments/quality_baseline/runs/checkpoint.pt",
    ],
)
def test_gitignore_actually_protects_dataset_paths(path):
    assert _git_check_ignore(path), f"{path} is NOT ignored by .gitignore"


def test_gitignore_does_not_blanket_ignore_all_png():
    """Small legitimate fixtures/docs images must remain committable."""
    assert not _git_check_ignore("docs/some_diagram.png")
