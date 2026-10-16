"""Locate and read finished run directories (used by the study notebooks)."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def list_runs(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> list[Path]:
    """Run directories of an experiment, oldest first (run IDs start with a UTC timestamp)."""
    runs = Path(output_root) / experiment_id / "runs"
    return sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []


def completed_runs(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> list[Path]:
    out = []
    for p in list_runs(experiment_id, output_root):
        m = p / "run_manifest.json"
        if m.is_file() and json.loads(m.read_text(encoding="utf-8")).get("status") == "completed":
            out.append(p)
    return out


def latest_completed_run(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> Path | None:
    runs = completed_runs(experiment_id, output_root)
    return runs[-1] if runs else None


def read_predictions(run_dir: str | Path, name: str = "predictions_validation.csv") -> dict[str, list]:
    """Columns of a predictions CSV as lists (strings for filename, floats otherwise)."""
    with (Path(run_dir) / name).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: ([r[k] for r in rows] if k == "filename" else [float(r[k]) for r in rows])
            for k in (rows[0] if rows else {})}
