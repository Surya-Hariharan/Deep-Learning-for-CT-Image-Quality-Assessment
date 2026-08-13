"""Experiment run reporting: what ran, with what config and seed, when.

PROJECT ADAPTATION -- not part of Ohashi's methodology, pure engineering
scaffolding so that every run (including non-training runs, like the dataset
planning stage) leaves a reproducible trail under experiments/.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ct_iqa.config.loader import project_root, random_seed
from ct_iqa.utils.paths import resolve


@dataclass
class ExperimentRecord:
    experiment_name: str
    script: str
    config_used: str | None = None
    seed: int = field(default_factory=random_seed)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    git_commit: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root(),
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def log_experiment(record: ExperimentRecord, experiments_root: Path | None = None) -> Path:
    """Write one run record to experiments/<name>/runs/<timestamp>.json."""
    if record.git_commit is None:
        record.git_commit = _git_commit()

    root = experiments_root or resolve("experiments")
    out_dir = root / record.experiment_name / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = record.started_at.replace(":", "").replace("-", "")
    out_path = out_dir / f"{stamp}.json"
    out_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return out_path
