"""Create the study-v3 FROZEN marker (hashes of the protocol, manifests, configs and result-producing code). Refuses to overwrite.

    python tools/create_freeze_v3.py
"""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FILES = [
    "docs/methodology/STUDY_V3_PROTOCOL.md",
    "experiments/study_v3/protocol/group_manifest.json", "experiments/study_v3/protocol/fold_manifest.json",
    "configs/base.yaml", "configs/experiments/a_reference.yaml", "configs/experiments/b_ranking_aware.yaml", "configs/experiments/c_uncertainty_aware.yaml",
    "src/ct_iqa/data/grouped.py", "src/ct_iqa/experiment/run_v3.py", "src/ct_iqa/experiment/run.py", "src/ct_iqa/experiment/grid.py",
    "src/ct_iqa/training/engine.py", "src/ct_iqa/training/checkpointing.py", "src/ct_iqa/models/registry.py", "src/ct_iqa/losses/ranking.py",
    "src/ct_iqa/losses/gaussian_nll.py", "src/ct_iqa/evaluation/intervals.py", "src/ct_iqa/evaluation/metrics.py",
    "tools/run_study_v3.py", "tools/select_b_v3.py", "tools/build_reports_v3.py", "tools/verify_study_v3.py", "tools/create_study_v3_manifests.py",
    "reports/study_v3/phase2_verification.json",
]


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main() -> int:
    out = Path("experiments/study_v3/protocol/FROZEN")
    if out.exists():
        raise SystemExit("FROZEN exists; a changed protocol needs a new version (v3.1), not an edit")
    ver = json.loads(Path("reports/study_v3/phase2_verification.json").read_text(encoding="utf-8"))
    if not ver["PASS"]:
        raise SystemExit("Phase 2 verification did not pass")
    marker = {"protocol_version": "study_v3 v3.0", "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "head": git("rev-parse", "HEAD"), "head_is_c10014f_cec8ebd_descendant": git("merge-base", "--is-ancestor", "cec8ebd", "HEAD") == "",
              "working_tree_dirty": bool(git("status", "--porcelain")), "python": sys.version.split()[0],
              "note": "Created after Phase 2 verification passed and before any selection or final run. Official test set is not part of any fold.",
              "sha256": {f: sha(f) for f in FILES},
              "planned_runs": {"selection_fold0_seed42": "A + 7 B grid points (8 runs)", "final": "A, B(selected), C x folds 0-4 x seeds 1,2,3 = 45 runs"}}
    out.write_text(json.dumps(marker, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in marker.items() if k != "sha256"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
