# Validation report (Stage 10)

Date: 2026-10-16. Branch `study-v2-rebuild` (from `main` at `821461ee24c7fea4d6c1b787c626cbde076ee998`). Committed
locally as five commits (`b742ed5` to `4cb61de`); not pushed. Environment: Windows, CPU only, torch 2.14.0, Python 3.10 venv created by `uv sync --extra dev`
(`uv.lock` unchanged).

## Tests run (exact outcomes)

| Command | Result |
|---|---|
| `pytest tests/unit` (baseline, before any code change) | 81 passed in 55 s |
| `pytest tests/unit tests/regression tests/integration/test_study_synthetic_end_to_end.py` (final) | **142 passed** in 55 s |
| `ct_iqa.cli validate-config` for A, B, C configs | exit 0 each: manifest verified, pretrained weights SHA-256 verified |
| `ct_iqa.cli --help` | prints usage for the four sub-commands |
| `tools/inspect_checkpoint_compatibility.py` | 6/6 checkpoints strict-structure compatible (320/320 keys, 0 missing, 0 unexpected, 0 shape/dtype mismatches, files unchanged) |
| Study notebooks 00-07, executed in memory with every training flag off | all executed without error; experiment notebooks printed "NO COMPLETED RUN FOUND (status PENDING)"; nothing saved |
| Markdown check (38 files): fences balanced, 83 relative links, 9 Mermaid blocks | all links resolve; Mermaid blocks were checked **structurally only** (header, brackets, quotes), not rendered |

Synthetic tests (fabricated data, mechanics only) are identified in their module docstrings:
`test_registry_and_bn_eps`, `test_study_config`, `test_three_way_split`, `test_losses`, `test_intervals`, `test_bootstrap`,
`test_runs_helpers`, `tests/regression/test_legacy_compatibility` (synthetic-weight equivalence part), and
`tests/integration/test_study_synthetic_end_to_end`. The checkpoint-structure regression test reads the six real
checkpoints but only compares keys and shapes.

## Boundary disclosure

While validating, I ran `pytest tests` (the whole directory) once, which included the pre-existing real-data integration
tests, contrary to the intention to keep real-data tests out of this stage. They (a) counted/read the real train and test
images (file counts and image-content hashes for the train/test overlap check), (b) performed **one optimizer step** on a
batch of 4 real training images with a freshly randomly initialized legacy model (not a checkpoint; nothing saved),
(c) loaded the real converted weights and ran forward passes on random noise. No checkpoint was loaded for inference, no
metric was computed from official test labels, and no file was written (verified: git status and `experiments/study_v2`
contain only intended files; all original artifact hashes unchanged). This is reported for transparency; the final
test selection above excludes those tests.

## What was NOT done

No study-v2 training run; no inference with any historical or new checkpoint; no official-test evaluation; no BatchNorm
epsilon impact measurement on historical predictions (remains unquantified); no push; Mermaid diagrams not
rendered; notebooks 03-07 not executed against real runs (none exist).

## Artifact preservation check

Backup: `C:\Users\surya\ct_iqa_backup`, 137 files, SHA-256 verified at creation. Re-check after the rebuild
(`sha256sum -c docs/audit/BACKUP_MANIFEST.sha256` from the repository root): the only mismatches are the intentionally
edited or moved files: `experiments/001_resnet50_baseline/config.yaml` (note added), `docs/README.md` (index rewritten),
`src/ct_iqa/config.py` (moved to
`config/legacy.py`), `src/ct_iqa/data/{loader,splits}.py`, `src/ct_iqa/evaluation/metrics.py`,
`src/ct_iqa/models/resnet50.py`, `src/ct_iqa/training/{checkpointing,optimizers}.py`. All six checkpoints, both weight
files, label files, `results/`, notebooks, `docs/paper`, `docs/replication`, tools and the other experiment files match.

## Known open items

- BatchNorm epsilon effect on 001-003 unquantified; RadImageNet TF-release input convention unconfirmed (ADR-002).
- Experiment B's grid and Experiment C's constant-width baseline are declared, not run.
- `OhashiResNet50` and the legacy notebook generators remain until their migration is verified.
- The `ct-iqa` console script is declared in `pyproject.toml` but requires reinstalling the package to appear; use
  `python -m ct_iqa.cli` meanwhile.
- `uv sync` recreated the ignored `src/ct_iqa.egg-info/`; it is generated and ignored.
