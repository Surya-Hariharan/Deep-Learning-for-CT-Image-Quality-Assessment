# Repository cleanup plan (Phase A: inventory, no changes applied)

Status: **proposal, awaiting review.** Nothing below has been executed. No file has been renamed, `.gitignore` is unchanged from HEAD, and `README.md` is not yet rewritten.
Commits `c10014f` and `cec8ebd` are untouched and will stay so (no rewrite, reset or force-push). Nothing is pushed or committed.

## 0. Facts verified during the audit (2026-10-18)

| Item | Result |
|---|---|
| Study v3 run manifests | 53 found, all `status: completed`, 0 failed |
| `reports/study_v3/aggregate_evaluation.csv` | A/B/C values match the reported table (PLCC .908/.923/.883, SROCC .914/.927/.886, KROCC .759/.779/.722, RMSE .454/.415/.510, MAE .358/.328/.390) |
| Pooled grouped split-conformal coverage | 0.919 (target 0.90) in `docs/experiments/STUDY_V3_RESULTS.md` |
| FROZEN marker hashes | 23 of 24 files match byte-for-byte. `tools/select_b_v3.py` differs, **as already documented** by `FROZEN_ADDENDUM_1.json` (`sha256_now` = f2688179..., matches the file today) |
| Test suite | **178 passed**, not the reported 185 (see section 7, issue 1) |
| Branches | `main`, `study-v2-rebuild`; remote `origin/main`. Large files tracked: none over 700 KB (pack 23 MiB) |
| Local-only bulk | `experiments/` 8.1 GB (study_v3 4.7 GB, study_v2 1.8 GB, 002 1.1 GB, 001/003 270 MB each, smoke 91 MB), 81 `.pt` files, all already ignored by extension |

`tools/verify_study_v3.py` was **not** run: it is a pre-freeze script that rewrites `reports/study_v3/phase2_verification.json`, a file whose hash is part of the freeze. The freeze was checked read-only by recomputing the hashes in `FROZEN`.

## 1. Current structure and main problems

- Root: `configs/ data/ docs/ experiments/ logs/ notebooks/ reports/ results/(empty) src/ tests/ tools/ weights/`.
- Naming conflict: the historical `experiments/001_resnet50_baseline`, `002_learning_rate_search`, `003_final_training` are **a baseline, a learning-rate search and a final retraining of the same MSE model**. They are *not* the ranking-aware (B) and uncertainty-aware (C) experiments, which live in `study_v2` and `study_v3` as A/B/C. The requested names `experiment_002_ranking_aware` / `experiment_003_uncertainty_aware` therefore collide with existing numbering. Decision needed (section 6).
- Study v3 code, protocol docs, manifests, reports and 53 runs are **all untracked**; the repository history does not yet contain the study the README should report.
- `experiments/study_v3_smoke/` (smoke test) is untracked and unignored.
- Two documents describe the project (`docs/project-overview.md` and the two `docs/methodology/*PROTOCOL*.md` families), plus `docs/experiment-protocol.md`; several "final" reports overlap (v2 `COMPARATIVE_ANALYSIS.md`, `ERROR_ANALYSIS.md`, v3 `STUDY_V3_RESULTS.md`).
- Legacy naming: `Ohashi`/`OhashiResNet50` appears in code, configs and 6 docs (kept; only the public README must not use it, section 5).

## 2. Artifact classification

| Class | Contents |
|---|---|
| A. Source code | `src/ct_iqa/**`, `tools/*.py`, `tests/**`, `pyproject.toml`, `uv.lock`, `.python-version` |
| B. Light research records | `configs/**`, `data/labels/ldct_iqac/*.json`, `experiments/**/{config*.json,config.snapshot.json,history.json,run_manifest.json,metrics_*.json}`, `experiments/study_v3/protocol/**`, `B_selection_record.json`, `run_registry.jsonl`, `docs/**` |
| C. Curated results | `reports/eda`, `reports/experiments`, `reports/preprocessing_probe`, `reports/audit`, `reports/study_v3/*` (largest 168 KB) |
| D. Large local-only | `**/checkpoint/*.pt`, `weights/pretrained/**/*.h5`, `data/raw/**`, `logs/`, `experiments/study_v3_smoke/`, per-run `predictions_*.csv` (decision in section 3) |
| E. Immutable (freeze) | the 24 files listed in `experiments/study_v3/protocol/FROZEN`, `FROZEN`, `FROZEN_ADDENDUM_1.json`, the manifests, every `run_manifest.json` and `config.snapshot.json`, `run_registry.jsonl`, `B_selection_record.json` |
| F. Disposable | `__pycache__/`, `.pytest_cache/`, `.benchmarks/`, `logs/*.err/*.out/*.ps1` queue logs (local, keep until user says otherwise), `experiments/study_v3_smoke/` (kept locally, ignored) |

## 3. Git tracking policy

| Category | Policy |
|---|---|
| Code, tests, configs, labels, docs, lock file | track |
| Study v3 protocol (`protocol/`, `FROZEN*`, `B_selection_record.json`, `run_registry.jsonl`) | **track** (evidence for the freeze; tiny) |
| Study v3 per-run `run_manifest.json`, `config.snapshot.json`, `metrics_*.json`, `history.json`, `calibration.json` | track (about 1 MB total; carry the code hashes and per-seed/per-fold provenance) |
| Study v3 per-run `predictions_*.csv` | **local-only** (reproducible per-image tables; ~8 MB). Their SHA-256 are already stored in each `run_manifest.json` (`artifact_sha256`), so integrity stays auditable. Needs approval |
| Curated `reports/**` tables and PNGs | track |
| Checkpoints `.pt`, pretrained `.h5`, raw/processed data, `logs/`, smoke runs, caches | local-only |
| Already-tracked bulky files (`experiments/study_v2/**/predictions_*.csv` etc.) | **leave tracked**; untracking needs your explicit approval and would only affect future checkouts, not history |

Proposed `.gitignore` change (specific, no blanket `*.csv`/`*.json`; this reverts my earlier overly broad rule, already undone):

```
experiments/*_smoke/
experiments/study_v3/**/predictions_*.csv
.env
.env.*
!.env.example
*.pem
*.key
.claude/settings.local.json
```

## 4. Frozen-study risks and decision

`FROZEN` pins these **paths**, so moving or renaming any of them changes what verification reads:

`docs/methodology/STUDY_V3_PROTOCOL.md`, `experiments/study_v3/protocol/{group,fold}_manifest.json`, `configs/base.yaml`, `configs/experiments/{a,b,c}_*.yaml`, `src/ct_iqa/{data/grouped,experiment/run_v3,experiment/run,experiment/grid,training/engine,training/checkpointing,models/registry,losses/ranking,losses/gaussian_nll,evaluation/intervals,evaluation/metrics}.py`, `tools/{run_study_v3,select_b_v3,build_reports_v3,verify_study_v3,create_study_v3_manifests}.py`, `reports/study_v3/phase2_verification.json`.
In addition every `run_manifest.json` stores `group_manifest_sha256`, `fold_manifest_sha256` and artifact hashes; configs and `src` are imported by those names.

**Decision: freeze-bound paths and the study_v3 directory are NOT renamed.** The alias `study_v3_group_aware` is provided by a human-readable index (`docs/reproducibility/EXPERIMENT_INDEX.md`) mapping it to `experiments/study_v3/`. Edits to any file in the freeze list are not planned, including `docs/methodology/STUDY_V3_PROTOCOL.md` (README and new docs link to it). After the cleanup the hash check above is re-run and its result reported exactly.

## 5. Proposed final structure and path mapping

Minimal moves only; results are not duplicated and no file content changes.

| Old path | New path | Reason / reference updates |
|---|---|---|
| `notebooks/00_environment_and_data_audit.ipynb` | `notebooks/00_environment_and_integrity.ipynb` | role: environment + data integrity. Update README, `docs/archived-artifacts.md`, backup manifest note |
| `notebooks/01_dataset_exploration.ipynb` | `notebooks/01_dataset_audit.ipynb` | EDA/audit |
| `notebooks/02_preprocessing_validation.ipynb` | unchanged | already clear |
| `notebooks/03_reference_baseline.ipynb` | `notebooks/03_baseline_experiment.ipynb` | |
| `notebooks/04_ranking_aware_experiment.ipynb` | unchanged | |
| `notebooks/05_uncertainty_experiment.ipynb` | unchanged | |
| `notebooks/06_comparative_analysis.ipynb`, `07_error_analysis_and_visualization.ipynb` | unchanged | |
| `experiments/001_resnet50_baseline` etc. | **unchanged** (historical, referenced by `configs/training.yaml`, `src/ct_iqa/config/legacy.py`, tests, `docs/audit/BACKUP_MANIFEST.*`, checkpoint-compat JSON) | renaming would break the backup manifest hashes and legacy tests |
| `experiments/study_v2`, `experiments/study_v3` | unchanged | study_v3 frozen; study_v2 referenced from committed reports |
| (new) `experiments/INDEX.md` | | what is in git vs local for 001/002/003, study_v2, study_v3, smoke; regeneration commands |
| `docs/*.md` (flat: dataset-and-labels, preprocessing-and-weights, model-architecture, evaluation-metrics, experiment-protocol, reproducibility, project-overview, limitations-and-threats, PROVENANCE, archived-artifacts) | `docs/dataset/`, `docs/methodology/`, `docs/reproducibility/` as appropriate, one canonical doc per topic; superseded ones get a "superseded by" banner, not deletion | link rewrite in README and docs |
| Study v3 untracked files | stay in place, become tracked after approval | |

The exact doc-by-doc moves are listed in the Phase B diff for review before applying; no doc is deleted. Notebook renames are the only renames proposed, because they are not in the freeze list. Result CSV renames (e.g. `final_aggregate_validation.csv`) are **not proposed**: they are referenced by the committed report builders (`tools/build_reports*.py`, two of which are freeze-hashed), so renaming would require editing frozen scripts. Descriptive names are instead given in the README results index.

## 6. Decisions needed from you

1. **Experiment numbering.** Keep 001-003 as the historical MSE baseline / LR search / final training and call the three-way comparison **Experiments A (baseline regression), B (ranking-aware), C (uncertainty-aware)** as the repo does? Or accept an alias `experiment_001/002/003` meaning A/B/C in the README only (my recommendation: README labels "Experiment 1/2/3 (A/B/C)", directories untouched)?
2. Approve keeping per-run `predictions_*.csv` for study_v3 local-only.
3. Approve moving docs into subfolders (reversible, no deletions).
4. Approve notebook renames.
5. The two lost test files (section 7).

## 7. Issues found

1. **Two untracked test files were lost: `tests/unit/test_grouped_splits.py` and `tests/unit/test_build_reports_v3.py`.** I deleted `tests/` earlier in this session at your request before this cleanup brief; they had never been committed, so git cannot restore them. The 25 tracked test files were restored (`git restore`). The suite now reports 178 passed versus the 185 reported (7 tests missing). Possible recovery: OneDrive version history/recycle bin for this folder. If unrecoverable, I can rewrite equivalent tests, clearly marked as post-hoc rewrites. These test files are not in the freeze list, so the freeze is unaffected.
2. `FROZEN` records `working_tree_dirty: true` and the Study v3 code was never committed: the freeze refers to `head` cec8ebd plus untracked files. The new commit will make hashes checkable, but the run manifests' `git` field will still show the earlier dirty state. This will be stated in the docs.
3. `select_b_v3.py` hash differs from the freeze by documented addendum 1 (verified).
4. Old and new numeric claims in `docs/experiments/*` must be re-checked line by line against `reports/study_v3/*.csv` when docs are consolidated.
5. The earlier `.gitignore` edits from this session (blanket `*.csv/*.json`) were reverted; the working-tree `.gitignore` equals HEAD.

## 8. README outline (to be written in Phase B)

Title and summary; problem; objectives and contributions; methodology; dataset and 0-4 label scale (only verified facts from `docs/dataset-and-labels.md` and `reports/eda/dataset_audit.json`); preprocessing (exact settings from `configs/preprocessing.yaml`); model and training; Experiments 1/2/3; Study v3 protocol and rationale; results tables (v3 table above with ±seed SD and group-bootstrap CI from `aggregate_evaluation.csv`; historical v2 table separate and labelled image-level); interpretation; conformal calibration (0.919 vs 0.90, wide intervals, constant-width comparator, official test coverage 0.76 retrospective); repository structure; installation (`uv sync`); tests / reproduction / report regeneration; artifact and data availability (dataset and checkpoints not redistributed); limitations; citation only if supported. No "Ohashi" in README text; scholarly attribution stays in `docs/PROVENANCE.md`, source and config comments.

## 9. Files proposed for removal

None proposed for deletion. Cache directories (`__pycache__`, `.pytest_cache`) are regenerated and already ignored.

## 10. Verification checklist (Phase B end)

- [ ] Recompute all 24 `FROZEN` hashes; report matches and the documented addendum difference
- [ ] `pytest` (expect 178, or 185 if the two files are recovered)
- [ ] Notebooks parse (`nbformat.read`), all notebook and doc links resolve
- [ ] `grep` for stale paths (`00_environment_and_data_audit`, `01_dataset_exploration`, `03_reference_baseline`, moved docs)
- [ ] `git check-ignore` on one file per category
- [ ] Review of `git status`/staged paths for data, checkpoints, logs, caches
- [ ] README tables diffed programmatically against `reports/study_v3/aggregate_evaluation.csv`
- [ ] `git log` still contains c10014f and cec8ebd, no history rewrite
- [ ] No commit, no push
