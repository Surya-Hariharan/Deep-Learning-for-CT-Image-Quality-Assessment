# Project Audit — 2026-08-14

Audit of the actual repository state (not prior assistant reports). Sources: direct
inspection of git metadata, config files, checkpoint state, running processes, and
three independent code/docs/notebook sweeps performed during this audit. Every claim
below is traceable to a file, command output, or process check made during this audit.

## 1. Executive Summary

This is a well-documented, config-driven research pipeline replicating Ohashi et al.'s
CT no-reference image-quality-assessment method, adapted from the paper's
DeepLesion+CQ500 source data to the locally available LDCT-IQAC dataset (1,000 images).
Phases 0–5 (foundation, methodology, dataset integration, degradation/VIF
implementation, VIF validation/pilots, production-pipeline engineering) are
essentially complete, well-tested, and thoroughly decision-documented. **Phase 6
(production dataset generation) is in progress but intentionally paused**: 5/1,000
references (845/169,000 records, 0.5%) generated and checkpointed; no process is
currently executing it. Phases 7–12 (EDA, model, training, evaluation, ablation,
final report) have not started; model/training code is fully specified but
intentionally gated (`build_model()`/`Trainer.fit()` raise unconditionally). Two
files are currently modified and uncommitted: a reference-level multiprocessing
execution path was added to the production generator and validated (determinism
proven byte-identical vs. the existing single-worker golden output) but **has not
been used to run production** — the checkpoint is untouched.

Repository hygiene is good (MIT LICENSE, thorough `.gitignore`, no tracked binaries,
161 tracked files, single `main` branch, clean environment separation) but has real
gaps: no CI, no `CITATION.cff`, no `CONTRIBUTING.md`, and one stale/misleading
function (`unresolved_prerequisites()` hardcodes "checkpoint not obtained" even
though the checkpoint file now exists on disk).

## 2. Current Phase

**CURRENT PHASE: PHASE 6 — Production Dataset Generation**
**CURRENT SUB-PHASE:** Execution, early and paused (0.5% of references), with an
optimization (worker-parallel execution) implemented and validated but not yet
applied to the real run.

**OVERALL PROJECT COMPLETION ESTIMATE:** roughly **30–35%** of the full 12-phase
project. This is a judgment call, not a file count: Phases 0–5 are essentially done
(they represent a large share of the *design and validation* effort), but Phases
6–10 (generation, EDA, training, evaluation) represent the largest share of
*compute time and remaining decisions*, and none of the compute-heavy phases has
materially progressed — production generation is at 0.5%, and nothing downstream of
it (EDA, training, evaluation) can start until it finishes. Treat this estimate as
directional, not a metric to optimize against.

## 3. Production Generation Status

Read directly from `data/interim/quality_iqa_checkpoint/progress_manifest.json` and
`completed_references.jsonl` at audit time (2026-08-14, ~07:02 UTC):

| Field | Value |
|---|---|
| run_id | `run_20260814T061614Z` |
| config_hash | `cbd4268950083b31136206796beb05460f54d81a16cbbedd0b50adda3ebee3e1` — **verified to still match** `sha256(configs/degradation.yaml + configs/quality/ohashi_ldctiqac.yaml)` recomputed live during this audit |
| seed | `20260813` |
| code_commit_hash | `47f5a9c1eebb56ed377294cf72a82b7976839a33` |
| started_at | 2026-08-14T06:16:14Z |
| updated_at (last checkpoint write) | 2026-08-14T06:21:22Z |
| status field | `"in_progress"` |
| references completed | **5 / 1,000 (0.5%)** |
| records written | **845 / 169,000 (0.5%)** |
| images on disk (`data/processed/quality_iqa`, all files) | 960 (845 belong to the 5 completed references' 169-file sets; the remainder are partial/stray files from reference `0005`, which was mid-processing when the run was paused — correctly **not** marked complete in the checkpoint, and safely regenerable) |
| elapsed wall time of the completed portion | ~5 minutes (06:16:14 → 06:21:22) to produce 5 references |
| time since last checkpoint update (audit time) | ~41 minutes |
| estimated remaining time | ~19.5h single-worker (per the 0.424 s/image pilot projection) or ~4.8h at the benchmarked/validated 6-worker configuration, **if and when workers is changed from the current default of 1** |
| throughput observed | ~1 reference/minute single-worker (5 refs in ~5 min), consistent with the 0.424 s/image baseline |

Record/image accounting: `845 = 5 × 169` records per reference (1 clean + 12 noise +
12 blur + 144 combined); the full run needs `1,000 × 169 = 169,000` records.

**No large or systematic gap between elapsed time and completed work was found** —
the checkpoint is small (5 refs) because the run was deliberately paused early for
the optimization work below, not because of a stall. Do not read "started ~46
minutes ago, only 5 references done" as underperformance; it reflects an
intentional pause, confirmed in section 4.

## 4. Production Generation Health

| Check | Result |
|---|---|
| Process alive | **No python process matching the generator is currently running.** The only `python.exe` processes found were VS Code's `isort`/`black` language servers (verified via `Get-CimInstance Win32_Process` command lines) — not the generator. |
| Checkpoint progressing | Not progressing right now, because nothing is running (see above) — not a stall, a pause. |
| Manifest shards present for all completed refs | Yes — `manifest_shards/0000.csv` … `0004.csv` exist, and `is_reference_complete()`'s hash check would catch any corruption; no mismatch found. |
| Duplicate records | None — `completed_references.jsonl` has exactly 5 unique `reference_id` entries. |
| Missing reference checkpoints | None for 0000–0004. |
| Configuration mismatch | None — config_hash matches live-recomputed hash (section 3). |
| Disk space | 136 GB free (`Get-PSDrive C`) against a ~28.3 GB projected full-run need — no concern. |
| Memory | Not applicable while idle; see the (separate, already-completed) multiprocessing benchmark work for worker-count memory figures if/when the run resumes with `--workers`. |
| `failures.jsonl` | Does not exist in the real checkpoint dir — no recorded failures. |
| `logs/execute_run.log` | Present but **empty (0 bytes)** — the generator only prints to stdout; nothing is actually captured to this configured log path. Worth noting as a real gap (section on code audit). |
| Recent log output | None available beyond the checkpoint's own `completed_at` timestamps, for the reason above. |

**CLASSIFICATION: PAUSED (intentional), not STALLED, not FAILED.** This matches the
explicit instruction from the prior session to pause after 5 references for
optimization work, and the checkpoint is internally consistent and immediately
resumable. No corrective action was taken or is needed.

## 5. Complete Research Status Matrix

| Component | Status | Evidence | Remaining Work |
|---|---|---|---|
| Research methodology | Complete | `docs/ohashi_methodology.md`, `docs/research_decisions.md` (S-01, S-02 + A-01…A-23) | None |
| Ohashi paper analysis | Complete | `docs/ohashi_methodology.md`, `docs/vif_investigation.md` | None |
| Dataset integration | Complete | `docs/dataset_adaptation.md` (DEV-01), `data/manifests/ldctiqa_manifest.csv` (1,000 rows) | None |
| Dataset audit | Complete | `docs/dataset_audit.md`/`.json`, generated 2026-08-13 | None |
| Reference splitting | Implemented | `src/ct_iqa/data/splitting.py`, `tests/test_splitting.py` | Re-verify actual split assignment once production manifest exists |
| Leakage prevention | Implemented | `check_group_leakage()` in `splitting.py`, exercised in generator's `--execute` refusal path | None |
| Noise degradation | Complete | `src/ct_iqa/degradation/gaussian_noise.py`, config-driven 12-level grid | None |
| Blur degradation | Complete | `src/ct_iqa/degradation/gaussian_blur.py`, 12-level grid | None |
| Combined degradation | Complete | `src/ct_iqa/degradation/combined.py`, order=`blur_then_noise` (A-16) | None |
| VIFp | Implemented, retained not used for labels | `src/ct_iqa/vif/metric.py` | None — kept only as a reference variant, never mixed with production labels |
| Wavelet/GSM VIF | Implemented, validated | `src/ct_iqa/vif/wavelet.py`, 22/22 property tests, r=0.99999999999989 vs. independent implementation (`docs/vif_implementation.md`) | Bit-exact match to Ohashi's original MATLAB run remains unverifiable (no MATLAB access) — documented limitation, not a blocker |
| VIF validation (pilots) | Complete | `docs/vif_pilot_report.md` (30 refs), `docs/vif_full_grid_pilot_report.md` (100 refs, full grid) | None |
| 30-reference pilot | Complete | `docs/vif_pilot_report.md` | None |
| 100-reference pilot | Complete | `docs/vif_full_grid_pilot_report.md`, recommendation B accepted 2026-08-14 | None |
| Production generator | Implemented, tested | `scripts/quality/generate_production_dataset.py`, `tests/test_production_generation.py` (18 tests) | Uncommitted: `--workers` multiprocessing path (validated, not yet committed or used) |
| Checkpointing | Implemented, tested | `src/ct_iqa/data/checkpoint.py`, reference-level, atomic writes | None |
| Preflight | Implemented | `scripts/preflight_generation.py` | Re-run before resuming `--execute` |
| Dry-run | Implemented | `generate_production_dataset.py --dry-run` | None |
| **Production generation** | **In progress, paused, 0.5%** | Section 3 | **995/1,000 references remaining — the current bottleneck for everything downstream** |
| Production dataset audit | Not started | — | Blocked on production generation completing |
| EDA | Not started (notebooks are skeletons) | `notebooks/01`–`06`, single markdown cell each, 0 code cells | Blocked on production generation |
| ResNet50 specification | Complete | `src/ct_iqa/models/resnet50.py` `ModelSpec`, `configs/quality/resnet50_vif.yaml` | None (spec only, not execution — see §7) |
| RadImageNet checkpoint | Obtained | `data/external/radimagenet/RadImageNet-ResNet50_notop.h5`, 94.85 MB, present on disk | Loader code to actually use it does not exist yet; `unresolved_prerequisites()` is stale (still reports it missing — code bug, see §6) |
| Training preprocessing | Specified, implemented as pure functions | `src/ct_iqa/preprocessing/{cropping,channels,normalization,pipeline}.py` | None for the functions themselves; not yet wired into a training loop (no framework) |
| Training environment | Provisioned, documented, isolated | `docs/training_environment.md`, Python 3.10.20 + TF 2.10.1 in `C:\Users\surya\.venvs\ct-iqa-tf210` (outside repo) | None |
| `build_model()` | **Intentionally gated** | `resnet50.py:96-107`, raises `ModelNotBuildable` unconditionally | Requires: framework installed in an environment that can run this code, then deliberate authorization to remove the gate |
| `Trainer` | Specified, gated | `src/ct_iqa/training/trainer.py`, `LEARNING_RATE_GRID`, `plan_lr_search()`, `select_best()` implemented; `.fit()` gated | Same as above |
| Training | Not started | — | Blocked on production dataset + model authorization |
| LR selection | Selection *logic* implemented, unexercised | `trainer.py:select_best()` | Needs real training runs to select from |
| Calibration | Implemented as a function, unexercised | `src/ct_iqa/evaluation/logistic_calibration.py` | Needs real model outputs |
| Stage 1 evaluation | Not started | `notebooks/08_test_evaluation.ipynb` skeleton | Blocked on trained model |
| Stage 2 evaluation | Not started | `notebooks/09_subjective_evaluation.ipynb` skeleton | Blocked on trained model; DEV-04 caveat must be applied |
| Stage 3 evaluation | Not started | Same notebook | Same |
| Ablations | Not started | — | Blocked on trained model |
| Error analysis | Not started | `notebooks/10_error_analysis.ipynb` skeleton | Blocked on trained model |
| Final reporting | Not started | — | Blocked on all of the above |
| Reproducibility | Strong | Seeds, config hashes, decision records, checkpoint provenance all present | CI/automated verification absent (§15) |

## 6. Code Audit

**No `TODO`/`FIXME`/`XXX` markers exist anywhere in `src/` or `scripts/`.** No
accidentally-unfinished function bodies, no dead commented-out code, no unreachable
branches were found. The codebase is "specification-as-code": nearly every module
is fully implemented for its own scope, with model construction/training
deliberately blocked.

**Intentional gates (correctly distinguished from incomplete code):**
- `src/ct_iqa/models/resnet50.py:104-107` — `build_model()` raises
  `ModelNotBuildable` unconditionally, listing blockers from
  `unresolved_prerequisites()`.
- `src/ct_iqa/training/trainer.py:63-68` — `Trainer.fit()` raises
  `TrainingNotAuthorized` unconditionally.
- `src/ct_iqa/training/checkpointing.py:23-30` — save/load raise
  `CheckpointingNotAuthorized` unconditionally.
- `scripts/quality/generate_production_dataset.py` — `--execute` requires an
  explicit flag and a passing preflight; a full run additionally requires the
  human decision this audit reaffirms not to make.
- `configs/quality/resnet50_vif.yaml:3-4` — explicit `# DO NOT TRAIN until...` comment.

**One real code-quality finding — stale/incorrect logic, not just documentation
drift:** `unresolved_prerequisites()` in `resnet50.py` **hardcodes** the string
`"RadImageNet ResNet50 checkpoint not obtained (weights_path unset)"` rather than
actually checking `spec.weights_path` or whether a checkpoint file exists on disk.
The checkpoint *is* now present (`data/external/radimagenet/RadImageNet-ResNet50_notop.h5`,
94.85 MB), so this blocker message is currently **factually wrong** even though the
overall gate (training still correctly blocked because no framework is installed) is
harmless in effect. This should be fixed to actually check for the weights path —
flagged as P1 below, not fixed here per the "report only" instruction.

**Genuinely empty/stub packages (not gated, just unimplemented placeholders for
future phases):** `src/confidence/`, `src/fusion/`, `src/growth/`,
`src/reporting/` — each is an empty `__init__.py` + `.gitkeep`. These are not dead
code (nothing references them), just unstarted future work areas; no action needed
now.

**Near-duplicate logic:** `vif_p()` (VIFp) vs. `vif_wavelet()` — intentional and
explicitly documented as non-interchangeable (different formulations, S-02), not an
accidental duplication.

**Logging gap:** `configs/degradation.yaml` configures `logging.log_dir`, but
`generate_production_dataset.py` never writes to it — output only goes to stdout.
`data/interim/quality_iqa_checkpoint/logs/execute_run.log` exists but is 0 bytes.
Minor, but worth fixing before a long unattended run so progress is recoverable
without a live terminal.

## 7. Model Pipeline Audit

| Item | Status | Evidence |
|---|---|---|
| RadImageNet ResNet50 checkpoint | Obtained, but not wired to a loader | File present on disk; no load code; `unresolved_prerequisites()` stale (§6) |
| 224×224 input | SPECIFIED-ONLY | `ModelSpec.input_size=224`; `resnet50_vif.yaml: input_shape: [224,224,3]` |
| Central crop | IMPLEMENTED (as a function) | `preprocessing/cropping.py:central_crop()` |
| Grayscale→3-channel | IMPLEMENTED (as a function) | `preprocessing/channels.py:grayscale_to_rgb()` |
| Fine-tuned (unfrozen) backbone | SPECIFIED-ONLY | `ModelSpec.freeze_backbone=False` (A-01/S-01); no framework model exists to actually freeze/unfreeze |
| GAP head | SPECIFIED-ONLY | `ModelSpec.head` tuple |
| Dropout(0.5) | SPECIFIED-ONLY | `DEFAULT_DROPOUT_PROBABILITY=0.5` (A-22) |
| Dense(1) | SPECIFIED-ONLY | `ModelSpec.head` tuple |
| Sigmoid | SPECIFIED-ONLY | `ModelSpec.head` tuple |
| MSE loss | IMPLEMENTED (metric function) / SPECIFIED for training use | `training/losses.py:mse_loss()`; `PlannedRun.loss="mse"` |
| Adam optimizer | SPECIFIED-ONLY | `trainer.py: OPTIMIZER="adam"` |
| Batch size 64 | SPECIFIED-ONLY | `trainer.py: BATCH_SIZE=64` |
| 30 epochs | SPECIFIED-ONLY | `trainer.py: EPOCHS=30` |
| LR candidate sweep | IMPLEMENTED (planning logic) | `LEARNING_RATE_GRID=(1e-2,1e-3,1e-4,1e-5)`, `plan_lr_search()` |
| Validation-MSE selection | IMPLEMENTED (pure selection function) | `trainer.py:select_best()` |
| `build_model()` execution | **BLOCKED** | Raises unconditionally |
| `Trainer.fit()` execution | **BLOCKED** | Raises unconditionally |

Every architectural/training-hyperparameter choice from the brief is present as
config/spec data and as the pure, unit-testable logic that will consume it (LR
selection, loss computation, preprocessing steps). What does **not** yet exist is
the framework-specific code that actually builds and trains a Keras/TF model — that
requires running inside the isolated TF 2.10.1 environment, which itself is
provisioned and documented but not yet wired up to this code. No training occurred
during this audit, and none should until that wiring is deliberately built and
authorized.

## 8. VIF Pipeline Audit

- Full wavelet-domain/GSM VIF (`vif_wavelet`) is implemented separately from
  pixel-domain VIFp (`vif_p`); `compute_vif()` requires the caller to name the
  variant explicitly, with no default.
- Two profiles exist: `"project"` (the only one authorized for LDCT-IQAC labels)
  and `"reference_crosscheck"` (parameter-matched to a vendored independent
  implementation, used only to cross-validate numerically, never for labels).
- Per-subband diagnostics are tracked (`cu_condition_number`, `g_max`, `ss_max`)
  and rolled into a flag string (`ok`, `UNSTABLE_CHANNEL_GAIN`,
  `ILL_CONDITIONED_COVARIANCE`, `NON_FINITE_VIF`, etc.) per decision A-19's
  flag-and-retain policy — bad computations are never silently dropped or corrected.
- Determinism is enforced via `condition_rng()`, a SHA-256-derived per-(reference,
  condition) RNG stream independent of process/worker/iteration order — verified
  empirically in the prior session's multiprocessing work (byte-identical output
  across serial and spawned-worker execution for a real completed reference).
- Tests exist: `tests/test_vif_wavelet.py` (identity, monotonicity, determinism,
  shape robustness, input validation, profile behavior), plus VIF exercised
  end-to-end in `tests/test_production_generation.py`.
- **No documentation was found that incorrectly calls VIFp the canonical Ohashi
  VIF** — multiple docs (`research_decisions.md`, `ohashi_methodology.md`,
  `vif_implementation.md`) explicitly warn against that conflation.

## 9. Notebook Audit

All 10 notebooks (`01_dataset_inventory.ipynb` … `10_error_analysis.ipynb`) share
an identical structure: **exactly one markdown cell, zero code cells, nothing
executed.** Each markdown cell follows the same template (Objective / Intended
analysis / Depends on / Project status) and ends with: *"STATUS: skeleton only --
no cells executed... Do not implement pipeline logic directly in this notebook --
import from `src/ct_iqa` and call into it."*

| Notebook | Status | Ready now? |
|---|---|---|
| 01_dataset_inventory | SKELETON | Yes — inputs (manifests) already exist |
| 02_ldct_iqac_eda | SKELETON | Yes — inputs already exist |
| 03_reference_validation | SKELETON | Yes — inputs already exist |
| 04_reference_split | SKELETON | Yes — split code already exists |
| 05_degradation_validation | SKELETON | Yes — degradation code already exists |
| 06_vif_analysis | SKELETON | **No** — needs the production manifest |
| 07_training | SKELETON | **No** — needs an authorized, trained model |
| 08_test_evaluation | SKELETON | **No** — needs a trained model |
| 09_subjective_evaluation | SKELETON | **No** — needs a trained model |
| 10_error_analysis | SKELETON | **No** — needs a trained model + predictions |

The existing 01–10 naming already matches a sensible EDA/pipeline workflow; **no
new notebooks are needed**, and none should be pre-filled with placeholder
analysis — that would misrepresent unexecuted work as done. Notebooks 01–05 could
legitimately be populated *today* against existing data (they don't depend on
production generation), but doing so was out of scope for this audit (report-only).
06 is correctly gated on production generation; 07–10 are correctly gated on model
training authorization.

## 10–11. Environment / Dependency Audit

**Two structurally separate environments, by design, and this is the correct
policy for this project — do not merge them:**

1. **Main dev/test environment** — Python 3.13, `pyproject.toml` (setuptools,
   package `ct-iqa`, `requires-python>=3.10`) + `requirements.txt` (numpy, scipy,
   pillow, PyYAML, pyarrow, matplotlib, jupyter, pytest, `pyrtools==1.0.10` for
   steerable pyramids). No `uv.lock`, no `environment.yml`.
2. **Legacy training environment** — Python 3.10.20 (uv-managed toolchain),
   TensorFlow 2.10.1 + Keras 2.10.0, isolated at `C:\Users\surya\.venvs\ct-iqa-tf210`
   (outside the repo, matching `.gitignore`'s `.venv/` pattern in spirit even
   though the actual path is external). Documented in `docs/training_environment.md`
   and `docs/radimagenet_environment_audit.md`. CPU-only confirmed (no GPU support
   validated for this TF version on this machine).

This satisfies the "don't force TF 2.10.1 into Python 3.13" requirement already —
no consolidation is needed on that axis. **Status: CLEAN, not duplicated or
contradictory**, but **not fully consolidated with a lockfile**: `uv.lock` is
absent, so the main environment's dependency versions are pinned only by
lower-bound (`>=`) constraints in `requirements.txt`, not exactly reproduced.

**Recommendation (report only, not applied):** adopt `uv + pyproject.toml +
uv.lock` for the main environment as the brief prefers, moving `requirements.txt`'s
contents into `pyproject.toml`'s `[project.dependencies]` and generating a
`uv.lock`. Keep `requirements-training.txt` (or the existing docs) as the
authoritative *record* of the training environment's contents, since that
environment is intentionally not uv/pyproject-managed inside this repo (it lives
outside the repo entirely, which is appropriate for a large, legacy, GPU/CPU
framework install).

## 12. Gitignore Audit

`.gitignore` (111 lines) is thorough and specifically targets this project's risk
surface: `data/raw/**`, `data/interim/**`, `data/processed/**` (with `.gitkeep`
allow-listing), medical/archive formats (`*.dcm`, `*.nii`, `*.zip`, `*.tar`, etc.),
model weights (`*.h5`, `*.ckpt`, `*.pt`, `*.pth`, `*.onnx`, `*.npy`, `*.npz`),
`checkpoints/`, TensorBoard/log artifacts, standard Python artifacts, and
OS/editor files. `git check-ignore -v` was used during the audit to confirm these
patterns actually match `data/raw/**`, `data/interim/**`, `data/processed/**`,
`checkpoints/`, and `.venv/` — all matched correctly.

`git ls-files` filtered for image/archive/weight extensions returned **zero
matches** — no tracked binaries. The only tracked `data/` content is
`data/manifests/*.{csv,parquet,header.json}` (metadata, not pixels), largest at
~242 KB.

**Minor gap:** no `.mypy_cache/`/`.ruff_cache/` entries, though `ruff` is
configured in `pyproject.toml` — low-priority, add if/when either tool is run
locally and generates a cache dir.

## 13. README Audit

`README.md` (405 lines) accurately describes: project title, research
objective/question, Ohashi methodology summary, the LDCT-IQAC dataset adaptation
(explicitly labeled a deviation), architecture, VIF, degradation grids, dataset
generation pipeline, environment/testing instructions, repository structure,
reproducibility notes, data policy, and citations (Ohashi 2025, Lee 2025 for
LDCT-IQAC). It correctly distinguishes Ohashi-specified vs. project-adaptation vs.
unknown/observed-pilot-behavior throughout (mirroring the taxonomy also used in
`configs/degradation.yaml` and `docs/research_decisions.md`).

**Its "Current status" block states dataset generation / model training / final
evaluation are all NOT YET EXECUTED** — this audit confirms that claim is still
accurate as of 2026-08-14: production generation is 0.5% complete (not "done" by
any reading), and no training/evaluation has occurred. **No correction is needed
here.**

## 14. License Audit

**LICENSE STATUS:** Present. MIT License, "Copyright (c) 2026 Surya Hariharan."

**Consideration:** MIT is appropriate for the *source code* in this repository. It
does **not** and should not be read as covering: (a) the LDCT-IQAC dataset itself
(license "not explicitly specified" upstream, per the README's own data-policy
section — correctly flagged as unresolved rather than assumed permissive), (b) the
RadImageNet ResNet50 checkpoint (third-party weights with their own license terms
not reproduced in this repo), or (c) the vendored reference VIF code under
`scripts/_reference_vif/` (third-party, demarcated via its own `NOTICE.md`). No
action needed beyond continuing to keep those three items out of the MIT grant's
practical scope (which the current repo structure already does, by not
redistributing any of them in git).

## 15. GitHub Repository Hygiene

| File | Status |
|---|---|
| README.md | PRESENT |
| LICENSE | PRESENT |
| .gitignore | PRESENT |
| CONTRIBUTING.md | MISSING — RECOMMENDED (low urgency; solo-authored so far) |
| CODE_OF_CONDUCT.md | MISSING — NOT NECESSARY yet (no external contributors) |
| SECURITY.md | MISSING — NOT NECESSARY (no deployed service/attack surface) |
| CITATION.cff | MISSING — **RECOMMENDED**: this is a research project with a citable methodology and an existing BibTeX block in the README; a `CITATION.cff` is cheap to add and is exactly what this kind of repo should have |
| CHANGELOG.md | MISSING — RECOMMENDED once the project reaches a release-like milestone (e.g., after production generation completes); not urgent now |
| `.github/workflows/` | MISSING — **RECOMMENDED**: a simple `ci.yml` running `pytest` on push would catch regressions like the one this audit's own prior-session work already caught (a hardcoded-constant bug in the new multiprocessing path, found only because a test was written) |
| `.github/ISSUE_TEMPLATE/` | MISSING — NOT NECESSARY yet |
| `.github/PULL_REQUEST_TEMPLATE.md` | MISSING — NOT NECESSARY yet (single-branch, no PR workflow in use) |

## 16. Reproducibility

Strong on the axes that matter most for this project: every production parameter
traces to a config file with an explicit provenance tag (OHASHI-SPECIFIED /
REFERENCE-IMPLEMENTATION-CONVENTION / PROJECT ADAPTATION / OBSERVED PILOT
BEHAVIOR), the global seed and per-condition RNG derivation are documented and
tested, the checkpoint records `config_hash` + `code_commit_hash` + `seed` and
refuses to resume under a mismatch, and the private dataset stays outside git
entirely (only metadata manifests are tracked). A researcher without access to the
private LDCT-IQAC data could reproduce the pipeline's *mechanics* (degradation,
VIF, splitting, checkpointing) against their own data, and could exactly reproduce
this project's results given access to the same data, by following
`docs/research_decisions.md` + the configs + `requirements.txt` +
`docs/training_environment.md`.

**Gap:** no CI means none of this is *automatically* re-verified on change; it
currently depends on a human running `pytest` before committing.

## 17. Project Directory Structure

Current structure is clean and already close to the suggested target:

```
.
├── configs/        (datasets/, degradation.yaml, quality/, preprocessing/, paths.yaml, ...)
├── docs/           (14 files: decision records, pilot reports, environment audits)
├── notebooks/      (10 skeleton notebooks, 01-10)
├── scripts/        (quality/, dataset/, evaluation/, preprocessing/, preflight, validate_*)
├── src/            (ct_iqa/ — the installable package; confidence/, fusion/, growth/, reporting/ — empty future-phase stubs)
├── tests/          (16 test files + conftest.py)
├── data/           (gitignored except data/manifests/*)
├── experiments/    (fusion/, growth_baseline/, quality_baseline/ — gitignored contents)
├── reports/        (generated audit/status reports)
├── pyproject.toml, requirements.txt, README.md, LICENSE, .gitignore
```

No `.github/` directory exists (§15). No `CITATION.cff`. Otherwise this already
matches the target structure reasonably well — **no large reorganization is
recommended.**

## 18–19. Workflow and Roadmap

See `docs/workflow.md` and `docs/roadmap.md` (created alongside this audit).
Neither existed before this audit.

## 20. Current Phase Assessment (restated per the requested format)

```
CURRENT PHASE: PHASE 6 — PRODUCTION DATASET GENERATION

References:
5 / 1,000  (0.5%)

Records:
845 / 169,000  (0.5%)

Images (all files on disk, incl. partial/stray from the in-flight reference
that was mid-processing when paused):
960, of which 845 belong to fully checkpointed references

Phase completion:
~0.5%  (by references/records completed — NOT by elapsed time)

Elapsed runtime of the completed portion: ~5 minutes
Time since run was paused: ~41 minutes (idle, no process running)
```

## 21. Next Phase Plan

Unchanged from the standard plan already implied by the repository's own docs and
notebook skeletons — restated here for completeness, not as new scope:

**Immediately after production generation completes:** production integrity audit
→ dataset statistics → reference score distribution → split verification → image
dimensions → intensity distributions → degradation distributions → VIF
distributions → diagnostic distribution → monotonicity → expert-score relationship
→ representative visual inspection → leakage audit. Notebooks `01`–`06` are the
natural home for this (01–05 can start earlier; 06 needs the finished manifest).
Only after EDA passes: Phase 8 (model implementation, gated), then Phase 9
(training), then Phase 10 (evaluation).

## 22. Confirmation: No Interference With Production

This audit performed **read-only** operations against
`data/interim/quality_iqa_checkpoint/` and `data/processed/quality_iqa/`: file
reads, a live process list, a config-hash recomputation, and disk-space checks.
Nothing was stopped (nothing was running), nothing was restarted, no config or
code affecting production behavior was changed during this audit, and no
checkpoint file was modified. The `--workers` multiprocessing code from the prior
session remains uncommitted and unused against the real checkpoint.

---

# Prioritized Action List

**P0 — Blocking (must happen before resuming/relying on affected components):**
- None found. The production checkpoint is valid and resumable as-is; nothing is
  broken.

**P1 — Important before the next phase:**
- Fix `unresolved_prerequisites()` in `src/ct_iqa/models/resnet50.py` to actually
  check `spec.weights_path`/file existence instead of hardcoding "not obtained" —
  it is currently reporting a false blocker.
- Decide and commit (or discard) the uncommitted `--workers` multiprocessing path
  in `scripts/quality/generate_production_dataset.py` / `tests/test_production_generation.py`
  before resuming production generation, so the run that actually executes is
  backed by a committed, reviewable code state.
- Wire `logging.log_dir` up so `execute_run.log` actually captures output before
  the next long unattended run — right now recovery after a crash depends entirely
  on the checkpoint files, with no textual log trail.
- Add a minimal `.github/workflows/ci.yml` running `pytest` — cheap, and would
  have caught the RECORDS_PER_REFERENCE/`len(conditions)` coupling bug found (and
  fixed) during the multiprocessing work automatically on every future change.

**P2 — Repository quality:**
- Add `CITATION.cff` (the README already has the BibTeX content to source it from).
- Add `uv.lock` (or equivalent) to fully pin the main dev environment.
- Add `.mypy_cache/`/`.ruff_cache/` to `.gitignore` if either tool starts being run
  locally.
- Add `CONTRIBUTING.md` once/if external contributors are anticipated.

**P3 — Nice-to-have:**
- `CHANGELOG.md`, once the project reaches its first release-like milestone
  (e.g., production generation complete).
- Populate notebooks `01`–`05` with real EDA against the existing (pre-production)
  data — they don't depend on production generation finishing, but this is
  genuinely optional and was correctly left undone rather than filled with
  meaningless placeholder cells.

**No action was taken on any of the above during this audit** — this document is a
report. Await explicit approval before implementing any P0–P3 item, resuming
production generation, or committing the pending multiprocessing changes.
