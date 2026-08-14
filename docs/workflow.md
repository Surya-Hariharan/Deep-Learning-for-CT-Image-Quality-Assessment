# Project Workflow

The complete lifecycle of this project, stage by stage. Written during the
2026-08-14 repository audit (`docs/project_audit_2026-08-14.md`) to record what
actually exists today — update this file as stages complete, don't let it drift
into aspirational documentation.

```
Dataset
  ↓
Dataset audit
  ↓
Reference split
  ↓
Degradation
  ↓
VIF labels
  ↓
Pilot validation
  ↓
Production generation   <-- CURRENT STAGE (0.5% complete, paused)
  ↓
Final dataset audit
  ↓
EDA
  ↓
Model construction
  ↓
Training
  ↓
LR selection
  ↓
Calibration
  ↓
Stage 1 evaluation
  ↓
Stage 2 subjective evaluation
  ↓
Stage 3 clinical evaluation
  ↓
Ablation
  ↓
Final analysis
```

## Dataset

- **INPUT:** LDCT-IQAC raw PNGs + expert MOS scores (external, not in git).
- **PROCESS:** manual acquisition, documented in `docs/dataset_adaptation.md` (DEV-01).
- **OUTPUT:** `data/raw/ldctiqa/LDCTiqa_png/*.png`, `data/manifests/ldctiqa_manifest.csv`.
- **STATUS:** Complete.
- **DEPENDENCIES:** None.
- **GATE TO NEXT STAGE:** Manifest exists and row count matches expected (1,000).

## Dataset audit

- **INPUT:** `ldctiqa_manifest.csv`.
- **PROCESS:** `scripts/audit_dataset.py` / `scripts/validate_dataset.py` — corruption,
  duplicate, and score-coverage checks.
- **OUTPUT:** `docs/dataset_audit.md`, `docs/dataset_audit.json`.
- **STATUS:** Complete (generated 2026-08-13).
- **DEPENDENCIES:** Dataset stage.
- **GATE TO NEXT STAGE:** Audit reports no blocking integrity issues.

## Reference split

- **INPUT:** reference `image_id` list.
- **PROCESS:** `src/ct_iqa/data/splitting.py: assign_splits()` — 60/20/20,
  reference-level (never image-level), seeded.
- **OUTPUT:** `split` assignment per reference, checked via `check_group_leakage()`.
- **STATUS:** Implemented and tested (`tests/test_splitting.py`); executed live
  inside the production generator at `--execute` time, not as a separate artifact.
- **DEPENDENCIES:** Dataset audit.
- **GATE TO NEXT STAGE:** `check_group_leakage().ok is True` (the generator refuses
  to proceed otherwise).

## Degradation

- **INPUT:** reference image, `Condition` (clean/noise/blur/noise_blur), config
  grids from `configs/degradation.yaml`.
- **PROCESS:** `src/ct_iqa/degradation/{gaussian_noise,gaussian_blur,combined}.py`.
- **OUTPUT:** degraded image array (in-memory; written to PNG by the caller).
- **STATUS:** Complete, validated in the 30- and 100-reference pilots.
- **DEPENDENCIES:** None beyond the reference image itself.
- **GATE TO NEXT STAGE:** N/A — consumed directly by VIF labeling.

## VIF labels

- **INPUT:** reference + degraded image pair.
- **PROCESS:** `src/ct_iqa/vif/labeling.py: compute_vif_with_diagnostics()`, variant
  `vif_wavelet`, profile `project` (A-18) — the only combination authorized for
  production labels.
- **OUTPUT:** `vif_score`, `diagnostic_status`, `max_channel_gain`,
  `covariance_condition_number` per record.
- **STATUS:** Complete, 22/22 property tests pass, cross-validated against an
  independent implementation to r=0.99999999999989 (`docs/vif_implementation.md`).
- **DEPENDENCIES:** Degradation stage.
- **GATE TO NEXT STAGE:** N/A — consumed directly by pilot validation / production.

## Pilot validation

- **INPUT:** subset of references (30, then 100) run through the full
  degradation+VIF pipeline.
- **PROCESS:** manual pilot runs, analyzed for finite-ness, monotonicity,
  diagnostic-flag rates, correlation with expert scores.
- **OUTPUT:** `docs/vif_pilot_report.md` (30-ref), `docs/vif_full_grid_pilot_report.md`
  (100-ref, full 168-condition grid).
- **STATUS:** Complete. Both pilots recommended "B — ready with documented
  conditions," formally accepted 2026-08-14 as the "Production Generation Decision"
  in `docs/research_decisions.md`.
- **DEPENDENCIES:** VIF labels stage.
- **GATE TO NEXT STAGE:** Recommendation B accepted + `scripts/preflight_generation.py`
  passes + an explicit human decision to invoke `--execute` (this gate is by
  design not automatic — see `generate_production_dataset.py`'s module docstring).

## Production generation — CURRENT STAGE

- **INPUT:** all 1,000 references, `configs/degradation.yaml`,
  `configs/quality/ohashi_ldctiqac.yaml`.
- **PROCESS:** `scripts/quality/generate_production_dataset.py --execute`,
  reference-level checkpointed (`src/ct_iqa/data/checkpoint.py`), one shard per
  reference, resumable. A validated (but not-yet-used) multiprocessing execution
  path exists for this stage — see `docs/project_audit_2026-08-14.md` §5, "Production
  generator" row.
- **OUTPUT:** `data/processed/quality_iqa/{references,degraded}/*.png`,
  `data/manifests/quality_iqa_manifest.{csv,parquet,header.json}` (assembled only
  once all 1,000 references complete).
- **STATUS:** **In progress, paused. 5/1,000 references (0.5%).** Checkpoint valid,
  no process currently running, no failures recorded.
- **DEPENDENCIES:** Pilot validation acceptance.
- **GATE TO NEXT STAGE:** All 1,000 references checkpointed complete AND the final
  manifest assembled (`assemble_final_manifest()` — happens automatically once
  `n_completed_total >= 1000`).

## Final dataset audit

- **INPUT:** `data/manifests/quality_iqa_manifest.csv` (169,000 rows).
- **PROCESS:** dataset statistics, reference score distribution, split
  verification, image dimensions, intensity distributions, degradation
  distributions, VIF distributions, diagnostic-flag distribution, monotonicity,
  expert-score relationship, representative visual inspection, leakage audit
  (see `docs/project_audit_2026-08-14.md` §21 for the full checklist).
- **OUTPUT:** an audit report analogous to `docs/dataset_audit.md`, but at
  production scale.
- **STATUS:** Not started (blocked on production generation).
- **DEPENDENCIES:** Production generation complete.
- **GATE TO NEXT STAGE:** No blocking integrity issues found.

## EDA

- **INPUT:** production manifest + final dataset audit.
- **PROCESS:** `notebooks/01_dataset_inventory.ipynb` through
  `06_vif_analysis.ipynb` — currently single-markdown-cell skeletons, each with an
  explicit "call into `src/ct_iqa`, don't reimplement" instruction.
- **OUTPUT:** populated notebooks with real statistics/plots.
- **STATUS:** Notebooks 01–05 are unblocked today (their inputs already exist) but
  intentionally left unpopulated; 06 is blocked on production generation.
- **DEPENDENCIES:** For 06: production generation. For 01–05: none (already met).
- **GATE TO NEXT STAGE:** EDA shows no disqualifying dataset issues.

## Model construction

- **INPUT:** `configs/quality/resnet50_vif.yaml`, RadImageNet checkpoint
  (`data/external/radimagenet/RadImageNet-ResNet50_notop.h5`, present), a
  framework-installed environment (TF 2.10.1, provisioned separately).
- **PROCESS:** `src/ct_iqa/models/resnet50.py: build_model()`.
- **OUTPUT:** a trainable model instance.
- **STATUS:** **Intentionally gated** — `build_model()` raises
  `ModelNotBuildable` unconditionally. This is a deliberate research safeguard,
  not incomplete code.
- **DEPENDENCIES:** EDA passing; a decision to lift the gate.
- **GATE TO NEXT STAGE:** Explicit authorization + the gate's blockers actually
  resolved (note: `unresolved_prerequisites()` currently over-reports one blocker
  that's already resolved — see audit §6 P1 item).

## Training

- **INPUT:** built model, production dataset, `Trainer` spec (`BATCH_SIZE=64`,
  `EPOCHS=30`, `OPTIMIZER="adam"`, `LOSS="mse"`).
- **PROCESS:** `src/ct_iqa/training/trainer.py: Trainer.fit()`.
- **OUTPUT:** trained weights, training history.
- **STATUS:** **Intentionally gated** — `Trainer.fit()` raises
  `TrainingNotAuthorized` unconditionally.
- **DEPENDENCIES:** Model construction.
- **GATE TO NEXT STAGE:** Explicit authorization.

## LR selection

- **INPUT:** training results across `LEARNING_RATE_GRID = (1e-2, 1e-3, 1e-4, 1e-5)`.
- **PROCESS:** `trainer.py: select_best()` — pure function, already implemented
  and unit-testable, just unexercised (no real results to select from yet).
- **OUTPUT:** chosen learning rate.
- **STATUS:** Logic implemented; not yet run.
- **DEPENDENCIES:** Training.
- **GATE TO NEXT STAGE:** A best LR selected.

## Calibration

- **INPUT:** trained model's raw outputs vs. `vif_score`.
- **PROCESS:** `src/ct_iqa/evaluation/logistic_calibration.py`.
- **OUTPUT:** calibrated score mapping.
- **STATUS:** Function implemented; not yet run.
- **DEPENDENCIES:** LR selection / a final trained model.
- **GATE TO NEXT STAGE:** Calibration fit converges.

## Stage 1 evaluation (objective, vs. VIF)

- **INPUT:** calibrated model, held-out `vif_score` labels.
- **PROCESS:** `notebooks/08_test_evaluation.ipynb` (skeleton), metrics via
  `src/ct_iqa/evaluation/metrics.py`.
- **STATUS:** Not started.
- **DEPENDENCIES:** Calibration.
- **GATE TO NEXT STAGE:** Metrics computed and reported.

## Stage 2 subjective evaluation (vs. expert MOS)

- **INPUT:** model predictions, expert scores.
- **PROCESS:** `notebooks/09_subjective_evaluation.ipynb` (skeleton) — must apply
  the DEV-04 caveat (Stage 2/3 here draw on the same LDCT-IQAC images, unlike
  Ohashi's separate datasets).
- **STATUS:** Not started.
- **DEPENDENCIES:** Stage 1 evaluation.
- **GATE TO NEXT STAGE:** Metrics computed with DEV-04 explicitly caveated.

## Stage 3 clinical evaluation

- **INPUT/PROCESS/OUTPUT:** same notebook/inputs as Stage 2, clinically-framed
  analysis per the Ohashi methodology.
- **STATUS:** Not started.
- **DEPENDENCIES:** Stage 2 evaluation.
- **GATE TO NEXT STAGE:** Complete.

## Ablation / error analysis

- **INPUT:** full evaluation results.
- **PROCESS:** `notebooks/10_error_analysis.ipynb` (skeleton), cross-checked
  against VIF diagnostic flags from production labeling.
- **STATUS:** Not started.
- **DEPENDENCIES:** Stage 1–3 evaluation.
- **GATE TO NEXT STAGE:** Complete.

## Final analysis

- **INPUT:** everything above.
- **PROCESS:** synthesis into a final research report.
- **STATUS:** Not started.
- **DEPENDENCIES:** Everything above.
- **GATE:** N/A — terminal stage.
