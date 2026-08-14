# Project Roadmap

Master phase roadmap, written during the 2026-08-14 repository audit
(`docs/project_audit_2026-08-14.md`). See `docs/workflow.md` for the fine-grained
stage-by-stage lifecycle within these phases.

---

## PHASE 0 — Foundation

**STATUS:** COMPLETED

- **ENTRY CRITERIA:** None (project start).
- **COMPLETED:** Repo scaffolding (`src/ct_iqa` installable package, `configs/`,
  `docs/`, `scripts/`, `tests/`), `pyproject.toml`, `requirements.txt`, `.gitignore`,
  `README.md`, `LICENSE` (MIT).
- **REMAINING:** None for this phase specifically. `.github/` (CI, `CITATION.cff`)
  is a cross-cutting hygiene gap carried forward as a P1/P2 item, not a Phase 0 blocker.
- **EXIT CRITERIA:** Installable package + test harness exist. **Met.**

## PHASE 1 — Methodology

**STATUS:** COMPLETED

- **ENTRY CRITERIA:** Phase 0 complete.
- **COMPLETED:** `docs/ohashi_methodology.md`, `docs/vif_investigation.md`
  (resolved which VIF formulation Ohashi's citation implies — wavelet-domain, not
  VIFp, decision S-02), `docs/research_decisions.md` decision register established.
- **REMAINING:** None.
- **EXIT CRITERIA:** Methodology transcribed and open questions (U-D01, U-D02,
  U-V01, U-M0x, etc.) enumerated. **Met.**

## PHASE 2 — Dataset

**STATUS:** COMPLETED

- **ENTRY CRITERIA:** Phase 1 complete.
- **COMPLETED:** LDCT-IQAC adopted as the DeepLesion+CQ500 substitute (DEV-01),
  `docs/dataset_adaptation.md`, dataset audit (`docs/dataset_audit.md`/`.json`),
  reference manifest (`data/manifests/ldctiqa_manifest.csv`, 1,000 rows), splitting
  + leakage-prevention code (`src/ct_iqa/data/splitting.py`, tested).
- **REMAINING:** None for the dataset itself. Split assignment gets re-verified
  live at production-generation time (already wired, not separate work).
- **EXIT CRITERIA:** Manifest present, audit passes, split/leakage logic tested.
  **Met.**

## PHASE 3 — VIF + Degradation

**STATUS:** COMPLETED

- **ENTRY CRITERIA:** Phase 2 complete.
- **COMPLETED:** Gaussian noise/blur/combined degradation
  (`src/ct_iqa/degradation/*`), pixel-domain VIFp (`vif/metric.py`) and
  wavelet-domain/GSM VIF (`vif/wavelet.py`) both implemented, `compute_vif()`
  requires an explicit variant (no silent default), per-subband diagnostics +
  flag-and-retain policy (A-19).
- **REMAINING:** None. Bit-exact match to Ohashi's original MATLAB implementation
  remains unverifiable (no MATLAB access) — a permanently documented limitation,
  not a task to complete.
- **EXIT CRITERIA:** Degradation grids match Ohashi's spec; VIF implementation
  structurally validated (22/22 property tests). **Met.**

## PHASE 4 — Validation

**STATUS:** COMPLETED

- **ENTRY CRITERIA:** Phase 3 complete.
- **COMPLETED:** 30-reference pilot (`docs/vif_pilot_report.md`), 100-reference
  full-grid pilot (`docs/vif_full_grid_pilot_report.md`, 16,800 images). Both
  recommended "B — ready with documented conditions." Cross-validated against an
  independent VIF implementation (r=0.99999999999989).
- **REMAINING:** None.
- **EXIT CRITERIA:** Pilot recommendation accepted. **Met** — accepted 2026-08-14
  as the "Production Generation Decision" in `docs/research_decisions.md`.

## PHASE 5 — Production Pipeline

**STATUS:** COMPLETED

- **ENTRY CRITERIA:** Phase 4 acceptance.
- **COMPLETED:** `scripts/quality/generate_production_dataset.py` (config-driven,
  no hardcoded production parameters), `src/ct_iqa/data/checkpoint.py`
  (reference-level, atomic, resumable), `scripts/preflight_generation.py`,
  `--dry-run` mode, 18 tests in `tests/test_production_generation.py`. A
  reference-level multiprocessing execution path was designed, implemented, and
  determinism-validated (byte-identical to single-worker golden output) in a prior
  session, but is **uncommitted** — see Phase 6 remaining work.
- **REMAINING:** Decide whether to commit the multiprocessing path before or
  separately from resuming Phase 6.
- **EXIT CRITERIA:** Generator + checkpoint infrastructure implemented and tested.
  **Met** (the multiprocessing addition is an optimization on top of an
  already-complete, already-exit-criteria-meeting Phase 5, not a blocker to
  calling this phase done).

## PHASE 6 — Production Dataset

**STATUS:** CURRENT (in progress, paused)

- **ENTRY CRITERIA:** Phase 5 complete + explicit human authorization to
  `--execute`. **Met** (authorization given, run started `run_20260814T061614Z`).
- **COMPLETED:** 5 / 1,000 references (845 / 169,000 records, 0.5%). Checkpoint
  valid, config_hash verified live against current configs, no failures recorded.
- **CURRENT:** Paused — no process running. Resuming requires re-invoking
  `--execute` (optionally with `--workers N` per the validated benchmark, once that
  code is committed).
- **REMAINING:** 995 / 1,000 references. At the single-worker baseline (~0.424
  s/image) this is ~19.5h; at the benchmarked 6-worker configuration, ~4.8h.
- **EXIT CRITERIA:** All 1,000 references checkpointed complete AND
  `assemble_final_manifest()` has produced `data/manifests/quality_iqa_manifest.csv`
  (169,000 rows). **Not met.**

## PHASE 7 — EDA

**STATUS:** REMAINING (partially unblocked)

- **ENTRY CRITERIA:** For notebook 06 (`06_vif_analysis.ipynb`): Phase 6 complete.
  For notebooks 01–05: already met today (their inputs already exist), but
  intentionally left unpopulated per this audit's report-only scope.
- **COMPLETED:** Notebook skeletons exist for all of 01–06, each with a clear
  "call into `src/ct_iqa`" instruction and explicit dependency statement.
- **REMAINING:** Populate 01–05 (unblocked) and 06 (blocked on Phase 6) with real
  analysis: dataset inventory, score distributions, reference validation, split
  visualization, degradation visualization, VIF distribution/monotonicity/
  expert-score correlation at production scale, diagnostic-flag distribution,
  leakage audit.
- **EXIT CRITERIA:** EDA finds no disqualifying dataset issues; production dataset
  audit report produced.

## PHASE 8 — Model

**STATUS:** REMAINING (spec complete, execution intentionally gated)

- **ENTRY CRITERIA:** Phase 7 (EDA) passing.
- **COMPLETED:** Full `ModelSpec` (`src/ct_iqa/models/resnet50.py`) — 224×224
  input, central crop, grayscale→3-channel, unfrozen backbone, GAP, Dropout(0.5),
  Dense(1), sigmoid — all present as config/spec data and as pure preprocessing
  functions. RadImageNet checkpoint obtained (94.85 MB, on disk). Isolated TF
  2.10.1 training environment provisioned and documented.
- **REMAINING:** Framework-specific model-building code (actually constructing the
  Keras/TF graph from `ModelSpec`); fix the stale `unresolved_prerequisites()`
  check (P1, audit §6); deliberate authorization to lift the `build_model()` gate.
- **EXIT CRITERIA:** `build_model()` succeeds and produces a model whose
  architecture matches `ModelSpec` exactly, inside the isolated training
  environment.

## PHASE 9 — Training

**STATUS:** REMAINING (intentionally gated)

- **ENTRY CRITERIA:** Phase 8 complete.
- **COMPLETED:** `Trainer` spec (`BATCH_SIZE=64`, `EPOCHS=30`, Adam, MSE,
  `LEARNING_RATE_GRID`), `plan_lr_search()`/`select_best()` implemented as pure,
  tested functions.
- **REMAINING:** `Trainer.fit()` itself (currently raises `TrainingNotAuthorized`
  unconditionally); actually running the LR sweep; selecting the best LR.
- **EXIT CRITERIA:** Trained model + selected LR + training history recorded.

## PHASE 10 — Evaluation

**STATUS:** REMAINING

- **ENTRY CRITERIA:** Phase 9 complete.
- **COMPLETED:** Calibration function (`logistic_calibration.py`), metrics
  functions (`evaluation/metrics.py`), evaluation notebook skeletons (08, 09).
- **REMAINING:** Calibration fit; Stage 1 (objective, vs. VIF), Stage 2
  (subjective, vs. expert MOS — apply DEV-04 caveat), Stage 3 (clinical) evaluation.
- **EXIT CRITERIA:** All three evaluation stages reported with metrics.

## PHASE 11 — Ablation/Error Analysis

**STATUS:** REMAINING

- **ENTRY CRITERIA:** Phase 10 complete.
- **COMPLETED:** Notebook skeleton (`10_error_analysis.ipynb`).
- **REMAINING:** Everything — ablation design, execution, error analysis
  cross-checked against VIF diagnostic flags.
- **EXIT CRITERIA:** Ablation/error analysis complete and reported.

## PHASE 12 — Final Research Package

**STATUS:** REMAINING

- **ENTRY CRITERIA:** Phase 11 complete.
- **COMPLETED:** None yet — but the raw material (decision records, pilot
  reports, this audit, workflow/roadmap docs) already forms a strong foundation
  for the final write-up's methodology section.
- **REMAINING:** Final research report synthesizing all phases; reproducibility
  package finalization (consider adding `CITATION.cff`, CI, `uv.lock` before this
  phase — see audit P1/P2 items).
- **EXIT CRITERIA:** Final report published/delivered.

---

## Summary Table

| Phase | Status |
|---|---|
| 0 — Foundation | COMPLETED |
| 1 — Methodology | COMPLETED |
| 2 — Dataset | COMPLETED |
| 3 — VIF + Degradation | COMPLETED |
| 4 — Validation | COMPLETED |
| 5 — Production Pipeline | COMPLETED |
| **6 — Production Dataset** | **CURRENT — 0.5%, paused** |
| 7 — EDA | REMAINING |
| 8 — Model | REMAINING (gated) |
| 9 — Training | REMAINING (gated) |
| 10 — Evaluation | REMAINING |
| 11 — Ablation/Error Analysis | REMAINING |
| 12 — Final Research Package | REMAINING |
