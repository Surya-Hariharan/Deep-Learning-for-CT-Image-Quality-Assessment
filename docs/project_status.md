# Project Status — Single Source of Truth

Last updated: 2026-08-14, after (1) a full repository audit that created
this document, (2) a RadImageNet checkpoint/environment feasibility audit
(`docs/radimagenet_environment_audit.md`), (3) resolution of the two
remaining baseline-model-specification items it surfaced (backbone
fine-tuning code/decision consistency, dropout-probability baseline), and
(4) provisioning the actual RadImageNet checkpoint and an isolated
TensorFlow training environment (`docs/training_environment.md`) — see
§12. Where this
document and any other doc disagree, treat divergence as a signal to fix the
other doc, not to trust this one blindly — but as of the date above, both
were reconciled.

This document reports the **current, actual, re-verified** state of the
repository. It does not take any earlier report's word for it — every claim
below was checked against the repository itself (code, tests, configs, or a
fresh script run) during this audit. Historical pilot/validation reports
(`docs/vif_pilot_report.md`, `docs/vif_full_grid_pilot_report.md`,
`docs/vif_implementation.md`) are **not** rewritten to match current status;
they remain accurate records of what was true when they were written, and
this document is where "what's true now" lives.

---

## 1. Project objective

Reproduce, as closely as the available data permits, the Ohashi et al.
(2025) No-Reference CT Image Quality Assessment (CT-NR-IQA) method: a
RadImageNet-pretrained ResNet50 regression model, trained on synthetically
degraded CT images labelled by VIF (Visual Information Fidelity), calibrated
against expert radiologist scores via a five-parameter logistic. This
repository implements one component (Component B) of a larger
quality-aware longitudinal lung-CT system; NGP-Net (Component A) and the
confidence/fusion components are out of scope here (see
`configs/growth/ngpnet.yaml`, status `BLOCKED - source paper and reference
implementation not available locally`, and `reports/research_decisions.md`).

## 2. Methodological basis

The primary source is Ohashi K, Nagatani Y, Yamazaki A, Yoshigoe M, Iwai K,
Uemura R, Shimomura M, Tanimura K, Ishida T, *"Development of a No-Reference
CT Image Quality Assessment Method Using RadImageNet Pre-trained Deep
Learning Models,"* Journal of Imaging Informatics in Medicine (2025). The
paper itself is not present in this repository or local filesystem; the
methodology is captured from the project brief's summary of it
(`docs/ohashi_methodology.md`) and, for VIF specifically, from the paper's
own reference-list citation (Sheikh & Bovik 2006), which was independently
resolved to a formulation family (S-02, `docs/research_decisions.md`).
**No claim in this repository asserts having read the primary paper
directly.**

Every methodological item is tagged with one of six categories, applied
consistently across `docs/research_decisions.md`,
`configs/degradation.yaml`, and `configs/quality/*.yaml`:

| Category | Meaning |
| --- | --- |
| OHASHI-SPECIFIED | stated in the project brief's summary of the paper |
| PROJECT-ADAPTATION | this project's deliberate choice, because the source is silent |
| REFERENCE-IMPLEMENTATION-CONVENTION | a documented external standard adopted (e.g. `vifvec.m`'s own constants, scipy's own defaults) |
| OBSERVED-PILOT-BEHAVIOR | an empirically measured property, not a parameter choice |
| UNKNOWN / NOT-SPECIFIED | the source is silent and no value has been adopted |
| EXPERIMENTAL-EXTENSION | intentionally beyond the Ohashi baseline (none exist in this component yet) |

No PROJECT-ADAPTATION or UNKNOWN item is described as OHASHI-SPECIFIED
anywhere in the audited documentation (re-verified this pass — see §9).

## 3. Dataset

**LDCT-IQAC** (1,000 CT PNG images, 512×512, uint8, 3 identical channels;
`expert_score` 0–4, mean of 5 radiologists on abdominal soft-tissue window)
replaces Ohashi's original 105-reference DeepLesion(90)+CQ500(15) set —
PROJECT-ADAPTATION, because DeepLesion/CQ500 are not present in this
environment (`docs/dataset_adaptation.md`). Verified this pass:
1,000/1,000 references present, 0 corrupted (fresh full-image
`PIL.Image.verify()` sweep via `scripts/preflight_generation.py`, see §11),
100% expert-score coverage. This is **methodological replication with an
alternative reference dataset, not an exact reproduction of Ohashi's own
dataset** — restated because it is the single most important caveat on
every downstream number this project produces.

## 4. Completed work

Re-verified this pass (tests, a fresh script run, or direct code
inspection — not assumed from prior reports):

- LDCT-IQAC integration, audit, and manifest (`docs/dataset_audit.md`,
  `data/manifests/ldctiqa_manifest.csv`) — 1,000/1,000 images.
- Reference-level 60/20/20 splitting with leakage checking
  (`src/ct_iqa/data/splitting.py`, `tests/test_splitting.py`).
- Gaussian noise, Gaussian blur, and combined degradation engines
  (`src/ct_iqa/degradation/`), fully OHASHI-SPECIFIED sigma grids,
  deterministic per-(reference, condition) RNG.
- Two independent VIF implementations: `vif_p()` (VIFp, pixel-domain,
  `src/ct_iqa/vif/metric.py`) and `vif_wavelet()` (full wavelet/GSM VIF,
  `src/ct_iqa/vif/wavelet.py`, decision A-15) — never aliased, never
  substituted for each other.
- VIF structural validation: 22 property tests
  (`tests/test_vif_wavelet.py`), an independent-implementation cross-check,
  and a parameter-equivalence validation (Pearson 0.99999999999989 once
  parameters are matched) — see §6.
- 30-reference pilot (990 degraded images, `docs/vif_pilot_report.md`) and
  100-reference full-168-condition-grid pilot (16,800 degraded images,
  `docs/vif_full_grid_pilot_report.md`) — see §10.
- Production pipeline: `configs/degradation.yaml` (authoritative, no
  unresolved nulls), `scripts/quality/generate_production_dataset.py`
  (checkpointed, streaming, dry-run), `src/ct_iqa/data/checkpoint.py`,
  `scripts/preflight_generation.py` — see §11.
- Evaluation infrastructure: MSE/PLCC/SROCC
  (`src/ct_iqa/evaluation/metrics.py`), five-parameter logistic calibration
  (`src/ct_iqa/evaluation/logistic_calibration.py`), experiment-run logging
  (`src/ct_iqa/evaluation/reporting.py`) — all implemented and tested
  against synthetic inputs, not yet run against real predictions (none
  exist).
- Model/training specification: `ModelSpec`
  (`src/ct_iqa/models/resnet50.py`), LR-search plan
  (`src/ct_iqa/training/trainer.py`) — deliberately spec-only;
  `build_model()`/`Trainer.fit()` raise unconditionally by design
  (decision A-14). As of 2026-08-14, every field of `ModelSpec` is fully
  resolved (backbone fine-tuning: S-01; dropout baseline: A-22) — see §12.
- Git/dataset safety: `scripts/verify_git_safety.py`,
  `.gitignore` rules, `tests/test_git_safety.py`.

## 5. Current work

Nothing is "in progress" mid-implementation as of this audit. The
production pipeline (generator/checkpoint/preflight/dry-run) reached a
tested, passing state this session and then intentionally stopped short of
execution.

## 6. Not started

- Full 1,000-reference / 169,000-image production dataset generation
  (pipeline ready; execution not authorized — see §12).
- RadImageNet ResNet50 checkpoint acquisition (U-M08).
- Model training of any kind (no framework installed; TensorFlow pinned in
  `requirements.txt` but commented out and not installed, per Ohashi's
  reported environment, Python ≤3.10 + TensorFlow 2.10.10 — incompatible
  with this repo's Python 3.13 development environment; a separate
  environment will be needed when training is authorized).
- Stage 1/2/3 evaluation against real predictions (no predictions exist).
- EDA notebook execution (`notebooks/01`–`10` are skeletons; see §13).

## 7. Blocked items

| Item | Blocked on |
| --- | --- |
| RadImageNet ResNet50 checkpoint (U-M08) | obtaining the actual checkpoint file; nothing in this repo can produce it |
| ~~Dropout probability (U-M01)~~ | **RESOLVED 2026-08-14** as PROJECT-ADAPTATION baseline `0.5` (decision A-22) — no longer blocking. See §12. |
| Model training (any) | RadImageNet checkpoint, a compatible TensorFlow environment, and explicit authorization — all three, not just one (model *specification* itself is now fully resolved, §12) |
| Bit-exact MATLAB VIF equivalence | no MATLAB installation reachable from this project, and Ohashi's paper names no specific function/toolbox to target — this is not resolvable from within this repository at all, not merely "not yet done" |
| Full production dataset generation | `scripts/preflight_generation.py` passing (currently does — see §11) **and** a separate, explicit human go-ahead, which this audit task explicitly withholds |

## 8. Known limitations

Restated from `docs/deviations_from_ohashi.md` and
`docs/research_decisions.md`, not re-litigated here:

- **DEV-01**: 1,000 LDCT-IQAC references replace Ohashi's 105
  DeepLesion+CQ500 references — scales every count, does not change the
  degradation/label/split mechanics.
- **DEV-02**: LDCT-IQAC references are not guaranteed pristine (they carry
  their own acquisition artifacts); "clean" is relative to the additional
  synthetic degradation only.
- **DEV-03**: `vif_wavelet()` is structurally validated
  (parameter-equivalence Pearson 0.99999999999989 against an independent
  implementation once parameters are matched) but **not** bit-exact-verified
  against Ohashi's own MATLAB R2024a run — no such ground truth is
  reachable from this project.
- **DEV-04**: Stage 2 (subjective) and Stage 3 (real-image) evaluation both
  draw on the same LDCT-IQAC references in this adaptation, whereas Ohashi
  used distinct datasets for each — narrows what Stage 3 can demonstrate
  about generalisation.
- Low-texture/near-flat images can destabilise the GSM channel-gain
  estimation (`UNSTABLE_CHANNEL_GAIN`); this is FLAGGED and RETAINED, never
  silently corrected (decision A-19) — see §9/§10.
- Combined noise+blur grid VIF labels show small, high-blur-concentrated
  local non-monotonicities (max +0.039 observed, none exceeding 0.05);
  documented as expected metric behavior, not corrected (decision A-20).
- The development/test environment (Microsoft Store Python 3.13.14, this
  audit's interpreter) differs from the Python 3.13.9 Anaconda environment
  `docs/vif_implementation.md` records `pyrtools` as originally verified
  under. `pyrtools==1.0.10` installs and passes identically under both
  (re-verified this pass — see §11); flagged here as an environment-drift
  fact worth knowing, not a defect.

## 9. Research decisions — integrity check

Re-audited this pass, all in `docs/research_decisions.md` unless noted:

- No PROJECT-ADAPTATION entry is labelled OHASHI-SPECIFIED, and vice versa
  (checked every row of the OHASHI-SPECIFIED / PROJECT ADAPTATION / NOT
  SPECIFIED BY OHASHI / EXPERIMENTAL EXTENSION tables).
- U-D01 (noise sigma units) and U-D02 (combination order) are resolved as
  PROJECT ADAPTATION (decisions A-17, A-16, both dated 2026-08-14) — struck
  through in the NOT SPECIFIED BY OHASHI table with an explicit pointer to
  the resolving decision, not silently deleted from that table.
- `vif_variant` is resolved to `vif_wavelet` (decision A-18) — the
  formulation *family* is OHASHI-SPECIFIED (S-02, from the paper's own
  citation), the specific Python implementation is PROJECT-ADAPTATION
  (A-15) and explicitly **not** claimed bit-exact against MATLAB (§8).
- `UNSTABLE_CHANNEL_GAIN` and all VIF diagnostic fields: FLAG, RETAIN, NEVER
  SILENTLY ALTER (decision A-19) — verified in code:
  `ct_iqa.vif.labeling.compute_vif_with_diagnostics` returns the flag
  alongside the score and never branches on it to change `vif_score`;
  `ProductionManifestRecord` retains `diagnostic_status`,
  `max_channel_gain`, `covariance_condition_number` as separate columns.
- Combined-grid local monotonicity violations: documented as
  OBSERVED-PILOT-BEHAVIOR (decision A-20), not corrected in code — verified
  no smoothing/clipping/monotonicity-enforcement code exists anywhere in
  `src/ct_iqa/vif/`.

## 10. Pilot results (unchanged from their original reports; restated for completeness)

**30-reference pilot** (`docs/vif_pilot_report.md`): 990 degraded images
(30 refs × 33 non-clean reduced-grid conditions), 990/990 (100%) finite
VIF, 0 NaN/Inf, 742/990 (75.0%) `UNSTABLE_CHANNEL_GAIN`-flagged (never
propagated into a non-finite or out-of-range score), 30/30 references fully
monotonic on both noise-only and blur-only sweeps.

**100-reference full-grid pilot** (`docs/vif_full_grid_pilot_report.md`):
16,800 degraded images (100 refs × full 168-condition grid: 12 noise + 12
blur + 144 combined), 16,800/16,800 (100%) finite VIF, 0 NaN/Inf/negative,
95.2% `UNSTABLE_CHANNEL_GAIN`-flagged overall (mechanistically traced to be
noise-driven, near-100% on every noise-bearing condition), 100/100
references fully monotonic on noise-only and blur-only sweeps, combined-grid
local violations present but small (max +0.0393, none exceeding 0.05) with
correct net direction in 100% of blur sweeps / 83% of noise sweeps. Runtime
7,124.9s (~118.75 min) for 16,800 images (0.424s/image mean); projected
~19.9h / ~28.3GB for the full 169,000-image run. **No production dataset
was generated by either pilot** — both wrote only to `data/interim/` (never
committed, verified again this pass, §12).

## 11. Production readiness

Re-run this pass, not assumed from the prior session:

- `configs/degradation.yaml`: no unresolved/null production parameters
  (verified programmatically by `scripts/preflight_generation.py`'s
  "no unresolved (null) production parameters" check).
- `scripts/quality/generate_production_dataset.py --dry-run`: **READY** —
  config load, dataset inspection (1,000/1,000 references), output-path and
  checkpoint-dir writability, disk-space check (140+ GB free vs ~28.3 GB
  projected), VIF import/compute self-test, checkpoint write/resume
  self-test, and a full single-reference (169-record) pipeline probe all
  pass; nothing under `data/processed/quality_iqa` or
  `data/manifests/quality_iqa_manifest.*` was touched.
- `scripts/preflight_generation.py`: **READY** — all 14 checks pass
  (dataset presence/count/corruption/expert-scores, config validity, VIF
  import, disk space, checkpoint writability, git safety, no conflicting
  run, expected-count arithmetic).
- Expected production output, restated explicitly:
  **1,000 references × 168 degraded conditions = 168,000 degraded images**,
  plus **1,000 reference-copy images** if reference copies are stored
  separately = **169,000 total image files**; the manifest carries
  **169,000 rows** (168,000 computed + 1,000 clean-identity rows at
  `vif_score=1.0`, not separately computed).
- **Full production generation has NOT been executed.** No checkpoint
  exists under `data/interim/quality_iqa_checkpoint/` beyond this audit's
  own throwaway self-test probes (created and deleted within the dry-run
  and preflight scripts themselves). No file exists under
  `data/processed/quality_iqa/` except pre-existing `.gitkeep` placeholders.
  No production manifest exists at `data/manifests/quality_iqa_manifest.csv`
  beyond the earlier **planning-stage** manifest (`vif_score` null
  throughout, written by `scripts/quality/build_ohashi_dataset.py
  --manifest-only`, predating this pipeline) — confirmed by inspecting that
  file's `vif_score` column this pass.

## 12. Model-training readiness

**Model architecture specification: RESOLVED, except external
checkpoint/environment provisioning. Training itself: NOT READY, not
attempted this pass.**

Resolved 2026-08-14 (`docs/radimagenet_environment_audit.md`,
`docs/research_decisions.md` "Baseline Model Specification Decision"):

- **Backbone fine-tuning** — `ModelSpec.freeze_backbone` now defaults to
  `False`, matching decision S-01 exactly (previously inconsistently left
  `None` in code despite S-01 already being resolved at the decision-record
  level — that inconsistency is now fixed, not merely documented around).
  Regression-guarded by `tests/test_models_training_specs.py::test_default_spec_backbone_is_fine_tuned_not_frozen`.
- **Dropout probability** — resolved to a documented PROJECT-ADAPTATION
  baseline, `0.5` (decision A-22), sourced from RadImageNet's own
  base-model training recipe (Mei et al. 2022 — the paper the pretrained
  checkpoint itself comes from), **not** from Ohashi's paper, which remains
  silent on the rate. Not chosen by any hyperparameter search — see DEV-05,
  `docs/deviations_from_ohashi.md`, for why this remains a genuine,
  acknowledged point of potential divergence from whatever Ohashi's own
  (unknown) rate actually was.
- The full baseline model specification (224×224 input, fine-tuned
  RadImageNet ResNet50, GAP → Dropout(0.5) → FC(1) → Sigmoid, VIF target,
  MSE/Adam/batch 64/30 epochs/LR sweep {1e-2..1e-5} selected by validation
  MSE) is now completely and explicitly stated in `ModelSpec`
  (`src/ct_iqa/models/resnet50.py`) and `configs/quality/resnet50_vif.yaml`
  — no remaining `null`/unresolved fields in either.

### Checkpoint and environment (resolved 2026-08-14, `docs/training_environment.md`)

```
TRAINING ENVIRONMENT:
READY (CPU-only)
```

- **RadImageNet checkpoint obtained** (U-M08 resolved): the official
  `RadImageNet-ResNet50_notop.h5` (94,852,768 bytes), from the official
  `BMEII-AI/RadImageNet` GitHub repo's own linked Google Drive release.
  SHA-256 recorded (no official checksum exists to verify against —
  locally computed only). Loads into `tf.keras.applications.ResNet50`
  with zero shape-mismatch errors; known backbone layer names confirmed
  present. Stored at `data/external/radimagenet/`, confirmed Git-ignored.
- **Isolated training environment provisioned**: Python 3.10.20 +
  `tensorflow==2.10.1` in a venv entirely outside this repository
  (`C:\Users\surya\.venvs\ct-iqa-tf210`), structurally incapable of being
  Git-tracked. The main project Python 3.13 environment is unmodified
  (re-verified: `tensorflow` still not importable there).
- **`tensorflow==2.10.10` (as recorded in `requirements.txt` and elsewhere,
  inherited from the project brief) does not exist on PyPI** — the 2.10.x
  line only ever shipped 2.10.0 and 2.10.1. `tensorflow==2.10.1` was
  installed instead, on the user's explicit direction after this was
  surfaced, not silently substituted. This is now a **known open item**:
  `requirements.txt`'s and other docs' "2.10.10" references should be
  corrected to "2.10.1" (or otherwise reconciled) the next time
  methodology documentation is revisited — not yet done, since this
  provisioning pass was scoped to environment/checkpoint work, not doc
  correction beyond what's recorded in `docs/training_environment.md`.
- **GPU is present but not usable by this TensorFlow install**: RTX 4060
  (8GB), driver-visible via `nvidia-smi`, but
  `tf.config.list_physical_devices('GPU')` returns `[]` — TensorFlow 2.10.1
  needs the CUDA 11.x Toolkit runtime + cuDNN 8.1 specifically, neither of
  which is installed system-wide (only the driver is). Installing them was
  explicitly out of scope for this pass (a system-level change beyond an
  isolated venv) — documented as an open, deliberately-not-performed step
  in `docs/training_environment.md` §8, with the exact missing DLLs and
  candidate isolated fixes recorded, not just "GPU broken."
- **CPU execution confirmed working**: a tensor-op smoke test
  (matrix multiply) produced the mathematically correct result on
  `/device:CPU:0`.

Still blocking actual training:

- Model construction itself: `build_model()` still raises unconditionally
  (decision A-14) — obtaining the checkpoint and provisioning the
  environment does not itself authorize construction; those remain
  separate, deliberate gates. `unresolved_prerequisites(DEFAULT_SPEC)`
  still returns 2 items (checkpoint-path wiring into `ModelSpec`, and
  framework-availability wiring into the *main* Python 3.13 environment —
  both intentionally not auto-satisfied by the isolated environment
  existing elsewhere on disk).
- No training data exists yet (§11) — training cannot start before
  production dataset generation regardless of the above.
- GPU training specifically additionally requires the CUDA/cuDNN
  installation described above (CPU training does not).
- Explicit human authorization to start training, separate from the
  environment being ready — same gate structure as production dataset
  generation (§13).

## 13. Recommended next phase

This task (repository audit + consistency + safety checkpoint) does **not**
authorize the next phase. For the record, the two candidate next phases,
in the order the pipeline is built for:

1. **Final dataset generation** — `python scripts/preflight_generation.py`
   (must show READY), then an explicit, separate human decision to run
   `python scripts/quality/generate_production_dataset.py --execute`
   (resumable, ~20h/~28GB, per decision record in
   `docs/research_decisions.md`, "Production Generation Decision").
2. **Post-generation dataset audit** — once (1) completes, a full-scale
   validity/diagnostic/monotonicity audit analogous to §10's pilots, but
   over the actual 169,000-record production manifest, before any training
   authorization is considered.

Model training (RadImageNet checkpoint acquisition, environment setup,
LR search execution) is a **separate, later** authorization gate, not
reachable from this document.

---

## Current phase / next phase (explicit, as required)

```
CURRENT PHASE:
Repository + methodology validation complete

NEXT PHASE:
Final dataset generation / final dataset audit
(NOT authorized by this task -- requires a separate, explicit go-ahead)
```
