# CT-IQA: Ohashi-Method No-Reference CT Image Quality Assessment

## Overview

This project implements a No-Reference CT Image Quality Assessment
(CT-NR-IQA) pipeline as part of a larger quality-aware longitudinal lung CT
follow-up system. The immediate objective (this repository) is to reproduce
a RadImageNet-pretrained ResNet50 quality-regression model that predicts a
continuous CT image-quality score, trained on synthetically degraded images
labelled by VIF (Visual Information Fidelity) and calibrated against expert
radiologist scores.

## Research objective

Replicate the Ohashi et al. CT-NR-IQA method as closely as the available
data permits: degrade CT reference images along Gaussian noise/blur grids,
label each degraded image with VIF against its own clean reference, train a
RadImageNet ResNet50 regressor on those labels, and calibrate/evaluate
against expert radiologist scores in three stages (synthetic, subjective,
real-image). This repository covers the CT-NR-IQA component only; NGP-Net
and the confidence/fusion components of the broader system are out of scope
here (`src/growth/`, `src/confidence/`, `src/fusion/` are stubs).

## Ohashi methodology

The methodology follows **Ohashi K, Nagatani Y, Yamazaki A, Yoshigoe M, Iwai
K, Uemura R, Shimomura M, Tanimura K, Ishida T. "Development of a No-Reference
CT Image Quality Assessment Method Using RadImageNet Pre-trained Deep
Learning Models." Journal of Imaging Informatics in Medicine (2025).**

The paper itself is not present in this repository or the local filesystem.
Everything tagged OHASHI-SPECIFIED throughout this project is sourced from
the project brief's own summary of the paper's methodology, not read
directly from the primary source — see `docs/research_decisions.md` for
what is and is not verifiable as a result, and `docs/ohashi_methodology.md`
for the full specification as briefed.

## Critical dataset distinction

**This project follows the Ohashi CT-NR-IQA methodology but uses LDCT-IQAC
as an alternative reference dataset. It is NOT an exact reproduction of the
original dataset.**

**The original Ohashi study used DeepLesion and CQ500** (105 reference
images: 90 + 15) as its source of clean CT reference slices for synthetic
degradation. **This project instead uses LDCT-IQAC** (1,000 CT PNG images
with expert perceptual quality scores), because DeepLesion and CQ500 are not
available in this environment while LDCT-IQAC is downloaded and fully
verified.

Every other part of the methodology — degradation grids, VIF as the
synthetic training target, the reference-level 60/20/20 split protocol, the
RadImageNet ResNet50 architecture, training configuration, five-parameter
logistic calibration, and the three-stage evaluation — is preserved exactly
as specified. See:

- `docs/ohashi_methodology.md` — what Ohashi specifies
- `docs/dataset_adaptation.md` — exactly how LDCT-IQAC replaces DeepLesion/CQ500
- `docs/deviations_from_ohashi.md` — the (small) list of genuine deviations
- `docs/research_decisions.md` — OHASHI-SPECIFIED / PROJECT-ADAPTATION /
  REFERENCE-IMPLEMENTATION-CONVENTION / OBSERVED-PILOT-BEHAVIOR / UNKNOWN /
  EXPERIMENTAL-EXTENSION, never mixed
- `docs/project_status.md` — the current, re-verified state of every
  component (single source of truth; read this first)

## Current methodology

```
LDCT-IQAC reference images (1,000, expert_score 0-4)
        |
Ohashi-style Gaussian degradation (noise / blur / noise+blur)
        |
Full wavelet-domain/GSM VIF labeling (vif_score, per degraded image;
expert_score kept separate)
        |
RadImageNet-pretrained ResNet50  (central crop -> 224x224 -> dropout -> FC(1) -> sigmoid)
        |
continuous quality prediction
        |
five-parameter logistic calibration
        |
PLCC / SROCC / MSE   (Stage 1: vs vif_score)
        |
expert-score evaluation   (Stage 2/3: vs expert_score)
```

| Element | Specification | Source |
| --- | --- | --- |
| Gaussian noise sigma grid | {1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 14, 21, 30, 50} | OHASHI-SPECIFIED |
| Gaussian blur sigma grid | {0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.4, 2.1, 3.0, 5.0} | OHASHI-SPECIFIED |
| Degradation families | noise-only, blur-only, noise+blur | OHASHI-SPECIFIED |
| Combined-degradation order | blur-then-noise | PROJECT-ADAPTATION (resolved 2026-08-14, decision A-16) |
| Noise sigma units | 8-bit grey levels, `sigma_scale=1.0` | PROJECT-ADAPTATION (resolved 2026-08-14, decision A-17) |
| Label | full wavelet/GSM VIF (vs the image's own clean reference) | formulation family OHASHI-SPECIFIED (S-02); Python implementation PROJECT-ADAPTATION (A-15) |
| Split | reference-level, 60/20/20 | OHASHI-SPECIFIED |
| Backbone | RadImageNet-pretrained ResNet50, fine-tuned | OHASHI-SPECIFIED (fine-tune status confirmed from paper text, S-01) |
| Input | central crop -> 224x224 | OHASHI-SPECIFIED |
| Head | Dropout -> FC(1) -> Sigmoid | OHASHI-SPECIFIED |
| Optimizer / batch / epochs | Adam / 64 / 30 | OHASHI-SPECIFIED |
| LR search | {1e-2, 1e-3, 1e-4, 1e-5}, selected by validation MSE | OHASHI-SPECIFIED |
| Calibration | five-parameter logistic, nonlinear least squares | OHASHI-SPECIFIED |
| Evaluation | MSE / PLCC / SROCC, three stages | OHASHI-SPECIFIED |
| Dropout rate | -- | UNKNOWN / NOT-SPECIFIED by Ohashi |

Full detail: `docs/ohashi_methodology.md`, `docs/research_decisions.md`.

## Dataset

- **1,000 CT PNG images**, 512x512, RGB (3 identical channels), uint8.
- **Expert scores 0-4** (continuous), the mean of 5 radiologists' ratings on
  abdominal soft-tissue window (350/40), step 0.2.
- **Continuous quality labels throughout** — this is a regression task.
- **Dataset location is local and machine-specific.** It resolves through
  `ct_iqa.utils.paths`, honouring the `LDCT_IQA_DATA_ROOT` environment
  variable if set, defaulting to `<repo>/data`. No machine-specific path is
  ever committed.
- **The actual images are intentionally excluded from Git** — see "Data
  policy" below.
- **Immutable.** Every loader in this repository (`ct_iqa.data.loader`,
  `scripts/quality/generate_production_dataset.py`) opens LDCT-IQAC source
  files read-only; nothing ever writes back to `data/raw/ldctiqa/` or
  `ground_truth_score.json`.
- Source: HuggingFace mirror `MikaJesse/LDCTiqa_png`, reformatted from the
  **LDCT-IQAC 2023 Grand Challenge**
  (https://ldctiqac2023.grand-challenge.org/). Full audit results:
  `docs/dataset_audit.md`.

## VIF implementation

Ohashi cites Sheikh & Bovik (2006), "Image information and visual quality"
[ref 27] — the original **wavelet-domain/GSM VIF**, not the simplified,
earlier (2005) pixel-domain **VIFp**. Both are implemented in this
repository, kept as two distinct, never-aliased variants:

- `src/ct_iqa/vif/wavelet.py::vif_wavelet()` — full wavelet-domain VIF
  (steerable pyramid + per-subband Gaussian Scale Mixture model), the
  formulation Ohashi's citation specifies. **This is the only variant used
  for label generation.**
- `src/ct_iqa/vif/metric.py::vif_p()` — VIFp, retained unchanged, never
  substituted for `vif_wavelet()`.

Validation performed (`docs/vif_implementation.md`, `docs/research_decisions.md`):

- 22 property tests (identity, monotonicity, determinism, degenerate-input
  rejection) — `tests/test_vif_wavelet.py`.
- An independent-implementation cross-check at default parameters (moderate
  agreement, Pearson 0.62 — expected, since the two implementations chose
  different parameters).
- A parameter-equivalence validation: once every tunable parameter is
  matched, the two implementations agree to Pearson 0.99999999999989 — this
  is evidence the *algorithm* is implemented correctly, not evidence that
  the *canonical parameter set* matches Ohashi's own MATLAB R2024a run.
- **Bit-exact numerical equivalence to Ohashi's MATLAB R2024a run is NOT
  established and is not expected to become established** — no MATLAB
  installation is reachable from this project, and the paper names no
  specific MATLAB function/toolbox to target.
- Low-texture/near-flat images can destabilise the GSM channel-gain
  estimation stage (`UNSTABLE_CHANNEL_GAIN`). The formal policy (decision
  A-19): **flag, retain, never silently alter.** Every production record
  keeps `vif_score`, `diagnostic_status`, `max_channel_gain`, and
  `covariance_condition_number` verbatim — nothing is clipped, replaced,
  discarded, or excluded on the basis of a diagnostic flag.

## Pilot validation

Two pilots, both completed, neither writing any production data:

- **30-reference pilot** (`docs/vif_pilot_report.md`): 990 degraded images,
  100% finite VIF, 0 NaN/Inf, 75.0% `UNSTABLE_CHANNEL_GAIN`-flagged (never
  propagated into a bad score), 100% monotonic on noise-only/blur-only
  sweeps.
- **100-reference full-grid pilot** (`docs/vif_full_grid_pilot_report.md`):
  16,800 degraded images (full 12+12+144-condition grid), 100% finite VIF,
  95.2% flag rate (traced to be noise-driven, not a defect), 100% monotonic
  on noise-only/blur-only sweeps, small (max +0.039) local non-monotonicity
  on the combined grid with correct net direction in the large majority of
  sequences. Runtime and storage projected the full 169,000-image run at
  ~19.9h / ~28.3GB, single-threaded.

Neither pilot's output was committed (`data/interim/**` is Git-ignored) or
used to generate any production label.

## Production pipeline

Implemented, tested, and **not yet executed**:

- `configs/degradation.yaml` — the single authoritative production config
  (sigma grids, resolved combination order, VIF variant/profile,
  checkpointing, output layout). No unresolved/null production parameters.
- `scripts/quality/generate_production_dataset.py` — reference-level
  checkpointed generator (`reference completed -> checkpoint -> next
  reference`), streams records to disk (never holds the full ~169,000-record
  set in memory), resumable after crash/restart, `--dry-run` mode.
- `src/ct_iqa/data/checkpoint.py` — atomic writes, SHA-256-verified manifest
  shards, corruption/incompleteness detection, config/seed-hash mismatch
  protection.
- `scripts/preflight_generation.py` — 14 checks (dataset presence/count/
  corruption, config validity, VIF import, disk space, checkpoint
  writability, Git safety, run-conflict detection, expected-count
  arithmetic); fails rather than proceeds on any critical check.

Run `python scripts/preflight_generation.py` and
`python scripts/quality/generate_production_dataset.py --dry-run` to verify
readiness at any time — neither writes production data.

## Current status

```
PRODUCTION DATASET GENERATION:
NOT YET EXECUTED

MODEL TRAINING:
NOT YET EXECUTED

FINAL EVALUATION:
NOT YET EXECUTED
```

See `docs/project_status.md` for the full, itemized component-by-component
status (COMPLETE / PARTIALLY COMPLETE / NOT STARTED / BLOCKED / UNKNOWN) and
the current test/preflight/git-safety results as of the last audit.

## Repository structure

```
configs/       dataset.yaml-equivalents (configs/datasets/*.yaml),
               degradation.yaml (production-authoritative), quality/*.yaml,
               growth/, preprocessing/, experiments/
docs/          ohashi_methodology.md, dataset_adaptation.md,
               deviations_from_ohashi.md, research_decisions.md,
               project_status.md (single source of truth for current state),
               vif_implementation.md, vif_pilot_report.md,
               vif_full_grid_pilot_report.md, dataset_audit.md
src/ct_iqa/    config/ data/ preprocessing/ degradation/ vif/ models/
               training/ evaluation/ utils/   -- the CT-IQA package
src/           growth/ confidence/ fusion/ reporting/  -- stubs for the
               broader project's other components (out of scope here)
scripts/       audit_dataset.py, build_manifest.py, validate_dataset.py,
               verify_git_safety.py, preflight_generation.py,
               quality/build_ohashi_dataset.py (legacy planning skeleton),
               quality/generate_production_dataset.py (production generator),
               quality/vif_pilot.py, quality/vif_full_grid_pilot.py,
               dataset/inspect_datasets.py
notebooks/     01_dataset_inventory .. 10_error_analysis (skeletons; logic
               lives in src/, notebooks call into it -- not executed yet)
tests/         config, dataset validation, manifest, splitting, degradation,
               VIF (metric + wavelet), checkpoint, production generation,
               metrics, preprocessing, models/training specs, git safety
data/          raw/ -> interim/ -> processed/  (NOT committed) ; manifests/ (committed)
experiments/   run outputs (NOT committed)
reports/       validation_report.json, dataset_inventory.md (partly generated)
```

`src/ct_iqa/` implements this component (Ohashi + LDCT-IQAC). `src/growth/`,
`src/confidence/`, `src/fusion/`, `src/reporting/` are stubs for the broader
project's other components, out of scope here.

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

Python ≥3.10 (developed and tested under Python 3.13). `pyrtools==1.0.10` is
required for `vif_wavelet()` (steerable-pyramid decomposition); the deep
learning stack (TensorFlow, pinned to match Ohashi's reported environment)
is intentionally commented out in `requirements.txt` and not installed —
see "Training" note below.

## Configuration

Every degradation parameter, split ratio, and pipeline setting lives in
`configs/*.yaml`, never hardcoded in source:

- `configs/paths.yaml` — project paths, global `random_seed`.
- `configs/datasets/*.yaml` — per-dataset metadata (LDCT-IQAC and six
  others inventoried but out of scope for this component).
- `configs/quality/ohashi_ldctiqac.yaml` — active reference/split spec for
  the LDCT-IQAC adaptation.
- `configs/degradation.yaml` — production-authoritative degradation/
  generation config (the generator reads this, not hardcoded values).
- `configs/quality/resnet50_vif.yaml` — model/training/calibration spec
  (spec only; training not authorized).

## Testing

```bash
python -m pytest tests -q
```

Covers config loading, dataset validation, manifest I/O, splitting/leakage,
degradation (noise/blur/combined), both VIF implementations, checkpoint
resume/corruption-detection, the production generator's per-reference
pipeline, evaluation metrics, calibration, preprocessing, and Git safety.

## Reproducibility

- **Configuration-first.** Every degradation parameter, split ratio, and
  training hyperparameter lives in `configs/*.yaml`, not hardcoded in source.
- **Deterministic seeds.** `configs/paths.yaml:random_seed` is consumed by
  every script that samples, shuffles, or splits. Per-image degradation RNG
  is derived from `SHA256(seed, reference_id, condition)`, so any single
  image regenerates identically regardless of run order.
- **Dataset manifests.** Every dataset has a CSV + Parquet + JSON-header
  manifest under `data/manifests/` (committed — it's metadata, not pixels).
- **Reference-level splitting, no data leakage.** The 60/20/20 split is
  computed over the 1,000 reference identities *before* generating
  degradations; every degraded variant inherits its reference's partition.
  Enforced by `ct_iqa.data.splitting.check_group_leakage` and tested.
- **Two labels, never conflated.** `expert_score` (LDCT-IQAC MOS) and
  `vif_score` (the Ohashi synthetic target) are separate fields throughout.
- **Checkpointed, resumable production generation.** Reference-level
  checkpoints with atomic writes and SHA-256-verified shards mean a
  crash/restart never regenerates a completed reference and never silently
  accepts a corrupted one (`src/ct_iqa/data/checkpoint.py`).
- **Experiment tracking.** `ct_iqa.evaluation.reporting.log_experiment`
  writes a config+seed+git-commit record for every run under
  `experiments/<name>/runs/`.
- **No machine-specific paths.** Everything resolves via `ct_iqa.utils.paths`;
  the dataset location is configurable through the `LDCT_IQA_DATA_ROOT`
  environment variable, never hardcoded.

## Data policy

**Raw and generated datasets are intentionally excluded from Git.**
`data/raw/`, `data/interim/`, `data/processed/`, and the production
checkpoint directory are `.gitignore`d (see that file for the full,
format-aware rule set — DICOM/NIfTI/archives/checkpoints/model weights are
excluded wherever they appear, not just inside one folder name). Only
`data/manifests/` (small CSV/Parquet/JSON metadata) is committed, with one
exception: the full `quality_iqa_manifest.{csv,parquet,header.json}`
(hundreds of thousands of rows once generated) stays local-only; a small
`quality_iqa_manifest.sample.csv` is committed instead.

Before every commit, run:

```bash
python scripts/verify_git_safety.py
```

It inspects `git ls-files` for anything that looks like tracked dataset
content or a checkpoint, reports `SAFE` or `POTENTIAL DATA LEAK`, and never
deletes anything itself — if something is already tracked, it prints the
exact `git rm --cached <path>` command to remove it from the index without
touching the local file.

## Research limitations

- **Not an exact reproduction.** The reference-image source (LDCT-IQAC vs
  DeepLesion+CQ500) differs from Ohashi's own study; every other mechanic
  (degradation grids, VIF target, split protocol, architecture, training
  config, calibration, evaluation structure) is preserved.
- **LDCT-IQAC references are not pristine** — they carry their own
  acquisition artifacts; VIF labels measure information loss relative to an
  already-imperfect reference, not a pristine acquisition (DEV-02).
  Absolute VIF values are not directly comparable to Ohashi's reported
  numbers; PLCC/SROCC remain the meaningful comparison.
- **VIF implementation is structurally validated, not bit-exact-verified
  against MATLAB.** No MATLAB ground truth is reachable from this project
  (DEV-03).
- **Stage 2 and Stage 3 evaluation share the same LDCT-IQAC references** in
  this adaptation, whereas Ohashi used distinct datasets for each — narrows
  what Stage 3 can demonstrate about generalisation (DEV-04).
- **Combined-grid VIF labels show small, characterised non-monotonicity**
  at high blur severity (max +0.039 observed, none exceeding 0.05) —
  documented as expected metric behavior, not a bug, and not corrected.
- **Multiple methodological parameters remain unresolved by Ohashi** and
  will need a recorded PROJECT-ADAPTATION decision before they can be used
  (dropout probability, exact grayscale-to-3-channel implementation,
  resize/interpolation before central crop, RadImageNet preprocessing,
  Adam betas/LR schedule/early stopping/checkpoint selection, calibration
  initial guess/bounds — see `docs/research_decisions.md`, "NOT SPECIFIED
  BY OHASHI").

See `docs/project_status.md` for the complete, current list.

## Citation

```bibtex
@article{ohashi2025ctiqa,
  title={Development of a No-Reference CT Image Quality Assessment Method Using
         RadImageNet Pre-trained Deep Learning Models},
  author={Ohashi, K and Nagatani, Y and Yamazaki, A and Yoshigoe, M and Iwai, K
          and Uemura, R and Shimomura, M and Tanimura, K and Ishida, T},
  journal={Journal of Imaging Informatics in Medicine},
  year={2025}
}

@article{lee2025ldctiqac,
  title={Low-dose computed tomography perceptual image quality assessment},
  author={Lee, Wonkyeong and Wagner, Fabian and Galdran, Adrian and Shi, Yongyi
          and Xia, Wenjun and Wang, Ge and Mou, Xuanqin and Ahamed, Md Atik
          and Imran, Abdullah Al Zubaer and Oh, Ji Eun and others},
  journal={Medical Image Analysis},
  volume={99},
  pages={103343},
  year={2025},
  publisher={Elsevier}
}
```

Dataset source: `MikaJesse/LDCTiqa_png` (HuggingFace), reformatted from the
LDCT-IQAC 2023 Grand Challenge (https://ldctiqac2023.grand-challenge.org/).
License terms: not explicitly specified on the HuggingFace listing — refer to
the original challenge site; see `configs/datasets/ldctiqa.yaml`.
