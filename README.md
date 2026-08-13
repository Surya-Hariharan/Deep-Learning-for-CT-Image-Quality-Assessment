# CT-IQA: Ohashi-Method No-Reference CT Image Quality Assessment

## Overview

This project implements a No-Reference CT Image Quality Assessment (CT-NR-IQA)
pipeline as part of a larger quality-aware longitudinal lung CT follow-up
system. The immediate objective (this repository) is to reproduce a
RadImageNet-pretrained ResNet50 quality-regression model that predicts a
continuous CT image-quality score, trained on synthetically degraded images
labelled by VIF (Visual Information Fidelity) and calibrated against expert
radiologist scores.

**Current phase:** repository architecture initialization only. No model has
been trained. No synthetic pixel has been generated. No raw data has been
modified or committed.

## Research basis

The methodology follows **Ohashi K, Nagatani Y, Yamazaki A, Yoshigoe M, Iwai
K, Uemura R, Shimomura M, Tanimura K, Ishida T. "Development of a No-Reference
CT Image Quality Assessment Method Using RadImageNet Pre-trained Deep Learning
Models." Journal of Imaging Informatics in Medicine (2025).**

That paper (and two other reference documents this project was expected to
draw on) were **not found anywhere in this repository or the local
filesystem** as of this initialization pass -- see `docs/research_decisions.md`
for what is and is not verifiable as a result. Everything tagged
OHASHI-SPECIFIED below is sourced from the project brief's own summary of the
paper's methodology, not from the primary source directly. Obtaining the
actual paper is the single highest-value next step (see `docs/research_decisions.md`).

## Important dataset distinction

**The original Ohashi study used DeepLesion and CQ500** (105 reference images:
90 + 15) as its source of clean CT reference slices for synthetic degradation.

**This project instead uses LDCT-IQAC** (1,000 CT PNG images with expert
perceptual quality scores) as an alternative reference dataset, because
DeepLesion and CQ500 are not available in this environment while LDCT-IQAC is
already downloaded and fully verified.

**This is therefore a methodological replication/adaptation of Ohashi et
al., not an exact dataset reproduction.** Every other part of the methodology
-- degradation grids, VIF as the synthetic training target, the
reference-level 60/20/20 split protocol, the RadImageNet ResNet50
architecture, training configuration, five-parameter logistic calibration, and
the three-stage evaluation -- is preserved exactly as specified. See:

- `docs/ohashi_methodology.md` -- what Ohashi specifies
- `docs/dataset_adaptation.md` -- exactly how LDCT-IQAC replaces DeepLesion/CQ500
- `docs/deviations_from_ohashi.md` -- the (small) list of genuine deviations
- `docs/research_decisions.md` -- OHASHI-SPECIFIED / PROJECT-ADAPTATION / UNKNOWN / EXPERIMENTAL-EXTENSION, never mixed

## Pipeline

```
LDCT-IQAC reference images (1,000, expert_score 0-4)
        |
Ohashi-style Gaussian degradation (noise / blur / noise+blur)
        |
VIF labeling  (vif_score, per degraded image; expert_score kept separate)
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

## Dataset

- **1,000 CT PNG images**, 512x512, RGB (3 identical channels), uint8.
- **Expert scores 0-4** (continuous), the mean of 5 radiologists' ratings on
  abdominal soft-tissue window (350/40), step 0.2.
- **Continuous quality labels throughout** -- this is a regression task. A
  binary `poor <= 2.0 < good` label may be derived later for project-specific
  analysis, but it is not part of the core architecture (see
  `docs/research_decisions.md`).
- **Dataset location is local and machine-specific.** It resolves through
  `ct_iqa.utils.paths`, honouring the `LDCT_IQA_DATA_ROOT` environment
  variable if set, defaulting to `<repo>/data`. No machine-specific path is
  ever committed.
- **The actual images are intentionally excluded from Git** -- see "Dataset
  privacy / Git policy" below.
- Source: HuggingFace mirror `MikaJesse/LDCTiqa_png`, reformatted from the
  **LDCT-IQAC 2023 Grand Challenge**
  (https://ldctiqac2023.grand-challenge.org/). Full audit results:
  `docs/dataset_audit.md`.

## Methodology

| Element | Specification | Source |
| --- | --- | --- |
| Gaussian noise sigma grid | {1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 14, 21, 30, 50} | OHASHI-SPECIFIED |
| Gaussian blur sigma grid | {0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.4, 2.1, 3.0, 5.0} | OHASHI-SPECIFIED |
| Degradation families | noise-only, blur-only, noise+blur | OHASHI-SPECIFIED |
| Label | VIF (vs the image's own clean reference) | OHASHI-SPECIFIED |
| Split | reference-level, 60/20/20 | OHASHI-SPECIFIED |
| Backbone | RadImageNet-pretrained ResNet50 | OHASHI-SPECIFIED |
| Input | central crop -> 224x224 | OHASHI-SPECIFIED |
| Head | Dropout -> FC(1) -> Sigmoid | OHASHI-SPECIFIED |
| Optimizer / batch / epochs | Adam / 64 / 30 | OHASHI-SPECIFIED |
| LR search | {1e-2, 1e-3, 1e-4, 1e-5}, selected by validation MSE | OHASHI-SPECIFIED |
| Calibration | five-parameter logistic, nonlinear least squares | OHASHI-SPECIFIED |
| Evaluation | MSE / PLCC / SROCC, three stages | OHASHI-SPECIFIED |
| Dropout rate | -- | **NOT SPECIFIED BY OHASHI** |
| Backbone frozen vs fine-tuned | -- | **NOT SPECIFIED BY OHASHI** |
| VIF variant (pixel vs wavelet) | -- | **NOT SPECIFIED BY OHASHI** |
| Combined-degradation order | -- | **NOT SPECIFIED BY OHASHI** |
| Noise sigma units (HU vs 8-bit) | -- | **NOT SPECIFIED BY OHASHI** |

Full detail: `docs/ohashi_methodology.md`, `docs/research_decisions.md`.

## Repository structure

```
configs/       dataset.yaml-equivalents (configs/datasets/*.yaml), degradation,
               model/training config (configs/quality/*.yaml), experiments
docs/          ohashi_methodology.md, dataset_adaptation.md,
               deviations_from_ohashi.md, research_decisions.md, dataset_audit.md
src/ct_iqa/    config/ data/ preprocessing/ degradation/ vif/ models/
               training/ evaluation/ utils/   -- the CT-IQA package
src/           growth/ confidence/ fusion/ reporting/  -- stubs for the
               broader project's other components (out of scope here)
scripts/       audit_dataset.py, build_manifest.py, validate_dataset.py,
               verify_git_safety.py, quality/build_ohashi_dataset.py,
               dataset/inspect_datasets.py
notebooks/     01_dataset_inventory .. 10_error_analysis (skeletons; logic
               lives in src/, notebooks call into it)
tests/         config, dataset validation, manifest, splitting, degradation,
               metrics, preprocessing, models/training specs, git safety
data/          raw/ -> interim/ -> processed/  (NOT committed) ; manifests/ (committed)
experiments/   run outputs (NOT committed)
reports/       validation_report.json, dataset_inventory.md (partly generated)
```

`src/ct_iqa/` implements this component (Ohashi + LDCT-IQAC). `src/growth/`,
`src/confidence/`, `src/fusion/`, `src/reporting/` are stubs for the broader
project's longitudinal-growth and confidence-fusion components, out of scope
for this initialization pass.

## Reproducibility

- **Configuration-first.** Every degradation parameter, split ratio, and
  training hyperparameter lives in `configs/*.yaml`, not hardcoded in source.
- **Deterministic seeds.** `configs/paths.yaml:random_seed` is consumed by
  every script that samples, shuffles, or splits. Per-image degradation RNG is
  derived from `SHA256(seed, reference_id, condition)`, so any single image
  regenerates identically regardless of run order.
- **Dataset manifests.** Every dataset has a CSV + Parquet + JSON-header
  manifest under `data/manifests/` (committed -- it's metadata, not pixels).
- **Reference-level splitting, no data leakage.** The 60/20/20 split is
  computed over the 1,000 reference identities *before* generating
  degradations; every degraded variant inherits its reference's partition.
  Enforced by `ct_iqa.data.splitting.check_group_leakage` and tested in
  `tests/test_splitting.py` and `tests/test_ldctiqac_ohashi_adaptation.py`.
- **Two labels, never conflated.** `expert_score` (LDCT-IQAC MOS) and
  `vif_score` (the Ohashi synthetic target) are separate fields throughout --
  tested explicitly.
- **Experiment tracking.** `ct_iqa.evaluation.reporting.log_experiment` writes
  a config+seed+git-commit record for every run under `experiments/<name>/runs/`.
- **No machine-specific paths.** Everything resolves via `ct_iqa.utils.paths`;
  the dataset location is configurable through the `LDCT_IQA_DATA_ROOT`
  environment variable, never hardcoded.

## Dataset privacy / Git policy

**Raw and processed datasets are intentionally excluded from Git.**
`data/raw/`, `data/interim/`, `data/processed/` are `.gitignore`d (see that
file for the full, format-aware rule set: DICOM/NIfTI/archives/checkpoints are
excluded wherever they appear, not just inside one folder name). Only
`data/manifests/` (small CSV/Parquet/JSON metadata) is committed.

Before every commit, run:

```bash
python scripts/verify_git_safety.py
```

It inspects `git ls-files` for anything that looks like tracked dataset
content or a checkpoint, reports `SAFE` or `POTENTIAL DATA LEAK`, and never
deletes anything itself -- if something is already tracked, it prints the
exact `git rm --cached <path>` command to remove it from the index without
touching the local file.

## Research status

- **Phase 1 -- Repository architecture** (this initialization): done.
- **Phase 2 -- Dataset EDA:** not started (notebook skeletons exist; see `notebooks/01`-`03`).
- **Phase 3 -- Synthetic degradation:** blocked on resolving NOT SPECIFIED BY OHASHI
  parameters (noise sigma units, combination order); engine is implemented and tested.
- **Phase 4 -- VIF labeling:** blocked on the VIF-variant question; VIFp implemented, not yet
  numerically cross-validated (see `docs/deviations_from_ohashi.md`, DEV-03).
- **Phase 5 -- Model training:** blocked -- no RadImageNet checkpoint, no framework installed,
  dropout rate and frozen/fine-tuned status unresolved. Not authorized at this phase regardless.
- **Phase 6 -- Evaluation / calibration:** implemented and tested against synthetic data;
  not yet run against real data.
- **Phase 7 -- Confidence fusion (novel contribution):** not started.

## Quick start

```bash
python scripts/dataset/inspect_datasets.py           # read-only inventory scan, all datasets
python scripts/audit_dataset.py                       # LDCT-IQAC-specific audit -> docs/dataset_audit.md
python scripts/build_manifest.py --hash               # build manifests
python scripts/validate_dataset.py                    # integrity + leakage gate
python scripts/quality/build_ohashi_dataset.py --manifest-only   # LDCT-IQAC-based synthetic-dataset plan
python scripts/verify_git_safety.py                   # confirm no dataset content is tracked
python -m pytest tests -q                              # test suite
```

`scripts/validate_dataset.py` exits non-zero on any failure and gates every
downstream stage; nothing trains until it passes.

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
License terms: not explicitly specified on the HuggingFace listing -- refer to
the original challenge site; see `configs/datasets/ldctiqa.yaml`.
