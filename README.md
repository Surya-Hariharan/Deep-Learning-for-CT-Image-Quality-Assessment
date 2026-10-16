# Quality-Aware Lung CT

**No-reference CT image quality assessment with a RadImageNet-pretrained ResNet50: a controlled comparison of regression, ranking-aware regression, and uncertainty-aware regression.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Backbone](https://img.shields.io/badge/Backbone-ResNet50-informational)
![Pretraining](https://img.shields.io/badge/Pretraining-RadImageNet-6f42c1)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
A no-reference model scores the diagnostic quality of a CT image (0 = worst, 4 = best, radiologist consensus) without a clean reference image. This repository trains and evaluates
three ResNet50 variants on the LDCT-IQAC data under one fixed, pre-declared protocol, and reports honestly what the evidence does and does not support.
It is a research code base for an intra-college study, not a clinical tool; predicting a quality score is not diagnostic performance.

## Research questions
| Experiment | Question | Outcome (this study) |
|---|---|---|
| **A** Reference regression | How accurately can ResNet50 + sigmoid head (MSE) predict the continuous score? | Test PLCC 0.868 (retrospective, 3 seeds) |
| **B** Ranking-aware regression | Does adding a pairwise ranking loss improve rank correlation without hurting absolute error? | **Not supported**: no improvement beyond noise; test RMSE slightly worse |
| **C** Uncertainty-aware regression | Does a Gaussian-NLL head with split-conformal intervals give useful, calibrated uncertainty? | Weak uncertainty signal; **coverage not preserved** on unseen anatomy (0.64 vs 0.90 target) |

Implemented capabilities (config-driven framework, immutable run records, tests) are distinguished below from experimentally demonstrated findings.

## Dataset
LDCT-IQAC: 1000 labeled training-set images (512x512 float32 TIFF, scores in [0, 4], 21 distinct values) and a 300-image official test set (25 distinct values). The images are **not** redistributed
here; only label files are tracked (`data/labels/ldct_iqac/`).
- **Split (frozen, image-level):** 800 train / 100 validation / 100 calibration, seed 2026, canonical hash `be5831c5...0a16` (`experiments/study_v2/protocol/split_manifest.json`, verified by the EDA).
- **Preprocessing:** 224x224 center crop (not resize), `2x - 1` scaling to [-1, 1], grayscale replicated to three channels; targets divided by 4 (sigmoid output), predictions scaled back to [0, 4].
- **Known data limitations (measured, see [DATASET_AND_EDA](docs/methodology/DATASET_AND_EDA.md)):** no patient identifiers; the labeled images are repeated renderings of the same anatomy at different quality
  levels, so **every validation and calibration image has a near-identical-anatomy partner in the training set**, while the test set has none. Validation metrics are therefore optimistic. No exact duplicates were found; all files decode.

## Methodology
- **Model:** ResNet50 (v1 stride placement, BatchNorm epsilon 1.001e-5 from the weight file) with RadImageNet weights (SHA-256 verified), global average pooling, dropout 0.5, one linear unit + sigmoid. C adds a second unit for clamped log-variance.
- **A:** MSE. **B:** MSE + lambda x pairwise margin ranking over in-batch pairs whose label gap is at least delta. **C:** heteroscedastic Gaussian NLL; split-conformal intervals mu +/- q sigma from the 100 calibration images (target coverage 0.90).
- **Training (identical for all):** Adam, lr 1e-3, batch 64, 30 epochs, no augmentation or scheduler; checkpoint = best validation PLCC.
- **B selection:** 7-point grid (lambda 0.03/0.1/0.3/1.0; then margin 0.02/0.1 and delta 0.5 at the stage-1 lambda), validation RMSE within 5% of A, highest mean(PLCC, SROCC, KROCC), paired bootstrap against A. Declared before any run.
- **Seeds:** 42 = selection (exploratory); 1, 2, 3 = independent final replicates for A, B (selected point) and C.
- **Test boundary:** the official test set is read only after a `FROZEN` marker exists, once per final model, and is reported as **retrospective** (it was examined by earlier historical experiments).
- Full protocol: [EXPERIMENT_PROTOCOL](docs/methodology/EXPERIMENT_PROTOCOL.md); preprocessing evidence: [PREPROCESSING](docs/methodology/PREPROCESSING.md).

## Results
Mean +/- SD over three final seeds, raw [0, 4] scale. Validation numbers share anatomy with training and are optimistic; test numbers are **retrospective**.

| Model | Partition (n) | PLCC | SROCC | KROCC | RMSE | MAE | MSE |
|---|---|---|---|---|---|---|---|
| A | validation (100) | 0.978 +/- 0.001 | 0.978 +/- 0.002 | 0.887 +/- 0.004 | 0.236 +/- 0.007 | 0.184 +/- 0.005 | 0.056 +/- 0.003 |
| B | validation (100) | 0.979 +/- 0.001 | 0.979 +/- 0.001 | 0.891 +/- 0.003 | 0.249 +/- 0.016 | 0.192 +/- 0.011 | 0.062 +/- 0.008 |
| C | validation (100) | 0.974 +/- 0.000 | 0.972 +/- 0.001 | 0.873 +/- 0.003 | 0.262 +/- 0.007 | 0.214 +/- 0.007 | 0.069 +/- 0.004 |
| A | official test (300), retrospective | 0.868 +/- 0.009 | 0.870 +/- 0.010 | 0.685 +/- 0.013 | 0.596 +/- 0.054 | 0.491 +/- 0.043 | 0.358 +/- 0.063 |
| B | official test (300), retrospective | 0.869 +/- 0.009 | 0.869 +/- 0.010 | 0.686 +/- 0.013 | 0.637 +/- 0.031 | 0.521 +/- 0.030 | 0.407 +/- 0.040 |
| C | official test (300), retrospective | 0.865 +/- 0.012 | 0.867 +/- 0.015 | 0.681 +/- 0.021 | 0.592 +/- 0.040 | 0.488 +/- 0.034 | 0.352 +/- 0.047 |

**Experiment C uncertainty** (conformal, target 0.90, 100 calibration images): validation coverage 0.94 (not independent evidence), **test coverage 0.64** (width 1.14); a constant-width interval from the same residuals reaches 0.54 on test (width 0.96).
Spearman(sigma, |error|): 0.13 validation, 0.09 test.

**Paired comparisons with A** (seed-averaged, image-level bootstrap, optimistic): B - A mean-correlation +0.002 (95% CI -0.005 to +0.009, validation) and +0.0003 (test); B test RMSE +0.041 (CI +0.027 to +0.055).
C - A: mean correlation -0.008 (validation, CI includes 0), test RMSE -0.004 (CI includes 0).

**Reading the results.** The ranking objective did not improve rank correlation and slightly increased error; the uncertainty head matches the baseline's point accuracy but its intervals are not calibrated on unseen anatomy.
A is the recommended default. Details, per-seed values, claim-to-evidence table: [COMPARATIVE_ANALYSIS](docs/experiments/COMPARATIVE_ANALYSIS.md), [ERROR_ANALYSIS](docs/experiments/ERROR_ANALYSIS.md), tables/figures in [`reports/experiments/`](reports/experiments/), EDA in [`reports/eda/`](reports/eda/).

*Historical experiments 001-003* (earlier protocol: 900/100 split, BatchNorm epsilon mismatch between training and test evaluation) reported test PLCC 0.867 (001) and 0.849 (003). They are preserved unchanged, are not directly comparable with the study above, and are not part of the claims here.

## Limitations and threats to validity
Image-level split with measured anatomy overlap (optimistic validation and bootstrap intervals); no patient IDs; retrospective test set; three seeds; B selection reused the validation set that also selects checkpoints;
`2x-1` is a labelled assumption for the weights' pretraining convention (a probe found it indistinguishable from caffe-style scaling); single dataset and scanner mix; C models aleatoric uncertainty only.
See [limitations-and-threats](docs/limitations-and-threats.md). Nothing here is evidence of clinical utility, and no claim of novelty is made.

## Repository structure
```
configs/        base.yaml + experiments/{a,b,c}_*.yaml (study v2); legacy YAMLs for 001-003
src/ct_iqa/     data, preprocessing, models, losses, training, evaluation (metrics, bootstrap, conformal), experiment (runs, grid, reports), cli
tools/          run_eda.py, preprocessing_probe.py, run_study.py, select_b.py, build_reports.py, create_split_manifest.py
tests/          unit / regression / integration
notebooks/      00-07: audit, EDA, preprocessing, experiments A/B/C, comparison, error analysis (training gated off by default)
docs/           methodology, experiment specs and analyses, decision records, audit, reproducibility
reports/        eda/, preprocessing_probe/, experiments/ (generated from raw run records)
experiments/    historical 001-003 (immutable) and study_v2/ run records (checkpoints are not committed)
weights/        pretrained weights (not committed)
```

## Reproducibility
Environment for the reported runs: Python 3.10, PyTorch 2.5.1 + CUDA 12.1, one RTX 4060 Laptop GPU (8 GB), Windows 11; about 6-8 minutes per 30-epoch run. Obtain the dataset and the RadImageNet weights as described in
[`docs/reproducibility.md`](docs/reproducibility.md) and [`weights/pretrained/radimagenet/resnet50/README.md`](weights/pretrained/radimagenet/resnet50/README.md), then:

```bash
pip install -e ".[dev]"
python -m pytest -q                                              # 176+ tests (synthetic mechanics; some need real data/weights)
python -m ct_iqa.cli validate-config --config configs/experiments/a_reference.yaml
python tools/run_eda.py && python tools/preprocessing_probe.py
python tools/run_study.py --exp A --seed 1 --purpose final       # one 30-epoch run (nothing trains on import)
python tools/build_reports.py                                    # regenerate all tables/figures from run records
```
Full command sequence (grid, selection, final seeds, freeze, retrospective test): [`docs/reproducibility.md`](docs/reproducibility.md). Every run writes an immutable directory with config snapshot, run manifest
(git state, hashes, versions, seed), history, predictions and artifact checksums.

## Scientific transparency
Selection rules were written before results; failed and exploratory runs are retained and labelled (one run failed on a file-lock error and was rerun as a new run); the official test set was never used for any choice;
differences within bootstrap noise are reported as such. Methodological sources are recorded in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Conclusion
With the available data, a plain ResNet50 regressor is as good as the ranking-aware and uncertainty-aware variants for point prediction; the ranking loss gave no demonstrated benefit, and the uncertainty estimates were weak
and not calibrated for unseen anatomy. The most important caveat is the anatomy overlap between training and validation/calibration images, which inflates validation results and calls for an anatomy-aware split before any stronger claim.

## License
Code is MIT-licensed ([`LICENSE`](LICENSE)). The LDCT-IQAC dataset and RadImageNet weights remain under their own terms and are not redistributed.
