# Comparative analysis: Experiments A, B, C (study v2)

All numbers are produced by `tools/build_reports.py` from raw run records (`reports/experiments/summary.json` and CSVs). Final comparison = seeds 1, 2, 3 (independent replicates);
seed 42 runs are selection/pilot runs and are not in the aggregate. Raw [0, 4] scale. **Validation** (n = 100) numbers are optimistic because every validation image has a same-anatomy
partner in the training set ([DATASET_AND_EDA](../methodology/DATASET_AND_EDA.md)); **official test** (n = 300, anatomy-disjoint) numbers are **retrospective**, because the test set was examined by historical experiments 001/003.
With 3 seeds the SDs below are rough spreads, not confidence statements.

## 1. Setup comparison
| | A | B | C |
|---|---|---|---|
| Head / output | sigmoid mean | sigmoid mean | sigmoid mean + clamped log-variance |
| Loss | MSE | MSE + 0.3 x pairwise hinge ranking (margin 0.05, min gap 0.5 raw units) | heteroscedastic Gaussian NLL |
| Selection | none (fixed recipe) | 7-point grid at seed 42, declared rule | none |
| Extra stage | - | - | split-conformal on 100 calibration images |
| Backbone, data, preprocessing, optimizer, epochs, batch | identical (ResNet50 RadImageNet, 800/100/100, 224 crop + `2x-1`, Adam 1e-3, 30 ep, bs 64, best val PLCC) | | |
| Final runs | 3 (seeds 1-3) | 3 | 3 |

## 2. Point prediction (mean +/- SD over seeds 1-3)
| | Partition | PLCC | SROCC | KROCC | MSE | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| A | validation | 0.9784 +/- 0.0007 | 0.9777 +/- 0.0018 | 0.8873 +/- 0.0041 | 0.0559 +/- 0.0034 | 0.1845 +/- 0.0051 | 0.2364 +/- 0.0072 |
| B | validation | 0.9790 +/- 0.0009 | 0.9790 +/- 0.0010 | 0.8912 +/- 0.0026 | 0.0623 +/- 0.0081 | 0.1917 +/- 0.0107 | 0.2493 +/- 0.0159 |
| C | validation | 0.9739 +/- 0.0001 | 0.9721 +/- 0.0007 | 0.8729 +/- 0.0035 | 0.0688 +/- 0.0038 | 0.2138 +/- 0.0069 | 0.2621 +/- 0.0072 |
| A | test (retro.) | 0.8678 +/- 0.0086 | 0.8699 +/- 0.0096 | 0.6849 +/- 0.0135 | 0.3577 +/- 0.0635 | 0.4912 +/- 0.0426 | 0.5964 +/- 0.0538 |
| B | test (retro.) | 0.8692 +/- 0.0090 | 0.8689 +/- 0.0098 | 0.6856 +/- 0.0126 | 0.4067 +/- 0.0397 | 0.5212 +/- 0.0302 | 0.6372 +/- 0.0313 |
| C | test (retro.) | 0.8648 +/- 0.0122 | 0.8674 +/- 0.0154 | 0.6811 +/- 0.0212 | 0.3519 +/- 0.0469 | 0.4882 +/- 0.0340 | 0.5923 +/- 0.0400 |

Validation-to-test PLCC drops from about 0.98 to about 0.87 for all three models; RMSE rises from about 0.24-0.26 to about 0.59-0.64. Per-seed tables: `reports/experiments/final_per_seed_*.csv`.

## 3. Paired comparisons with A (seed-averaged predictions, 2000 image-level bootstrap resamples; intervals optimistic)
Primary metric: mean of PLCC, SROCC, KROCC. Two comparisons, so the Bonferroni interval is 97.5%.

| Comparison | Partition | Metric | Difference | 95% CI | 97.5% CI | Excludes 0 at 97.5%? |
|---|---|---|---|---|---|---|
| B - A | validation | mean corr | +0.0019 | [-0.0054, +0.0090] | [-0.0064, +0.0100] | no |
| B - A | validation | RMSE | +0.0130 | [-0.0124, +0.0403] | [-0.0164, +0.0440] | no |
| B - A | test | mean corr | +0.0003 | [-0.0037, +0.0044] | [-0.0043, +0.0051] | no |
| B - A | test | RMSE | +0.0408 | [+0.0265, +0.0546] | [+0.0243, +0.0570] | **yes (B worse)** |
| C - A | validation | mean corr | -0.0082 | [-0.0175, +0.0006] | [-0.0189, +0.0017] | no |
| C - A | validation | RMSE | +0.0258 | [+0.0071, +0.0434] | [+0.0055, +0.0461] | **yes (C worse)** |
| C - A | test | mean corr | -0.0031 | [-0.0093, +0.0032] | [-0.0103, +0.0039] | no |
| C - A | test | RMSE | -0.0041 | [-0.0166, +0.0085] | [-0.0188, +0.0105] | no |

## 4. Answers to the research questions
1. **Baseline accuracy.** A reaches validation PLCC 0.978 / RMSE 0.236 and, on the anatomy-disjoint retrospective test set, PLCC 0.868 / SROCC 0.870 / KROCC 0.685 / RMSE 0.596 (three seeds; SD of PLCC about 0.009). The test figures are the more realistic estimate of generalization.
2. **Does the ranking objective improve rank correlation?** Not demonstrated. Mean correlation changed by +0.002 (validation) and +0.0003 (test), well inside the bootstrap intervals; the seed-42 grid showed the same (+0.0018, half-width 0.0058). The hypothesis is **not supported**; the data are also compatible with a small true effect that 100/300 images and 3 seeds cannot resolve.
3. **Does uncertainty-aware regression give useful, calibrated uncertainty?** Only weakly and not reliably. sigma correlates weakly with absolute error (Spearman about 0.13 validation, 0.09 test, strongly seed-dependent). Conformal intervals reach nominal coverage on validation (0.94, not independent evidence) but only about 0.64 on the anatomy-disjoint test set (target 0.90), so they are **not calibrated under shift**. A constant-width interval from the same residuals does worse on test coverage (0.54) with a narrower width, so the learned sigma adds some coverage at the cost of width; usefulness is not established.
4. **Trade-offs.** B moved rank correlations marginally up and RMSE up (worse on test, significant by the declared interval); the ranking term does not buy accuracy and costs some. C's point accuracy is equal to A's on test and slightly lower on validation, so uncertainty output comes at little accuracy cost, but its coverage guarantee does not transfer to new anatomy.
5. **Supported vs hypothesis.** See the claim-to-evidence table below.

No overall winner is declared: A is the recommended default (simplest, best validation RMSE, equal test correlation); C is the only model that outputs an uncertainty, with the caveats above; B is not recommended.

## 5. Claim-to-evidence table
| Claim | Status | Evidence | Limitation |
|---|---|---|---|
| ResNet50 with a sigmoid regression head predicts the LDCT-IQAC quality score with test PLCC about 0.87 | Supported (retrospective) | A seeds 1-3, `final_per_seed_test_retrospective.csv` | test set previously seen by 001/003; one dataset |
| Pairwise ranking loss improves rank correlation | Not supported | B vs A, section 3; grid in `b_grid_selection_seed42.csv` | low power; 3 seeds; one grid |
| Pairwise ranking loss worsens absolute error | Weakly supported (test RMSE +0.04, CI excludes 0); not on validation | section 3 | image-level bootstrap optimistic; selection of epoch noisy |
| Gaussian-NLL head matches baseline point accuracy | Supported on test (difference within noise); slightly worse on validation | section 3 | 3 seeds |
| Predicted sigma tracks error | Weakly supported / seed-dependent | Spearman 0.03-0.24 | aleatoric only; weak |
| Conformal intervals are calibrated | **Not supported** under anatomy shift (test coverage 0.64 vs 0.90) | `exp_c_uncertainty_per_seed.csv` | validation coverage not independent; n = 100 calibration |
| Validation metrics estimate generalization | **Contradicted**: anatomy overlap, validation PLCC 0.98 vs test 0.87 | EDA anatomy groups, this table | image-level split frozen by protocol |
| `2x-1` is the right pretraining convention | Unresolved; probe finds `2x-1` and caffe-style indistinguishable | PREPROCESSING.md | probe is task evidence, not provenance |

## 6. Threats to validity
Image-level split with anatomy overlap (optimistic validation, narrow bootstrap intervals, checkpoint selection fits the overlap); retrospective test set; 3 seeds; B selection reused the validation set that also picked checkpoints; a clipped conformal comparison and a width-matched comparison were not exhaustively explored; CUDA nondeterminism; the working tree was dirty (recorded in manifests); historical 001-003 used a different BN epsilon and 900/100 split, so no claim of direct comparability is made. The word "novel" is not used: the contribution is an empirical, controlled comparison on one dataset, and no literature review for novelty has been performed.
