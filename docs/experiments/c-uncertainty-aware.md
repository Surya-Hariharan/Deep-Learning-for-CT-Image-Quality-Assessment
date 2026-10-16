# Experiment C: uncertainty-aware ResNet50 regression

ID `C_uncertainty_aware_resnet50` | config [c_uncertainty_aware.yaml](../../configs/experiments/c_uncertainty_aware.yaml) |
notebook `notebooks/05_uncertainty_experiment.ipynb` | status: **EXECUTED** (pilot seed 42 + final seeds 1-3; results below)

## 1. Objective
Predict, alongside the quality score, a per-image predictive variance, and turn it into prediction intervals whose
coverage is checked on data not used to fit them.

## 2. Research question and hypotheses
RQ1: does heteroscedastic Gaussian training keep point accuracy comparable to A? RQ2: does predicted sigma correlate
with absolute error? RQ3: do split-conformal intervals reach about 90% empirical coverage on an independent partition,
and at what width?
H-C1: point metrics within the bootstrap interval of A. H-C2: positive Spearman(sigma, |error|). H-C3: coverage near
0.90 on validation. All three can fail; with 100 calibration images, coverage control is coarse.

## 3. Rationale
A single Gaussian head `(mu, log sigma^2)` models data-dependent (aleatoric) noise, including inter-rater disagreement
that is not observed here (only averaged scores are available). It does **not** estimate epistemic (model)
uncertainty; no ensembling or MC dropout is used, so none is claimed.

## 4. Architecture and outputs
`resnet50_gaussian`: shared backbone, dropout 0.5, `fc` -> sigmoid -> `mu_n in [0,1]`, `fc_logvar` -> clamp[-10, 2] ->
`s_n` (normalized-space log-variance). `ModelOutput(mean, log_var)`. The clamp keeps `exp(-s)` and `exp(s)` finite
(unit-tested at the bounds with finite gradients). Diagram: [MODEL_ARCHITECTURE](../model-architecture.md) section 4.

## 5. Loss
`L_C = (1/(2N)) sum_i [exp(-s_i) (y_i - mu_i)^2 + s_i]` in the normalized space (constant omitted); matches
`torch.nn.GaussianNLLLoss` (eps = 0) in a unit test. Code: `losses/gaussian_nll.py`.

## 6. Data and selection
Frozen manifest: train 800, validation 100 (selection by validation PLCC of `mu`, same rule as A), calibration 100
(never used for training or selection). Validation NLL is also recorded per epoch (`val_loss`).

## 7. Calibration (split-conformal, the single predeclared primary method)
Target coverage 0.90 fixed in `configs/base.yaml` before any result. Convert `mu = 4 mu_n`, `sigma = 4 exp(s_n/2)`;
on the calibration partition compute `r_i = |y_i - mu_i| / sigma_i`; `k = ceil((n+1) * 0.9)` (n = 100 -> k = 91);
`q` = k-th smallest `r_i`, or +inf (unbounded interval) if `k > n`; interval `mu +/- q sigma`. Reported both
unclipped and clipped to [0, 4]. Because labels lie in [0, 4], clipping cannot change whether a label is covered; it
only narrows the interval (width and fraction of clipped intervals are reported). Evaluation of the interval is on the
validation partition (independent of calibration). Guarantee: marginal coverage >= 0.90 under exchangeability, which is
**not established** here. Never fitted on official test labels. Code: `evaluation/intervals.py`; edge cases
(`k > n`, ties, float noise in `k`, scale conversion) are unit-tested.

## 8. Metrics
Point: PLCC, SROCC, KROCC, MSE, MAE, RMSE of `mu` (raw, [0, 4]). Distribution/intervals: Gaussian NLL in the [0, 4]
scale (labeled; the training loss is in the normalized scale, so values are not comparable between scales), empirical
coverage, mean/median width, Spearman(sigma, |error|), coverage by sigma tercile (conditional-coverage diagnostic).
Point and uncertainty metrics are reported separately and never combined.

## 9. Baselines
A (same split, same selection). Interval baseline: a constant-width conformal interval from A's residuals on the
calibration partition (computed by `tools/build_reports.py`, see Results) shows whether the learned sigma adds anything.

## 10. Reproducibility and artifacts
As A, plus `calibration.json` (rank k, q, unbounded flag, coverage, widths, NLL, terciles).

## 11. Results
(Results: see the Results section at the end of this document.)

## 12. Interpretation rules
Nominal 90% coverage is not clinical reliability. Report n for calibration (100) and evaluation (100); the standard
error of an empirical coverage near 0.9 with n = 100 is about 0.03, so deviations of that size are noise. Do not tune
the quantile, coverage target or clamp bounds after seeing calibration results.

## 13. Limitations
Aleatoric only; targets are rater averages (5-6 raters), so label noise differs between train (5 raters) and test (6);
image-level split; small calibration set; conformal guarantee is marginal, not per-image or per-quality-level.

## 14. Completion checklist
- [ ] run completed; `calibration.json` present; q finite
- [ ] separate point and uncertainty tables filled with run ID
- [ ] constant-width baseline computed
- [ ] limitations restated next to the results

## Results (executed 2026-10-16; `reports/experiments/exp_c_uncertainty_per_seed.csv`, `exp_c_uncertainty.png`)
Selection/pilot seed 42 (validation): PLCC 0.9762, SROCC 0.9765, KROCC 0.8870, MSE 0.1033, MAE 0.2584, RMSE 0.3214 (epoch 30); conformal q = 3.41, validation coverage 0.93.

Final seeds 1-3 (mean +/- SD):

| Partition | PLCC | SROCC | KROCC | MSE | MAE | RMSE |
|---|---|---|---|---|---|---|
| validation (100) | 0.9739 +/- 0.0001 | 0.9721 +/- 0.0007 | 0.8729 +/- 0.0035 | 0.0688 +/- 0.0038 | 0.2138 +/- 0.0069 | 0.2621 +/- 0.0072 |
| official test (300), retrospective | 0.8648 +/- 0.0122 | 0.8674 +/- 0.0154 | 0.6811 +/- 0.0212 | 0.3519 +/- 0.0469 | 0.4882 +/- 0.0340 | 0.5923 +/- 0.0400 |

Uncertainty (target coverage 0.90; split-conformal on the 100 calibration images, k = 91; q = 2.14 / 1.72 / 2.16 for seeds 1 / 2 / 3). The constant-width comparator is the conformal
quantile of the *same model's* absolute residuals on the same calibration images (the protocol text said "A's residuals"; the implementation uses C's own point residuals, which isolates the value of the learned sigma).

| Quantity | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---|---|---|
| validation coverage (n=100; clipped identical) | 0.96 | 0.92 | 0.95 | 0.943 |
| validation mean width, learned sigma | 1.02 | 1.04 | 1.08 | 1.05 |
| validation coverage / width, constant-width | 0.94 / 0.99 | 0.92 / 0.97 | 0.93 / 0.93 | 0.930 / 0.96 |
| Spearman(sigma, abs error), validation | 0.239 | 0.031 | 0.124 | 0.131 |
| validation coverage by sigma tercile (low/mid/high) | .94/.97/.97 | .82/.97/.97 | .85/1.0/1.0 | low-sigma tercile under-covers |
| **test (300, retrospective) coverage, learned sigma** | 0.660 | 0.627 | 0.620 | **0.636** |
| test mean width, learned sigma | 1.06 | 1.13 | 1.24 | 1.14 |
| test coverage / width, constant-width | 0.627 / 0.99 | 0.550 / 0.97 | 0.457 / 0.93 | 0.544 / 0.96 |
| Spearman(sigma, abs error), test | 0.145 | 0.012 | 0.111 | 0.090 |

Interpretation:
1. Point accuracy of C is similar to A on test (PLCC 0.865 vs 0.868; RMSE difference -0.004, interval [-0.017, +0.009], not distinguishable) and slightly worse on validation (RMSE +0.026, interval excludes zero).
2. The predicted sigma has only a weak, seed-dependent positive association with absolute error (Spearman 0.03-0.24 on validation, 0.01-0.15 on test).
3. Validation coverage is near or above nominal (0.92-0.96) but validation shares anatomy with calibration and training, so this is **not independent evidence**. On the anatomy-disjoint test set coverage falls to about 0.64 against a 0.90 target, for both learned-sigma and constant-width intervals (the latter worse, 0.54, at a narrower width). The conformal exchangeability assumption is violated by the shift, so the intervals cannot be called calibrated for new anatomy.
4. Learned sigma gives higher test coverage than the constant-width comparator but at larger mean width (1.14 vs 0.96); a width-matched comparison was not performed, so a benefit of the learned sigma is not established. Only aleatoric uncertainty is modelled.
