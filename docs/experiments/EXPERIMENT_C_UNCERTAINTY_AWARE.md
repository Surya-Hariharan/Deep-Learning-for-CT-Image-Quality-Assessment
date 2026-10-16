# Experiment C: uncertainty-aware ResNet50 regression

ID `C_uncertainty_aware_resnet50` | config [c_uncertainty_aware.yaml](../../configs/experiments/c_uncertainty_aware.yaml) |
notebook `notebooks/05_uncertainty_experiment.ipynb` | status: **NOT RUN**

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
(unit-tested at the bounds with finite gradients). Diagram: [MODEL_ARCHITECTURE](../MODEL_ARCHITECTURE.md) section 4.

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
calibration partition (to be computed in the comparison notebook, PENDING) shows whether the learned sigma adds anything.

## 10. Reproducibility and artifacts
As A, plus `calibration.json` (rank k, q, unbounded flag, coverage, widths, NLL, terciles).

## 11. Results
| Run ID | PLCC | SROCC | KROCC | RMSE | NLL (0-4) | Coverage (val) | Mean width | Spearman(sigma,err) | Status |
|---|---|---|---|---|---|---|---|---|---|
| - | - | - | - | - | - | - | - | - | **PENDING** |

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
