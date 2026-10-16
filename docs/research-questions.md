# Research questions

Hypotheses are statements to be tested; none has been tested yet. Results belong only in
[RESULTS_AND_COMPARISON](results-and-comparison.md).

## Primary question

With a shared RadImageNet-initialized ResNet50 backbone on LDCT-IQAC, do (B) a pairwise ranking term added to MSE and
(C) heteroscedastic Gaussian training with split-conformal intervals change the prediction and ordering of
radiologist quality scores, or provide useful uncertainty, relative to the reference regression (A)?

## Secondary questions

1. **RQ-A** How does the reference model behave on the frozen partitions, and how far is it from historical 001 given
   the corrected BatchNorm epsilon and new split? (Descriptive.)
2. **RQ-B** Does any ranking weight `lambda > 0` raise SROCC/KROCC on validation beyond bootstrap uncertainty while RMSE
   stays within 5% of A?
3. **RQ-C1** Does Gaussian NLL training preserve point accuracy relative to A?
4. **RQ-C2** Is predicted sigma positively associated with absolute error?
5. **RQ-C3** Do split-conformal intervals calibrated on 100 images attain about 90% coverage on an independent partition,
   at what width, and is coverage similar across predicted-sigma terciles?

## Hypotheses (H0 = no effect)

| ID | Hypothesis | Would be supported by | Would be refuted / not supported by |
|---|---|---|---|
| H-B | Ranking supervision improves rank correlation | paired 95% interval for SROCC/KROCC difference > 0 and RMSE guard met | interval includes 0, or RMSE guard violated |
| H-C1 | Point accuracy of C is comparable to A | all metric differences' intervals include 0 | C clearly worse on RMSE/PLCC |
| H-C2 | sigma tracks error | Spearman(sigma, abs error) > 0 with interval excluding 0 | near 0 or negative |
| H-C3 | conformal coverage is near nominal | validation coverage within about 0.06 of 0.90 (two standard errors, n = 100) | outside, or strongly sigma-dependent |

## Evaluation criteria

Metrics in [EVALUATION_METRICS](evaluation-metrics.md) on the validation partition; comparisons by paired bootstrap;
the official test set is evaluated once per frozen model and is retrospective.

## Not in scope

Other backbone families; patient-level generalization (not testable without patient IDs); clinical utility; epistemic
uncertainty; novelty claims about the losses themselves.

## Expected outcomes (uncertain, stated to avoid post-hoc rationalization)

Differences between A and B are likely small relative to validation noise (100 images); C's point metrics are expected
to be close to A's; conformal coverage with n = 100 is expected to be roughly nominal with sampling error near 0.03.
