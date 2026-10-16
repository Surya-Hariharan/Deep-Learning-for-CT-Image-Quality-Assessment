# Limitations and threats to validity

| Threat | Detail | Mitigation / status |
|---|---|---|
| Small data | 800 training images; validation and calibration have 100 each | paired bootstrap; no ranking on sub-interval differences; conformal coverage SE about 0.03 |
| Image-level leakage | **measured** (EDA, `reports/eda/dataset_audit.json`): all validation and calibration images have a near-identical-anatomy partner in train (124 groups at r>0.999, 0 singletons); test images have none. Validation PLCC about 0.98 vs test about 0.87 | wording "image-level split"; frozen split not altered; validation metrics and bootstrap intervals described as optimistic; conformal coverage fails on the anatomy-disjoint test set (0.64 vs 0.90); group-aware re-split is future work |
| Prior test exposure | the official test set evaluated experiments 001 and 003 and their comparison | never used for choices in study v2; any use labeled retrospective; code-level guard |
| Historical BatchNorm mismatch | 001-003 trained at eps 1e-3, tested at 1e-5; source convention is 1.001e-5 | study v2 uses 1.001e-5 throughout; effect on 001-003 unquantified (no inference run) |
| Preprocessing convention | `2x - 1` documented for RadImageNet's PyTorch release; TF `.h5` convention not stated by any source found | flagged open in ADR-002; crop vs RadImageNet's resize is a deliberate deviation |
| Weight provenance | file hash recorded; RadImageNet training procedure not verified; training-time epsilon inferred from exported config | documented |
| Rater differences | training labels average five radiologists, test labels six (inferred from granularity); label noise differs by split | disclosed; not modelled |
| Averaged labels | per-rater scores unavailable, so inter-rater variance is not observable | Experiment C estimates aleatoric spread only through its own fit |
| Aleatoric only | a single deterministic Gaussian head does not capture epistemic uncertainty | stated in model and experiment docs |
| Conformal assumptions | marginal coverage needs exchangeability, unverified; clipping to [0, 4] changes interval width, not coverage of labels in [0, 4]; n = 100 gives coarse control | both interval versions reported; terciles as conditional diagnostic |
| Selection noise | selecting an epoch by PLCC on 100 images is noisy and slightly optimistic for the selected metric | same rule for all experiments; seed repeats suggested |
| Single split | one partition, one seed unless repeated | any other split is a separately labeled sensitivity analysis |
| Determinism | CUDA nondeterminism; CPU/GPU differences | seeds, dedicated shuffle generator, recorded environment |
| External validity | one dataset, one anatomy (abdominal CT), two artifact types | no claim beyond LDCT-IQAC |
| Clinical meaning | predicting a radiologist's 0-4 quality score is not diagnostic utility; nominal 90% intervals are not clinical safety | stated wherever intervals are reported |
| Synthetic tests | unit/integration tests use fabricated data and verify mechanics only | each such test says so |

Additional, from the executed study: three final seeds only (SD is a rough spread); B selection reused the validation set that also selects checkpoints; the constant-width interval comparator uses C's own point residuals and no width-matched comparison was made;
test results are retrospective (set previously seen by 001/003); runs were made with a dirty working tree (recorded in manifests); one first-attempt B run failed on a Windows file-lock error and was rerun as a new run.
