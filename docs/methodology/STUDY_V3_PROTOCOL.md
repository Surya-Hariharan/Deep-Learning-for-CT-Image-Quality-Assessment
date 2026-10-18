# Study v3 protocol: group-aware evaluation of Experiments A, B, C

Version `study_v3` (protocol document v3.0). Written before any v3 model was trained. Any change after the freeze requires a new versioned protocol (v3.1...), not an edit.
Background: [SPLIT_AND_CALIBRATION_AUDIT](../audit/SPLIT_AND_CALIBRATION_AUDIT.md). Study v2 (commits `c10014f`, `cec8ebd`, directory `experiments/study_v2/`) is preserved unchanged and remains the historical record.
All v3 artifacts live under `experiments/study_v3/`, `reports/study_v3/` and `docs/methodology/STUDY_V3_*`; nothing in v2 is overwritten.

## 1. What v3 changes and why
v2 split the 1000 labeled images at image level, but the images are repeated renderings of the same content at different quality levels. Validation and calibration images therefore had near-identical siblings in training,
which made validation metrics optimistic and made the conformal calibration set unrepresentative of unseen content. v3 keeps every group of near-identical images inside one partition and
evaluates on held-out groups by grouped 5-fold cross-validation, with group-level resampling for uncertainty. Models, preprocessing, optimizer, epochs and selection rules are those of v2 (section 5).

## 2. Groups (image-similarity components; NOT patient, scan or slice identifiers)
- **Algorithm** (`ct_iqa.data.grouped`, `tools/create_study_v3_manifests.py`): each of the 1000 labeled images is resized (PIL bilinear) to 64x64 from the full 512x512 float image and flattened; Pearson correlation r between all pairs;
  graph edge where r > **0.99**; groups = connected components; ids `G000..G053` ordered by smallest file name. Software versions are recorded in the manifest.
- **Why 0.99:** it is the most conservative threshold for which no single group exceeds 120 images (0.98 gives groups up to 130 and only 37 groups; 0.95 gives 16 groups up to 250 images, which cannot support train/validation/calibration/evaluation roles).
  At 0.9995 the labeled set resolves into 98 groups of exactly 10 images plus one of 20, i.e. one group per rendered source; 0.99 additionally merges visually similar different sources. The threshold was chosen from this structure and the feasibility requirement alone, before any model result existed.
- **Result:** 54 groups: 34 of 10 images, 12 of 20, 1 of 30, 5 of 40, 1 of 70, 1 of 120. Maximum correlation between images of different groups: 0.9898 (< 0.99 by construction). Median nearest-other-group correlation 0.9795.
- **Limitations (stated wherever v3 results appear):**
  1. No patient/scan identifiers exist; groups are similarity components. Patient-level independence between partitions is **not** established.
  2. Different groups can still be quite similar (r up to 0.99). The official test set is much further from the labeled images (max r 0.942, median 0.896), so v3 held-out data remain **closer to the training data than the official test set**. Expected consequence (a prediction, not a hypothesis test): v3 metrics fall between v2 validation (PLCC about 0.98) and v2 test (about 0.87).
  3. With 54 groups the effective number of independent units for any group-level statistic is at most 54 (fewer if groups share patients).
  4. The 120- and 70-image groups are always in train when not evaluated (section 3).

## 3. Folds (5-fold grouped cross-validation; manifests are the source of truth)
- `experiments/study_v3/protocol/group_manifest.json` (sha256 `fab04204...5d79`) and `fold_manifest.json` (sha256 `9c0fe67b...33a5`), created once and never overwritten.
- **Evaluation partitions:** groups sorted by decreasing size (ties in seeded random order, seed 3030) are greedily placed in the bin with the fewest images (5 bins); every group is evaluated in exactly one fold.
- **Validation and calibration** (fold k): groups larger than 40 images stay in train; from the remaining non-evaluation groups, validation then calibration are drawn in a seeded random order (seed 3030 + 1000 + k) until each has at least 100 images; the rest is train.
- **Sizes (groups / images):**

| Fold | Train | Validation | Calibration | Evaluation |
|---|---|---|---|---|
| 0 | 31 / 590 | 8 / 100 | 7 / 110 | 8 / 200 |
| 1 | 30 / 590 | 5 / 100 | 8 / 110 | 11 / 200 |
| 2 | 29 / 600 | 8 / 100 | 6 / 100 | 11 / 200 |
| 3 | 26 / 600 | 9 / 100 | 7 / 100 | 12 / 200 |
| 4 | 30 / 580 | 6 / 120 | 6 / 100 | 12 / 200 |

- **Roles:** train = optimizer updates; validation = checkpoint selection and B hyperparameter selection only; calibration = Experiment C conformal fitting only (never selection); evaluation = held-out groups reported after training, never used for selection or calibration of that fold.
- **Integrity checks (verified before the freeze, `tools/verify_study_v3.py`, plus unit tests):** group manifest and fold manifest hashes; groups partition the 1000 images exactly once; within every fold the four partitions are group-disjoint and file-disjoint and cover all images; every group is evaluated exactly once; run manifests record both hashes.
- **Official test set (300 images):** in no fold and in no decision. It was examined before (historical 001/003 and study v2). v3 reports it only as a **descriptive retrospective** evaluation of each finished model after the freeze; it cannot be confirmatory evidence.

## 4. Fixed settings (identical for A, B, C; inherited from v2 `configs/base.yaml`)
ResNet50 (`bn_eps` 1.001e-5, RadImageNet weights SHA-256 `45cba0d5...a0f785`), 224 center crop, `2x-1`, gray replicated to 3 channels, dropout 0.5, Adam lr 1e-3, batch 64, **30 epochs**, no augmentation or scheduler, targets y/4 with sigmoid output,
checkpoint = highest validation PLCC (raw scale, ties keep earlier epoch, 1-based epochs). Training sets are about 590 images (v2: 800). Validation-set noise is large (5-9 groups); this is accepted and reported, not tuned away.

## 5. Experiments and seeds
- **A:** MSE. **C:** Gaussian NLL, clamped log-variance in [-10, 2], aleatoric only. **B:** MSE + lambda x pairwise margin ranking, in-batch pairs with label gap >= delta.
- **B selection (declared now):** grid and rule as in v2 (`ct_iqa.experiment.grid`): stage 1 lambda in {0.03, 0.1, 0.3, 1.0} at margin 0.05 / delta 0.25; stage 2 at the stage-1 lambda: margin 0.02, margin 0.1, delta 0.5; seed 42; **fold 0 only** (train on fold-0 train, select on fold-0 validation; fold-0 evaluation groups are never used for selection).
  Rule: RMSE <= 1.05 x A's (A = fold 0, seed 42); highest mean(PLCC, SROCC, KROCC) among guarded points (ties keep the earlier point); paired **group** bootstrap (resample fold-0 validation groups, 2000, seed 0) of the mean-correlation difference; "improvement demonstrated" only if the difference exceeds the interval half-width.
  **Regardless of that outcome the best guarded point is retrained for the final runs** (declared now, so the A-vs-B comparison rests on replicates). If no point passes the guard, B final runs use the stage-1 lambda 0.3 / margin 0.05 / delta 0.25 and this is reported as a deviation.
  Caveat: with 8 validation groups the selection is weak evidence; groups that are evaluation groups of folds 1-4 sit in fold-0 train/validation, so the selected hyperparameter carries mild (low-dimensional) information from other folds' evaluation data. This is disclosed, not corrected.
- **Final runs:** A, B (selected point) and C on folds 0-4 x seeds **1, 2, 3** = 45 runs. Seed 42 runs are selection runs and are excluded from final aggregates. Smoke runs (`experiments/study_v3_smoke/`) are never reported.
- **Failures:** a failed run stays recorded as failed; it is rerun as a new run with identical settings; selection and aggregation refuse ambiguous duplicates. A defect that needs a protocol change stops execution and creates protocol v3.1.

## 6. Metrics and aggregation
- Point metrics on raw [0, 4] scores: PLCC, SROCC, KROCC, MSE, MAE, RMSE. **Primary metric:** mean of PLCC, SROCC, KROCC.
- **Pooled per seed:** for each model and seed, the five folds' evaluation predictions (each image predicted exactly once, by a model that never saw its group) are pooled (1000 images) and the metrics computed on the pool. Per-fold metrics are reported as a sensitivity view. Across seeds: mean and SD (3 seeds; SD is a rough spread of training randomness).
- **Group-level uncertainty:** resample the 54 groups with replacement (2000 draws, seed 0), take all images of each sampled group, recompute the pooled metrics for each seed and average over seeds within the draw. This gives group-resampling intervals for each model and for paired differences B - A and C - A (same draws).
  Seeds are **not** treated as independent observations and are not resampled; seed-to-seed variability (SD) and evaluation-group uncertainty (bootstrap) are reported separately. Predictions of different seeds on the same images are never pooled as if independent.
- **Inference rule:** a difference is called demonstrated only if the Bonferroni-adjusted 97.5% interval (two comparisons) of the primary metric excludes zero; otherwise "no demonstrated difference". Because groups are not patients and there are at most 54 units, even these intervals may be optimistic and are described as group-level, not patient-level.

## 7. Experiment C: conformal calibration and its assumptions
Target coverage 0.90 (fixed). Nonconformity r_i = |y_i - mu_i| / sigma_i with sigma = 4 exp(s/2); interval mu +/- q sigma (unclipped and clipped to [0, 4] reported).
- **Primary (P), grouped split conformal:** q fitted on the fold's calibration groups (all their images, n about 100, k = ceil((n+1) 0.9)), evaluated on the fold's evaluation groups. **Assumption:** calibration groups and evaluation groups are exchangeable draws of groups from one population. Within a group images are strongly dependent, so the effective calibration size is the number of groups (6-8), the guarantee is marginal and approximate, and per-fold realized coverage is expected to vary widely.
- **Secondary (S), grouped cross-conformal:** for fold k, q is the k'-th smallest r among the out-of-fold evaluation residuals of the other four folds (same model family and seed; n about 800, k' = ceil((n+1) 0.9)), evaluated on fold k's evaluation groups. Assumption: the five fold-models and their held-out groups are exchangeable; models differ across folds, so the guarantee is approximate.
- **Reported (both):** pooled coverage with covered counts (k/n), per-fold coverage, group-bootstrap interval for pooled coverage, mean/median width (unclipped and clipped), coverage by sigma tercile, Spearman(sigma, |error|), Gaussian NLL, and the constant-width comparator (same model's absolute residuals on the same calibration images for P, other folds' residuals for S). A width-matched comparison is not claimed.
- q is never tuned using evaluation or test coverage. Official-test coverage uses each fold's P quantile and is descriptive only.

## 8. Comparison with historical results
The final report lists side by side: study v3 pooled evaluation metrics; study v2 validation (optimistic) and retrospective test metrics; historical 001/003 test metrics. Differences in data (about 590 vs 800 vs 900 training images), partition type (held-out groups vs validation vs the examined test set) and BatchNorm epsilon (001/003) are stated; no direct comparability is claimed.

## 9. What v3 can and cannot establish
Can: how A, B, C behave on held-out groups of near-identical images, under a protocol where calibration is fitted on groups that were not seen in training; whether v2's conclusions (no ranking benefit, miscalibrated intervals) persist when train/evaluation share no similarity-groups; group-level uncertainty of those comparisons.
Cannot: patient-level generalization (no identifiers); confirmatory test evidence (the official test set was examined); generalization beyond this dataset; novelty; conformal validity for data less similar to training than the groups here (the official test set is such a case). Held-out groups are closer to the training data than the official test set is.

## 10. Freeze
Before the first selection run: `experiments/study_v3/protocol/FROZEN` records sha256 of this document, both manifests, the base and experiment configs, and the code that produces or analyzes results (`src/ct_iqa/data/grouped.py`, `src/ct_iqa/experiment/run_v3.py`, `tools/run_study_v3.py`, `tools/select_b_v3.py`, `tools/build_reports_v3.py`, `tools/verify_study_v3.py`), the HEAD commit and the dirty state. The final-run launch list is generated from this protocol.
