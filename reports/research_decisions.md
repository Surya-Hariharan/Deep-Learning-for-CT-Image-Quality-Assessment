# Research Decisions Log

Three categories, never mixed:

- **VERIFIED FROM PAPER** — stated in the source (paper, official repository, or
  the project brief that quotes it). Reproduce exactly.
- **PROJECT DECISION** — the source is silent; we chose, and the choice is
  recorded here with its rationale so it can be revisited or reported.
- **UNKNOWN / REQUIRES VERIFICATION** — the source is silent or ambiguous and we
  have *not* chosen. Nothing downstream may assume a value.

A fourth marker, **DERIVED**, is used for facts forced by arithmetic on verified
numbers. Derived facts are strong but are not the same as read-from-source, and
each one names the check that would confirm it.

Last updated: 2026-08-13.

---

## VERIFIED FROM PAPER

### Component B — Ohashi et al. (2025), CT IQA

| ID | Item | Value |
| --- | --- | --- |
| V-01 | Reference images | 105 total: 90 DeepLesion + 15 CQ500 |
| V-02 | Gaussian noise sigma grid | 1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 14, 21, 30, 50 (12 levels) |
| V-03 | Gaussian blur sigma grid | 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.4, 2.1, 3.0, 5.0 (12 levels) |
| V-04 | Degradation conditions | noise-only, blur-only, and noise+blur combinations |
| V-05 | Total degraded dataset size | 17,745 images |
| V-06 | Label | VIF of each degraded image against its clean reference |
| V-07 | Split ratios | 60 / 20 / 20 → 10,647 / 3,549 / 3,549 images |
| V-08 | Split unit | reference-image identity, never individual degraded images |
| V-09 | Backbone in scope | ResNet50 only for now (not all four Ohashi backbones) |
| V-10 | Pretraining | RadImageNet |
| V-11 | Input shape | 224 × 224 × 3 |
| V-12 | Head | GAP → Dropout → Dense(1) → Sigmoid |
| V-13 | Loss | MSE |
| V-14 | Optimizer | Adam |
| V-15 | Batch size | 64 |
| V-16 | Epochs | 30 |
| V-17 | LR search grid | 1e-2, 1e-3, 1e-4, 1e-5 |
| V-18 | Best LR for ResNet50 | 1e-3 |
| V-19 | Calibration | five-parameter logistic, nonlinear least squares |
| V-20 | Calibration form | `Y_L = b1·(1/2 − 1/(1+exp(b2·(Y−b3)))) + b4·Y + b5` |
| V-21 | Evaluation metrics | PLCC and SROCC |
| V-22 | First replication target | predicted quality vs VIF (objective, not subjective) |

### Component A — NGP-Net

| ID | Item | Value |
| --- | --- | --- |
| V-30 | Dataset | PNG longitudinal pulmonary nodule dataset |
| V-31 | Stated scale | 103 patients, 378 CT exams, 226 nodules, ≥3 timepoints, 2–64 month irregular intervals |

**Nothing else about NGP-Net is verified.** No paper PDF and no reference
implementation is present in this repository, so every architectural,
preprocessing and training detail is UNKNOWN (see U-20…U-29).

---

## DERIVED (arithmetically forced — confirm against the paper text)

### D-A1 — The clean reference is almost certainly a dataset member

105 × (12 noise + 12 blur + 144 combined) = **17,640**, which is exactly **105
short** of the reported 17,745 — a deficit of exactly one image per reference.

The parsimonious explanation is that each clean reference is itself included as
a sample, with maximal quality (VIF = 1.0 against itself). Two independent
checks support it:

- 105 × 169 = 17,745 exactly.
- 63 / 21 / 21 references × 169 = 10,647 / 3,549 / 3,549 — the published split
  sizes, exactly, with no rounding.

Both checks are asserted in `tests/test_degradation.py` and
`tests/test_splits.py`, and reproduced by
`scripts/quality/build_ohashi_dataset.py`.

**Status:** not confirmed against the paper text. `build_grid(include_clean=False)`
reproduces the 17,640 variant if this turns out to be wrong. **Confirm by:**
reading the dataset-construction section of Ohashi et al.

### D-A2 — Reference-level split sizes are 63 / 21 / 21

Follows from D-A1 and V-07. Recorded as exact counts in
`configs/quality/ohashi_degradation.yaml` so that ratio rounding can never drift.

### D-A3 — VIF target range is (0, 1)

Consistent with the sigmoid output head (V-12) and with VIF's own range.

---

## PROJECT DECISION

| ID | Decision | Rationale |
| --- | --- | --- |
| D-01 | Global seed = `20260813`, single source in `configs/paths.yaml` | one seed, consumed by every script, so any split or degradation is reproducible |
| D-02 | **SUPERSEDED 2026-08-13** ~~LDCT-IQAC assigned role `subjective_iqa_validation`, not "reference images"~~ → LDCT-IQAC is now the ACTIVE Ohashi reference-image source (`role: ohashi_reference_images`), an explicit PROJECT ADAPTATION replacing DeepLesion+CQ500. See `docs/dataset_adaptation.md` and `docs/deviations_from_ohashi.md` for the full rationale, the taxonomy used there (OHASHI-SPECIFIED / PROJECT ADAPTATION / NOT SPECIFIED BY OHASHI / EXPERIMENTAL EXTENSION), and the resulting count changes (169,000 vs 17,745 total images). | user directive: use LDCT-IQAC as the reference-image dataset while preserving Ohashi methodology otherwise |
| D-03 | LDCT-IQAC data relocated to `data/raw/ldctiqa/`, contents unaltered | keeps the raw layout uniform; verified byte-identical after the move |
| D-04 | Datasets keep separate raw/interim/processed trees; never merged | preserves per-dataset provenance and preprocessing |
| D-05 | Degradation ids follow `<REF>_noise_<σ>_blur_<σ>` | stable, human-readable, collision-free (tested) |
| D-06 | Reference ids: `DL_###` (DeepLesion), `CQ_###` (CQ500) | source is recoverable from the id alone |
| D-07 | Per-image RNG derived from `SHA256(seed, reference_id, condition)` | regenerating one image reproduces the exact noise realisation, independent of iteration order or worker count |
| D-08 | Blur uses `truncate=4.0`, `mode="nearest"` | scipy convention; recorded because the source does not state kernel extent or boundary handling |
| D-09 | Processed IQA images stored as PNG | lossless — JPEG would add an uncontrolled distortion on top of the studied ones |
| D-10 | Calibration initial guess derived from the data (`b1`=target range, `b2`=1/std, `b3`=mean, `b4`=0, `b5`=target mean) | the paper gives no initialisation; data-driven beats unexplained constants |
| D-11 | Calibration uses a 4-point multi-start, best SSE wins | one start fixes the sign of `b1` and fails on inverse-S predictors; affects optimiser robustness only, never the published model form |
| D-12 | Calibration exponent clipped to ±500 | prevents `np.exp` overflow during optimisation without changing the fitted function |
| D-13 | Missing metadata written as `null`, never imputed | an unknown spacing stays unknown |
| D-14 | `apply_condition` requires `order` and `sigma_scale` as explicit arguments | makes the two unresolved Ohashi ambiguities impossible to leave implicit |
| D-15 | `compute_vif` requires an explicit `variant` and has no default | choosing silently would bake an unverified choice into every label |
| D-16 | The IQA generator refuses `--execute` while any blocker is open | a wrong dataset generated reproducibly is worse than none |
| D-17 | VIFp implementation follows Sheikh & Bovik's reference `vifp_mscale` (4 scales, σ_nsq = 2.0) | a documented standard rather than an ad-hoc reimplementation — but see U-04, the variant question is separate |

---

## UNKNOWN / REQUIRES VERIFICATION

Nothing below may be assumed by any script. The IQA generator enumerates
U-01…U-05 as blockers and refuses to produce images while they stand.

### Component B — Ohashi IQA

| ID | Unknown | Why it matters | How to resolve |
| --- | --- | --- | --- |
| U-01 | Which 90 DeepLesion + 15 CQ500 slices, and by what selection rule | different slices → different VIF distribution → different model | paper methods section |
| U-02 | Reference preprocessing before degradation (windowing, HU clipping, resize) | sets the intensity scale that noise sigma is relative to | paper methods section |
| U-03 | Noise sigma units — HU or 8-bit grey levels | σ=50 is mild in HU and catastrophic in 8-bit; changes every label | paper; or infer from the reported VIF range once references exist |
| U-04 | VIF variant — pixel-domain VIFp or wavelet-domain VIF | the two disagree numerically; determines every training label | paper; check for a cited implementation |
| U-05 | Combined-degradation order — blur-then-noise or noise-then-blur | noise-then-blur partially smooths the noise away; measurably different images (tested) | paper |
| U-06 | Dropout rate | **explicitly flagged by the project brief**; must become a PROJECT DECISION before training | paper; else choose and record here |
| U-07 | Whether the RadImageNet backbone is frozen or fine-tuned | **explicitly flagged by the project brief**; large effect on results | paper |
| U-08 | Intensity normalisation for RadImageNet inputs | RadImageNet preprocessing differs from ImageNet | RadImageNet documentation |
| U-09 | Resize interpolation to 224 × 224 | interacts directly with the blur degradation | paper |
| U-10 | Data augmentation | assume none until verified — do not add any | paper |
| U-11 | Adam betas, LR schedule, early stopping, checkpoint selection | reproducibility of the training curve | paper |
| U-12 | Whether splits were stratified by source dataset | affects DeepLesion/CQ500 balance per partition | paper |
| U-13 | Bit depth of stored degraded images | 8-bit quantisation is itself a distortion | follows from U-02 |
| U-14 | Clipping after noise addition | determines whether σ=50 saturates | paper |
| U-15 | RadImageNet ResNet50 checkpoint not yet obtained | cannot build the model at all | RadImageNet release |

### Component A — NGP-Net

| ID | Unknown |
| --- | --- |
| U-20 | Input representation (2D / 2.5D / 3D), patch size, resampled spacing, intensity window |
| U-21 | Number of timepoints consumed, and how irregular intervals are encoded |
| U-22 | Backbone and temporal module architecture |
| U-23 | Parameter count ("lightweight" is not a specification) |
| U-24 | Output head — growth regression vs growth classification |
| U-25 | Loss, optimizer, LR, batch size, epochs, augmentation |
| U-26 | Split protocol and cross-validation scheme |
| U-27 | Evaluation metrics and the reported baseline to replicate against |
| U-28 | PNG dataset file format, identifier scheme and annotation format |
| U-29 | Definition of the growth label itself |

**All of Component A is blocked** until the paper or the official repository is
available. `configs/growth/ngpnet.yaml` is deliberately all-nulls.

### Component C — confidence and fusion

| ID | Unknown |
| --- | --- |
| U-40 | Peeters et al. ensemble size and entropy formulation as applied to a growth model |
| U-41 | How a regression output yields an entropy-based uncertainty (vs a classification output) |
| U-42 | The Q + U fusion function — **this is the novel contribution and is intentionally undesigned until both modules reproduce** |
| U-43 | Accept / Flag / Reject thresholds, and what clinical evidence would justify them |

---

## Open questions for the supervisor

1. Was the LDCT-IQAC dataset downloaded intentionally? It fits the project's
   theme but fills none of the six planned roles, and its lack of patient
   identifiers limits how it can be used.
2. Is a copy of the NGP-Net paper or repository available? Component A cannot
   start without one.
3. Is full-text access to Ohashi et al. (2025) available? Resolving U-01…U-05
   is the single highest-value unblock in the project.
