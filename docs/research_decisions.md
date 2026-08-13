# Research Decisions — Ohashi/LDCT-IQAC Component

Scoped to Component B (the Ohashi-method CT-NR-IQA replication). For NGP-Net
(Component A) and the confidence/fusion components, see
`reports/research_decisions.md`, which predates this document and uses a
different but compatible taxonomy (VERIFIED FROM PAPER / DERIVED / PROJECT
DECISION / UNKNOWN). This file uses the taxonomy fixed for this component:

- **OHASHI-SPECIFIED** — stated in the project brief's summary of Ohashi et al.
- **PROJECT ADAPTATION** — our deliberate choice, most centrally the LDCT-IQAC
  dataset swap and its direct consequences.
- **NOT SPECIFIED BY OHASHI** — the source is silent; must not be guessed.
- **EXPERIMENTAL EXTENSION** — intentionally beyond the Ohashi baseline; none
  exist yet at this phase.

These four categories are never mixed within one entry.

Last updated: 2026-08-13.

---

## OHASHI-SPECIFIED

See `docs/ohashi_methodology.md` for the full list (degradation
grids, VIF as target, 60/20/20 reference-level split, ResNet50 + RadImageNet +
central crop + dropout + FC + sigmoid, Adam/64/30/MSE, LR grid {1e-2..1e-5},
five-parameter logistic calibration, PLCC/SROCC, three-stage evaluation).

## PROJECT ADAPTATION

| ID | Decision | Rationale |
| --- | --- | --- |
| A-01 | LDCT-IQAC (1,000 images) replaces DeepLesion+CQ500 (105 images) as the reference-image source | DeepLesion/CQ500 not present in this repository; LDCT-IQAC is present and fully verified — see `docs/dataset_adaptation.md` |
| A-02 | `expert_score` (LDCT-IQAC MOS) and `vif_score` kept as separate manifest fields, never conflated | `vif_score` remains the sole synthetic-stage training target per Ohashi; `expert_score` serves Stages 2–3 evaluation only |
| A-03 | Reference identity = real LDCT-IQAC `image_id`, not a synthetic placeholder | the dataset is actually present, so the plan should reflect real data rather than fabricated ids |
| A-04 | Split unit for the 60/20/20 split is `image_id` itself, no patient-level grouping | matches Ohashi's own split unit (the reference image); LDCT-IQAC carries no patient identifiers regardless (see `configs/datasets/ldctiqa.yaml`) |
| A-05 | Global seed = `20260813`, single source in `configs/paths.yaml` | reproducible reference-level split and per-image RNG |
| A-06 | Per-(reference, condition) RNG derived from `SHA256(seed, reference_id, condition)` | regenerating one image reproduces the exact noise realisation, independent of iteration order |
| A-07 | Blur uses `truncate=4.0`, `mode="nearest"` | scipy convention; recorded because boundary handling is not specified |
| A-08 | Processed images stored as 8-bit PNG | matches the LDCT-IQAC source encoding; lossless |
| A-09 | Calibration initial guess is data-driven, with a 4-point multi-start | one start fixes the sign of b1 and fails on inverse-S predictors; see `src/ct_iqa/evaluation/logistic_calibration.py` |
| A-10 | Calibration exponent clipped to ±500 | prevents overflow without changing the fitted function |
| A-11 | `apply_condition` requires `order` and `sigma_scale`, `compute_vif` requires `variant` — no defaults | makes NOT SPECIFIED BY OHASHI choices impossible to leave implicit |
| A-12 | The dataset generator refuses `--execute` while any NOT SPECIFIED BY OHASHI blocker is open | a wrong 169,000-image dataset generated reproducibly is worse than none |
| A-13 | VIFp implementation follows Sheikh & Bovik's reference `vifp_mscale` (4 scales, σ_nsq=2.0) | a documented standard rather than an ad-hoc reimplementation — variant choice is separate, see U-V01 below |
| A-14 | `src/ct_iqa/models/`, `src/ct_iqa/training/` define specifications only; `build_model()` and `Trainer.fit()` raise unconditionally | training is not authorized at this phase; the spec must not be quietly runnable |

## NOT SPECIFIED BY OHASHI

The dataset generator (`scripts/quality/build_ohashi_dataset.py`) enumerates
U-D01–U-D03 as blockers and refuses `--execute` while any remain.

| ID | Unknown | Why it matters | How to resolve |
| --- | --- | --- | --- |
| U-D01 | Noise sigma units — HU or 8-bit grey levels | σ=50 is mild in HU, severe in 8-bit; changes every degraded image | read the paper; LDCT-IQAC is 8-bit, so `sigma_scale=1.0` is the natural adaptation choice once confirmed |
| U-D02 | Combined-degradation order — blur-then-noise or noise-then-blur | noise-then-blur partially smooths the noise away; measurably different images (tested in `tests/test_degradation.py`) | read the paper |
| U-V01 | VIF variant — pixel-domain VIFp or wavelet-domain VIF | the two disagree numerically; determines every training label | read the paper; cross-validate `src/ct_iqa/vif/metric.py` against a MATLAB or trusted-package reference (DEV-03) |
| U-M01 | Dropout probability | flagged explicitly in the brief; large effect on regularisation | read the paper; else choose and record as a new PROJECT ADAPTATION entry before training |
| U-M02 | Frozen vs fine-tuned RadImageNet backbone | flagged explicitly in the brief; large effect on results | read the paper |
| U-M03 | Exact grayscale-to-3-channel implementation | affects every input pixel | read the paper (channel replication is used here as A-13-style documented adaptation until confirmed) |
| U-M04 | Resize-before-crop behaviour, interpolation | interacts with the blur degradation | read the paper |
| U-M05 | RadImageNet preprocessing / intensity normalisation | differs from standard ImageNet preprocessing | RadImageNet documentation |
| U-M06 | Adam betas, LR schedule, early stopping, checkpoint selection | reproducibility of the training curve | read the paper |
| U-M07 | Split stratification by source dataset | not applicable to a single-source (LDCT-IQAC-only) adaptation, but relevant if DeepLesion/CQ500 are later added back | read the paper |
| U-M08 | RadImageNet ResNet50 checkpoint | cannot build the model at all without it | obtain the RadImageNet release |

## EXPERIMENTAL EXTENSION

None yet. This baseline intentionally stays within the Ohashi + LDCT-IQAC
adaptation described above; extensions (additional backbones, alternative
calibration methods, augmentation) are out of scope until the baseline
reproduces.
