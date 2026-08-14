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

Two items resolved directly from the source paper text (previously carried as
NOT SPECIFIED, based on a secondhand brief that didn't include this detail):

| ID | Item | Evidence |
| --- | --- | --- |
| S-01 (was U-M02) | RadImageNet backbone is **fine-tuned**, not frozen | Paper: "By fine-tuning these pre-trained models with our IQA dataset..." — explicit, not inferred |
| S-02 (was part of U-V01) | VIF formulation is **wavelet-domain VIF** (Sheikh & Bovik 2006, GSM-over-wavelet-subbands), not pixel-domain VIFp | Paper cites ref [27] = Sheikh & Bovik, "Image information and visual quality," IEEE TIP 15(2):430-444, 2006 — the original VIF paper, not the earlier VIFp approximation. See "VIF Implementation Status" below — the *formulation family* is resolved, but exact MATLAB implementation parameters are not (tracked as U-V01, narrowed). |

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
| A-16 | Combined-degradation order = `blur_then_noise`. Resolves U-D02. | Ohashi's Methods/Fig. 3 caption never states the order; `blur_then_noise` was already the pilots' provisional choice (`scripts/quality/vif_pilot.py`, `scripts/quality/vif_full_grid_pilot.py`) and is retained rather than silently changed at production time. Rationale: it models a physical acquisition chain where optical/reconstruction blur precedes final sensor/quantization noise, which is a defensible (not verified) mental model, not a claim about Ohashi's own pipeline. `noise_then_blur` measurably differs (`tests/test_degradation.py::test_combination_order_actually_matters`), so this choice materially affects every one of the 14,400/reference combined labels. Recorded here as PROJECT ADAPTATION, not OHASHI-SPECIFIED. |
| A-17 | Noise sigma is interpreted directly in 8-bit grey-level units (`sigma_scale=1.0`, `sigma_units="8bit_grey_levels"`). Resolves U-D01. | Ohashi's pipeline degrades already-saved 8-bit grayscale images; LDCT-IQAC is native 8-bit PNG, so treating the published sigma grid as already being in image units is the natural adaptation. This was already the pilots' provisional value, retained rather than changed at production time. Not a claim that Ohashi's own MATLAB run used the same units — genuinely unverifiable without their code. |
| A-18 | VIF variant for production label generation = `vif_wavelet`, `profile="project"` (never `"reference_crosscheck"`, never `vifp`). | S-02 establishes the wavelet-domain VIF formulation family is OHASHI-SPECIFIED; A-15 documents the implementation is structurally validated (parameter-equivalence Pearson 0.99999999999989) and behaves sanely across 16,800 full-grid-pilot images (zero NaN/Inf/negative/out-of-range, `docs/vif_full_grid_pilot_report.md` §5). `vifp` remains available but is never substituted — see `ct_iqa.vif.labeling.compute_vif`, which still requires the variant to be named explicitly at every call site, including the production generator. |
| A-19 | `UNSTABLE_CHANNEL_GAIN` (and every other VIF diagnostic flag) is FLAGGED, RETAINED, and NEVER used to silently alter, clip, replace, or exclude a record. Every production record retains `vif_score`, `diagnostic_status`, `max_channel_gain`, and `covariance_condition_number` verbatim. | `docs/vif_full_grid_pilot_report.md` §6/§16.1: the flag fires on 100% of noise-bearing conditions (95.2% of the full grid) — any exclusion rule would eliminate virtually the entire dataset, and the flag was mechanistically traced to a real, non-bug property of GSM channel-gain estimation on near-zero-variance regions, not a computation error. Silently altering flagged labels would hide this from downstream training/evaluation without fixing anything. |
| A-20 | Combined noise+blur grid label noise (small, local, non-monotonic VIF steps) is documented as expected metric behavior, not corrected. | `docs/vif_full_grid_pilot_report.md` §10: 41.3% of the full 12x12 combined-grid sequences show at least one local step increase, concentrated at high blur (sigma>=1.4); every violation is small (max +0.0393, none exceeding 0.05) and net direction across the full severity range remains correct in 100% of blur sweeps and 83% of noise sweeps. `vif_wavelet()` is not smoothed, clipped, or post-processed to force monotonicity — doing so would fabricate label structure the metric itself does not produce. |
| A-22 | Dropout probability baseline = 0.5. Resolves U-M01. | Ohashi's paper states only "a Dropout layer was added ... to prevent overfitting" -- no rate; not OHASHI-SPECIFIED. Sourced from Mei X, Liu Z, Robson PM, et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning," *Radiology: Artificial Intelligence* 2022;4(5):e210315 -- the paper that produced the RadImageNet checkpoint this project's ResNet50 backbone is pretrained from. That paper's own transfer-learning head is architecturally identical in shape to Ohashi's (global average pooling -> dropout -> output layer) and states explicitly: "A global average pooling layer, a dropout layer at a rate of 0.5, and the output layer activated by the softmax function were added after the CNNs." This is a REFERENCE-IMPLEMENTATION-CONVENTION value from the checkpoint's own origin paper, adopted here as a single documented PROJECT-ADAPTATION baseline -- not chosen by any hyperparameter search or validation-metric comparison (deliberately out of scope until the baseline architecture is established; any future dropout ablation is a separate experiment). No Ohashi source code, supplementary material, or public repository was found (searched 2026-08-14) to confirm or contradict this value against Ohashi's own actual choice -- it remains unverified relative to Ohashi specifically, only traceable to the checkpoint's own origin. Recorded in `src/ct_iqa/models/resnet50.py::DEFAULT_DROPOUT_PROBABILITY` and `configs/quality/resnet50_vif.yaml:model.dropout_rate`. **Alternatives considered and rejected at this stage:** (a) leaving it `None`/unresolved -- rejected because the audit task requires a committed baseline, and "genuinely unspecified" should not be conflated with "left implicit forever"; (b) a common ImageNet-transfer-learning default of 0.2-0.3 -- rejected in favor of the value from the paper that actually produced this specific checkpoint's own recipe, which is a stronger, traceable justification than a generic convention; (c) running a small dropout sweep now -- explicitly rejected per this decision's own scope (would introduce an experimental degree of freedom before the baseline architecture exists). |
| A-21 | Noise is not clipped to the display range during degradation (`clip_after: false`, `configs/degradation.yaml`); the 8-bit clip only happens at PNG-encode time. | NOT SPECIFIED BY OHASHI whether intermediate clipping occurs. VIF is computed on the full-precision degraded array before any 8-bit quantization, so clipping earlier would silently alter the label; clipping only at the point pixels are actually written as 8-bit PNG keeps the label and the saved image consistent without adding an extra, unrecorded transformation in between. |
| A-15 | Full wavelet-domain VIF is implemented as `vif_wavelet()` (`src/ct_iqa/vif/wavelet.py`), built on `pyrtools` (v1.0.10) for the steerable-pyramid decomposition, structurally following Sheikh & Bovik's released reference algorithm (`vifvec.m` / `refparams_vecgsm.m` / `vifsub_est_M.m`). Existing `vif_p()` (VIFp) is unchanged and kept as a separate, never-aliased variant. This is a PROJECT ADAPTATION, not a claim of reproducing Ohashi's exact MATLAB run: Ohashi's paper does not name a specific MATLAB function/toolbox (see U-V01), so every implementation-level parameter (pyramid orientation count, GSM block size, `σ_nsq`, boundary handling, exact subband selection, numerical stabilizers) had to be sourced from Sheikh's *released* reference code rather than from the Ohashi paper itself — full parameter-by-parameter provenance is in `docs/vif_implementation.md`. **Not yet used for LDCT-IQAC label generation** — see status note below. | S-02 establishes the formulation *family* (wavelet-domain VIF) is OHASHI-SPECIFIED; everything about *how* to compute it in Python is this project's own choice, made because no MATLAB run is available to copy or verify against (`docs/vif_investigation.md`) |

## NOT SPECIFIED BY OHASHI

The dataset generator (`scripts/quality/build_ohashi_dataset.py`) enumerates
U-D01–U-D03 as blockers and refuses `--execute` while any remain.

| ID | Unknown | Why it matters | How to resolve |
| --- | --- | --- | --- |
| U-D01 | ~~Noise sigma units — HU or 8-bit grey levels~~ — **RESOLVED, see A-17** | σ=50 is mild in HU, severe in 8-bit; changes every degraded image | Resolved 2026-08-14 as PROJECT ADAPTATION (A-17): `sigma_scale=1.0`, `sigma_units="8bit_grey_levels"`, recorded in `configs/degradation.yaml`. Still not a claim that this matches Ohashi's own MATLAB run. |
| U-D02 | ~~Combined-degradation order — blur-then-noise or noise-then-blur~~ — **RESOLVED, see A-16** | noise-then-blur partially smooths the noise away; measurably different images (tested in `tests/test_degradation.py`) | Resolved 2026-08-14 as PROJECT ADAPTATION (A-16): `order="blur_then_noise"`, recorded in `configs/degradation.yaml`. Not stated anywhere in the paper text (Methods or Fig. 3 caption) — genuinely still open on Ohashi's side; this project adopts a value rather than leaving it implicit. |
| U-V01 | Exact wavelet-domain VIF implementation matching Ohashi's MATLAB R2024a run: specific MATLAB function/toolbox, decomposition levels/subbands, σ_nsq, boundary handling | the *formulation family* is now resolved (wavelet-domain VIF, not VIFp — see S-02 above and "VIF Implementation Status" below), but these implementation-level parameters still are not, and VIF numerically depends on them | paper only says "MATLAB R2024a" — no function/toolbox named; would need Ohashi's code (not public per the paper's Data Availability statement) or a documented-standard wavelet VIF implementation adopted as a new PROJECT ADAPTATION entry |
| U-M01 | ~~Dropout probability~~ — **RESOLVED, see A-22** | flagged explicitly in the brief; large effect on regularisation | Resolved 2026-08-14 as PROJECT ADAPTATION (A-22): baseline `dropout_probability=0.5`, sourced from RadImageNet's own training recipe (not Ohashi's paper, which remains silent on the rate). Not chosen by any hyperparameter search. |
| U-M03 | Exact grayscale-to-3-channel implementation | affects every input pixel | not stated in the paper (channel replication is used here as A-13-style documented adaptation until confirmed) |
| U-M04 | Resize-before-crop behaviour, interpolation | interacts with the blur degradation | paper states images are "cropped to the central region according to the input size" with no resize step mentioned — suggests direct center-crop from 512×512, but interpolation/resize is not explicitly ruled out |
| U-M05 | RadImageNet preprocessing / intensity normalisation | differs from standard ImageNet preprocessing | not stated in the paper; RadImageNet's own documentation would need to be consulted |
| U-M06 | Adam beta parameters, LR schedule, early stopping, checkpoint selection | reproducibility of the training curve | not stated in the paper beyond optimizer=Adam, batch=64, epochs=30, LR grid {1e-2..1e-5} |
| U-M07 | Split stratification by source dataset | not applicable to a single-source (LDCT-IQAC-only) adaptation, but relevant if DeepLesion/CQ500 are later added back | not stated in the paper |
| U-M08 | RadImageNet ResNet50 checkpoint | cannot build the model at all without it | obtain the RadImageNet release |
| U-S01 | Stage-2 (subjective, artificially-degraded) 210-image composition doesn't fully reconcile with Table 2's "5 noise levels + 5 blur levels × 6 references" (=60) | can't exactly replicate the 210-image evaluation set without knowing what the other ~150 images are (presumably combined noise+blur conditions, counts unstated) | not stated in the paper; Table 2 and the "210 artificially degraded images" figure are both given but not reconciled in the text |

## VIF Implementation Status — Known Mismatch (not yet corrected)

This section exists to make the VIF question impossible to skim past. It is
required reading before anyone touches `src/ct_iqa/vif/metric.py`.

**What the Ohashi citation specifies:** Ohashi et al. cite Sheikh & Bovik
(2006), "Image information and visual quality," IEEE Trans Image Process
15(2):430-444 [ref 27]. This is the original **VIF** paper: it models
wavelet-subband coefficients of both reference and distorted images as a
Gaussian Scale Mixture (GSM), fits a per-subband channel-distortion model,
and combines mutual information across wavelet scales/subbands/orientations
to produce the final score.

**What our current implementation does:** `src/ct_iqa/vif/metric.py`
implements **VIFp** (`vifp_mscale`, per decision A-13) — a simplified,
**pixel/spatial-domain** variant from an earlier (2005) Sheikh & Bovik
paper. VIFp skips the wavelet decomposition and GSM subband modeling
entirely in favor of a spatial-domain multi-scale Gaussian-pyramid
computation.

**Why they should not currently be considered equivalent:** VIFp and
wavelet-domain VIF are two distinct, separately-published formulations that
are known to diverge numerically — VIFp is markedly less sensitive to blur
than full wavelet-domain VIF, and the two handle noise differently because
wavelet-domain VIF estimates noise variance per subband rather than
globally in the pixel domain. This is exactly the noise-vs-blur asymmetry
the paper itself reports in Table 5 ("all models tended to demonstrate
lower correlation coefficients for noise than for blur") — a result that is
partly an artifact of *which* VIF formulation produced the labels. Because
VIF is the sole synthetic-stage training target, this is not a cosmetic
naming difference: it would change the value of every one of the 169,000
labels this project would generate, and therefore the score the model
learns to predict. For that reason VIFp must not be informally referred to
as "VIF," relabeled, or treated as a drop-in equivalent anywhere in this
codebase or its docs.

**What remains to be verified about Ohashi's actual MATLAB R2024a run:**

- Which specific MATLAB function or toolbox was called — the paper names
  only "MATLAB R2024a," not a function, toolbox, or file.
- Number of wavelet decomposition levels/subbands used.
- The σ_nsq (visual noise) parameter value.
- Any 8-bit-specific handling (block size, boundary/edge treatment) that
  would affect a 512×512 or center-cropped input.
- Whether Ohashi's own earlier work (ref [28], Ohashi et al. 2023,
  "Applicability evaluation of full-reference image quality assessment
  methods for computed tomography images," J Digit Imaging 36:2623-2634)
  specifies more implementation detail — it is cited as the source of the
  DMOS data reused in this paper's Stage 2 evaluation, and may also document
  which VIF implementation their group has standardized on. Not yet
  checked; would need to be read directly.

**Status (updated 2026-08-13, after parameter-equivalence validation):**
`src/ct_iqa/vif/metric.py` (VIFp) was never modified. `vif_wavelet()`
(`src/ct_iqa/vif/wavelet.py`) now exists under decision A-15, built on
`pyrtools` and structurally following Sheikh's released reference
implementation. It passes 22 property-based tests
(`tests/test_vif_wavelet.py`: identity, noise/blur monotonicity,
determinism, shape handling, degenerate-input rejection).

A first cross-check against a second, independently-authored full
wavelet-domain VIF implementation (`scripts/validate_vif.py`) showed only
moderate numerical correlation (Pearson ≈ 0.62) between our canonical
("project") configuration and that implementation. Rather than stop there,
a **parameter-equivalence validation** was run
(`scripts/validate_vif_parameter_equivalence.py`) to determine whether that
moderate disagreement was caused by an implementation error or by
documented parameter differences. Every tunable parameter was itemized
(`σ_nsq`, orientation/subband selection, window-size-to-level mapping,
aggregation method, numerical stabilizer, eigenvalue-floor regularisation —
full table in `docs/vif_implementation.md` §13) and a second, explicit
`"reference_crosscheck"` profile was built that matches the independent
implementation's choices as closely as technically possible, without
altering the canonical `"project"` profile in any way.

**Result: once parameters are matched, the two implementations agree to
within floating-point precision** — Pearson r = 0.99999999999989, Spearman
r = 1.0, MAE = 4.6×10⁻⁷, max absolute difference = 9.6×10⁻⁷, across 15
finite test cases (natural-like/CT-like/high-texture images × identity /
noise / blur). This is strong evidence that `vif_wavelet()`'s core
algorithm — pyramid decomposition, GSM covariance/eigenvalue estimation,
linear distortion-channel fitting, mutual-information aggregation — is
implemented correctly, and that the earlier moderate disagreement was
fully attributable to the two implementations' independently-made,
individually-defensible parameter choices, not to a bug in either one.
One caveat on this conclusion: the specific `lev_mode="reversed_pair"`
setting needed to match the independent implementation reproduces what
appears to be an inversion in *that* implementation's own window-size
bookkeeping (it assigns the largest correlation window to the *finest*
pyramid level and the smallest to the *coarsest*, backwards from
`vifvec.m`'s own formula) — meaning the near-perfect agreement confirms
our code can reproduce *that* implementation's specific behaviour
precisely, which is exactly what a parameter-equivalence test needs to
show, but does not by itself vouch for that behaviour being correct
relative to Sheikh & Bovik's original algorithm. Full numbers, the
parameter-difference table, and the per-image comparison are in
`docs/vif_implementation.md` §13.

A second finding from this same validation pass: **the earlier
"smooth-image" instability was traced to a specific stage, not left as an
unexplained bad number.** Diagnostic instrumentation
(`vif_wavelet_diagnostics()`) shows two compounding causes on a
near-flat/low-complexity test image: (1) the GSM block-covariance matrix
is severely ill-conditioned (condition numbers ~10¹⁶–10¹⁸) at every
pyramid level for this image, an intrinsic property of a smooth image's
near-rank-deficient local block statistics, not a numerical accident; and
(2) at the finest pyramid levels specifically, near-zero local reference
variance in the distortion-channel estimation causes the fitted channel
gain `g` to explode (observed up to ~2212), which inflates the
numerator (`g²·s·λ`) far more than the denominator (`s·λ`, no `g`
dependence), producing VIF ≫ 1. Both this project's implementation and
the independent implementation are unstable on this same test image
(ours: large finite value; theirs: `NaN`) — a shared property of the
algorithm family on this class of input, not a defect unique to either
port. Ohashi's own reference images are structured CT anatomy, not
synthetic smooth gradients, making this an unlikely but **unconfirmed**
risk for the actual LDCT-IQAC dataset.

This is judged sufficient to consider Route A (§1–§9 above) validated as a
**structurally correct implementation** of the wavelet-domain VIF
algorithm family, with the specific known failure mode above now
understood rather than merely observed. It is explicitly **not** judged
sufficient to generate the 169,000 LDCT-IQAC training labels yet — doing
so is a large, hard-to-reverse action, and no implementation of this
metric family (ours or the independent one) has ever been checked against
a MATLAB ground-truth run, because none exists in this project's reach.
Label generation requires a separate, explicit go-ahead after this
validation state has been reviewed — it does not follow automatically from
"the tests pass" or from "the cross-check now agrees." Former U-V01
remains split as before: the formulation family is OHASHI-SPECIFIED
(S-02); the implementation-level parameters are PROJECT-ADAPTATION (A-15,
itemized in `docs/vif_implementation.md`), not OHASHI-SPECIFIED and not
bit-exact-verified against MATLAB.

### VIF STATUS

```
Algorithm:
    IMPLEMENTED

Formulation:
    FULL WAVELET/GSM VIF

Reference implementation:
    TRACED TO SHEIKH/BOVIK VIF REFERENCE CODE (vifvec.m, refparams_vecgsm.m,
    vifsub_est_M.m -- LIVE lab, UT Austin)

Unit/property tests:
    PASS (22/22, tests/test_vif_wavelet.py)

Independent implementation cross-check (default parameters):
    PARTIAL PASS (Pearson 0.62, Spearman 0.56 -- correct direction,
    moderate numerical agreement)

Parameter-equivalence validation:
    PASS (Pearson 0.99999999999989, Spearman 1.0, once parameters matched
    -- confirms implementation is structurally correct; prior disagreement
    was parameterization-related, not a bug)

Numerical equivalence (project profile vs any external ground truth):
    NOT ESTABLISHED

MATLAB R2024a equivalence:
    NOT ESTABLISHED (no MATLAB installation available in this project;
    Ohashi's paper names no specific function/toolbox to target)

Known failure mode:
    DIAGNOSED (low-texture/near-flat images destabilise the GSM covariance
    and channel-gain estimation stages -- shared with the independent
    implementation, not unique to this port; unconfirmed relevance to
    actual LDCT-IQAC images)

Dataset label generation:
    READY WITH DOCUMENTED CONDITIONS (per docs/vif_full_grid_pilot_report.md
    Section 17, "B"), production pipeline preparation authorized -- see
    "Production Generation Decision" below. Full 1,000-reference /
    169,000-image generation itself remains a SEPARATE, EXPLICIT
    authorization gate (scripts/preflight_generation.py must pass and a
    human must issue the go-ahead) -- it does not follow automatically from
    this status.
```

## Production Generation Decision (2026-08-14)

Following the 100-reference full-grid pilot (`docs/vif_full_grid_pilot_report.md`),
the pilot's recommendation **B — READY WITH SPECIFIC DOCUMENTED CONDITIONS** is
accepted. The specific conditions from that report's §17 are addressed as follows:

1. Flag-and-retain handling for `UNSTABLE_CHANNEL_GAIN` (and all diagnostic
   flags) is adopted as decision A-19 above.
2. Combined-grid label noise is documented as expected metric behavior, not a
   defect — decision A-20 above.
3. U-D01 and U-D02 are resolved as PROJECT ADAPTATION — decisions A-17 and
   A-16 above — and recorded in `configs/degradation.yaml`.
4. Checkpointing/resumability is added to the production generator
   (`scripts/quality/generate_production_dataset.py`) — reference-level
   checkpoints, streamed manifest writes, no full in-memory record list. See
   that script's module docstring for the design.

**This authorizes preparing the production pipeline (config, generator,
checkpointing, preflight, dry-run) — it does NOT authorize starting the full
1,000-reference / 169,000-image generation run.** That is a separate,
explicit go-ahead, gated on `scripts/preflight_generation.py` passing and a
human decision to proceed, per the instruction that produced this section.

## Baseline Model Specification Decision (2026-08-14)

Following `docs/radimagenet_environment_audit.md` (the RadImageNet
checkpoint/environment feasibility audit), two remaining architecture-spec
items are resolved:

1. **`ModelSpec.freeze_backbone` inconsistency fixed.** S-01 already
   resolved, at the decision-record level, that the backbone is
   fine-tuned, not frozen — but `src/ct_iqa/models/resnet50.py`'s
   `ModelSpec.freeze_backbone` still defaulted to `None`, contradicting
   that resolution in code. Fixed: `ModelSpec.freeze_backbone` now
   defaults to `False` (fine-tuned), matching S-01 exactly. No new
   interpretation was introduced — `False` is the direct, literal encoding
   of "not frozen." `tests/test_models_training_specs.py::test_default_spec_backbone_is_fine_tuned_not_frozen`
   guards against this regressing back to `None` or flipping to `True`.
2. **Dropout probability resolved as decision A-22** (see the PROJECT
   ADAPTATION table above) — baseline `0.5`, sourced from RadImageNet's own
   training recipe, not Ohashi's paper.

**This resolves the model *specification* only.** `build_model()` still
raises unconditionally (decision A-14) — the only remaining blockers are
the RadImageNet checkpoint itself (U-M08) and the absence of an installed,
compatible deep-learning framework (`docs/radimagenet_environment_audit.md`).
Neither downloading the checkpoint, installing TensorFlow, provisioning a
new environment, generating the production dataset, nor training is
authorized by this decision.

## EXPERIMENTAL EXTENSION

None yet. This baseline intentionally stays within the Ohashi + LDCT-IQAC
adaptation described above; extensions (additional backbones, alternative
calibration methods, augmentation) are out of scope until the baseline
reproduces.
