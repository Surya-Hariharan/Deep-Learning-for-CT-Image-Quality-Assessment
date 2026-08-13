# Deviations From Ohashi

Only genuine deviations are listed here — cases where this project's actual
behaviour differs from `docs/ohashi_methodology.md`. Unresolved
parameters that Ohashi simply doesn't specify are NOT deviations; those are
tracked in `docs/research_decisions.md` under NOT SPECIFIED BY OHASHI, and must
not be pre-empted by a value here.

## DEV-01 — Reference-image source and count

**What differs:** Ohashi's 105 references (90 DeepLesion + 15 CQ500) are
replaced with the 1,000-image LDCT-IQAC dataset.

**Consequence:** the synthetic dataset total scales from 17,745 to 169,000
images; the 60/20/20 reference split scales from 63/21/21 to 600/200/200
references (101,400/33,800/33,800 images). All other degradation and split
mechanics are unchanged — see `docs/dataset_adaptation.md` for the full table.

**Status:** intentional, user-directed. Not a limitation to fix.

## DEV-02 — Reference images are not guaranteed pristine

**What differs:** Ohashi's DeepLesion/CQ500 references are treated as clean
CT slices. LDCT-IQAC images already contain sparse-view streak artifacts and
low-dose noise from their own acquisition protocol — that is the dataset's
entire premise (see `data/raw/ldctiqa/SOURCE_README.md`).

**Consequence:** every VIF label in this project measures information loss
*relative to* an LDCT-IQAC image, not relative to a pristine acquisition. The
absolute VIF values are therefore not directly comparable to Ohashi's reported
numbers; PLCC/SROCC (rank- and correlation-based) remain the meaningful
comparison.

**Status:** documented, not corrected. Correcting it would require a
genuinely pristine reference dataset (e.g., the original DeepLesion/CQ500),
which is not currently available in this repository.

## DEV-03 — VIF implementation is a from-scratch Python port

**What differs:** Ohashi's paper used MATLAB R2024a for VIF (Section 12 of the
brief). `src/ct_iqa/vif/metric.py` is a from-scratch Python implementation of the
pixel-domain VIFp variant (Sheikh & Bovik's `vifp_mscale` reference), not a
call into the MATLAB toolchain.

**Consequence:** small numerical differences versus a MATLAB run are possible
even with an identical formula, from floating-point and filter-boundary
implementation details. **Not yet cross-validated** against a MATLAB reference
run or a known-good Python package (e.g. `sewar`, `piqa`) on a shared test
image — this is an open validation task, not a resolved one.

**Status:** open. See `docs/research_decisions.md`, item U-V01.

## DEV-04 — Real-image evaluation set differs from Ohashi's

**What differs:** Ohashi's Stage 3 (real clinical-image evaluation) used
DeepLesion/CQ500 real images. This project's Stage 3 reuses the same
LDCT-IQAC references (unmodified) against their own `expert_score`, since a
separate real-clinical evaluation set is not available.

**Consequence:** Stage 2 (subjective evaluation) and Stage 3 (real-image
evaluation) draw from the same underlying images in this project, whereas
Ohashi's original design used distinct datasets for each. This narrows what
Stage 3 can demonstrate about generalisation beyond the training distribution.

**Status:** documented limitation, not corrected. See
`configs/quality/resnet50_vif.yaml:evaluation.stage_3_real_image`.

## Non-deviations (recorded here to prevent accidental "fixing")

- Sigma grids, condition families, split protocol structure, model
  architecture, training hyperparameters, and calibration form are all
  unchanged from `docs/ohashi_methodology.md`. Do not alter these to
  make the LDCT-IQAC numbers "look nicer."
- `expert_score` is not a deviation-in-waiting to be merged with `vif_score`.
  Keeping them separate is the point (see `docs/dataset_adaptation.md`).
