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

## DEV-03 — VIF implementation is the wrong formulation family (confirmed mismatch)

**What differs:** the paper's citation for VIF — Sheikh & Bovik (2006),
"Image information and visual quality," IEEE Trans Image Process
15(2):430-444 [ref 27] — is the original **wavelet-domain VIF** (GSM model
over wavelet subbands). `src/ct_iqa/vif/metric.py` implements **VIFp**
(`vifp_mscale`, decision A-13), a different, simplified **pixel-domain**
formulation from an earlier (2005) Sheikh/Bovik paper. This was previously
recorded as an unresolved MATLAB-vs-Python porting question ("read the
paper"); having now read the paper, it is a formulation-family mismatch, not
a floating-point/toolchain nuance.

**Consequence:** VIFp and wavelet-domain VIF are known to diverge
numerically, particularly on blur sensitivity and noise handling (wavelet
VIF estimates noise variance per subband; VIFp does not). Since VIF is the
sole synthetic-stage training target, every one of the 169,000 labels this
project would generate depends on which formulation is used. This is graded
above ordinary floating-point drift — it is a different metric, not a
different implementation of the same metric. Full writeup, including what
remains unverified about Ohashi's exact MATLAB call, is in
`docs/research_decisions.md` → "VIF Implementation Status."

**Status (updated 2026-08-13, after parameter-equivalence validation):**
partially addressed. `src/ct_iqa/vif/metric.py` (VIFp) remains unmodified
and unrenamed. A full wavelet-domain VIF implementation, `vif_wavelet()`
(`src/ct_iqa/vif/wavelet.py`, decision A-15), now exists alongside it as a
distinct, explicitly-named variant
(`ct_iqa.vif.labeling.compute_vif(..., variant="vif_wavelet")`) — see
`docs/vif_implementation.md` for its full specification and
`docs/research_decisions.md` (decision A-15, "VIF STATUS" block) for its
validation status.

The initial cross-check against an independent implementation showed only
moderate numerical agreement (Pearson 0.62). A follow-up
parameter-equivalence validation (`scripts/validate_vif_parameter_equivalence.py`)
showed that once every identified parameter difference is matched, the two
implementations agree to within floating-point precision (Pearson
0.99999999999989) — strong evidence the moderate disagreement was caused
by documented, defensible parameter choices, not an implementation bug.

**Still blocking for actual LDCT-IQAC label generation**: this validates
that the *code* is structurally correct, not that the *canonical parameter
set* (drawn from `vifvec.m`, since Ohashi names no specific MATLAB
function) matches whatever Ohashi's own MATLAB R2024a run actually
computed — no MATLAB ground truth exists anywhere in this project's reach,
and none is expected to. No LDCT-IQAC labels have been computed with
either variant. See `docs/research_decisions.md`, items S-02 (formulation
family, OHASHI-SPECIFIED) and A-15/U-V01 (implementation, PROJECT-ADAPTATION,
structurally validated, not bit-exact-verified against MATLAB).

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
