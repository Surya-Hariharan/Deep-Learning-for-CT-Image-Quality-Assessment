# VIF Methodology Investigation

Dedicated investigation into the VIF-implementation question flagged in
`docs/research_decisions.md` ("VIF Implementation Status") and
`docs/deviations_from_ohashi.md` (DEV-03). This document is the single
source of truth for that question. It changes no code: `src/ct_iqa/vif/metric.py`
is unmodified, no VIF labels have been computed, and no synthetic dataset
has been generated.

Investigation date: 2026-08-13.

---

## 1. Ohashi's VIF reference — exactly what the paper says

From the Labeling subsection (Materials and Methods):

> "The 17,745 degraded images were labeled with quality scores using the
> visual information fidelity (VIF) [27] metric, an FR-IQA method that
> quantifies the mutual information shared between the reference and
> degraded images. VIF was selected due to its high correlation with
> subjective CT image quality assessments [28]. The VIF scores were
> calculated using MATLAB R2024a (The MathWorks, Inc., Natick, MA, USA)."

Reference [27] in the bibliography:

> "Sheikh HR, Bovik AC: Image information and visual quality. IEEE Trans
> Image Process 15(2):430-444, 2006."

Reference [28] (cited only as evidence VIF correlates with subjective CT
assessment, not as a second implementation source, but worth noting since
it's the same author group and reuses data from this very paper — see
Acknowledgements: *"the subjective assessment data ... used in this study
were also utilized in our previous research [28] to investigate
correlations with nine full-reference image quality assessment methods"*):

> "Ohashi K, Nagatani Y, Yoshigoe M, Iwai K, Tsuchiya K, Hino A, Kida Y,
> Yamazaki A, Ishida T: Applicability evaluation of full-reference image
> quality assessment methods for computed tomography images. J Digit
> Imaging 36:2623-2634, 2023."

No other detail is given anywhere in the paper: no MATLAB function name, no
toolbox, no parameter values, no mention of "VIFp" as a distinct or reduced
variant. "VIF" is used as if it names one specific, unambiguous metric.

## 2. The Sheikh & Bovik (2006) formulation, and why it is not one metric

This is the single most important technical fact this investigation
surfaces: **Sheikh & Bovik did not publish one VIF implementation — they
published two, under the same research program, that are commonly confused:**

| | Full VIF (the 2006 IEEE TIP paper, ref [27]) | VIFp (pixel-domain reduction) |
| --- | --- | --- |
| Domain | Wavelet/steerable-pyramid subband coefficients | Raw pixel intensities |
| Decomposition | Steerable pyramid (typically via Simoncelli's `matlabPyrTools`, `buildSpyr`), multiple orientation bands per scale | Gaussian low-pass pyramid, no orientation bands |
| Statistical model | Each subband modeled as a Gaussian Scale Mixture (GSM): coefficients = scalar field × Gaussian vector, capturing heavy-tailed natural-image statistics per orientation/scale | Local image patches modeled directly as correlated Gaussian random fields in the pixel domain — no GSM, no orientation decomposition |
| Distortion (channel) model | Per-subband, per-orientation linear-plus-additive-noise model between reference and distorted subband coefficients | Single per-scale linear-plus-additive-noise model between reference and distorted pixel neighborhoods |
| Output | Ratio of mutual information (distorted vs. reference) integrated over all subbands, scales, and orientations | Ratio of mutual information integrated over 4 isotropic Gaussian-pyramid scales only |
| Canonical MATLAB reference name | `vifvec.m` (sometimes bundled as `vif.m`) — publicly released by Sheikh alongside the paper | `vifp_mscale.m` — released as a simplified companion function on the same page |
| Reported sensitivity | More sensitive to blur (captures orientation-specific frequency loss); the metric actually cited in the 2006 paper's headline results | Markedly less sensitive to blur; faster, simpler, widely used as a "good enough" approximation in later IQA literature |

**The IEEE TIP 15(2):430-444, 2006 paper title is exactly "Image
information and visual quality" — this is the full, wavelet/GSM
formulation.** VIFp was introduced as a lower-complexity companion in the
same released code bundle but was not the subject of a dedicated
peer-reviewed publication with that title; papers that cite "VIF [Sheikh &
Bovik 2006]" are, by the citation itself, pointing at the full
steerable-pyramid/GSM metric unless they explicitly say "pixel-domain VIF"
or "VIFp." Ohashi's paper does neither — it says "VIF," cites only the 2006
paper, and never uses the word "pixel."

## 3. Current implementation — what `src/ct_iqa/vif/metric.py` actually does

Read directly from the source (`vif_p()`):

- Operates purely on 2-D pixel arrays; no wavelet or pyramid *orientation*
  decomposition — only an isotropic Gaussian blur/downsample pyramid.
- 4 scales (`range(1, 5)`), Gaussian window side `2**(4-scale+1)+1`,
  standard deviation `side/5`.
- At each scale after the first, both images are low-pass filtered and
  downsampled by 2 (`[::2, ::2]`) — a standard Gaussian pyramid, not a
  steerable pyramid, and no orientation subbands are produced at any scale.
- Local statistics (`mu`, `sigma1_sq`, `sigma2_sq`, `sigma12`) are computed
  by Gaussian-weighted local moments directly on pixel intensities.
- `g` (channel gain) and `sv_sq` (residual noise variance) are computed
  per-pixel from these moments — a per-pixel/per-scale channel model, not a
  per-subband GSM model.
- `sigma_nsq = 2.0` (module constant `SIGMA_NSQ`), matching the constant
  used in the commonly-circulated `vifp_mscale.m` reference port.
- Final score: `sum(log10(1 + g^2 * sigma1_sq/(sv_sq+sigma_nsq))) / sum(log10(1 + sigma1_sq/sigma_nsq))`,
  summed over the 4 scales — the VIFp aggregation formula.

This is a faithful, careful port of `vifp_mscale.m` — the docstring already
says as much. It is **not** an implementation of the full wavelet/GSM VIF
from the cited 2006 paper: there is no steerable pyramid, no orientation
bands, and no GSM parameter estimation anywhere in this file.

## 4. Side-by-side comparison

| Property | Ohashi-cited VIF (ref [27], full) | Current `vif_p()` (VIFp) |
| --- | --- | --- |
| Decomposition | Steerable pyramid, multiple orientations/scale | Gaussian pyramid, isotropic, no orientations |
| Statistical model | GSM per subband | Local Gaussian moments per pixel neighborhood |
| Scales | Typically 4 (paper-dependent) | 4 (hardcoded) |
| Orientation bands | Yes (several per scale, steerable-pyramid dependent) | None |
| sigma_nsq | Not documented per-metric in general practice (varies by MATLAB release/toolbox default, commonly ~0.4 in `vifvec.m`) | 2.0 (VIFp convention) |
| Known behavior | More blur-sensitive | Less blur-sensitive |
| Public MATLAB name | `vifvec.m` / `vif.m` | `vifp_mscale.m` |
| Status in this project | Cited target, not implemented | Implemented, mismatched |

The practical consequence, already noted in `docs/research_decisions.md`:
Ohashi's own Table 5 shows every model scoring lower on noise-only than
blur-only degradation. If the training labels were in fact generated by the
full wavelet VIF (more blur-sensitive, orientation-aware), that asymmetry
is partly a property of the *label generator*, not purely a property of the
learned models — a detail this project cannot currently reproduce with
VIFp, whose blur/noise sensitivity profile is different by construction.

## 5. Search of available project dependencies for a wavelet-domain VIF

Checked the active Python environment (`C:\Users\surya\anaconda3`, the
interpreter this repo currently develops against per `requirements.txt`):

| Package | Status | Relevance |
| --- | --- | --- |
| `pywt` (PyWavelets) | **Installed**, v1.8.0 | Provides generic discrete wavelet transforms (DWT/SWT). Does **not** provide a steerable pyramid — the specific, orientation-selective decomposition the original VIF reference code (`vifvec.m`) is built on. A DWT-based VIF would be a *different* approximation again, not a faithful reproduction of the cited paper. |
| `scikit-image` | Installed, v0.26.0 | No VIF implementation of any kind (checked: no `vif`/`VIF` symbol in its `metrics` module). |
| `opencv-python` (`cv2`) | Installed, v5.0.0 | No VIF implementation. |
| `sewar` | **Not installed** | Third-party FR-IQA package; implements `vifp` (pixel-domain) only — same formulation family as our current code, would not resolve the mismatch. |
| `piq` / `piqa` / `pyiqa` | **Not installed** | PyTorch-based IQA libraries; some (`pyiqa`, `piq`) advertise a "VIF" metric, but their implementations need direct source inspection before trusting them as the full steerable-pyramid/GSM formulation — several open-source "VIF" ports in the wild are actually VIFp under a shortened name. Not verified in this investigation; flagged as a candidate requiring code-level audit, not adoption.
| `pyrtools` | **Not installed**, not currently a project dependency | The actively-maintained Python port of Simoncelli's `matlabPyrTools` (Laboratory for Computational Vision, NYU) — the same pyramid toolbox the original MATLAB VIF reference code depends on. This is the standard route to a *faithful* Python re-implementation of the cited formulation, not a pre-built VIF metric itself. |
| MATLAB / MATLAB Engine API for Python | Not installed, no MATLAB license currently established in this environment | Would allow calling Sheikh's original `vifvec.m` (or a modern reimplementation) directly if the reference code and a MATLAB license were obtained — the maximally faithful route, at the cost of reintroducing a MATLAB dependency this project has otherwise avoided. |

No package currently importable in this project's environment implements
the full wavelet/GSM VIF formulation. Nothing has been installed as part of
this investigation, per instruction.

## 6. Candidate implementation routes

### Route A — Re-implement full VIF from the published equations, using `pyrtools` for the steerable pyramid

- **What it is:** write `vif_wavelet()` from scratch, following Sheikh &
  Bovik (2006) equations 1–14 directly, using `pyrtools.pyramids.SteerablePyramidSpace`
  (or equivalent) for the decomposition and implementing the GSM parameter
  estimation (regression for `g`, `sigma_v^2` per subband) and the final
  mutual-information ratio ourselves.
- **Advantages:** most faithful available route without MATLAB; auditable,
  version-controlled, testable against known VIF properties (monotonic
  decrease with blur/noise, invariance under identical images, etc.);
  consistent with this project's existing practice of implementing metrics
  from scratch against documented formulas (as VIFp already is).
- **Disadvantages:** highest implementation risk — GSM parameter
  estimation and steerable-pyramid boundary handling have historically been
  a source of subtle bugs in independent VIF ports; no ground-truth MATLAB
  output available in this repository to validate against; `pyrtools` is a
  new dependency requiring installation and Windows-compatibility
  verification (not yet checked in this environment).
- **Reproducibility:** high once validated, but "once validated" is doing
  real work here — needs a held-out test image with either a published
  reference VIF score or a cross-check against a second independent
  implementation.

### Route B — Adopt a third-party Python "VIF" implementation (`pyiqa`, `piq`, or similar) after auditing its source

- **Advantages:** less implementation work if a package genuinely
  implements the full formulation; maintained by a broader community; some
  of these libraries (`pyiqa` in particular) are widely used in recent IQA
  literature and may already be validated against MATLAB reference outputs
  by their authors.
- **Disadvantages:** unverified in this investigation whether any given
  package's "VIF" is full VIF or a relabeled VIFp — this is a known,
  common point of confusion in the wider ecosystem, which is precisely the
  failure mode this investigation exists to avoid repeating. Introduces a
  dependency whose internals this project does not control or fully audit
  by default. GPU/PyTorch-based implementations (`piq`, `pyiqa`) may
  compute in a different numeric path (float32, GPU nondeterminism) than a
  CPU/MATLAB baseline, adding a further reproducibility variable.
- **Reproducibility:** contingent entirely on how well the specific
  package's implementation has itself been validated — not yet established
  by this investigation.

### Route C — Call the original MATLAB reference code via a MATLAB Engine or a one-time offline export

- **Advantages:** maximal fidelity — this would be the literal code family
  the cited paper is built on (`vifvec.m`), if a copy of the reference
  implementation can be sourced (Sheikh's LIVE lab has historically
  distributed it) and a MATLAB license obtained.
- **Disadvantages:** reintroduces the MATLAB dependency this project has
  otherwise deliberately avoided (see `requirements.txt` header); requires
  a MATLAB license not currently available in this environment; poor fit
  for generating 169,000 labels as part of a reproducible Python pipeline;
  Ohashi's own Data Availability statement says their *code* is "currently
  not publicly available," so even this route would rely on a third-party
  MATLAB VIF release, not Ohashi's exact script.
- **Reproducibility:** high fidelity to the *algorithm family*, poor fit
  for this project's tooling and licensing constraints.

### Route D — Keep VIFp, but reclassify it explicitly as a documented project substitution rather than a silent bug

- **Advantages:** zero new dependencies, zero new implementation risk,
  keeps the fast, already-tested, already-integrated metric.
- **Disadvantages:** does not resolve the underlying fidelity question;
  every training label would carry a known, documented divergence from
  what the cited paper actually used, specifically in blur sensitivity —
  the exact axis Ohashi's own Table 5 highlights as behaviorally important.
- **Reproducibility:** internally reproducible (deterministic, tested), but
  not faithful to the cited source — this is a "faithful to *a* metric,
  not faithful to *the* metric" outcome, and would need to be treated as a
  first-class, permanent deviation in `docs/deviations_from_ohashi.md`
  rather than a temporary placeholder.

## 7. Numerical / reproducibility considerations that apply regardless of route

- Whichever route is chosen, this project has **no MATLAB R2024a output on
  any of our images** to validate against — Ohashi's code and raw VIF
  outputs are not published. Validation will necessarily be indirect: known
  VIF properties (monotonicity, identical-image ⇒ 1.0, degenerate-region
  handling), cross-checks between two independently-sourced
  implementations, and/or comparison to published VIF behavior on standard
  IQA benchmark images (e.g., LIVE database) where literature values exist.
- GSM parameter estimation in full VIF involves local regression that can
  be numerically unstable in flat or low-variance image regions — exactly
  the kind of region the current VIFp code already special-cases (see the
  `sigma1_sq < 1e-10` branches in `vif_p()`). Any full-VIF implementation
  will need equivalent degenerate-region handling, and the reference
  formulation does not spell this out precisely either — expect this to be
  its own documented implementation choice, not a settled fact.
- Whatever is chosen must be applied identically and deterministically to
  all 169,000 planned images; a GPU-based implementation (Route B, some
  packages) risks introducing run-to-run nondeterminism that a from-scratch
  NumPy/SciPy implementation (consistent with the current VIFp code's
  style) would not.

## 8. Classification

**CONFIRMED FROM OHASHI**
- The paper cites Sheikh & Bovik (2006), "Image information and visual
  quality," IEEE TIP 15(2):430-444, as the source of their VIF metric.
- VIF (whichever exact form) is computed against each degraded image's own
  clean reference and used as the sole synthetic-stage training label.
- Computation was done in MATLAB R2024a. No function/toolbox name given.

**STRONGLY SUPPORTED BY CITATION (not stated outright, but the citation is
specific enough to make the alternative implausible)**
- The cited paper is the full wavelet/steerable-pyramid/GSM VIF paper, not
  a paper describing VIFp. Papers that mean VIFp conventionally cite it as
  such or reference a different, later source; Ohashi's paper does neither.
- Therefore the *intended* metric family is full VIF, not VIFp — this is
  now recorded as decision S-02 in `docs/research_decisions.md`.

**IMPLEMENTATION CHOICE (project must decide; paper does not determine
this)**
- Which of Routes A–D to take.
- If Route A: exact steerable-pyramid parameters (number of orientation
  bands, filter family), GSM estimation method, degenerate-region handling,
  numerical stabilizing constants.
- If Route B: which package, and the outcome of auditing its source against
  the full-VIF formulation before trusting it.
- sigma_nsq and any other constants not fixed by the paper itself.

**STILL UNKNOWN (genuinely unresolved, not this project's to decide)**
- Ohashi's exact MATLAB function/toolbox name and version.
- Their exact parameter values (steerable pyramid orientation count,
  sigma_nsq, etc.), if they differ from whatever public reference
  implementation they built on.
- Whether they used the original Sheikh-released MATLAB code unmodified or
  a locally adapted version — their own ref [28] (Ohashi et al. 2023) might
  contain more detail but has not been read as part of this investigation.

---

## Recommendation

**USE THIS VIF IMPLEMENTATION: Route A — a from-scratch `vif_wavelet()`
implementation of the full steerable-pyramid/GSM VIF, built on `pyrtools`
for the pyramid decomposition, following Sheikh & Bovik (2006) directly,
validated against known VIF properties and (if obtainable) a second
independent implementation before it is trusted for label generation.**

Why this is the most defensible choice for this project, not just the most
"complete" one:

1. **It targets the metric actually cited**, not a same-family
   approximation. The citation evidence in Section 2 makes VIFp a poor
   match for what Ohashi's paper says it did; adopting VIFp permanently
   (Route D) would mean every training label in this replication measures
   something the source paper did not use.
2. **It stays inside this project's existing engineering discipline.**
   `vif_p()` is already a documented, from-scratch, formula-faithful,
   tested implementation with no opaque third-party dependency — Route A
   extends that same standard to the full formulation rather than trading
   it for an unaudited external package (Route B) or a licensing
   dependency this project has deliberately avoided (Route C, MATLAB).
3. **It is auditable and reproducible in the way this project already
   requires.** A from-scratch NumPy/SciPy/`pyrtools` implementation runs
   deterministically on CPU, matching the reproducibility bar the current
   VIFp code and the seeded-RNG degradation pipeline (`A-05`, `A-06`) are
   built around — unlike GPU-based third-party libraries, which risk
   nondeterminism across runs.
4. **It does not foreclose Route B as a cross-check.** Once a candidate
   Python package's source has been audited and confirmed to genuinely
   implement full VIF (not a relabeled VIFp), it becomes a useful
   *validation* target for the from-scratch implementation rather than a
   replacement for it — cheaper confidence than trying to source MATLAB
   and a license for Route C.

This recommendation is **not yet actioned**. Before any implementation
work begins, this choice should be recorded as a new PROJECT ADAPTATION
entry in `docs/research_decisions.md` (with an ID, e.g. `A-15`), since it
is a genuine project decision the paper does not make for us — the paper
tells us *which metric family* (Section 2 of this document), not *how to
implement it in Python*. `src/ct_iqa/vif/metric.py` remains unmodified
until that decision is formally recorded and a validation plan (Section 7)
is agreed.

---

## Addendum (2026-08-13): Route A implemented and validated — see follow-on documents

This recommendation was accepted and actioned as decision **A-15**
(`docs/research_decisions.md`). The implementation itself, its exact
parameters, and the results of the cross-check proposed in Section 7 above
are documented separately rather than folded back into this file:

- **`docs/vif_implementation.md`** — the practical specification for
  `vif_wavelet()`: exact pyramid/GSM/channel-model parameters, each
  classified as OHASHI-SPECIFIED / REFERENCE-IMPLEMENTATION-CONVENTION /
  PROJECT-ADAPTATION / UNKNOWN, plus known numerical limitations
  discovered during validation (notably: both this implementation and an
  independently-sourced reference implementation become unstable on
  near-flat, low-texture images — a shared property of the algorithm
  family, not a bug in either port).
- **`src/ct_iqa/vif/wavelet.py`** — the implementation, built on
  `pyrtools` (steerable pyramid decomposition) and Sheikh & Bovik's
  released MATLAB reference code (`vifvec.m` / `refparams_vecgsm.m` /
  `vifsub_est_M.m`, mirrored at
  https://github.com/sattarab/image-quality-tools) as the structural
  template.
- **`tests/test_vif_wavelet.py`** and **`scripts/validate_vif.py`** — the
  property-test suite and the cross-check against a second, independent
  full wavelet-domain VIF implementation
  (https://github.com/abhinaukumar/vif), respectively.

VIFp (`src/ct_iqa/vif/metric.py`) was not modified, renamed, or aliased.
No LDCT-IQAC labels have been computed with either variant; that remains a
separately-gated decision (see `docs/research_decisions.md`, decision
A-15's status note).
