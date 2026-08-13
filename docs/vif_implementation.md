# VIF Implementation Specification — `vif_wavelet()`

Practical specification for `src/ct_iqa/vif/wavelet.py::vif_wavelet()`, the
full wavelet-domain VIF implementation adopted under decision A-15
(`docs/research_decisions.md`). Read `docs/vif_investigation.md` first for
*why* this route was chosen; this document is *how* it is built.

`vif_p()` (`src/ct_iqa/vif/metric.py`, VIFp) is unchanged and remains
available as a separate variant. The two are never aliased.

---

## 1. Mathematical formulation

VIF measures the ratio of mutual information the human visual system (HVS)
could extract from the distorted image about the reference, relative to
what it could extract from the reference image alone, under a Gaussian
Scale Mixture (GSM) model of natural-image wavelet statistics and an
additive-noise HVS model:

```
VIF = sum_subbands( I(C ; F | s) ) / sum_subbands( I(C ; E | s) )
```

where, per subband, `C` is the GSM field of the reference coefficients,
`s` the local GSM scalar multiplier, `E` the HVS-noise-corrupted
reference perception, and `F` the HVS-noise-corrupted perception of the
reference *after* passing through a fitted linear "distortion channel"
(`distorted ≈ g * reference + noise`) that models how the distortion
process transformed the reference subband into the distorted one. In closed
form, per subband and per GSM eigenvalue `λ_j`:

```
numerator_term   = log2(1 + g^2 * s * λ_j / (σ_v^2 + σ_nsq))
denominator_term = log2(1 + s * λ_j / σ_nsq)
```

summed over all pixels, eigenvalues, and subbands. This is exactly the
formula in Sheikh & Bovik's released reference code (`vifvec.m`, §2
below) and is what `vif_wavelet()` implements.

## 2. Pyramid construction

| Parameter | Value | Classification |
| --- | --- | --- |
| Decomposition | Steerable pyramid | OHASHI-SPECIFIED (the cited paper's title formulation) |
| Levels/scales | 4 | REFERENCE-IMPLEMENTATION-CONVENTION (`vifvec.m`: `buildSpyr(im, 4, ...)`) |
| Orientations per level | 6 | REFERENCE-IMPLEMENTATION-CONVENTION (`vifvec.m`: `'sp5Filters'`; `pyrtools` equivalent: `order=5`) |
| Boundary handling | `reflect1` | REFERENCE-IMPLEMENTATION-CONVENTION (`vifvec.m`'s explicit argument; also `pyrtools`'s own default) |
| Subbands used | 2 of 6 orientations per level, all 4 levels (8 total) | REFERENCE-IMPLEMENTATION-CONVENTION for the *count*; PROJECT-ADAPTATION for *which two* — see §9 |
| Library | `pyrtools.pyramids.SteerablePyramidSpace` | PROJECT-ADAPTATION (Python substitute for MATLAB's `matlabPyrTools`/`buildSpyr`; same underlying algorithm family, not the same codebase) |

`vifvec.m`'s literal subband list, `[4 7 10 13 16 19 22 25]` (1-based MATLAB
indices into a 26-band pyramid: 1 highpass residual + 4×6 oriented bands +
1 lowpass residual), selects orientations 3 and 6 (1-based) — i.e. indices
2 and 5 (0-based) — out of each level's 6 orientations. `vif_wavelet()`
uses `ORIENTATIONS_USED = (2, 5)` to match this pattern. Whether
`pyrtools`'s orientation-channel numbering is rotationally aligned with
`matlabPyrTools`'s is **not independently verified** — see §12.

## 3. GSM estimation

Per subband, per `refparams_vecgsm.m`:

1. Crop the subband to a multiple of the block size `M`.
2. Collect **all** overlapping `M×M` blocks (every pixel position) as
   `M²`-dimensional vectors; estimate their sample mean and covariance
   matrix `Cu` (`M²×M²`).
3. Eigendecompose `Cu` → `M²` eigenvalues `λ_1..λ_{M²}`.
4. Collect **non-overlapping** `M×M` blocks on a decimated grid; for each,
   compute the scalar field `s[n] = block_n^T · Cu⁻¹ · block_n / M²`.

| Parameter | Value | Classification |
| --- | --- | --- |
| Block size `M` | 3 | REFERENCE-IMPLEMENTATION-CONVENTION (`refparams_vecgsm.m`, `vifsub_est_M.m`: `M=3`) |
| Covariance estimator | sample mean/covariance over all overlapping blocks | REFERENCE-IMPLEMENTATION-CONVENTION |
| Eigen-decomposition | `numpy.linalg.eigvalsh` (symmetric) | PROJECT-ADAPTATION (MATLAB's `eig`; mathematically equivalent for a symmetric PSD matrix, ordering differs but is irrelevant since all eigenvalues are summed) |
| Grid indexing order | row-major (`numpy`) | PROJECT-ADAPTATION (MATLAB uses column-major `reshape`; does not change the resulting statistics or final score — see the docstring of `_reference_gsm_field` — but the (row, col) address of an individual field value is not guaranteed to match a MATLAB run's) |
| Covariance matrix inverse | `numpy.linalg.pinv` | PROJECT-ADAPTATION (MATLAB uses `inv`; `pinv` is used defensively for near-singular `Cu`, which occurs on very low-texture subbands — see §10) |

## 4. Distortion/channel model

Per subband, per `vifsub_est_M.m`, a locally-windowed linear regression
between reference and distorted subband coefficients:

```
g   = cov(ref, dist) / var(ref)            # channel gain
σ_v² = var(dist) - g · cov(ref, dist)      # residual ("noise") variance
```

computed with a box-filter window whose size grows with pyramid level
(`winsize = 2^level + 1`), via correlate-then-downsample.

| Parameter | Value | Classification |
| --- | --- | --- |
| Window size | `2^level + 1` (level 1..4) | REFERENCE-IMPLEMENTATION-CONVENTION |
| Window shape | flat/box (`ones(winsize, winsize)`) | REFERENCE-IMPLEMENTATION-CONVENTION |
| Correlate-then-downsample primitive | `pyrtools.corrDn` | PROJECT-ADAPTATION, but a strong one: `pyrtools.corrDn` is compiled from the same C source as `matlabPyrTools`'s `corrDn`, the exact function `vifsub_est_M.m` calls |
| Sampling grid (`winstep`, `winstart`, `winstop`) | step=`M`, start=`floor(M/2)`, stop sized to keep the grid within bounds | REFERENCE-IMPLEMENTATION-CONVENTION for the *scheme*; PROJECT-ADAPTATION for the exact 0-based index arithmetic translating MATLAB's 1-based formula (`winstart=floor(M/2)+1`, `winstop=size-ceil(M/2)+1`) — not independently verified to produce identically-shaped output to a MATLAB run |
| Degenerate handling (near-zero variance, negative gain) | exact branch-by-branch port of `vifsub_est_M.m`'s clipping logic | REFERENCE-IMPLEMENTATION-CONVENTION |

## 5. Mutual-information calculation and aggregation

Exact port of `vifvec.m`'s final loop: for each subband, each GSM
eigenvalue, accumulate
`log2(1 + g²·s·λ/(σ_v²+σ_nsq))` (numerator) and `log2(1 + s·λ/σ_nsq)`
(denominator) over all valid pixel positions; sum across all subbands;
divide.

| Parameter | Value | Classification |
| --- | --- | --- |
| `σ_nsq` (HVS noise variance) | 0.4 | REFERENCE-IMPLEMENTATION-CONVENTION (`vifvec.m`: `sigma_nsq=0.4`) — **note this is deliberately different from VIFp's `SIGMA_NSQ=2.0`**; the two constants are not interchangeable and using one in place of the other is a distinct bug class, not a rounding choice |
| Log base | 2 | REFERENCE-IMPLEMENTATION-CONVENTION (`vifvec.m` uses `log2`) — mathematically this cancels in the final ratio regardless of base, so this is a cosmetic-only match, not a numerically load-bearing one |
| Aggregation | `sum` over all pixels/eigenvalues/subbands, then a single division | REFERENCE-IMPLEMENTATION-CONVENTION |

## 6. Numerical stability

| Mechanism | Value/behaviour | Classification |
| --- | --- | --- |
| Degenerate-variance tolerance | `1e-15` | REFERENCE-IMPLEMENTATION-CONVENTION (`vifsub_est_M.m`: `tol = 1e-15`) |
| Negative-variance clipping | `max(variance, 0)` before use | REFERENCE-IMPLEMENTATION-CONVENTION |
| Non-positive-definite `Cu` | `pinv` instead of `inv` | PROJECT-ADAPTATION (defensive; see §10, low-texture-region caveat) |
| Zero total denominator | returns `NaN` | PROJECT-ADAPTATION, consistent with `vif_p()`'s existing documented behaviour for a fully flat reference (`src/ct_iqa/vif/metric.py`: *"A completely flat reference carries no information to preserve."*) |

## 7. Boundary handling

`reflect1` throughout — both the pyramid decomposition (`SteerablePyramidSpace(..., edge_type="reflect1")`) and the windowed correlation
(`corrDn(..., edge_type="reflect1")`). REFERENCE-IMPLEMENTATION-CONVENTION,
matching `vifvec.m`'s explicit choice at both call sites. In addition, an
explicit offset-crop is applied per subband after computing `g`, `σ_v²`,
and `s`, removing `ceil(((winsize-1)/2)/M)` pixels from each border, per
`vifvec.m`'s own boundary-safety logic — REFERENCE-IMPLEMENTATION-CONVENTION.

## 8. Input requirements

- 2-D array, both inputs the same shape.
- No `NaN`/`Inf` values (rejected explicitly, `ValueError`).
- Minimum size: the coarsest of the 4 pyramid levels (decimated by `2³`)
  must, after cropping to a multiple of `M=3`, still exceed that level's
  distortion-channel window (`2⁴+1=17`) with margin — concretely, both
  image dimensions must be **≥ 184 pixels**. All of Ohashi's own reference
  images (512×512) and the LDCT-IQAC images (also 512×512, per
  `docs/dataset_adaptation.md`) comfortably satisfy this; this only
  matters for small synthetic test images. PROJECT-ADAPTATION (the exact
  threshold formula), derived directly from the algorithm's own structure
  rather than chosen arbitrarily.
- Intensity scale: same convention as `vif_p()` — both images on the same
  scale (e.g. both 0–255 8-bit); the algorithm itself is scale-sensitive
  through `σ_nsq`, exactly as VIFp is through its own `σ_nsq`.

## 9. Output definition

A scalar float. `1.0` for identical inputs (by construction: `g=1`,
`σ_v²≈0`, so numerator ≈ denominator per subband). Decreases toward 0 as
the distorted image loses information relative to the reference. Unlike
VIFp, is **not** guaranteed to stay within `[0, 1]` in all cases — see the
low-texture caveat in §10. `NaN` for a fully flat reference (no
information to measure loss against).

## 10. Known limitations and open caveats

- **Low-texture / near-flat images can produce VIF > 1 or non-monotonic
  scores.** Verified empirically (`scripts/validate_vif.py`, "smooth"
  category): a smooth quadratic-bowl test image produced VIF ≈ 7–11 and
  *increased* with added noise, instead of decreasing. Root cause: on a
  near-flat subband, the block covariance matrix `Cu` is close to
  singular, so `s` (and therefore both numerator and denominator terms)
  become numerically unstable. **This is not unique to this
  implementation** — the independent reference implementation used for
  cross-checking (§11) produced `NaN` (from a negative log argument) on
  the identical test image, via a different but related numerical
  failure mode. This is treated as a genuine, documented property of the
  algorithm family on this class of input, not a bug to silently patch
  around. Ohashi's own reference images are structured CT anatomy, not
  synthetic smooth gradients, so this caveat is unlikely to be triggered
  by the actual LDCT-IQAC dataset — but it has not been checked against
  the real dataset yet (that check is out of scope for this phase; see
  `docs/research_decisions.md` guardrails).
- **Grid-alignment between the GSM field (`s`) and the distortion-channel
  fields (`g`, `σ_v²`) is self-consistent but not verified pixel-address-
  identical to a MATLAB run.** Both are cropped to a common shape before
  combination (`min(ss.shape, g.shape)`); this is a defensive
  PROJECT-ADAPTATION, not present in the MATLAB code (which relies on
  its own indexing arithmetic producing exactly matching shapes without
  needing a min-crop). In practice, in all tests run so far, the two
  fields' shapes have matched exactly without the min-crop ever
  triggering a non-trivial reduction — but this has not been proven in
  general.

## 11. Validation strategy

Two layers, both already executed (see final report in the conversation
this document accompanies):

1. **Property tests** (`tests/test_vif_wavelet.py`, 22 tests): identity ≈
   1.0 across four image categories; monotonic decrease under increasing
   noise and under increasing blur; determinism (repeated calls, and
   memory-layout independence); explicit rejection of shape mismatches,
   non-2D input, NaN/Inf, and undersized images; a documented degenerate
   case (fully flat image → `NaN`, matching `vif_p()`'s own convention);
   and one directional cross-variant check (full wavelet VIF is more
   blur-sensitive than VIFp on the same image pair, which the paper's own
   Table 5 noise/blur asymmetry predicts).
2. **Cross-check against an independent implementation**
   (`scripts/validate_vif.py`; results in §11 below and the accompanying
   final report). No numerical exact-match target exists (no MATLAB
   ground truth available in this environment), so this is a
   rank/direction check, not an equality check.

## 12. Reproducibility information

| Item | Value |
| --- | --- |
| `pyrtools` version | `1.0.10` (pinned in `requirements.txt`) |
| Install date / environment | 2026-08-13, Anaconda Python 3.13.9, Windows 11, `win_amd64` wheel |
| Determinism | Fully deterministic given identical inputs — no RNG, no GPU, pure NumPy/SciPy/`pyrtools` CPU computation. Verified independent of array memory layout (C- vs Fortran-order) in `tests/test_vif_wavelet.py::test_determinism_independent_of_array_memory_layout` |
| Dtype/precision | `float64` throughout (`np.asarray(..., dtype=np.float64)` at the public entry point) |
| Bit-exact MATLAB parity | **Not established.** No MATLAB installation is available in this environment (see `docs/vif_investigation.md` §5, Route C). This implementation is a structurally faithful port of the *released* reference algorithm, not a verified numerical match to it, and *a fortiori* not a verified match to whatever Ohashi's own MATLAB R2024a run actually computed (which used an unnamed function/toolbox — see `docs/research_decisions.md`, item U-V01) |

## 13. Parameter-equivalence validation (2026-08-13)

Follow-up to §11's cross-check, which showed only moderate agreement
(Pearson 0.62) between `vif_wavelet(..., profile="project")` and the
independent implementation (`scripts/_reference_vif/vif_utils.py`). This
section determines whether that gap is explained by known parameter
differences or indicates an implementation error.

### 13.1 Two profiles

`vif_wavelet()` now accepts a `profile` argument (`"project"` |
`"reference_crosscheck"` | a custom `VIFProfile`). `"project"` is
byte-for-byte the canonical configuration used throughout this document —
confirmed unchanged by regression test after the refactor (identical
output to five decimal places on a fixed test image before/after:
`0.3469591058671248` → `0.3469591058671254`, the ~1e-12 residual coming
from switching `eigvalsh` → `eigh` internally, not from any parameter
change). It remains the only profile permitted for LDCT-IQAC labelling.

`"reference_crosscheck"` reconfigures every tunable parameter to match
`vif_utils.py`'s `vif(..., wavelet='steerable')` as closely as technically
possible:

| Parameter | project (canonical) | reference_crosscheck | Same? |
| --- | --- | --- | --- |
| `sigma_nsq` | 0.4 | 0.1 | No |
| orientations used | (2, 5) 0-based | (0, 3) 0-based | No |
| window-size-to-level mapping | `direct` (matches `vifvec.m`) | `reversed_pair` (matches the independent implementation) | No |
| aggregation | `sum` across all pixels/subbands | `mean` per subband + additive stabilizer | No |
| numerical stabilizer | none | `1e-4` added to each subband's numerator/denominator | No |
| eigenvalue floor | none (`pinv` on raw Cu) | `1e-15` floor, Cu reconstructed from clipped eigenbasis, then `inv` | No |
| tolerance, pyramid height/order, block size, boundary | identical | identical | Yes |

One entry needs a caveat, not just a checkmark: **`reversed_pair` is not a
defensible alternative reading of Sheikh & Bovik (2006) — it reproduces
what looks like an actual inversion bug in the independent implementation's
own code.** Traced directly (see `scripts/validate_vif_parameter_equivalence.py`,
section 1): that implementation selects 8 subbands fine-to-coarse, then
calls `.reverse()` on the list before assigning window sizes by list
position. Net effect: it assigns the *smallest* correlation window
(winsize=3) to the *coarsest* pyramid level and the *largest* window
(winsize=17) to the *finest* level — backwards from `vifvec.m`'s own
formula (`lev=ceil((sub-1)/6)` on a fine-to-coarse list), which assigns
larger windows to coarser, smaller subbands, and smaller windows to finer,
larger ones. `reference_crosscheck` reproduces this deliberately, because
the point of this validation is to test whether *matching* the
independent implementation (quirks included) closes the numerical gap —
not to adopt this behaviour as an improvement.

### 13.2 Three-way comparison

20 controlled image pairs (4 categories × identity/noise(σ=10,30)/blur(σ=1.4,5.0),
identical to §11's set) run through:

- **A** — `vif_wavelet(..., profile="project")`
- **B** — `vif_wavelet(..., profile="reference_crosscheck")`
- **C** — the independent implementation directly

| case | A (project) | B (reference_crosscheck) | C (independent) |
| --- | --- | --- | --- |
| natural-like / identity | 1.0000 | 1.0000 | 1.0000 |
| natural-like / noise(σ=10) | 0.6226 | 0.4885 | 0.4885 |
| natural-like / noise(σ=30) | 0.4611 | 0.2577 | 0.2577 |
| natural-like / blur(σ=1.4) | 0.3470 | 0.8877 | 0.8877 |
| natural-like / blur(σ=5.0) | 0.2015 | 0.5990 | 0.5990 |
| ct-like / identity | 1.0000 | 1.0000 | 1.0000 |
| ct-like / noise(σ=10) | 0.7055 | 0.4862 | 0.4862 |
| ct-like / noise(σ=30) | 0.5068 | 0.2587 | 0.2587 |
| ct-like / blur(σ=1.4) | 0.3169 | 0.7885 | 0.7885 |
| ct-like / blur(σ=5.0) | 0.0711 | 0.4018 | 0.4018 |
| high-texture / identity | 1.0000 | 1.0000 | 1.0000 |
| high-texture / noise(σ=10) | 0.6368 | 0.4471 | 0.4471 |
| high-texture / noise(σ=30) | 0.3041 | 0.1811 | 0.1811 |
| high-texture / blur(σ=1.4) | 0.2100 | 0.5413 | 0.5413 |
| high-texture / blur(σ=5.0) | 0.0080 | 0.1719 | 0.1719 |
| smooth / identity | 1.0000 | NaN | NaN |
| smooth / noise(σ=10) | 10.8461 | NaN | NaN |
| smooth / noise(σ=30) | 11.6099 | NaN | NaN |
| smooth / blur(σ=1.4) | 0.9982 | NaN | NaN |
| smooth / blur(σ=5.0) | 0.9262 | NaN | NaN |

Every finite B value visually matches C to 4 decimal places already in this
table. Summary statistics (15 finite cases, "smooth" excluded — see §13.4):

| Metric | A (project) vs C | B (reference_crosscheck) vs C |
| --- | --- | --- |
| Pearson r | 0.6239 | 0.99999999999989 |
| Spearman r | 0.5607 | 1.0000 |
| MAE | 0.2236 | 4.6×10⁻⁷ |
| RMSE | 0.2756 | 5.6×10⁻⁷ |
| Max absolute difference | 0.5408 (natural-like / blur σ=1.4) | 9.6×10⁻⁷ |

### 13.3 Interpretation

B agrees with C to within floating-point noise. This is strong, direct
evidence that `vif_wavelet()`'s core computation — pyramid decomposition
via `pyrtools`, GSM block-covariance/eigenvalue estimation, the linear
distortion-channel regression via `pyrtools.corrDn`, and the final
mutual-information aggregation — contains no structural implementation
error: given the *same* configuration as the independent implementation,
it produces (as far as this test can measure) the *same* numbers. The
moderate A-vs-C disagreement documented in §11 is therefore fully
attributable to the five documented parameter/structural differences in
§13.1, not to a bug in either implementation.

This does **not** establish that `project`'s parameters (drawn from
`vifvec.m`) are what Ohashi's own MATLAB R2024a run actually used — that
remains unknown and unknowable without Ohashi's code or a MATLAB
comparison (§12). What it establishes is narrower and still valuable: the
Python implementation of the *algorithm*, independent of which specific
parameter set is dialled in, is behaving correctly.

### 13.4 Smooth/low-variance failure — stage localisation

`vif_wavelet_diagnostics()` was run on the "smooth" test image
(quadratic-bowl gradient) against `img + noise(σ=10)`, for both profiles,
recording per-subband: pyramid coefficient std, GSM covariance condition
number, eigenvalue range, channel-model (`g`, `σ_v²`) range, and each
subband's numerator/denominator contribution.

Findings, read directly from the trace (`project` profile):

| Level (0=finest) | `cu_cond` | `g_max` | subband num | subband den |
| --- | --- | --- | --- | --- |
| 0 | ~2–3×10¹⁶ | **2212 / 2426** | ~52,000 each | ~748 each |
| 1 | ~1–2×10¹⁷ | 79–92 | ~5,900–6,000 | ~2,840–2,850 |
| 2 | ~1.1–1.2×10¹⁷ | 2.9–3.1 | ~695–703 | ~1,640 |
| 3 | ~1.9–2.8×10¹⁷ | ~1.0 | ~158–163 | ~285 |

Two distinct, compounding causes, not one:

1. **GSM covariance ill-conditioning at every level** (condition numbers
   10¹⁶–10¹⁸ throughout, including the coarsest level where the signal is
   not small). This is an intrinsic property of this test image: a smooth
   quadratic surface's local `3×3` pixel blocks are nearly perfectly
   linearly correlated (adjacent pixels in a slowly-varying gradient carry
   almost the same information), so the block covariance matrix `Cu` is
   close to rank-deficient regardless of how much raw signal energy is
   present. This is the GSM estimation stage, not the pyramid
   decomposition or channel-model stage.
2. **Channel-gain blow-up at the finest levels specifically** (`g` up to
   ~2212 at level 0, decaying to ~1.0 by level 3). This is the
   distortion-channel estimation stage: the finest subbands of a smooth
   image carry almost no local variance (`ss_x` near the `1e-15`
   tolerance floor), so `g = cov_xy / (ss_x + tol)` can become enormous
   even with the tolerance guard, because `cov_xy` need not shrink as
   fast as `ss_x` does. Since the numerator term scales with `g²` and the
   denominator does not depend on `g` at all, this asymmetry is exactly
   what pushes the finest levels' contribution (num≈52,000 vs den≈748)
   to dominate the total ratio, producing VIF ≫ 1.

Both causes trace to genuine structural properties of this specific class
of input (near-flat, low local variance almost everywhere), not to an
implementation bug — and both stages are exact ports of `refparams_vecgsm.m`
and `vifsub_est_M.m`'s own math, run with no defensive deviation beyond
what those files themselves already specify (the `tol` floor, the
`max(variance, 0)` clip). The independent implementation fails
differently but for the same underlying reason (`NaN` rather than a large
finite value — its `1e-4` additive stabilizer and `mean`-based aggregation
change the failure's *symptom*, not its *cause*).

**Practical read for this project:** Ohashi's actual reference images are
structured CT anatomy (head/neck/thorax/lung/abdomen/pelvis/limb), not
synthetic smooth gradients, so this specific failure mode is unlikely to
be triggered by real LDCT-IQAC images — but that is an expectation, not a
verified fact, since no LDCT-IQAC image has been run through
`vif_wavelet()` in this validation phase (per the explicit constraint that
the dataset remains untouched). This risk should be checked, not assumed
away, before any future label-generation run — e.g. by running
`vif_wavelet_diagnostics()` over a small sample of real LDCT-IQAC
reference images first and confirming none trip the same `cu_cond`/`g`
blow-up pattern documented here.
