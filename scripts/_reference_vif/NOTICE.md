# Third-party reference code — validation use only

`vif_utils.py` in this directory is vendored, unmodified, from
https://github.com/abhinaukumar/vif (`vif_utils.py`, fetched 2026-08-13).
It is **not** a project dependency, not imported by `src/ct_iqa`, and not
used anywhere in the actual pipeline. Its sole purpose is
`scripts/validate_vif.py`'s cross-check of `ct_iqa.vif.wavelet.vif_wavelet`
against a second, independently-authored full wavelet-domain VIF
implementation, per decision A-15's validation requirement
(docs/research_decisions.md, docs/vif_investigation.md §7).

## Why this repository

It is a genuine full wavelet-domain/GSM VIF implementation (not a
relabeled VIFp): it uses `pyrtools.pyramids.SteerablePyramidSpace` for the
steerable-pyramid decomposition (4 levels, order 5 — the same
`pyrtools` primitive and the same pyramid configuration this project's
`ct_iqa/vif/wavelet.py` uses), performs GSM block-covariance estimation
via `im2col` + `np.cov` + eigendecomposition, and fits the same
linear signal-plus-noise distortion-channel model Sheikh & Bovik's
`vifsub_est_M.m` does (see the comparison in `docs/vif_implementation.md`).

## Known parameter differences from this project's `vif_wavelet()`

Both implementations aim at the same cited formulation but were built
independently, and neither is confirmed bit-exact against Ohashi's MATLAB
run (no MATLAB reference output is available to either). Specific
differences that will produce non-identical scores even on the same image
pair:

- `sigma_nsq = 0.1` here vs. `0.4` in this project (and in the original
  `vifvec.m` release this project's implementation follows).
- Subband selection: this code takes every 3rd pyramid-dictionary key
  (`list(pyr.keys())[1:-2:3]`), landing on orientations 0 and 3 (0-based)
  of each level's 6; this project's `vif_wavelet()` follows `vifvec.m`'s
  literal `subbands=[4 7 10 13 16 19 22 25]`, landing on orientations 2
  and 5. Both pick 2 of 6 orientations per level, 4 levels — but not the
  same two.
- Aggregation uses `np.mean` per subband plus a `1e-4` additive
  stabilizer on both numerator and denominator terms; this project's
  `vif_wavelet()` uses `np.sum` with no additive stabilizer, matching
  `vifvec.m` exactly.

None of this makes either implementation "wrong" — both are defensible,
independently-motivated readings of the same published algorithm family.
The point of the cross-check is to confirm the two agree in *direction and
rank* (both should show the same qualitative response to noise/blur/
identity), not to expect numerical equality. See
`docs/vif_implementation.md` for the full cross-check results.

## License note

`vif_utils.py` carries no explicit license header in the upstream
repository at the time of vendoring. It is kept here strictly for
local, non-distributed validation (comparing two implementations'
output on synthetic test images) and is not incorporated into this
project's actual VIF computation path, imported by `src/ct_iqa`, or
shipped as part of any dataset-labelling process.
