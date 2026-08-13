"""Full wavelet-domain VIF (Sheikh & Bovik, 2006) -- PROJECT ADAPTATION A-15.

Ohashi et al. cite Sheikh HR, Bovik AC, "Image information and visual
quality," IEEE Trans Image Process 15(2):430-444, 2006 [their ref 27] for
VIF. That paper's own released MATLAB reference code (Sheikh's LIVE lab,
UT Austin, ``vifvec.m`` + ``refparams_vecgsm.m`` + ``vifsub_est_M.m``,
mirrored at
https://github.com/sattarab/image-quality-tools/tree/master/metrix_mux/metrix/vif/vifvec_release)
decomposes both images with a steerable pyramid, models each subband's
coefficients as a Gaussian Scale Mixture (GSM), fits a linear
signal-plus-noise "distortion channel" model between the reference and
distorted subbands, and aggregates a mutual-information ratio across
subbands and GSM eigenvalues. This module is a structurally faithful
Python port of that released algorithm, built on ``pyrtools`` for the
steerable-pyramid decomposition (the same C-backed ``corrDn`` primitive
that the original MATLAB code uses is reused via ``pyrtools.corrDn``).

This is NOT verified bit-exact against a MATLAB run -- no MATLAB
installation is available in this environment (see
docs/vif_investigation.md and docs/vif_implementation.md). Every
parameter below is labelled with its provenance:

  * OHASHI-SPECIFIED       -- stated in the Ohashi paper itself.
  * REFERENCE-IMPLEMENTATION-CONVENTION -- taken directly from Sheikh's
    released MATLAB code (cited above), not invented by this project.
  * PROJECT-ADAPTATION     -- a choice this project made because the
    exact MATLAB behaviour could not be reproduced or verified here.
  * UNKNOWN                -- genuinely undetermined; see
    docs/vif_investigation.md.

See docs/vif_implementation.md for the full specification and
docs/research_decisions.md (decision A-15) for why this route was chosen
over VIFp, a third-party package, or a MATLAB call.

Two named PROFILES are available (``vif_wavelet(..., profile=...)``):

  * "project" (default) -- the canonical Ohashi-adaptation configuration
    described above. This is the ONLY profile that may ever be used for
    LDCT-IQAC label generation. It is not altered by the existence of the
    other profile.
  * "reference_crosscheck" -- reconfigures every tunable parameter to
    match, as closely as technically possible, a second, independently
    authored full wavelet-domain VIF implementation
    (https://github.com/abhinaukumar/vif, vendored for validation only at
    scripts/_reference_vif/vif_utils.py). This exists SOLELY to answer
    one question: is the earlier moderate numerical disagreement between
    "project" and that independent implementation caused by a bug in this
    project's code, or fully explained by documented parameter and
    structural differences? It is not a metric candidate for anything
    else and must never be used for dataset labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ProfileName = Literal["project", "reference_crosscheck"]


@dataclass(frozen=True)
class VIFProfile:
    """A named, fully-explicit bundle of every tunable VIF-wavelet parameter.

    Every field here is either REFERENCE-IMPLEMENTATION-CONVENTION (taken
    from a specific, cited piece of released code) or PROJECT-ADAPTATION
    (this project's own choice where no such code exists). None are
    OHASHI-SPECIFIED -- Ohashi's paper does not go to this level of detail
    for any implementation. See docs/vif_implementation.md.
    """

    name: str
    sigma_nsq: float
    orientations_used: tuple[int, int]
    #: "direct": distortion-channel window size grows with actual pyramid
    #: coarseness (level 0=finest -> winsize=3, level 3=coarsest ->
    #: winsize=17), matching vifvec.m's own ``lev=ceil((sub-1)/6)`` formula
    #: applied to a fine-to-coarse subband ordering.
    #: "reversed_pair": window size is assigned by position in a
    #: coarse-to-fine-reversed, paired subband list, which assigns the
    #: LARGEST window to the FINEST level and the SMALLEST window to the
    #: COARSEST level -- the opposite of vifvec.m's own convention. This
    #: is not an alternative reading of the algorithm; it reproduces what
    #: https://github.com/abhinaukumar/vif's ``vif()`` actually computes
    #: (traced explicitly in docs/vif_implementation.md), used here only
    #: to test whether replicating it closes the numerical gap.
    lev_mode: Literal["direct", "reversed_pair"]
    #: "sum": accumulate every per-pixel, per-eigenvalue term with a plain
    #: sum across the whole subband and across all subbands (vifvec.m).
    #: "mean_stabilized": average per-pixel terms within each subband
    #: (giving every subband equal weight regardless of pixel count, not
    #: an extensive/additive combination), add a small additive stabilizer
    #: to each subband's numerator and denominator total, then combine
    #: subbands (also as a mean, which is proportional to a sum here since
    #: every subband contributes one term -- see docs/vif_implementation.md).
    #: This reproduces abhinaukumar/vif's aggregation, not vifvec.m's.
    aggregation: Literal["sum", "mean_stabilized"]
    stabilizer: float = 0.0
    #: If set, GSM covariance eigenvalues below this floor are clipped to
    #: the floor and the covariance matrix is reconstructed from the
    #: clipped eigenvalues before inversion (abhinaukumar/vif's approach
    #: to a near-singular Cu). If None, this project's own defensive
    #: `pinv`-based handling is used instead (see _reference_gsm_field).
    eigenvalue_floor: float | None = None
    tolerance: float = 1e-15
    pyramid_height: int = 4
    pyramid_order: int = 5
    block_size: int = 3
    boundary: str = "reflect1"
    log_base: float = field(default=2.0)


#: The canonical, Ohashi-adaptation configuration (decision A-15). This is
#: the ONLY profile permitted for any future LDCT-IQAC label generation.
PROJECT_PROFILE = VIFProfile(
    name="project",
    sigma_nsq=0.4,
    orientations_used=(2, 5),
    lev_mode="direct",
    aggregation="sum",
    stabilizer=0.0,
    eigenvalue_floor=None,
    tolerance=1e-15,
    pyramid_height=4,
    pyramid_order=5,
    block_size=3,
    boundary="reflect1",
    log_base=2.0,
)

#: Mirrors https://github.com/abhinaukumar/vif's ``vif(..., wavelet='steerable')``
#: as closely as technically possible. VALIDATION USE ONLY -- see the
#: module docstring and docs/vif_implementation.md. Never use for labelling.
REFERENCE_CROSSCHECK_PROFILE = VIFProfile(
    name="reference_crosscheck",
    sigma_nsq=0.1,
    orientations_used=(0, 3),
    lev_mode="reversed_pair",
    aggregation="mean_stabilized",
    stabilizer=1e-4,
    eigenvalue_floor=1e-15,
    tolerance=1e-15,
    pyramid_height=4,
    pyramid_order=5,
    block_size=3,
    boundary="reflect1",
    log_base=2.0,  # cosmetic only: log base cancels in the final ratio.
)

_PROFILES: dict[str, VIFProfile] = {
    "project": PROJECT_PROFILE,
    "reference_crosscheck": REFERENCE_CROSSCHECK_PROFILE,
}

# Backwards-compatible module-level constants (canonical/"project" values).
PYRAMID_HEIGHT = PROJECT_PROFILE.pyramid_height
PYRAMID_ORDER = PROJECT_PROFILE.pyramid_order
NUM_ORIENTATIONS = PYRAMID_ORDER + 1
ORIENTATIONS_USED = PROJECT_PROFILE.orientations_used
BLOCK_SIZE = PROJECT_PROFILE.block_size
SIGMA_NSQ = PROJECT_PROFILE.sigma_nsq
TOLERANCE = PROJECT_PROFILE.tolerance
BOUNDARY = PROJECT_PROFILE.boundary


def _reference_gsm_field(
    y: np.ndarray, block_size: int, eigenvalue_floor: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-block GSM covariance eigenvalues and the local "S field".

    Direct port of refparams_vecgsm.m for a single subband. Returns
    ``(eigenvalues, ss)`` where ``eigenvalues`` are the block_size**2
    eigenvalues of the block covariance matrix Cu, and ``ss`` is the
    per-non-overlapping-block scalar field
    ``ss[n] = block_n^T @ inv(Cu) @ block_n / block_size**2``.

    PROJECT-ADAPTATION: internal grid indexing is row-major (numpy
    convention) rather than MATLAB's column-major ``reshape`` order.
    This does not change the reference GSM statistics (mean/covariance
    over a bag of samples is order-invariant) or the final VIF score
    (which sums over all spatial positions), but the specific (row,
    col) address of a given "ss" value is not guaranteed to match the
    MATLAB implementation's addressing -- only self-consistency within
    this port is guaranteed, which is what the aggregation requires.

    If ``eigenvalue_floor`` is given, eigenvalues below it are clipped to
    the floor and Cu is reconstructed from the clipped eigenbasis before
    inversion -- abhinaukumar/vif's regularisation for a near-singular Cu
    (relevant on low-texture subbands; see docs/vif_implementation.md).
    Otherwise ``np.linalg.pinv`` is used directly on the unclipped Cu.
    """
    m = block_size
    rows, cols = y.shape
    r = (rows // m) * m
    c = (cols // m) * m
    y = y[:r, :c]

    overlap_samples = []
    for k in range(m):
        for j in range(m):
            block = y[k : r - (m - 1 - k), j : c - (m - 1 - j)]
            overlap_samples.append(block.reshape(-1))
    u = np.stack(overlap_samples, axis=0)
    mu = u.mean(axis=1, keepdims=True)
    u_centered = u - mu
    cu = (u_centered @ u_centered.T) / u.shape[1]

    eigenvalues, eigenvectors = np.linalg.eigh(cu)

    if eigenvalue_floor is not None:
        eigenvalues = np.where(eigenvalues < eigenvalue_floor, eigenvalue_floor, eigenvalues)
        cu = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        cu_inv = np.linalg.inv(cu)
    else:
        cu_inv = np.linalg.pinv(cu)

    grid_r, grid_c = r // m, c // m
    block_samples = []
    for k in range(m):
        for j in range(m):
            block_samples.append(y[k::m, j::m][:grid_r, :grid_c].reshape(-1))
    b = np.stack(block_samples, axis=0)

    projected = cu_inv @ b
    ss = np.sum(projected * b, axis=0) / (m * m)
    ss = ss.reshape(grid_r, grid_c)

    return eigenvalues, ss


def _distortion_channel_field(
    reference: np.ndarray,
    distorted: np.ndarray,
    winsize: int,
    block_size: int,
    boundary: str,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear channel-model gain ``g`` and residual noise variance ``vv``.

    Direct port of vifsub_est_M.m for a single subband, using
    ``pyrtools.corrDn`` -- the same correlate-then-downsample primitive
    (compiled from the original matlabPyrTools C source) that the MATLAB
    reference code calls directly. ``winsize`` is passed in explicitly
    (rather than derived from the pyramid level inside this function) so
    that the caller's ``lev_mode`` choice controls it -- see VIFProfile.
    """
    from pyrtools import corrDn

    m = block_size
    win = np.ones((winsize, winsize))

    rows, cols = reference.shape
    r = (rows // m) * m
    c = (cols // m) * m
    ref = reference[:r, :c]
    dist = distorted[:r, :c]

    winstep = (m, m)
    winstart = (m // 2, m // 2)
    winstop = (r - ((m + 1) // 2) + 1, c - ((m + 1) // 2) + 1)

    mean_x = corrDn(ref, win / win.sum(), edge_type=boundary, step=winstep, start=winstart, stop=winstop)
    mean_y = corrDn(dist, win / win.sum(), edge_type=boundary, step=winstep, start=winstart, stop=winstop)
    cov_xy = (
        corrDn(ref * dist, win, edge_type=boundary, step=winstep, start=winstart, stop=winstop)
        - win.sum() * mean_x * mean_y
    )
    ss_x = (
        corrDn(ref * ref, win, edge_type=boundary, step=winstep, start=winstart, stop=winstop)
        - win.sum() * mean_x**2
    )
    ss_y = (
        corrDn(dist * dist, win, edge_type=boundary, step=winstep, start=winstart, stop=winstop)
        - win.sum() * mean_y**2
    )

    ss_x = np.maximum(ss_x, 0.0)
    ss_y = np.maximum(ss_y, 0.0)

    g = cov_xy / (ss_x + tol)
    vv = (ss_y - g * cov_xy) / win.sum()

    below_x = ss_x < tol
    g = np.where(below_x, 0.0, g)
    vv = np.where(below_x, ss_y, vv)
    ss_x = np.where(below_x, 0.0, ss_x)

    below_y = ss_y < tol
    g = np.where(below_y, 0.0, g)
    vv = np.where(below_y, 0.0, vv)

    negative_g = g < 0.0
    vv = np.where(negative_g, ss_y, vv)
    g = np.maximum(g, 0.0)

    vv = np.maximum(vv, tol)

    return g, vv


def _resolve_profile(profile: ProfileName | VIFProfile) -> VIFProfile:
    if isinstance(profile, VIFProfile):
        return profile
    try:
        return _PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown VIF profile: {profile!r}; known profiles: {sorted(_PROFILES)}") from exc


def _validate_inputs(ref: np.ndarray, dist: np.ndarray, cfg: VIFProfile) -> None:
    if ref.shape != dist.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {dist.shape}")
    if ref.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {ref.shape}")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(dist)):
        raise ValueError("inputs must not contain NaN or Inf")

    min_side = min(ref.shape)
    coarsest_winsize = 2**cfg.pyramid_height + 1
    required = 2 ** (cfg.pyramid_height - 1) * (coarsest_winsize + 2 * cfg.block_size)
    if min_side < required:
        raise ValueError(
            f"image too small for a {cfg.pyramid_height}-level steerable pyramid with "
            f"block size {cfg.block_size}: smallest side is {min_side}, need at least {required}"
        )


def _iter_subbands(cfg: VIFProfile) -> list[tuple[tuple[int, int], int]]:
    """Yield ``((level, orientation), winsize)`` for every subband this
    profile uses, with ``winsize`` already resolved per ``cfg.lev_mode``.
    """
    fine_to_coarse = [
        (level, orientation) for level in range(cfg.pyramid_height) for orientation in cfg.orientations_used
    ]

    if cfg.lev_mode == "direct":
        # vifvec.m: lev = ceil((sub-1)/6) on a fine-to-coarse subband
        # ordering -> window grows with actual pyramid coarseness.
        return [(key, 2 ** (level + 1) + 1) for level, key in ((lvl, (lvl, ori)) for lvl, ori in fine_to_coarse)]

    if cfg.lev_mode == "reversed_pair":
        # abhinaumar/vif: subbands collected fine->coarse, then reversed to
        # coarse->fine, then lev = ceil((i+1)/2) over that reversed order.
        # Net effect: the window size assigned to a given (level, orientation)
        # is inverted relative to "direct" -- see VIFProfile.lev_mode.
        coarse_to_fine = list(reversed(fine_to_coarse))
        return [(key, 2 ** (int(np.ceil((i + 1) / 2))) + 1) for i, key in enumerate(coarse_to_fine)]

    raise ValueError(f"unknown lev_mode: {cfg.lev_mode!r}")


def vif_wavelet_diagnostics(
    reference: np.ndarray, distorted: np.ndarray, profile: ProfileName | VIFProfile = "project"
) -> list[dict]:
    """Per-subband diagnostic trace of every numerical stage.

    Returns one dict per subband with: pyramid coefficient statistics,
    GSM covariance conditioning, eigenvalue range, channel-model (g, vv)
    ranges, and the numerator/denominator each subband contributed. Used
    to localise exactly which stage destabilises on a low-texture image
    (see docs/vif_implementation.md, "smooth-image instability" section)
    rather than treating the whole function as a black box.
    """
    import pyrtools as pt

    cfg = _resolve_profile(profile)
    ref = np.asarray(reference, dtype=np.float64)
    dist = np.asarray(distorted, dtype=np.float64)
    _validate_inputs(ref, dist, cfg)

    pyr_ref = pt.pyramids.SteerablePyramidSpace(
        ref, height=cfg.pyramid_height, order=cfg.pyramid_order, edge_type=cfg.boundary
    )
    pyr_dist = pt.pyramids.SteerablePyramidSpace(
        dist, height=cfg.pyramid_height, order=cfg.pyramid_order, edge_type=cfg.boundary
    )

    records = []
    for (level, orientation), winsize in _iter_subbands(cfg):
        key = (level, orientation)
        y = pyr_ref.pyr_coeffs[key]
        yn = pyr_dist.pyr_coeffs[key]

        rows, cols = y.shape
        r = (rows // cfg.block_size) * cfg.block_size
        c = (cols // cfg.block_size) * cfg.block_size
        y_cropped = y[:r, :c]
        overlap_samples = []
        for k in range(cfg.block_size):
            for j in range(cfg.block_size):
                block = y_cropped[k : r - (cfg.block_size - 1 - k), j : c - (cfg.block_size - 1 - j)]
                overlap_samples.append(block.reshape(-1))
        u = np.stack(overlap_samples, axis=0)
        cu = np.cov(u)
        cond = float(np.linalg.cond(cu))

        eigenvalues, ss = _reference_gsm_field(y, cfg.block_size, cfg.eigenvalue_floor)
        g, vv = _distortion_channel_field(y, yn, winsize, cfg.block_size, cfg.boundary, cfg.tolerance)

        h = min(ss.shape[0], g.shape[0])
        w = min(ss.shape[1], g.shape[1])
        ss_c, g_c, vv_c = ss[:h, :w], g[:h, :w], vv[:h, :w]
        offset = int(np.ceil(((winsize - 1) / 2) / cfg.block_size))
        if offset > 0 and h > 2 * offset and w > 2 * offset:
            ss_c = ss_c[offset:-offset, offset:-offset]
            g_c = g_c[offset:-offset, offset:-offset]
            vv_c = vv_c[offset:-offset, offset:-offset]

        num = 0.0
        den = 0.0
        for lam in eigenvalues:
            num += float(np.sum(np.log2(1.0 + (g_c**2) * ss_c * lam / (vv_c + cfg.sigma_nsq))))
            den += float(np.sum(np.log2(1.0 + ss_c * lam / cfg.sigma_nsq)))

        records.append(
            {
                "level": level,
                "orientation": orientation,
                "winsize": winsize,
                "coeff_std_ref": float(np.std(y)),
                "coeff_std_dist": float(np.std(yn)),
                "cu_condition_number": cond,
                "eigenvalue_min": float(np.min(eigenvalues)),
                "eigenvalue_max": float(np.max(eigenvalues)),
                "ss_min": float(np.min(ss_c)) if ss_c.size else float("nan"),
                "ss_max": float(np.max(ss_c)) if ss_c.size else float("nan"),
                "g_min": float(np.min(g_c)) if g_c.size else float("nan"),
                "g_max": float(np.max(g_c)) if g_c.size else float("nan"),
                "vv_min": float(np.min(vv_c)) if vv_c.size else float("nan"),
                "vv_max": float(np.max(vv_c)) if vv_c.size else float("nan"),
                "subband_num": num,
                "subband_den": den,
            }
        )
    return records


def vif_wavelet(
    reference: np.ndarray,
    distorted: np.ndarray,
    profile: ProfileName | VIFProfile = "project",
) -> float:
    """Full wavelet-domain VIF between a reference and a distorted image.

    Structurally follows Sheikh & Bovik's released reference MATLAB
    implementation (steerable pyramid + per-subband GSM + linear
    distortion channel + mutual-information ratio); see the module
    docstring and docs/vif_implementation.md for the full specification
    and provenance of every constant used. NOT verified bit-exact
    against a MATLAB run.

    ``profile`` selects the parameter set: "project" (default) is the
    canonical Ohashi-adaptation configuration (decision A-15) -- the only
    one permitted for LDCT-IQAC label generation. "reference_crosscheck"
    exists solely to test whether an independent implementation's
    parameter choices explain a prior numerical disagreement; see the
    module docstring. A caller may also pass a custom ``VIFProfile``.

    Both inputs must be 2-D, the same shape, finite, and on the same
    intensity scale. Returns a scalar; identical inputs give
    approximately 1.0.
    """
    import pyrtools as pt

    cfg = _resolve_profile(profile)
    ref = np.asarray(reference, dtype=np.float64)
    dist = np.asarray(distorted, dtype=np.float64)
    _validate_inputs(ref, dist, cfg)

    pyr_ref = pt.pyramids.SteerablePyramidSpace(
        ref, height=cfg.pyramid_height, order=cfg.pyramid_order, edge_type=cfg.boundary
    )
    pyr_dist = pt.pyramids.SteerablePyramidSpace(
        dist, height=cfg.pyramid_height, order=cfg.pyramid_order, edge_type=cfg.boundary
    )

    subband_nums = []
    subband_dens = []

    for (level, orientation), winsize in _iter_subbands(cfg):
        key = (level, orientation)
        y = pyr_ref.pyr_coeffs[key]
        yn = pyr_dist.pyr_coeffs[key]

        eigenvalues, ss = _reference_gsm_field(y, cfg.block_size, cfg.eigenvalue_floor)
        g, vv = _distortion_channel_field(y, yn, winsize, cfg.block_size, cfg.boundary, cfg.tolerance)

        h = min(ss.shape[0], g.shape[0])
        w = min(ss.shape[1], g.shape[1])
        ss_c, g_c, vv_c = ss[:h, :w], g[:h, :w], vv[:h, :w]

        offset = int(np.ceil(((winsize - 1) / 2) / cfg.block_size))
        if offset > 0 and h > 2 * offset and w > 2 * offset:
            ss_c = ss_c[offset:-offset, offset:-offset]
            g_c = g_c[offset:-offset, offset:-offset]
            vv_c = vv_c[offset:-offset, offset:-offset]

        sub_num = 0.0
        sub_den = 0.0
        for lam in eigenvalues:
            if cfg.aggregation == "sum":
                sub_num += float(np.sum(np.log2(1.0 + (g_c**2) * ss_c * lam / (vv_c + cfg.sigma_nsq))))
                sub_den += float(np.sum(np.log2(1.0 + ss_c * lam / cfg.sigma_nsq)))
            elif cfg.aggregation == "mean_stabilized":
                sub_num += float(np.mean(np.log2(1.0 + (g_c**2) * ss_c * lam / (vv_c + cfg.sigma_nsq))))
                sub_den += float(np.mean(np.log2(1.0 + ss_c * lam / cfg.sigma_nsq)))
            else:
                raise ValueError(f"unknown aggregation: {cfg.aggregation!r}")

        if cfg.aggregation == "mean_stabilized":
            sub_num += cfg.stabilizer
            sub_den += cfg.stabilizer

        subband_nums.append(sub_num)
        subband_dens.append(sub_den)

    if cfg.aggregation == "sum":
        num_total = float(np.sum(subband_nums))
        den_total = float(np.sum(subband_dens))
    else:
        num_total = float(np.mean(subband_nums))
        den_total = float(np.mean(subband_dens))

    if den_total <= 0.0:
        return float("nan")
    return num_total / den_total
