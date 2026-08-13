"""Validation suite for the full wavelet-domain VIF implementation
(``ct_iqa.vif.wavelet.vif_wavelet``, decision A-15).

These tests check documented *properties* of VIF (identity, monotonicity
under increasing degradation, determinism, robustness to degenerate
regions, explicit input validation) -- not invented exact numerical
values. No MATLAB reference is available in this environment to compare
against (see docs/vif_investigation.md), so exact-value assertions would
be unfounded; property-based assertions are the honest bar here.

The wavelet pyramid needs a minimum image size (the coarsest of 4 levels
must still exceed its distortion-channel window after cropping to a
multiple of the GSM block size) -- see ``ct_iqa.vif.wavelet``'s size
guard. All images in this file are >= 256x256 for that reason; the
64/128-pixel phantoms used for VIFp's tests in test_metrics.py are too
small for the wavelet variant by design, not by oversight.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ct_iqa.vif.metric import vif_p
from ct_iqa.vif.wavelet import vif_wavelet


def _phantom(size: int = 256, seed: int = 0) -> np.ndarray:
    """A textured test image with several scales of structure.

    Not perfectly flat anywhere (a perfectly flat image has zero
    reference information and is a documented, separate degenerate case
    -- see test_fully_flat_image_returns_nan_not_a_wrong_number).
    """
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 4 * np.pi, size), np.linspace(0, 4 * np.pi, size))
    structure = 40.0 * np.sin(x) * np.cos(y) + 20.0 * np.sin(3 * x + y)
    base = 100.0 + structure
    return base + rng.normal(scale=3.0, size=(size, size))


def _ct_like_phantom(size: int = 256, seed: int = 1) -> np.ndarray:
    """A CT-like grayscale image: a soft circular body outline plus
    piecewise-uniform internal structures, similar in spirit to Ohashi's
    reference images (Fig. 2)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    body = np.where(r < size * 0.4, 60.0, 5.0)
    organ = np.where(((yy - cy * 0.9) ** 2 + (xx - cx * 1.1) ** 2) < (size * 0.12) ** 2, 25.0, 0.0)
    return body + organ + rng.normal(scale=2.0, size=(size, size))


def _high_texture_phantom(size: int = 256, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=128.0, scale=40.0, size=(size, size))


def _smooth_phantom(size: int = 256) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return 128.0 + 50.0 * (x**2 + y**2)


# -------------------------------------------------------------- A. identity


def test_identity_gives_vif_approximately_one():
    img = _phantom()
    assert vif_wavelet(img, img) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "image_fn",
    [_phantom, _ct_like_phantom, _high_texture_phantom],
    ids=["natural-like", "ct-like", "high-texture"],
)
def test_identity_holds_across_image_types(image_fn):
    img = image_fn()
    assert vif_wavelet(img, img) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------- B. noise monotonicity


def test_vif_decreases_with_noise_severity():
    img = _phantom()
    rng = np.random.default_rng(10)
    sigmas = (2, 10, 30)
    scores = [vif_wavelet(img, img + rng.normal(scale=s, size=img.shape)) for s in sigmas]
    assert scores[0] > scores[1] > scores[2]


def test_vif_decreases_with_noise_severity_on_ct_like_image():
    img = _ct_like_phantom()
    rng = np.random.default_rng(11)
    sigmas = (2, 10, 30)
    scores = [vif_wavelet(img, img + rng.normal(scale=s, size=img.shape)) for s in sigmas]
    assert scores[0] > scores[1] > scores[2]


# -------------------------------------------------------- C. blur monotonicity


def test_vif_decreases_with_blur_severity():
    img = _phantom()
    sigmas = (0.5, 2.0, 5.0)
    scores = [vif_wavelet(img, gaussian_filter(img, s)) for s in sigmas]
    assert scores[0] > scores[1] > scores[2]


def test_vif_decreases_with_blur_severity_on_ct_like_image():
    img = _ct_like_phantom()
    sigmas = (0.5, 2.0, 5.0)
    scores = [vif_wavelet(img, gaussian_filter(img, s)) for s in sigmas]
    assert scores[0] > scores[1] > scores[2]


# ------------------------------------------------------ D. combined degradation


def test_combined_degradation_is_deterministic_and_bounded():
    img = _phantom()
    rng = np.random.default_rng(20)
    noisy_blurred = gaussian_filter(img, 1.4) + rng.normal(scale=8.0, size=img.shape)

    first = vif_wavelet(img, noisy_blurred)
    second = vif_wavelet(img, noisy_blurred)

    assert first == second
    assert np.isfinite(first)
    # A degraded image must score below an identical (undegraded) pair;
    # this is the only bound VIF guarantees here. Whether combined
    # noise+blur scores above or below either degradation applied alone is
    # NOT a guaranteed property of this metric (order and interaction
    # between the channel-model fits for each distortion can go either
    # way) -- not asserted.
    assert first < vif_wavelet(img, img)


# ------------------------------------------------------------ E. determinism


def test_determinism_same_pair_same_score():
    ref = _phantom(seed=5)
    dist = gaussian_filter(ref, 1.0)
    scores = [vif_wavelet(ref, dist) for _ in range(3)]
    assert scores[0] == scores[1] == scores[2]


def test_determinism_independent_of_array_memory_layout():
    """A Fortran-ordered copy of the same data must give the same score."""
    ref = _phantom(seed=6)
    dist = gaussian_filter(ref, 1.0)
    score_c = vif_wavelet(np.ascontiguousarray(ref), np.ascontiguousarray(dist))
    score_f = vif_wavelet(np.asfortranarray(ref), np.asfortranarray(dist))
    assert score_c == pytest.approx(score_f, abs=1e-9)


# ------------------------------------------------------------ F. shape handling


def test_512x512_input_produces_a_finite_score():
    img = _ct_like_phantom(size=512)
    distorted = gaussian_filter(img, 1.4) + np.random.default_rng(7).normal(scale=10, size=img.shape)
    score = vif_wavelet(img, distorted)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.5  # VIF > 1 is possible for pathological cases but should be near [0,1]


def test_non_square_input_is_supported():
    rng = np.random.default_rng(8)
    img = rng.normal(loc=100, scale=20, size=(300, 260))
    assert vif_wavelet(img, img) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------- G. flat / low-variance regions


def test_mostly_flat_image_with_small_textured_region_gives_finite_score():
    """A locally-flat background should not poison the score with NaN/Inf,
    as long as the image carries reference information somewhere."""
    img = np.full((256, 256), 50.0)
    img[100:156, 100:156] += 30.0 * np.sin(np.linspace(0, 6 * np.pi, 56))
    distorted = gaussian_filter(img, 1.0)
    score = vif_wavelet(img, distorted)
    assert np.isfinite(score)


def test_fully_flat_image_returns_nan_not_a_wrong_number():
    """A perfectly flat reference carries zero information (0/0 in the VIF
    ratio) -- NaN is the mathematically correct output here, matching
    vif_p's own documented behaviour for the same degenerate case. This is
    not the "no NaN/Inf" property from item G (which concerns *regions*
    within an otherwise informative image); it is a distinct, expected
    degenerate case worth pinning down explicitly.
    """
    flat = np.full((256, 256), 50.0)
    assert np.isnan(vif_wavelet(flat, flat))
    assert np.isnan(vif_wavelet(flat, flat + 5.0))


def test_low_contrast_image_gives_finite_score():
    img = 100.0 + np.random.default_rng(9).normal(scale=0.5, size=(256, 256))
    distorted = img + np.random.default_rng(90).normal(scale=0.3, size=img.shape)
    score = vif_wavelet(img, distorted)
    assert np.isfinite(score)


# ------------------------------------------------------------- H. invalid inputs


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        vif_wavelet(_phantom(256), _phantom(300))


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        vif_wavelet(np.zeros((256, 256, 3)), np.zeros((256, 256, 3)))


def test_rejects_nan_input():
    img = _phantom()
    bad = img.copy()
    bad[10, 10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        vif_wavelet(img, bad)


def test_rejects_inf_input():
    img = _phantom()
    bad = img.copy()
    bad[10, 10] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        vif_wavelet(img, bad)


def test_rejects_image_too_small_for_the_pyramid():
    small = np.random.default_rng(0).normal(size=(64, 64))
    with pytest.raises(ValueError, match="too small"):
        vif_wavelet(small, small)


# ----------------------------------------------- cross-variant sanity (not a
# ----------------------------------------------- cross-validation, see script)


# ------------------------------------------------------------------ profiles


def test_default_profile_is_project():
    """The canonical Ohashi-adaptation configuration is the default -- a
    caller who forgets to pass `profile=` must get the labelling-safe
    configuration, never the validation-only one."""
    img = _phantom()
    default = vif_wavelet(img, gaussian_filter(img, 1.4))
    explicit = vif_wavelet(img, gaussian_filter(img, 1.4), profile="project")
    assert default == explicit


def test_reference_crosscheck_profile_differs_from_project():
    """The two profiles must not silently collapse to the same behaviour --
    that would defeat the entire point of the parameter-equivalence check."""
    img = _ct_like_phantom()
    distorted = gaussian_filter(img, 1.4)
    project_score = vif_wavelet(img, distorted, profile="project")
    crosscheck_score = vif_wavelet(img, distorted, profile="reference_crosscheck")
    assert project_score != pytest.approx(crosscheck_score, rel=1e-3)


def test_reference_crosscheck_profile_still_gives_identity_one():
    img = _phantom()
    assert vif_wavelet(img, img, profile="reference_crosscheck") == pytest.approx(1.0, abs=1e-6)


def test_unknown_profile_name_is_rejected():
    img = _phantom()
    with pytest.raises(ValueError, match="unknown VIF profile"):
        vif_wavelet(img, img, profile="made_up_profile")  # type: ignore[arg-type]


def test_wavelet_vif_diverges_from_vifp_under_blur():
    """The two formulations are not interchangeable: full wavelet VIF is
    expected to be markedly more blur-sensitive than VIFp (this is the
    exact property the paper's own noise/blur asymmetry -- Table 5 -- turns
    on; see docs/vif_investigation.md, Section 4). This test only checks
    the two variants disagree in the expected direction, not by how much;
    an exact numerical comparison belongs in scripts/validate_vif.py.
    """
    img = _ct_like_phantom(size=256)
    blurred = gaussian_filter(img, 2.0)

    wavelet_score = vif_wavelet(img, blurred)
    vifp_score = vif_p(img, blurred)

    assert wavelet_score < vifp_score
