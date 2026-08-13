"""Validation script for the full wavelet-domain VIF implementation.

Runs `ct_iqa.vif.wavelet.vif_wavelet` against a small set of controlled,
locally-generated test images (NOT the LDCT-IQAC dataset -- see
docs/research_decisions.md decision A-15 and the instruction this script
was built under: "do not download a large external dataset merely for
testing" and "dataset must remain untouched").

Two things are checked:

1. Property-based sanity across image categories: identity, noise
   monotonicity, blur monotonicity (the same properties
   tests/test_vif_wavelet.py checks, run here for a human-readable report
   rather than pass/fail assertions).
2. A cross-check against an independent, vendored full wavelet-domain VIF
   implementation (scripts/_reference_vif/vif_utils.py, from
   https://github.com/abhinaukumar/vif -- see
   scripts/_reference_vif/NOTICE.md for its provenance and known parameter
   differences from this project's implementation).

This script performs NO dataset generation, NO label computation for
LDCT-IQAC, and NO training. It is a one-off validation aid.

Usage:
    python scripts/validate_vif.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "_reference_vif"))

from ct_iqa.vif.wavelet import vif_wavelet  # noqa: E402

try:
    from vif_utils import vif as reference_vif  # noqa: E402

    _REFERENCE_AVAILABLE = True
    _REFERENCE_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _REFERENCE_AVAILABLE = False
    _REFERENCE_ERROR = str(exc)


# --------------------------------------------------------------- test images


def natural_like(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 4 * np.pi, size), np.linspace(0, 4 * np.pi, size))
    structure = 40.0 * np.sin(x) * np.cos(y) + 20.0 * np.sin(3 * x + y)
    return 100.0 + structure + rng.normal(scale=3.0, size=(size, size))


def ct_like(size: int = 256, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    body = np.where(r < size * 0.4, 60.0, 5.0)
    organ = np.where(((yy - cy * 0.9) ** 2 + (xx - cx * 1.1) ** 2) < (size * 0.12) ** 2, 25.0, 0.0)
    return body + organ + rng.normal(scale=2.0, size=(size, size))


def high_texture(size: int = 256, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=128.0, scale=40.0, size=(size, size))


def smooth(size: int = 256) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return 128.0 + 50.0 * (x**2 + y**2)


CATEGORIES = {
    "natural-like": natural_like,
    "ct-like": ct_like,
    "high-texture": high_texture,
    "smooth": smooth,
}


# ------------------------------------------------------------ property report


def run_property_checks() -> None:
    print("=" * 78)
    print("1. PROPERTY CHECKS (identity, noise / blur monotonicity)")
    print("=" * 78)
    rng = np.random.default_rng(42)

    for name, make in CATEGORIES.items():
        img = make()
        identity = vif_wavelet(img, img)

        noise_scores = [vif_wavelet(img, img + rng.normal(scale=s, size=img.shape)) for s in (2, 10, 30)]
        blur_scores = [vif_wavelet(img, gaussian_filter(img, s)) for s in (0.5, 2.0, 5.0)]

        noise_monotonic = noise_scores[0] > noise_scores[1] > noise_scores[2]
        blur_monotonic = blur_scores[0] > blur_scores[1] > blur_scores[2]

        print(f"\n[{name}]")
        print(f"  identity (expect ~1.0)      : {identity:.6f}")
        print(f"  noise sigma=2,10,30         : {[round(s, 4) for s in noise_scores]}  monotonic={noise_monotonic}")
        print(f"  blur  sigma=0.5,2,5         : {[round(s, 4) for s in blur_scores]}  monotonic={blur_monotonic}")


# --------------------------------------------------------------- cross-check


def run_cross_check() -> None:
    print()
    print("=" * 78)
    print("2. CROSS-CHECK AGAINST INDEPENDENT IMPLEMENTATION")
    print("=" * 78)

    if not _REFERENCE_AVAILABLE:
        print(
            "\nNo independent implementation available -- import of "
            f"scripts/_reference_vif/vif_utils.py failed: {_REFERENCE_ERROR}"
        )
        print("This limitation is documented explicitly; see docs/vif_implementation.md.")
        return

    print(
        "\nReference: https://github.com/abhinaukumar/vif (vendored at "
        "scripts/_reference_vif/vif_utils.py; see NOTICE.md for known "
        "parameter differences -- sigma_nsq, subband selection, aggregation."
    )

    rng = np.random.default_rng(7)
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name, make in CATEGORIES.items():
        ref = make()
        pairs.append((f"{name} / identity", ref, ref.copy()))
        pairs.append((f"{name} / noise(sigma=10)", ref, ref + rng.normal(scale=10, size=ref.shape)))
        pairs.append((f"{name} / noise(sigma=30)", ref, ref + rng.normal(scale=30, size=ref.shape)))
        pairs.append((f"{name} / blur(sigma=1.4)", ref, gaussian_filter(ref, 1.4)))
        pairs.append((f"{name} / blur(sigma=5.0)", ref, gaussian_filter(ref, 5.0)))

    rows = []
    for label, ref_img, dist_img in pairs:
        ours = vif_wavelet(ref_img, dist_img)
        try:
            theirs = reference_vif(ref_img, dist_img, wavelet="steerable")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [{label}] reference implementation raised: {exc}")
            continue
        abs_diff = abs(ours - theirs)
        rel_diff = abs_diff / max(abs(theirs), 1e-9)
        rows.append((label, ours, theirs, abs_diff, rel_diff))

    print(f"\n{'case':32s} {'ours':>10s} {'reference':>10s} {'abs_diff':>10s} {'rel_diff':>10s}")
    for label, ours, theirs, abs_diff, rel_diff in rows:
        theirs_str = f"{theirs:10.4f}" if np.isfinite(theirs) else f"{'nan':>10s}"
        diff_str = f"{abs_diff:10.4f} {rel_diff:10.2%}" if np.isfinite(abs_diff) else f"{'nan':>10s} {'nan':>10s}"
        print(f"{label:32s} {ours:10.4f} {theirs_str} {diff_str}")

    # Exclude non-finite cases (e.g. the "smooth" category -- see below) from
    # the correlation/summary statistics rather than letting one NaN poison
    # every aggregate; those cases are reported separately, not hidden.
    finite_rows = [r for r in rows if np.isfinite(r[1]) and np.isfinite(r[2])]
    dropped = [r[0] for r in rows if r not in finite_rows]

    ours_arr = np.array([r[1] for r in finite_rows])
    theirs_arr = np.array([r[2] for r in finite_rows])

    if len(finite_rows) >= 2:
        pearson_r, _ = pearsonr(ours_arr, theirs_arr)
        spearman_r, _ = spearmanr(ours_arr, theirs_arr)
    else:
        pearson_r = spearman_r = float("nan")

    worst = max(finite_rows, key=lambda r: r[3]) if finite_rows else None
    mean_abs = float(np.mean([r[3] for r in finite_rows])) if finite_rows else float("nan")
    mean_rel = float(np.mean([r[4] for r in finite_rows])) if finite_rows else float("nan")

    print("\nSummary (finite cases only):")
    print(f"  n_cases (finite / total) : {len(finite_rows)} / {len(rows)}")
    if dropped:
        print(f"  dropped (non-finite)     : {dropped}")
    print(f"  Pearson correlation      : {pearson_r:.4f}")
    print(f"  Spearman correlation     : {spearman_r:.4f}")
    print(f"  mean absolute difference : {mean_abs:.4f}")
    print(f"  mean relative difference : {mean_rel:.2%}")
    if worst is not None:
        print(f"  largest disagreement     : {worst[0]}  (ours={worst[1]:.4f}, reference={worst[2]:.4f}, "
              f"abs_diff={worst[3]:.4f})")

    if dropped:
        print(
            "\nNote on dropped cases: the 'smooth' category is a near-flat "
            "quadratic bowl with very low local variance almost everywhere. "
            "Both implementations become numerically unstable on it (our "
            "implementation: VIF > 1 and non-monotonic under increasing "
            "noise; the reference implementation: NaN, from a negative "
            "log argument caused by an ill-conditioned GSM covariance "
            "matrix). This is a genuine, shared degenerate case -- not a "
            "bug unique to either implementation -- and is recorded as an "
            "open caveat in docs/vif_implementation.md rather than "
            "papered over here."
        )
    print(
        "\nInterpretation: exact numerical agreement is NOT expected (see "
        "scripts/_reference_vif/NOTICE.md for the specific parameter "
        "differences between the two implementations). What matters is "
        "whether the two agree on ranking/direction -- a high rank "
        "correlation here means both implementations respond to "
        "noise/blur/identity the same way, which is the property this "
        "cross-check exists to establish. See docs/vif_implementation.md "
        "for the recorded result and what it does and does not prove."
    )


if __name__ == "__main__":
    run_property_checks()
    run_cross_check()
