"""Parameter-equivalence validation for vif_wavelet().

Answers one specific question: is the moderate numerical disagreement
between this project's canonical vif_wavelet() ("project" profile) and the
independent implementation (scripts/_reference_vif/vif_utils.py) explained
by documented parameter/structural differences, or does it indicate an
implementation error?

Method: run the same 20 controlled image pairs used in
scripts/validate_vif.py through three configurations --

  A. vif_wavelet(..., profile="project")             -- canonical, unchanged
  B. vif_wavelet(..., profile="reference_crosscheck") -- every tunable
     parameter set to match the independent implementation as closely as
     technically possible (see VIFProfile.REFERENCE_CROSSCHECK_PROFILE
     and its docstring for exactly what "as closely as possible" means,
     including one case -- lev_mode -- where matching the independent
     implementation means reproducing what looks like an inversion bug in
     its window-size-to-pyramid-level mapping, not a legitimate
     alternative convention; this is called out explicitly, not hidden)
  C. the independent implementation directly

-- and reports A-vs-C and B-vs-C separately: Pearson r, Spearman r, MAE,
RMSE, max absolute difference, plus a full per-image table.

Also investigates the "smooth" low-texture failure mode by stage
(coefficient magnitude -> covariance conditioning -> GSM eigenvalues ->
channel model -> per-subband mutual information) using
vif_wavelet_diagnostics(), for both the "project" and
"reference_crosscheck" profiles, so the failure can be localised rather
than just observed as a bad final number.

Performs NO dataset generation, NO LDCT-IQAC label computation, NO
training. One-off validation aid only.

Usage:
    python scripts/validate_vif_parameter_equivalence.py
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

from ct_iqa.vif.wavelet import vif_wavelet, vif_wavelet_diagnostics  # noqa: E402

try:
    from vif_utils import vif as reference_vif  # noqa: E402

    _REFERENCE_AVAILABLE = True
    _REFERENCE_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _REFERENCE_AVAILABLE = False
    _REFERENCE_ERROR = str(exc)


# --------------------------------------------------------------- test images
# Identical generators to scripts/validate_vif.py, so the 20 image pairs are
# the exact same ones already used for the earlier (profile-less) cross-check.


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


def build_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, make in CATEGORIES.items():
        ref = make()
        pairs.append((f"{name} / identity", ref, ref.copy()))
        pairs.append((f"{name} / noise(sigma=10)", ref, ref + rng.normal(scale=10, size=ref.shape)))
        pairs.append((f"{name} / noise(sigma=30)", ref, ref + rng.normal(scale=30, size=ref.shape)))
        pairs.append((f"{name} / blur(sigma=1.4)", ref, gaussian_filter(ref, 1.4)))
        pairs.append((f"{name} / blur(sigma=5.0)", ref, gaussian_filter(ref, 5.0)))
    return pairs


# ----------------------------------------------------------------- section 1


def print_parameter_diff_table() -> None:
    print("=" * 100)
    print("1. PARAMETER / CONFIGURATION DIFFERENCES: project vs reference_crosscheck")
    print("=" * 100)
    from ct_iqa.vif.wavelet import PROJECT_PROFILE, REFERENCE_CROSSCHECK_PROFILE

    fields = [
        "sigma_nsq",
        "orientations_used",
        "lev_mode",
        "aggregation",
        "stabilizer",
        "eigenvalue_floor",
        "tolerance",
        "pyramid_height",
        "pyramid_order",
        "block_size",
        "boundary",
    ]
    print(f"\n{'parameter':22s} {'project (canonical)':28s} {'reference_crosscheck':28s} {'same?':>6s}")
    for f in fields:
        a = getattr(PROJECT_PROFILE, f)
        b = getattr(REFERENCE_CROSSCHECK_PROFILE, f)
        print(f"{f:22s} {str(a):28s} {str(b):28s} {'yes' if a == b else 'NO':>6s}")

    print(
        "\nNote on 'lev_mode': this is not a difference of degree. 'direct' "
        "(project) matches vifvec.m's own window-size formula applied to a "
        "fine-to-coarse subband order. 'reversed_pair' "
        "(reference_crosscheck) reproduces what the independent "
        "implementation's code actually computes after it reverses its "
        "subband list before assigning window sizes -- which inverts which "
        "pyramid level gets the largest window relative to vifvec.m. This "
        "looks like an artifact of that implementation's own code, not an "
        "equally-valid alternative reading of Sheikh & Bovik (2006). It is "
        "reproduced here anyway, deliberately, because the goal of this "
        "script is to test whether MATCHING the independent implementation "
        "(bugs and all) closes the gap -- which tells us whether our "
        "disagreement is explained by known differences, not whether "
        "'reference_crosscheck' is a better configuration to keep."
    )
    print(
        "\nAlso not tabulated above (present in the independent implementation's "
        "code, not expressible as a single VIFProfile field): natural log vs "
        "log2 aggregation (cancels in the final ratio, not a real "
        "difference) and dtype (float32 in im2col-based moment computation "
        "vs float64 throughout in this project -- see section 3)."
    )


# ----------------------------------------------------------------- section 2


def run_three_way_comparison() -> list[dict]:
    print()
    print("=" * 100)
    print("2. THREE-WAY COMPARISON: A=project, B=reference_crosscheck, C=independent")
    print("=" * 100)

    if not _REFERENCE_AVAILABLE:
        print(f"\nIndependent implementation unavailable: {_REFERENCE_ERROR}")
        return []

    pairs = build_pairs()
    rows = []
    for label, ref_img, dist_img in pairs:
        a = vif_wavelet(ref_img, dist_img, profile="project")
        b = vif_wavelet(ref_img, dist_img, profile="reference_crosscheck")
        try:
            c = reference_vif(ref_img, dist_img, wavelet="steerable")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [{label}] independent implementation raised: {exc}")
            c = float("nan")
        rows.append({"label": label, "A_project": a, "B_refcompat": b, "C_independent": c})

    header = f"{'case':32s} {'A_project':>12s} {'B_refcompat':>12s} {'C_independent':>14s}"
    print(f"\n{header}")
    for row in rows:
        c_str = f"{row['C_independent']:.4f}" if np.isfinite(row["C_independent"]) else "nan"
        print(f"{row['label']:32s} {row['A_project']:12.4f} {row['B_refcompat']:12.4f} {c_str:>14s}")

    return rows


def _stats(x: np.ndarray, y: np.ndarray) -> dict:
    diff = x - y
    pearson_r, _ = pearsonr(x, y) if len(x) >= 2 else (float("nan"), None)
    spearman_r, _ = spearmanr(x, y) if len(x) >= 2 else (float("nan"), None)
    return {
        "n": len(x),
        "pearson": pearson_r,
        "spearman": spearman_r,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def report_pairwise_stats(rows: list[dict]) -> None:
    print()
    print("=" * 100)
    print("3. SUMMARY STATISTICS (finite cases only, 'smooth' category excluded -- see section 4)")
    print("=" * 100)

    finite = [r for r in rows if np.isfinite(r["C_independent"]) and "smooth" not in r["label"]]
    dropped = [r["label"] for r in rows if r not in finite]

    a = np.array([r["A_project"] for r in finite])
    b = np.array([r["B_refcompat"] for r in finite])
    c = np.array([r["C_independent"] for r in finite])

    stats_ac = _stats(a, c)
    stats_bc = _stats(b, c)

    print(f"\nExcluded from statistics ({len(dropped)} cases): {dropped}")

    print("\nA. canonical (project) vs independent:")
    for k, v in stats_ac.items():
        print(f"   {k:12s}: {v}")

    print("\nB. reference_crosscheck vs independent:")
    for k, v in stats_bc.items():
        print(f"   {k:12s}: {v}")

    print("\nInterpretation:")
    if stats_bc["pearson"] > stats_ac["pearson"] + 0.15 and stats_bc["mae"] < stats_ac["mae"] * 0.6:
        print(
            "  reference_crosscheck agrees with the independent implementation "
            "MARKEDLY better than project does. This supports the hypothesis "
            "that the earlier moderate disagreement was parameterization- "
            "and structure-related, not an implementation error in "
            "vif_wavelet()'s core algorithm."
        )
    elif stats_bc["pearson"] > stats_ac["pearson"] and stats_bc["mae"] < stats_ac["mae"]:
        print(
            "  reference_crosscheck agrees somewhat better than project does, "
            "but the improvement is partial, not dramatic. This is weak-to-"
            "moderate evidence that parameterization explains PART of the "
            "gap; residual disagreement may still include structural or "
            "implementation differences not captured by VIFProfile's "
            "parameters (e.g. exact corrDn window/grid alignment, exact "
            "orientation-channel correspondence between pyrtools and "
            "matlabPyrTools)."
        )
    else:
        print(
            "  reference_crosscheck does NOT clearly agree with the independent "
            "implementation better than project does. This does NOT confirm "
            "an implementation error, but it does mean the parameter "
            "differences identified in section 1 do not fully explain the "
            "original disagreement -- further investigation is warranted "
            "before treating either implementation as a validated ground "
            "truth for the other."
        )


# ----------------------------------------------------------------- section 4


def investigate_smooth_failure() -> None:
    print()
    print("=" * 100)
    print("4. SMOOTH / LOW-VARIANCE FAILURE: stage-by-stage diagnostic")
    print("=" * 100)

    ref = smooth()
    rng = np.random.default_rng(99)
    dist = ref + rng.normal(scale=10, size=ref.shape)

    for profile_name in ("project", "reference_crosscheck"):
        print(f"\n--- profile={profile_name} ---")
        records = vif_wavelet_diagnostics(ref, dist, profile=profile_name)
        header = (
            f"{'lvl':>3s} {'ori':>3s} {'win':>4s} {'coeff_std':>10s} {'cu_cond':>12s} "
            f"{'eig_min':>10s} {'eig_max':>10s} {'ss_min':>10s} {'ss_max':>10s} "
            f"{'g_min':>8s} {'g_max':>8s} {'vv_min':>10s} {'vv_max':>10s} {'num':>8s} {'den':>8s}"
        )
        print(header)
        for r in records:
            print(
                f"{r['level']:3d} {r['orientation']:3d} {r['winsize']:4d} "
                f"{r['coeff_std_ref']:10.2e} {r['cu_condition_number']:12.2e} "
                f"{r['eigenvalue_min']:10.2e} {r['eigenvalue_max']:10.2e} "
                f"{r['ss_min']:10.2e} {r['ss_max']:10.2e} "
                f"{r['g_min']:8.2f} {r['g_max']:8.2f} "
                f"{r['vv_min']:10.2e} {r['vv_max']:10.2e} "
                f"{r['subband_num']:8.2f} {r['subband_den']:8.2f}"
            )

        total_num = sum(r["subband_num"] for r in records)
        total_den = sum(r["subband_den"] for r in records)
        print(f"  total_num={total_num:.4f}  total_den={total_den:.4f}  ratio={total_num / total_den if total_den else float('nan'):.4f}")

    print(
        "\nDiagnosis: the 'smooth' test image is a quadratic bowl -- almost "
        "no high-frequency energy anywhere except a broad, slowly-varying "
        "gradient. Read the coeff_std and cu_cond columns above: on the "
        "finest levels (lvl 0-1), pyramid coefficient std and covariance "
        "conditioning tell you whether the subband is carrying real texture "
        "or is dominated by near-zero coefficients (cu_cond >> 1e6 signals "
        "a near-singular covariance, i.e. the GSM stage is where instability "
        "originates, not the pyramid decomposition or the channel-model "
        "stage downstream of it). See docs/vif_implementation.md for the "
        "recorded conclusion from this run's actual numbers."
    )


if __name__ == "__main__":
    print_parameter_diff_table()
    rows = run_three_way_comparison()
    if rows:
        report_pairwise_stats(rows)
    investigate_smooth_failure()
