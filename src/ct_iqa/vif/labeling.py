"""VIF labelling dispatch for the synthetic degradation dataset.

OHASHI-SPECIFIED: VIF is the labelling method, and (per decision S-02 in
docs/research_decisions.md) the cited formulation is Sheikh & Bovik's full
wavelet-domain/GSM VIF -- not VIFp. Two families exist and they do not agree
numerically:

  * vif_wavelet -- the cited full wavelet-domain (steerable pyramid + GSM)
    formulation, implemented in ``ct_iqa.vif.wavelet`` per decision A-15.
    NOT yet used for dataset labelling -- see docs/vif_implementation.md
    for its current validation status before relying on it.
  * vifp -- the pixel-domain approximation from the same authors' reference
    code, which is what most Python packages ship as "vifp" (implemented in
    ``ct_iqa.vif.metric``). Kept available, unchanged, never silently
    substituted for the wavelet variant.

``compute_vif`` requires the caller to name the variant explicitly -- it
never defaults or silently substitutes one for the other.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from ct_iqa.vif.metric import vif_p
from ct_iqa.vif.wavelet import vif_wavelet, vif_wavelet_diagnostics

VIFVariant = Literal["vifp", "vif_wavelet"]


def compute_vif(
    reference: np.ndarray,
    distorted: np.ndarray,
    variant: VIFVariant,
) -> float:
    """Compute VIF under an explicitly named variant.

    ``variant`` has no default on purpose: choosing one silently would bake an
    unverified methodological decision into every label in the dataset.
    """
    if variant == "vifp":
        return vif_p(reference, distorted)
    if variant == "vif_wavelet":
        return vif_wavelet(reference, distorted)
    raise ValueError(f"unknown VIF variant: {variant!r}")


# Diagnostic flag thresholds -- FLAGS, not corrections (decision A-19,
# docs/research_decisions.md: "flag, retain, never silently alter"). Values
# unchanged from the 30- and 100-reference pilots
# (docs/vif_pilot_report.md, docs/vif_full_grid_pilot_report.md) so
# production labels are directly comparable to the validated pilot runs.
CU_COND_FLAG_THRESHOLD = 1e12
G_FLAG_THRESHOLD = 50.0
VIF_UPPER_FLAG_THRESHOLD = 1.5


class VIFWithDiagnostics(NamedTuple):
    vif_score: float
    diagnostic_status: str
    max_channel_gain: float
    covariance_condition_number: float


def _classify_diagnostics(records: list[dict], vif_score: float) -> tuple[str, float, float]:
    flags = []
    if not np.isfinite(vif_score):
        flags.append("NON_FINITE_VIF")
    elif vif_score < 0:
        flags.append("NEGATIVE_VIF")
    elif vif_score > VIF_UPPER_FLAG_THRESHOLD:
        flags.append("VIF_ABOVE_EXPECTED_RANGE")

    max_cu_cond = max((r["cu_condition_number"] for r in records), default=float("nan"))
    max_g = max((r["g_max"] for r in records), default=float("nan"))

    if np.isfinite(max_cu_cond) and max_cu_cond > CU_COND_FLAG_THRESHOLD:
        flags.append("ILL_CONDITIONED_COVARIANCE")
    if np.isfinite(max_g) and max_g > G_FLAG_THRESHOLD:
        flags.append("UNSTABLE_CHANNEL_GAIN")
    if any(not np.isfinite(r["ss_max"]) for r in records):
        flags.append("NON_FINITE_INTERMEDIATE")
    if any(r["ss_max"] <= 1e-10 for r in records):
        flags.append("NEAR_ZERO_VARIANCE_SUBBAND")

    return (";".join(flags) if flags else "ok", max_g, max_cu_cond)


def compute_vif_with_diagnostics(
    reference: np.ndarray,
    distorted: np.ndarray,
    *,
    variant: VIFVariant = "vif_wavelet",
    profile: str = "project",
) -> VIFWithDiagnostics:
    """Compute VIF plus the diagnostic fields the production manifest
    retains verbatim for every record (decision A-19): ``diagnostic_status``,
    ``max_channel_gain``, ``covariance_condition_number``.

    Only defined for ``variant="vif_wavelet"`` (the only variant with a
    per-subband diagnostic trace). Derives the score directly from
    :func:`vif_wavelet_diagnostics`'s per-subband totals rather than calling
    :func:`vif_wavelet` separately -- confirmed identical to 1e-9 precision
    in the 100-reference full-grid pilot
    (``docs/vif_full_grid_pilot_report.md``, "Runtime" section); only valid
    because the "project"/"reference_crosscheck" profiles aggregate via a
    plain sum.
    """
    if variant != "vif_wavelet":
        raise ValueError(
            f"compute_vif_with_diagnostics only supports variant='vif_wavelet', got {variant!r}"
        )
    try:
        records = vif_wavelet_diagnostics(reference, distorted, profile=profile)
    except ValueError as exc:
        return VIFWithDiagnostics(
            vif_score=float("nan"), diagnostic_status=f"COMPUTATION_ERROR:{exc}",
            max_channel_gain=float("nan"), covariance_condition_number=float("nan"),
        )
    total_num = sum(r["subband_num"] for r in records)
    total_den = sum(r["subband_den"] for r in records)
    score = total_num / total_den if total_den > 0 else float("nan")
    status, max_g, max_cu_cond = _classify_diagnostics(records, score)
    return VIFWithDiagnostics(
        vif_score=score, diagnostic_status=status,
        max_channel_gain=max_g, covariance_condition_number=max_cu_cond,
    )
