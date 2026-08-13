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

from typing import Literal

import numpy as np

from ct_iqa.vif.metric import vif_p
from ct_iqa.vif.wavelet import vif_wavelet

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
