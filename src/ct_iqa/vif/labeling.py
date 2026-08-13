"""VIF labelling dispatch for the synthetic degradation dataset.

OHASHI-SPECIFIED: VIF is the labelling method. NOT SPECIFIED BY OHASHI: which
VIF *variant*. Two families are in common use and they do not agree
numerically:

  * VIF   -- Sheikh & Bovik's original wavelet-domain (steerable pyramid) formulation.
  * VIFp  -- the pixel-domain approximation from the same authors' reference code,
             which is what most Python packages ship as "vifp" (implemented in
             ``ct_iqa.vif.metric``).

``compute_vif`` requires the caller to name the variant, raising for the
wavelet variant until it is implemented and the choice is confirmed against
the paper -- it never silently substitutes one for the other.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ct_iqa.vif.metric import vif_p

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
        raise NotImplementedError(
            "Wavelet-domain VIF is not implemented. The variant used by Ohashi et al. "
            "is NOT SPECIFIED BY OHASHI in the project brief -- resolve it against the "
            "paper before labelling, and record the outcome in docs/research_decisions.md."
        )
    raise ValueError(f"unknown VIF variant: {variant!r}")
