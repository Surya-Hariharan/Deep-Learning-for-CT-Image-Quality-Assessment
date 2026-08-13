"""VIF (Visual Information Fidelity) — the OHASHI-SPECIFIED Full-Reference
training target for the synthetic degradation dataset.

Two variants are available and are never aliased to one another:
``vif_p`` (VIFp, pixel-domain) and ``vif_wavelet`` (full wavelet-domain/GSM
VIF, decision A-15 -- the formulation Ohashi's citation actually points to).
See docs/vif_implementation.md for the wavelet variant's validation status
before using it for dataset labelling.
"""

from ct_iqa.vif.labeling import VIFVariant, compute_vif
from ct_iqa.vif.metric import SIGMA_NSQ, vif_p
from ct_iqa.vif.wavelet import SIGMA_NSQ as WAVELET_SIGMA_NSQ
from ct_iqa.vif.wavelet import vif_wavelet

__all__ = [
    "compute_vif",
    "VIFVariant",
    "vif_p",
    "SIGMA_NSQ",
    "vif_wavelet",
    "WAVELET_SIGMA_NSQ",
]
