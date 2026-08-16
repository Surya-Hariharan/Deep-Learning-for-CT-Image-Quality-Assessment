"""Inference helper for NGP-Net.

A thin wrapper around `NGPNet.ngpnet()` for growth prediction outside the
training loop. It does not add or change model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import NGPNet


@dataclass
class GrowthPrediction:
    ''' Outputs of a single NGP-Net inference call. '''

    image: Tensor    # (B,1,...) predicted scan
    mask: Tensor     # (B,1,...) hard nodule mask (argmax of logits)
    logits: Tensor   # (B,num_classes,...) segmentation logits (soft)
    field: Tensor    # (B,spatial_dim,...) deformation field


class NGPNetPredictor:
    ''' Loads an `NGPNet` and runs no-grad growth prediction. '''

    def __init__(self, model: NGPNet, device: str = "cpu"):
        self.device = device
        self.model = model.to(device).eval()

    @torch.no_grad()
    def predict(self,
                earlier_scan: Tensor,
                later_scan: Tensor,
                observed_interval: Tensor,
                target_interval: Tensor) -> GrowthPrediction:
        ''' Args:
        * `earlier_scan`, `later_scan`: (B,1,...) images at t0, t1.
        * `observed_interval`: months from t0 to t1, shape (B,).
        * `target_interval`: months from t1 to the prediction target, shape (B,).
        '''
        im0 = earlier_scan.to(self.device)
        im1 = later_scan.to(self.device)
        tm0 = observed_interval.to(self.device)
        tm1 = target_interval.to(self.device)

        image, logits, field = self.model.ngpnet(im0, im1, tm0, tm1)
        mask = logits.argmax(dim=1, keepdim=True)
        return GrowthPrediction(image=image, mask=mask, logits=logits, field=field)
