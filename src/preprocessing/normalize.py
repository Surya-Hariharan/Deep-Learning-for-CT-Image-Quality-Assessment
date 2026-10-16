"""Pixel-value normalization for CT images.

Extracted from `ct_iqa.data.ldct_iqac` with no change in behavior. Maps the
LDCT-IQAC TIFFs' native [0, 1] float pixel range to RadImageNet's own
[-1, 1] convention (`(pixel - 127.5) * 2 / 255` for [0, 255] input,
algebraically identical to `2 * x - 1` once `pixel` is already scaled to
[0, 1]) -- confirmed from RadImageNet's official reference code
(`BMEII-AI/RadImageNet/pytorch_example.ipynb`); the Ohashi paper itself does
not state a normalization formula for its own preprocessing (see
docs/replication/deviations.md).

This is pixel-space normalization only -- it is unrelated to LABEL/score
normalization (`ct_iqa.data.ldct_iqac.normalize_score`), which maps the
LDCT-IQAC quality score onto the model's [0, 1] Sigmoid output space and is
dataset-specific, not a generic preprocessing concern.
"""

from __future__ import annotations

import numpy as np


def to_radimagenet_range(array: np.ndarray) -> np.ndarray:
    """Map pixel values from the source [0, 1] float range to [-1, 1]."""
    return 2.0 * array - 1.0
