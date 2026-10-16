"""Output heads for the shared ResNet50 backbone.

Heads are described here as plain layer factories and pure functions so that the model
classes in `ct_iqa.models.registry` can register the layers under the attribute names
`dropout`, `fc` (and `fc_logvar` for the Gaussian head). Those names keep the
regression model's `state_dict` keys identical to the historical `OhashiResNet50`
(`backbone.*`, `fc.weight`, `fc.bias`), which is what lets checkpoints 001-003 load.

Regression head (Experiments A and B):
    features(2048) -> Dropout(p) -> Linear(2048, 1) -> Sigmoid -> mean in [0, 1]

Gaussian head (Experiment C), heteroscedastic mean / log-variance:
    features(2048) -> Dropout(p) -> Linear(2048, 1) -> Sigmoid            -> mean mu_n in [0, 1]
    features(2048) -> Dropout(p) -> Linear(2048, 1) -> clamp(LOGVAR_MIN, LOGVAR_MAX) -> s_n
All quantities are in the NORMALIZED [0, 1] score space; variance = exp(s_n). The clamp
keeps exp(-s_n) and exp(s_n) finite. This is a single deterministic head, so it models
aleatoric (data) uncertainty only, not epistemic uncertainty.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURE_DIM = 2048

# Bounds on the predicted log-variance in the normalized [0, 1] score space.
# exp(-10) ~ 4.5e-5 (std ~ 0.007, i.e. 0.03 score units) and exp(2) ~ 7.4 (std ~ 2.7, far wider than
# the [0, 1] range). They only guard numerical stability; they are not tuned.
LOGVAR_MIN = -10.0
LOGVAR_MAX = 2.0


def build_head_layers(kind: str, dropout_p: float) -> dict[str, nn.Module]:
    """Return the named layers for a head kind: "regression" or "gaussian"."""
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
    layers: dict[str, nn.Module] = {
        "dropout": nn.Dropout(p=dropout_p),
        "fc": nn.Linear(FEATURE_DIM, 1),
    }
    if kind == "gaussian":
        layers["fc_logvar"] = nn.Linear(FEATURE_DIM, 1)
    elif kind != "regression":
        raise ValueError(f"Unknown head kind {kind!r}; expected 'regression' or 'gaussian'")
    return layers


def clamp_log_var(raw: torch.Tensor) -> torch.Tensor:
    """Bound the raw log-variance output so exp(+/- s) stays finite."""
    return torch.clamp(raw, min=LOGVAR_MIN, max=LOGVAR_MAX)
