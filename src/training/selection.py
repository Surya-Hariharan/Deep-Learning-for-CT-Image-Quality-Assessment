"""Checkpoint-selection policies, kept separate from the training loop.

Selection uses VALIDATION metrics only; the engine never passes test or calibration data here.
A new epoch replaces the incumbent only on a strict improvement, so ties keep the earlier epoch.
A NaN metric (e.g. PLCC of constant predictions) is never an improvement.
Epochs are 1-based throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# metric name -> (key in the per-epoch metrics dict, higher_is_better)
_POLICIES: dict[str, tuple[str, bool]] = {
    "val_plcc": ("val_plcc", True),
    "val_srocc": ("val_srocc", True),
    "val_loss": ("val_loss", False),
}


@dataclass
class Selector:
    policy: str
    best_epoch: int = 0                       # 1-based; 0 means nothing selected yet
    best_value: float = field(default=math.nan)

    def __post_init__(self) -> None:
        if self.policy != "final_epoch" and self.policy not in _POLICIES:
            raise ValueError(f"unknown selection policy {self.policy!r}")

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Return True if `epoch` becomes the selected checkpoint."""
        if self.policy == "final_epoch":
            self.best_epoch, self.best_value = epoch, math.nan
            return True
        key, higher = _POLICIES[self.policy]
        value = metrics[key]
        if value is None or math.isnan(value):
            return False
        improved = self.best_epoch == 0 or (value > self.best_value if higher else value < self.best_value)
        if improved:
            self.best_epoch, self.best_value = epoch, value
        return improved
