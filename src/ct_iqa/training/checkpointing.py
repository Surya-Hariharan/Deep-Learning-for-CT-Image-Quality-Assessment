"""Checkpoint-selection policy.

NOT SPECIFIED BY OHASHI: whether the reported model is the best-validation-MSE
epoch or the last epoch. This module defines the policy as data and refuses to
save/load a real checkpoint until training is authorized and the policy is
resolved -- it must not silently default to one behaviour.
"""

from __future__ import annotations

from typing import Literal

CheckpointPolicy = Literal["best_validation_mse", "last_epoch"]

#: NOT SPECIFIED BY OHASHI -- left unset deliberately.
CONFIGURED_POLICY: CheckpointPolicy | None = None


class CheckpointingNotAuthorized(RuntimeError):
    pass


def save_checkpoint(*args, **kwargs) -> None:
    raise CheckpointingNotAuthorized(
        "Checkpoint saving is not authorized at this phase (no training has run). "
        "CONFIGURED_POLICY is also NOT SPECIFIED BY OHASHI and must be set first."
    )


def load_checkpoint(*args, **kwargs) -> None:
    raise CheckpointingNotAuthorized("No checkpoint exists yet -- training has not run.")
