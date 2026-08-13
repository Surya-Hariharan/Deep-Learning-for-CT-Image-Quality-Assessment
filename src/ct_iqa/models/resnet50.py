"""RadImageNet-pretrained ResNet50 quality-regression model.

OHASHI-SPECIFIED architecture:

    Input -> central crop -> 224x224 -> RadImageNet ResNet50 -> Dropout
          -> Fully Connected(1) -> Sigmoid -> quality score

This module defines the specification as data (``ModelSpec``) and DELIBERATELY
DOES NOT build or train a real model: no RadImageNet checkpoint has been
obtained, no deep-learning framework is installed in this environment, and the
project brief prohibits training at this phase. ``build_model`` documents the
prerequisites and raises until they are satisfied, rather than silently
falling back to a substitute (e.g. ImageNet weights), which would no longer be
an Ohashi replication.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """The architecture as specified, with unresolved items left ``None``."""

    backbone: str = "resnet50"                    # OHASHI-SPECIFIED
    pretrained_weights: str = "radimagenet"        # OHASHI-SPECIFIED
    input_size: int = 224                          # OHASHI-SPECIFIED
    crop_strategy: str = "central_crop"            # OHASHI-SPECIFIED
    head: tuple[str, ...] = ("dropout", "fully_connected_1", "sigmoid")  # OHASHI-SPECIFIED
    dropout_probability: float | None = None       # NOT SPECIFIED BY OHASHI
    freeze_backbone: bool | None = None            # NOT SPECIFIED BY OHASHI
    grayscale_to_rgb_method: str = "channel_replication"  # PROJECT ADAPTATION


DEFAULT_SPEC = ModelSpec()


class ModelNotBuildable(RuntimeError):
    """Raised by build_model() while prerequisites are unmet."""


def unresolved_prerequisites(spec: ModelSpec = DEFAULT_SPEC) -> list[str]:
    """Everything that must be resolved before a real model can be built."""
    blockers = [
        "RadImageNet ResNet50 checkpoint not obtained (weights_path unset)",
        "No deep-learning framework (tensorflow/torch) installed in this environment",
    ]
    if spec.dropout_probability is None:
        blockers.append("dropout_probability NOT SPECIFIED BY OHASHI -- must become a PROJECT ADAPTATION before training")
    if spec.freeze_backbone is None:
        blockers.append("freeze_backbone NOT SPECIFIED BY OHASHI -- must become a PROJECT ADAPTATION before training")
    return blockers


def build_model(spec: ModelSpec = DEFAULT_SPEC):
    """Construct the trainable model. NOT IMPLEMENTED at this phase.

    Raises ``ModelNotBuildable`` unconditionally right now, listing every
    unresolved prerequisite, so this function can be called safely from
    architecture-validation code without ever training anything by accident.
    """
    blockers = unresolved_prerequisites(spec)
    raise ModelNotBuildable(
        "Model construction is not authorized at this phase. Unresolved:\n  - "
        + "\n  - ".join(blockers)
    )
