"""RadImageNet-pretrained ResNet50 quality-regression model.

OHASHI-SPECIFIED architecture:

    Input -> central crop -> 224x224 -> RadImageNet ResNet50 (fine-tuned)
          -> Global Average Pooling -> Dropout -> Fully Connected(1)
          -> Sigmoid -> quality score

Trained against ``vif_score`` (the synthetic-stage target) with MSE loss,
Adam optimizer, batch size 64, 30 epochs, and a learning-rate sweep over
{1e-2, 1e-3, 1e-4, 1e-5} selected by validation MSE — see
``src/ct_iqa/training/trainer.py``. All OHASHI-SPECIFIED per
``docs/ohashi_methodology.md``.

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

# PROJECT-ADAPTATION baseline for dropout_probability, resolving U-M01
# (docs/research_decisions.md). Ohashi's own paper states only that "a
# Dropout layer was added ... to prevent overfitting" -- no rate. This value
# is NOT read from Ohashi's paper (not OHASHI-SPECIFIED) and was NOT chosen
# by any validation/hyperparameter search (explicitly out of scope until the
# baseline architecture is established, per the decision record).
#
# Source: Mei X, Liu Z, Robson PM, et al. "RadImageNet: An Open Radiologic
# Deep Learning Research Dataset for Effective Transfer Learning."
# Radiology: Artificial Intelligence 2022;4(5):e210315 -- the paper that
# produced the exact RadImageNet checkpoint this project's ResNet50 backbone
# is pretrained from. Its own transfer-learning head is architecturally
# identical in shape to Ohashi's (global average pooling -> dropout ->
# output layer) and states explicitly: "A global average pooling layer, a
# dropout layer at a rate of 0.5, and the output layer activated by the
# softmax function were added after the CNNs." This is a
# REFERENCE-IMPLEMENTATION-CONVENTION value from the checkpoint's own
# origin paper, adopted here as a PROJECT-ADAPTATION baseline because
# Ohashi's paper is silent. It is not verified that Ohashi's own transfer
# learning retained this exact value; no Ohashi source code, supplementary
# material, or public repository was found (searched 2026-08-14) to confirm
# or contradict it. Full writeup, alternatives considered, and the
# resolution rationale: docs/research_decisions.md, decision A-22.
DEFAULT_DROPOUT_PROBABILITY = 0.5


@dataclass(frozen=True)
class ModelSpec:
    """The architecture as specified. Every field is now resolved to an
    explicit value -- see docs/research_decisions.md decisions S-01 (backbone
    fine-tuning) and A-22 (dropout baseline) for how ``freeze_backbone`` and
    ``dropout_probability`` were resolved, and why neither resolution is
    read as an Ohashi-specified fact."""

    backbone: str = "resnet50"                    # OHASHI-SPECIFIED
    pretrained_weights: str = "radimagenet"        # OHASHI-SPECIFIED
    input_size: int = 224                          # OHASHI-SPECIFIED
    crop_strategy: str = "central_crop"            # OHASHI-SPECIFIED
    head: tuple[str, ...] = (
        "global_average_pooling", "dropout", "fully_connected_1", "sigmoid",
    )  # OHASHI-SPECIFIED (GAP is implicit in Ohashi's own head description
       # and explicit in RadImageNet's own recipe this backbone comes from)
    dropout_probability: float = DEFAULT_DROPOUT_PROBABILITY  # PROJECT-ADAPTATION (A-22), resolves U-M01
    freeze_backbone: bool = False                  # OHASHI-SPECIFIED (S-01): confirmed fine-tuned, not frozen --
                                                     # "By fine-tuning these pre-trained models with our IQA dataset"
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
