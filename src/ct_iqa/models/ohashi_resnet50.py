"""Ohashi-style RadImageNet-ResNet50 for no-reference CT image quality regression.

This is section B of the architecture split: modifications specific to
Ohashi et al., layered on top of the generic `ResNet50Backbone`
(`ct_iqa.models.resnet50`).

What the paper explicitly specifies (implemented here):
    - Backbone: ResNet50
    - Input: 224x224
    - Pretraining: RadImageNet (NOT ImageNet)
    - Classification head replaced with a regression head:
          features -> Dropout -> Dense(1) -> Sigmoid
    - Sigmoid replaces the original Softmax output activation

What the paper does NOT specify, and is therefore an explicit, documented
implementation choice rather than a claimed paper detail:
    - The exact Dropout probability. `dropout_p` has no built-in default
      here that pretends to be "the Ohashi value" -- it must be passed
      explicitly by the caller (see `OhashiResNet50.__init__`).
    - Whether input channels are grayscale or replicated to 3 channels.
      The LDCT-IQAC images used in this project are single-channel
      (see `ct_iqa.data.ldct_iqac`), so this module accepts single-channel
      input and internally replicates it to 3 channels before the stem,
      matching the common convention for applying RGB-pretrained backbones
      (including RadImageNet's own training convention) to grayscale
      medical images. This is an implementation choice, not a paper detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from ct_iqa.models.resnet50 import ResNet50Backbone

logger = logging.getLogger(__name__)

INPUT_SIZE = 224  # Explicit in the paper.


@dataclass
class WeightLoadReport:
    """Result of attempting to load RadImageNet backbone weights."""

    weights_path: str | None
    loaded: bool
    matched_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        if not self.loaded:
            return f"RadImageNet weights NOT loaded. {self.note}"
        return (
            f"RadImageNet weights loaded from {self.weights_path}: "
            f"{len(self.matched_keys)} matched, "
            f"{len(self.missing_keys)} missing, "
            f"{len(self.unexpected_keys)} unexpected. {self.note}"
        ).strip()


class OhashiResNet50(nn.Module):
    """ResNet50 backbone + Ohashi-style IQA regression head.

        input (B, 1, 224, 224)
            -> replicate to 3 channels
            -> ResNet50Backbone -> (B, 2048)
            -> Dropout(p=dropout_p)
            -> Linear(2048, 1)
            -> Sigmoid
            -> (B,)   quality prediction in [0, 1]

    The [0, 1] sigmoid output is a normalized score. Mapping it to the
    LDCT-IQAC [0, 4] score range is a dataset-level decision handled in
    `ct_iqa.data.ldct_iqac` (NOT in this model), per Part 8 of the task:
    the model's output space and the dataset's label space are kept
    explicitly separate and documented independently.

    Parameters
    ----------
    dropout_p:
        Dropout probability before the final Dense layer. The paper states
        a Dropout layer is present but does not specify its probability.
        This is REQUIRED (no silent default) to force the caller to make
        and record an explicit, documented choice.
    in_channels:
        Number of channels in the raw input image before internal 3-channel
        replication. LDCT-IQAC images are single-channel float TIFFs, so
        this defaults to 1.
    """

    def __init__(self, dropout_p: float, in_channels: int = 1) -> None:
        super().__init__()
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")

        self.in_channels = in_channels
        self.backbone = ResNet50Backbone(in_channels=3)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(ResNet50Backbone.OUT_FEATURES, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channel(s), got shape {tuple(x.shape)}"
            )
        if x.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"Expected {INPUT_SIZE}x{INPUT_SIZE} input, got spatial shape {tuple(x.shape[-2:])}"
            )

        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.fc(features)
        prediction = self.sigmoid(logits)
        return prediction.squeeze(-1)

    def load_radimagenet_weights(
        self, weights_path: str | None, strict: bool = False
    ) -> WeightLoadReport:
        """Load RadImageNet-pretrained backbone weights from a local checkpoint.

        Weights are NEVER downloaded automatically. `weights_path` must
        point to a local state-dict file (`.pt`/`.pth`) whose keys are
        expected to align with `self.backbone`'s `state_dict()` keys
        (optionally prefixed with `backbone.`). Produce this file from the
        official Keras RadImageNet-ResNet50 `.h5` release with
        `python -m ct_iqa.models.radimagenet_weights` -- see that module's
        docstring for the exact Keras-to-PyTorch mapping (kernel
        transpose, conv-bias-into-BN-running_mean folding) and for why
        `ct_iqa.models.resnet50.ResNet50Backbone` places its stride-2 convs
        on the 1x1 reduce conv rather than the 3x3 conv (matching Keras'
        ResNet50, not torchvision's "v1.5" convention) -- without that,
        weight *shapes* would still match and loading would silently
        succeed while producing a semantically wrong feature extractor.

        This method loads ONLY backbone weights -- it never touches
        `self.fc` (the regression head), which by construction cannot come
        from RadImageNet (RadImageNet's own head is a 1000-way/multi-class
        classifier, not this project's regression head).

        If `weights_path` is None, this method does NOT fall back to any
        other pretrained weights (e.g. torchvision ImageNet weights) -- it
        reports that the backbone remains randomly initialized. Silently
        substituting ImageNet weights for RadImageNet weights would
        invalidate the replication and is explicitly disallowed.

        On a successful load, every `nn.BatchNorm2d` in the backbone has its
        `eps` set to `1e-3` (Keras `BatchNormalization`'s default), matching
        the epsilon the loaded running_mean/running_var were computed under
        -- PyTorch's default (`1e-5`) would otherwise introduce a small,
        avoidable numerical mismatch relative to the original Keras model.
        """
        if weights_path is None:
            note = (
                "No weights_path provided. Backbone remains randomly initialized. "
                "RadImageNet weights were not found bundled in this repository as of "
                "this audit -- obtain them separately and pass an explicit path."
            )
            logger.warning(note)
            return WeightLoadReport(weights_path=None, loaded=False, note=note)

        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Accept either raw backbone keys or keys prefixed with "backbone."
        stripped = {}
        for k, v in state_dict.items():
            stripped[k[len("backbone.") :] if k.startswith("backbone.") else k] = v

        backbone_keys = set(self.backbone.state_dict().keys())
        matched = {k: v for k, v in stripped.items() if k in backbone_keys}
        missing = sorted(backbone_keys - matched.keys())
        unexpected = sorted(set(stripped.keys()) - backbone_keys)

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict RadImageNet weight load failed: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys."
            )

        self.backbone.load_state_dict(matched, strict=False)

        if matched:
            # Match Keras BatchNormalization's default epsilon, which the
            # loaded running_mean/running_var were computed under.
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eps = 1e-3

        # `num_batches_tracked` is a non-learned inference bookkeeping
        # counter (not present in the converted checkpoint by design -- see
        # ct_iqa.models.radimagenet_weights); left at its
        # freshly-initialized value of 0, it does not indicate an
        # incomplete weight load. Report it separately from genuinely
        # missing learned parameters (weight/bias/running_mean/running_var).
        missing_learned = [k for k in missing if not k.endswith("num_batches_tracked")]
        missing_counters = [k for k in missing if k.endswith("num_batches_tracked")]

        note = ""
        if not matched:
            note = "0 keys matched -- weights file is almost certainly incompatible."
        elif missing_learned:
            note = f"{len(missing_learned)} learned backbone keys were left at random init."
        if missing_counters:
            note = (note + " " if note else "") + (
                f"{len(missing_counters)} num_batches_tracked counters left at 0 "
                f"(expected -- not learned parameters)."
            )

        report = WeightLoadReport(
            weights_path=weights_path,
            loaded=bool(matched),
            matched_keys=sorted(matched.keys()),
            missing_keys=missing,
            unexpected_keys=unexpected,
            note=note,
        )
        logger.info(report.summary())
        return report

    def verify_backbone_loaded(self, reference_state_dict: dict) -> bool:
        """Sanity-check that the current backbone weights differ from a reference (e.g. a fresh random init).

        Returns True if at least one parameter tensor differs from the
        reference, which is a necessary (not sufficient) check that
        `load_radimagenet_weights` actually changed something.
        """
        current = self.backbone.state_dict()
        for key, ref_tensor in reference_state_dict.items():
            if key not in current:
                continue
            if not torch.equal(current[key], ref_tensor):
                return True
        return False
