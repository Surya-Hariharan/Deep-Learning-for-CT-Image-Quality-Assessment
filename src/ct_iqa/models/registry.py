"""Model registry for the ResNet50 family.

Every registered model shares `ResNet50Backbone` (RadImageNet-compatible, ResNet "v1"
stride placement) and differs only in its head and output. No non-ResNet50 backbone is
registered. Construction is fully determined by `ModelConfig`: BatchNorm epsilon
(`bn_eps`) is applied at construction and is not changed by weight loading.

Models return a `ModelOutput` in the NORMALIZED [0, 1] score space; mapping to the
[0, 4] label scale is done by `ct_iqa.data.ldct_iqac.denormalize_score`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ct_iqa.config.schema import ConfigError, ModelConfig
from ct_iqa.models.backbone import ResNet50Backbone
from ct_iqa.models.heads import build_head_layers, clamp_log_var
from ct_iqa.models.weights import WeightLoadReport, load_backbone_weights


@dataclass
class ModelOutput:
    mean: torch.Tensor                      # (B,), in [0, 1]
    log_var: torch.Tensor | None = None     # (B,), normalized-space log-variance (Gaussian head only)


class ResNet50Model(nn.Module):
    """Shared backbone + head. Attribute names (`backbone`, `dropout`, `fc`, `fc_logvar`)
    are chosen so the regression variant's state_dict keys equal the historical
    `OhashiResNet50` keys."""

    def __init__(self, head_kind: str, dropout_p: float, in_channels: int, bn_eps: float,
                 image_size: int = 224) -> None:
        super().__init__()
        self.head_kind = head_kind
        self.in_channels = in_channels
        self.image_size = image_size
        self.backbone = ResNet50Backbone(in_channels=3, bn_eps=bn_eps)
        for name, layer in build_head_layers(head_kind, dropout_p).items():
            setattr(self, name, layer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channel(s), got shape {tuple(x.shape)}")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} input, got {tuple(x.shape[-2:])}")
        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.dropout(self.backbone(x))
        mean = self.sigmoid(self.fc(features)).squeeze(-1)
        if self.head_kind == "gaussian":
            return ModelOutput(mean=mean, log_var=clamp_log_var(self.fc_logvar(features)).squeeze(-1))
        return ModelOutput(mean=mean)


_REGISTRY: dict[str, str] = {"resnet50_regression": "regression", "resnet50_gaussian": "gaussian"}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg: ModelConfig, image_size: int = 224) -> tuple[ResNet50Model, WeightLoadReport | None]:
    """Build a registered model from config; load and checksum-verify pretrained weights if configured."""
    if cfg.name not in _REGISTRY:
        raise ConfigError(f"Unknown model {cfg.name!r}; registered: {list_models()}")
    model = ResNet50Model(_REGISTRY[cfg.name], cfg.dropout_p, cfg.in_channels, cfg.bn_eps, image_size)
    report = None
    if cfg.pretrained_weights is not None:
        report = load_backbone_weights(model.backbone, cfg.pretrained_weights,
                                       expected_sha256=cfg.pretrained_sha256, strict=True)
    return model, report
