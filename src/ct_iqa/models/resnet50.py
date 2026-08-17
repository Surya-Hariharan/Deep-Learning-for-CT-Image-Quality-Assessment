"""Standard ResNet50 backbone (He et al., 2015), implemented from scratch.

`torchvision` is not installed in this project's environment, so the
backbone is implemented directly in PyTorch rather than imported. This is
section A of the architecture split described in the project task:
generic ResNet50 knowledge, not anything specific to Ohashi et al.

Architecture (standard ResNet50, bottleneck variant):

    Stem:      7x7 conv, stride 2, 64 channels -> BN -> ReLU
               3x3 max pool, stride 2
    Stage 1:   3 bottleneck blocks,  64 ->  256 channels, stride 1
    Stage 2:   4 bottleneck blocks, 128 ->  512 channels, stride 2
    Stage 3:   6 bottleneck blocks, 256 -> 1024 channels, stride 2
    Stage 4:   3 bottleneck blocks, 512 -> 2048 channels, stride 2
    Head:      global average pool -> 2048-d feature vector

Each bottleneck block is 1x1 conv (reduce) -> 3x3 conv -> 1x1 conv (expand,
x4 channel multiplier), with a skip connection (identity, or a 1x1
projection conv when the shape changes) added before the final ReLU. The
stride for each stage (when > 1) is applied on the 3x3 conv of the first
block in that stage, matching the standard "ResNet v1.5" convention used by
most modern reference implementations (including torchvision).

This module intentionally stops at the pooled 2048-d feature vector. The
original 1000-way ImageNet classification head is NOT implemented here —
task-specific heads (e.g. the Ohashi IQA regression head) live in
`ct_iqa.models.ohashi_resnet50` and are built on top of this backbone.
"""

from __future__ import annotations

import torch
from torch import nn

_EXPANSION = 4


class Bottleneck(nn.Module):
    """Standard ResNet bottleneck residual block (1x1 -> 3x3 -> 1x1, x4 expansion)."""

    expansion = _EXPANSION

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


def _make_stage(in_channels: int, mid_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers = [Bottleneck(in_channels, mid_channels, stride=stride)]
    out_channels = mid_channels * _EXPANSION
    for _ in range(num_blocks - 1):
        layers.append(Bottleneck(out_channels, mid_channels, stride=1))
    return nn.Sequential(*layers)


class ResNet50Backbone(nn.Module):
    """Standard ResNet50 feature extractor, ending in global-average-pooled 2048-d features.

    No classification head is included. `forward` returns a
    `(batch, 2048)` tensor of pooled features.
    """

    OUT_FEATURES = 2048
    BLOCKS_PER_STAGE = (3, 4, 6, 3)

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        b1, b2, b3, b4 = self.BLOCKS_PER_STAGE
        self.stage1 = _make_stage(64, 64, b1, stride=1)
        self.stage2 = _make_stage(256, 128, b2, stride=2)
        self.stage3 = _make_stage(512, 256, b3, stride=2)
        self.stage4 = _make_stage(1024, 512, b4, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x)
        return torch.flatten(x, 1)
