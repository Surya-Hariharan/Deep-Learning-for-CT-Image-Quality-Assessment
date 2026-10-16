"""Pretrained-weight loading with checksum verification.

`load_backbone_weights` loads a converted RadImageNet state-dict into a
`ResNet50Backbone`. Unlike the historical `OhashiResNet50.load_radimagenet_weights`, it
NEVER modifies BatchNorm epsilon: epsilon is a construction-time property
(`bn_eps`, see docs/decisions/0002-batchnorm-and-preprocessing.md).

The Keras `.h5` -> PyTorch conversion itself lives in `ct_iqa.models.radimagenet_weights`
(unchanged); it is re-exported here as `convert`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from ct_iqa.models.radimagenet_weights import convert  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

# SHA-256 of the files used by the historical experiments 001-003 (recorded 2026-10-16, see
# weights/pretrained/radimagenet/resnet50/CHECKSUMS.sha256).
RADIMAGENET_H5_SHA256 = "1f1d1c9fe6ab6980daa5e528145fe0db249c55415f8b3e1f3e3b1624d184f159"
RADIMAGENET_BACKBONE_PT_SHA256 = "45cba0d53f1d0ec1e1e2ae569a985b56a21da71c4a7271328b01ed5ea9a0f785"


@dataclass
class WeightLoadReport:
    weights_path: str | None
    loaded: bool
    sha256: str | None = None
    sha256_verified: bool = False
    matched_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def missing_learned_keys(self) -> list[str]:
        return [k for k in self.missing_keys if not k.endswith("num_batches_tracked")]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_backbone_weights(
    backbone: nn.Module,
    weights_path: str | Path,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> WeightLoadReport:
    """Load a converted state-dict into `backbone`; verify the file's SHA-256 when given.

    Raises `FileNotFoundError` for a missing file (no silent fallback to ImageNet or to
    random initialization), `ValueError` on a checksum mismatch, and, when `strict`,
    `RuntimeError` if any learned key is missing or any key is unexpected.
    `num_batches_tracked` counters are not part of the converted file and are reported
    separately; they are bookkeeping, not learned parameters.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found at {str(path)!r}. Obtain and convert them "
            "(see weights/pretrained/radimagenet/resnet50/README.md) or set "
            "model.pretrained_weights to null to run with a random-init backbone."
        )
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {str(path)!r}: expected {expected_sha256}, found {digest}"
        )

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("backbone."):] if k.startswith("backbone.") else k): v for k, v in state.items()}

    expected_keys = set(backbone.state_dict().keys())
    matched = {k: v for k, v in state.items() if k in expected_keys}
    missing = sorted(expected_keys - matched.keys())
    unexpected = sorted(set(state) - expected_keys)
    report = WeightLoadReport(
        weights_path=str(path),
        loaded=bool(matched),
        sha256=digest,
        sha256_verified=expected_sha256 is not None,
        matched_keys=sorted(matched),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
    if strict and (report.missing_learned_keys or unexpected or not matched):
        raise RuntimeError(
            f"Strict weight load failed: {len(report.missing_learned_keys)} learned keys missing, "
            f"{len(unexpected)} unexpected, {len(matched)} matched."
        )
    backbone.load_state_dict(matched, strict=False)
    logger.info(
        "Loaded %d tensors from %s (sha256 %s, %s)",
        len(matched), path, digest[:12], "verified" if report.sha256_verified else "not checked",
    )
    return report
