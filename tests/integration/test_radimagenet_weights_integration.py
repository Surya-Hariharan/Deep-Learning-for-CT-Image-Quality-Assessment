"""Integration tests for RadImageNet weight conversion/loading against real,
locally-downloaded weight files.

Split out of `tests/unit/test_radimagenet_weights.py` (see
docs/internal/repository_architecture_audit.md, section 3): both weight files here
are gitignored and not part of the repository, so these tests are skipped
automatically unless a contributor has fetched/converted the real files
locally (see weights/pretrained/radimagenet/resnet50/README.md).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ct_iqa.models.ohashi_resnet50 import OhashiResNet50
from ct_iqa.models.resnet50 import ResNet50Backbone

REPO_ROOT = Path(__file__).resolve().parents[2]
_WEIGHTS_DIR = REPO_ROOT / "weights" / "pretrained" / "radimagenet" / "resnet50"
H5_PATH = _WEIGHTS_DIR / "RadImageNet-ResNet50_notop.h5"
CONVERTED_PT_PATH = _WEIGHTS_DIR / "radimagenet_resnet50_backbone.pt"

requires_h5 = pytest.mark.skipif(
    not H5_PATH.exists(), reason=f"RadImageNet .h5 not present at {H5_PATH} (gitignored, local-only)"
)
requires_converted_pt = pytest.mark.skipif(
    not CONVERTED_PT_PATH.exists(),
    reason=(
        f"converted checkpoint not present at {CONVERTED_PT_PATH}; "
        f"run `python -m ct_iqa.models.radimagenet_weights`"
    ),
)


@requires_h5
def test_conversion_script_produces_full_key_coverage():
    from ct_iqa.models.radimagenet_weights import convert

    converted = convert(H5_PATH)
    backbone_keys = set(ResNet50Backbone(in_channels=3).state_dict().keys())
    learned_keys = {k for k in backbone_keys if not k.endswith("num_batches_tracked")}

    assert set(converted.keys()) == learned_keys
    for k in learned_keys:
        expected_shape = ResNet50Backbone(in_channels=3).state_dict()[k].shape
        assert converted[k].shape == expected_shape, f"shape mismatch at {k}"


@requires_converted_pt
def test_real_radimagenet_weights_load_cleanly():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    reference = copy.deepcopy(model.backbone.state_dict())

    report = model.load_radimagenet_weights(str(CONVERTED_PT_PATH))

    assert report.loaded is True
    assert len(report.matched_keys) == 265
    assert report.unexpected_keys == []
    assert all(k.endswith("num_batches_tracked") for k in report.missing_keys)
    assert model.verify_backbone_loaded(reference) is True


@requires_converted_pt
def test_forward_pass_with_real_radimagenet_weights():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    model.load_radimagenet_weights(str(CONVERTED_PT_PATH))
    model.eval()

    x = torch.randn(4, 1, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4,)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    assert not torch.isnan(out).any()
