"""Unit tests for RadImageNet weight loading (`OhashiResNet50.load_radimagenet_weights`).

These exercise the loading mechanics -- key matching, missing/unexpected
reporting, BN epsilon correction, head isolation -- against a hand-built
state dict shaped like a real converted checkpoint. No external files are
needed; these always run.

Real-file integration tests (the actual conversion script and the actual
downloaded RadImageNet `.h5`) live in
`tests/integration/test_radimagenet_weights_integration.py` and are skipped
if the weight files aren't present locally (gitignored, not part of the
repository).
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ct_iqa.models.ohashi_resnet50 import OhashiResNet50
from ct_iqa.models.resnet50 import ResNet50Backbone


def _fake_pretrained_state_dict() -> dict[str, torch.Tensor]:
    """A state dict shaped exactly like a real converted RadImageNet checkpoint:
    every learned backbone key (weight/bias/running_mean/running_var), no
    `num_batches_tracked` (matching `ct_iqa.models.radimagenet_weights.convert`'s
    output), with values distinct from a fresh random init.
    """
    backbone = ResNet50Backbone(in_channels=3)
    state_dict = backbone.state_dict()
    return {
        k: v.clone() + 1.0  # shift every value so it's provably different from a fresh init
        for k, v in state_dict.items()
        if not k.endswith("num_batches_tracked")
    }


def test_matching_state_dict_loads_with_no_unexpected_keys(tmp_path):
    fake_weights = _fake_pretrained_state_dict()
    path = tmp_path / "fake_radimagenet.pt"
    torch.save(fake_weights, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    report = model.load_radimagenet_weights(str(path))

    assert report.loaded is True
    assert len(report.matched_keys) == len(fake_weights)
    assert report.unexpected_keys == []
    # only num_batches_tracked counters should be "missing"
    assert all(k.endswith("num_batches_tracked") for k in report.missing_keys)
    assert len(report.missing_keys) == 53  # one per BatchNorm2d in the backbone


def test_loaded_weights_actually_change_the_backbone(tmp_path):
    fake_weights = _fake_pretrained_state_dict()
    path = tmp_path / "fake_radimagenet.pt"
    torch.save(fake_weights, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    reference = copy.deepcopy(model.backbone.state_dict())

    model.load_radimagenet_weights(str(path))

    assert model.verify_backbone_loaded(reference) is True
    # and the values match what we asked to load, not just "something changed"
    loaded_sd = model.backbone.state_dict()
    for k, v in fake_weights.items():
        assert torch.equal(loaded_sd[k], v)


def test_regression_head_untouched_by_backbone_load(tmp_path):
    fake_weights = _fake_pretrained_state_dict()
    path = tmp_path / "fake_radimagenet.pt"
    torch.save(fake_weights, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    fc_before = copy.deepcopy(model.fc.state_dict())

    model.load_radimagenet_weights(str(path))

    fc_after = model.fc.state_dict()
    for k in fc_before:
        assert torch.equal(fc_before[k], fc_after[k])


def test_bn_epsilon_set_to_keras_default_after_successful_load(tmp_path):
    fake_weights = _fake_pretrained_state_dict()
    path = tmp_path / "fake_radimagenet.pt"
    torch.save(fake_weights, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    # before loading, PyTorch's default eps (not Keras's) is in effect
    assert any(m.eps != 1e-3 for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d))

    model.load_radimagenet_weights(str(path))

    assert all(m.eps == 1e-3 for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d))


def test_bn_epsilon_unchanged_when_no_weights_loaded():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    model.load_radimagenet_weights(weights_path=None)
    assert all(m.eps == 1e-5 for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d))


def test_unexpected_keys_are_reported_not_silently_dropped(tmp_path):
    fake_weights = _fake_pretrained_state_dict()
    fake_weights["stage1.0.totally_bogus_key"] = torch.zeros(3)
    path = tmp_path / "fake_radimagenet.pt"
    torch.save(fake_weights, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    report = model.load_radimagenet_weights(str(path))

    assert "stage1.0.totally_bogus_key" in report.unexpected_keys


def test_genuinely_incompatible_checkpoint_reports_zero_matched(tmp_path):
    path = tmp_path / "bogus.pt"
    torch.save({"not.a.real.key": torch.zeros(3)}, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    report = model.load_radimagenet_weights(str(path))

    assert report.loaded is False
    assert report.matched_keys == []
    assert "incompatible" in report.note


def test_missing_weights_file_raises_actionable_filenotfounderror(tmp_path):
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    missing_path = str(tmp_path / "does_not_exist.pt")

    with pytest.raises(FileNotFoundError, match="configs/model.yaml"):
        model.load_radimagenet_weights(missing_path)


def test_strict_mode_raises_on_missing_or_unexpected(tmp_path):
    fake_weights = _fake_pretrained_state_dict()
    path = tmp_path / "fake_radimagenet.pt"
    torch.save(fake_weights, path)

    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    with pytest.raises(RuntimeError):
        # num_batches_tracked keys are always "missing" from a converted
        # checkpoint by design, so strict=True must raise here.
        model.load_radimagenet_weights(str(path), strict=True)
