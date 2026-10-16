"""Unit tests for the model registry, heads and BatchNorm epsilon handling.

All tests use SYNTHETIC random tensors and randomly initialized weights; they verify structure and
numerical behavior, not scientific performance.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ct_iqa.config.schema import ConfigError, ModelConfig
from ct_iqa.models.backbone import ResNet50Backbone
from ct_iqa.models.heads import LOGVAR_MAX, LOGVAR_MIN
from ct_iqa.models.registry import ModelOutput, build_model, list_models
from ct_iqa.models.weights import load_backbone_weights, sha256_file

EPS = 1.001e-5


def _cfg(name="resnet50_regression", **kw):
    return ModelConfig(name=name, dropout_p=0.5, bn_eps=kw.pop("bn_eps", EPS), **kw)


def _bn_eps(module: nn.Module) -> set[float]:
    return {m.eps for m in module.modules() if isinstance(m, nn.BatchNorm2d)}


def test_registry_lists_only_resnet50_family():
    assert list_models() == ["resnet50_gaussian", "resnet50_regression"]


def test_unknown_model_rejected():
    with pytest.raises(ConfigError):
        build_model(_cfg(name="densenet121"))


def test_bn_eps_applied_to_every_batchnorm_at_construction():
    model, _ = build_model(_cfg(bn_eps=EPS))
    assert _bn_eps(model) == {EPS}
    assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == 53


def test_backbone_default_eps_is_pytorch_default_for_legacy_callers():
    assert _bn_eps(ResNet50Backbone()) == {1e-5}


def test_eps_unchanged_by_weight_loading_and_checkpoint_reload(tmp_path):
    src, _ = build_model(_cfg(bn_eps=EPS))
    path = tmp_path / "synthetic_backbone.pt"
    torch.save(src.backbone.state_dict(), path)
    digest = sha256_file(path)

    model, report = build_model(_cfg(bn_eps=EPS, pretrained_weights=str(path), pretrained_sha256=digest))
    assert report is not None and report.sha256_verified and report.sha256 == digest
    assert _bn_eps(model) == {EPS}                       # loading must not touch eps

    # Rebuilding for evaluation from the SAME config gives the same eps (no train/eval drift).
    rebuilt, _ = build_model(_cfg(bn_eps=EPS))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert _bn_eps(rebuilt) == _bn_eps(model) == {EPS}


def test_checksum_mismatch_and_missing_file_fail_loudly(tmp_path):
    model, _ = build_model(_cfg())
    path = tmp_path / "w.pt"
    torch.save(model.backbone.state_dict(), path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_backbone_weights(model.backbone, path, expected_sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        load_backbone_weights(model.backbone, tmp_path / "absent.pt")


def test_regression_head_shape_range_and_state_dict_keys():
    model, _ = build_model(_cfg())
    model.eval()
    out = model(torch.randn(2, 1, 224, 224))
    assert isinstance(out, ModelOutput) and out.log_var is None
    assert out.mean.shape == (2,)
    assert torch.all((out.mean >= 0) & (out.mean <= 1))
    keys = set(model.state_dict())
    assert {"fc.weight", "fc.bias"} <= keys and not any(k.startswith("head.") for k in keys)


def test_gaussian_head_finite_bounded_log_variance():
    model, _ = build_model(_cfg(name="resnet50_gaussian"))
    model.eval()
    out = model(torch.randn(3, 1, 224, 224) * 50)       # extreme input to stress the clamp
    assert out.mean.shape == out.log_var.shape == (3,)
    assert torch.isfinite(out.mean).all() and torch.isfinite(out.log_var).all()
    assert torch.all(out.log_var >= LOGVAR_MIN) and torch.all(out.log_var <= LOGVAR_MAX)
    assert torch.isfinite(torch.exp(-out.log_var)).all()


def test_input_validation():
    model, _ = build_model(_cfg())
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 224, 224))
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 128, 128))


def test_single_channel_equals_explicit_replication():
    model, _ = build_model(_cfg())
    model.eval()
    x = torch.randn(2, 1, 224, 224)
    model3, _ = build_model(ModelConfig(name="resnet50_regression", dropout_p=0.5, bn_eps=EPS, in_channels=3))
    model3.load_state_dict(model.state_dict())
    model3.eval()
    assert torch.allclose(model(x).mean, model3(x.repeat(1, 3, 1, 1)).mean, atol=1e-6)
