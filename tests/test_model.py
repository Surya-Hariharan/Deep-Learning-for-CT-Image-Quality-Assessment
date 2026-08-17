import pytest
import torch

from ct_iqa.models.ohashi_resnet50 import INPUT_SIZE, OhashiResNet50
from ct_iqa.models.resnet50 import ResNet50Backbone


def test_backbone_output_shape():
    backbone = ResNet50Backbone(in_channels=3)
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, ResNet50Backbone.OUT_FEATURES)


def test_dropout_p_is_required():
    with pytest.raises(TypeError):
        OhashiResNet50()  # no default allowed -- must be explicit


def test_dropout_p_validated():
    with pytest.raises(ValueError):
        OhashiResNet50(dropout_p=1.5)


def test_forward_pass_output_shape():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    model.eval()
    x = torch.randn(4, 1, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,)


def test_forward_pass_output_range_is_sigmoid_bounded():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    model.eval()
    x = torch.randn(4, 1, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = model(x)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_no_nan_output():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    model.eval()
    x = torch.randn(4, 1, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


def test_wrong_input_size_raises():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    x = torch.randn(1, 1, 32, 32)
    with pytest.raises(ValueError):
        model(x)


def test_wrong_channel_count_raises():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with pytest.raises(ValueError):
        model(x)


def test_load_radimagenet_weights_none_reports_not_loaded():
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    report = model.load_radimagenet_weights(weights_path=None)
    assert report.loaded is False
