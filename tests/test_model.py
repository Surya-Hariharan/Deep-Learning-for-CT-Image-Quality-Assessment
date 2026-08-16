"""NGP-Net construction, parameter count, and forward pass."""

import pytest
import torch

from quality_aware_lung_ct.ngpnet import NGPNet

EXPECTED_PARAM_COUNT = 1_984_821


def test_construction_needs_no_cli_parsing():
    model = NGPNet(img_size=64, feat_dim=8)
    assert isinstance(model, torch.nn.Module)


def test_parameter_count():
    model = NGPNet(img_size=64, feat_dim=8)
    n_params = sum(p.nelement() for p in model.parameters())
    assert n_params == EXPECTED_PARAM_COUNT


def test_forward_pass_cpu():
    model = NGPNet(img_size=64, feat_dim=8).eval()
    im0 = torch.randn(1, 1, 64, 64, 64)
    im1 = torch.randn(1, 1, 64, 64, 64)
    tm0 = torch.randint(0, 64, (1,))
    tm1 = torch.randint(0, 64, (1,))

    with torch.no_grad():
        image, mask = model(im0, im1, tm0, tm1)

    assert image.shape == (1, 1, 64, 64, 64)
    assert mask.shape == (1, 1, 64, 64, 64)


def test_ngpnet_raw_heads_shapes():
    model = NGPNet(img_size=64, feat_dim=8, num_classes=2).eval()
    im0 = torch.randn(2, 1, 64, 64, 64)
    im1 = torch.randn(2, 1, 64, 64, 64)
    tm0 = torch.randint(0, 64, (2,))
    tm1 = torch.randint(0, 64, (2,))

    with torch.no_grad():
        image, logits, field = model.ngpnet(im0, im1, tm0, tm1)

    assert image.shape == (2, 1, 64, 64, 64)
    assert logits.shape == (2, 2, 64, 64, 64)
    assert field.shape == (2, 3, 64, 64, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass_cuda():
    model = NGPNet(img_size=64, feat_dim=8).eval().cuda()
    im0 = torch.randn(1, 1, 64, 64, 64, device="cuda")
    im1 = torch.randn(1, 1, 64, 64, 64, device="cuda")
    tm0 = torch.randint(0, 64, (1,), device="cuda")
    tm1 = torch.randint(0, 64, (1,), device="cuda")

    with torch.no_grad():
        image, mask = model(im0, im1, tm0, tm1)

    assert image.device.type == "cuda"
    assert mask.device.type == "cuda"
