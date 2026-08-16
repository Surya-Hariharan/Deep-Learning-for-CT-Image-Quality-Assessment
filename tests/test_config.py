"""NGPNetConfig behavior: defaults, CLI round-trip, device resolution."""

from quality_aware_lung_ct.ngpnet.config import NGPNetConfig, config_from_args


def test_defaults_match_author_values():
    config = NGPNetConfig()
    assert config.seed == 35202
    assert config.batch_size == 4
    assert config.max_epochs == 200
    assert config.in_size == 64
    assert config.feat_dim == 8
    assert config.depths == (2, 2, 2, 2)


def test_config_from_args_overrides_defaults():
    config = config_from_args(["--batch-size", "2", "--data-root", "/tmp/png"])
    assert config.batch_size == 2
    assert config.data_root == "/tmp/png"
    assert config.max_epochs == NGPNetConfig().max_epochs  # untouched default


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = NGPNetConfig(device="cuda")
    assert config.resolve_device() == "cpu"
