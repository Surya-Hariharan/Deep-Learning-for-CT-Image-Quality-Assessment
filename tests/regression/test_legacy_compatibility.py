"""Regression tests: the new registry model versus the historical `OhashiResNet50`, and the
historical checkpoints' structure.

Synthetic-weight tests build both models, copy one state_dict into the other and compare outputs on
SYNTHETIC random input at equal BatchNorm epsilon. The checkpoint test is STRUCTURAL ONLY (keys and
shapes, strict load into a fresh model); it never runs a forward pass on checkpoint weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ct_iqa.config.schema import ModelConfig
from ct_iqa.models.ohashi_resnet50 import OhashiResNet50
from ct_iqa.models.registry import build_model

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = [
    "experiments/001_resnet50_baseline/checkpoint/best.pt",
    "experiments/002_learning_rate_search/lr_1e-2/checkpoint/best.pt",
    "experiments/002_learning_rate_search/lr_1e-3/checkpoint/best.pt",
    "experiments/002_learning_rate_search/lr_1e-4/checkpoint/best.pt",
    "experiments/002_learning_rate_search/lr_1e-5/checkpoint/best.pt",
    "experiments/003_final_training/checkpoint/final.pt",
]


def test_registry_regression_model_matches_legacy_class_on_synthetic_weights():
    torch.manual_seed(0)
    legacy = OhashiResNet50(dropout_p=0.5, in_channels=1).eval()      # legacy eps = 1e-5 (no weight load)
    new, _ = build_model(ModelConfig(name="resnet50_regression", dropout_p=0.5, bn_eps=1e-5))
    assert set(legacy.state_dict()) == set(new.state_dict())
    new.load_state_dict(legacy.state_dict(), strict=True)
    new.eval()
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        assert torch.allclose(legacy(x), new(x).mean, atol=1e-6)


@pytest.mark.parametrize("rel", CHECKPOINTS)
def test_historical_checkpoint_structure_loads_strictly(rel):
    path = ROOT / rel
    if not path.is_file():
        pytest.skip(f"checkpoint not present locally (gitignored): {rel}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    model, _ = build_model(ModelConfig(name="resnet50_regression", dropout_p=0.5, bn_eps=1.001e-5))
    model.load_state_dict(state, strict=True)       # raises on any missing/unexpected key or shape mismatch
