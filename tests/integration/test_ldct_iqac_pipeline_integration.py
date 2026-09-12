"""Integration tests against the REAL local LDCT-IQAC dataset.

Unlike `tests/unit/test_dataset.py` (synthetic `tmp_path` fixtures), these
tests exercise the actual files under `data/raw/ldct_iqac/` and
`data/labels/ldct_iqac/`. They are skipped if that data isn't present
locally (e.g. a fresh clone before dataset setup -- see README.md,
"Dataset setup").

Added by the scientific pipeline audit (see
docs/replication/reproducibility.md) to cover two invariants that were not
previously tested against real data: (1) no train/test leakage by
filename, pixel content, or label key, and (2) a full real-data training
step (dataset -> preprocessing -> DataLoader -> model -> loss -> backward
-> optimizer.step) actually runs end to end.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from ct_iqa.config import ExperimentConfig
from ct_iqa.data.ldct_iqac import SCORE_MAX, SCORE_MIN, LDCTIQACDataset, normalize_score
from ct_iqa.data.loader import build_dataloaders
from ct_iqa.models.ohashi_resnet50 import OhashiResNet50
from ct_iqa.training.losses import build_loss
from ct_iqa.training.optimizers import build_optimizer
from ct_iqa.training.trainer import train_one_step

_config = ExperimentConfig()
requires_real_dataset = pytest.mark.skipif(
    not (
        __import__("pathlib").Path(_config.train_image_dir).is_dir()
        and __import__("pathlib").Path(_config.test_image_dir).is_dir()
    ),
    reason="real LDCT-IQAC dataset not present locally (gitignored; see README.md)",
)


def _image_hashes(directory) -> set[str]:
    from pathlib import Path

    directory = Path(directory)
    hashes = set()
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            with Image.open(p) as im:
                hashes.add(hashlib.md5(np.array(im).tobytes()).hexdigest())
    return hashes


@requires_real_dataset
def test_expected_dataset_counts():
    train_ds = LDCTIQACDataset(_config.train_image_dir, _config.train_json_path)
    test_ds = LDCTIQACDataset(_config.test_image_dir, _config.test_json_path)
    assert len(train_ds) == 1000
    assert len(test_ds) == 300


@requires_real_dataset
def test_no_filename_overlap_between_train_and_test():
    from pathlib import Path

    train_names = {p.name for p in Path(_config.train_image_dir).iterdir() if p.is_file()}
    test_names = {p.name for p in Path(_config.test_image_dir).iterdir() if p.is_file()}
    assert train_names & test_names == set()


@requires_real_dataset
def test_no_identical_image_content_between_train_and_test():
    train_hashes = _image_hashes(_config.train_image_dir)
    test_hashes = _image_hashes(_config.test_image_dir)
    assert train_hashes & test_hashes == set()


@requires_real_dataset
def test_no_label_key_overlap_between_train_and_test():
    import json
    from pathlib import Path

    train_labels = json.loads(Path(_config.train_json_path).read_text())
    test_labels = json.loads(Path(_config.test_json_path).read_text())
    assert set(train_labels.keys()) & set(test_labels.keys()) == set()


@requires_real_dataset
def test_real_dataloader_batch_shape_and_range():
    config = ExperimentConfig(batch_size=8)
    train_loader, val_loader = build_dataloaders(config)
    images, raw_scores = next(iter(train_loader))

    assert images.shape[1:] == (1, 224, 224)
    assert images.dtype == torch.float32
    assert images.min() >= -1.0 - 1e-6 and images.max() <= 1.0 + 1e-6
    assert raw_scores.shape == (images.shape[0],)
    assert raw_scores.min() >= 0.0 and raw_scores.max() <= 4.0


@requires_real_dataset
def test_score_normalization_constants_bound_the_real_dataset():
    """SCORE_MIN/SCORE_MAX must actually bound the real dataset's scores --
    normalize_score must never be asked to map a value outside [0, 1]
    (its Sigmoid-compatible output range) on real data."""
    train_ds = LDCTIQACDataset(_config.train_image_dir, _config.train_json_path)
    test_ds = LDCTIQACDataset(_config.test_image_dir, _config.test_json_path)
    all_scores = train_ds.raw_scores + test_ds.raw_scores

    assert min(all_scores) >= SCORE_MIN
    assert max(all_scores) <= SCORE_MAX

    normalized = [normalize_score(s) for s in all_scores]
    assert min(normalized) >= 0.0
    assert max(normalized) <= 1.0


@requires_real_dataset
def test_one_real_training_step_end_to_end():
    """Real image -> preprocessing -> DataLoader -> model -> loss -> backward -> optimizer.step()."""
    config = ExperimentConfig(batch_size=4, dropout_p=0.5)
    train_loader, _ = build_dataloaders(config)
    batch = next(iter(train_loader))

    model = OhashiResNet50(dropout_p=config.dropout_p, in_channels=config.in_channels)
    optimizer = build_optimizer(model, config)
    criterion = build_loss(config)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss = train_one_step(model, batch, optimizer, criterion, device="cpu")

    assert loss == loss  # not NaN
    assert loss != float("inf")
    assert all(p.grad is not None for p in model.parameters())
    after = dict(model.named_parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_configured_radimagenet_path_loads_and_matches_backbone():
    """configs/model.yaml's radimagenet_weights_path must actually be usable
    end to end: ExperimentConfig.from_yaml_files() -> OhashiResNet50 ->
    load_radimagenet_weights() -> a real forward pass. Skipped (not failed)
    if the gitignored weight file isn't present on this machine -- this
    project never silently falls back to ImageNet weights either way."""
    from pathlib import Path

    config = ExperimentConfig.from_yaml_files()
    if config.radimagenet_weights_path is None:
        pytest.skip("configs/model.yaml has radimagenet_weights_path: null")
    if not Path(config.radimagenet_weights_path).is_file():
        pytest.skip(f"configured weights file not present locally: {config.radimagenet_weights_path}")

    model = OhashiResNet50(dropout_p=config.dropout_p, in_channels=config.in_channels)
    report = model.load_radimagenet_weights(config.radimagenet_weights_path)

    assert report.loaded is True
    assert report.unexpected_keys == []
    assert all(k.endswith("num_batches_tracked") for k in report.missing_keys)

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, config.in_channels, 224, 224))
    assert torch.isfinite(out).all()
    assert bool((out >= 0).all() and (out <= 1).all())
