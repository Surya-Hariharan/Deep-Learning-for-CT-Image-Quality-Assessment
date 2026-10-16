"""DataLoader construction for the LDCT-IQAC dataset.

Extracted from `ct_iqa.training.trainer` with no change in behavior. Builds
`torch.utils.data.DataLoader` instances around
`ct_iqa.data.ldct_iqac.LDCTIQACDataset` and `ct_iqa.data.splits.train_val_split`.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from ct_iqa.config import ExperimentConfig
from ct_iqa.data.ldct_iqac import LDCTIQACDataset
from ct_iqa.data.splits import train_val_split


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) from the training split only.

    The validation subset is carved out of `LDCTIQACDataset(train_image_dir,
    train_json_path, ...)` using `ct_iqa.data.splits.train_val_split` seeded
    from `config.seed`, so the split is exactly reproducible given the same
    seed and `val_fraction`. The test set is not referenced here at all.
    """
    full_train_dataset = LDCTIQACDataset(
        image_dir=config.train_image_dir,
        json_path=config.train_json_path,
        image_size=config.image_size,
    )
    train_subset, val_subset = train_val_split(
        full_train_dataset, config.val_fraction, config.seed
    )

    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def build_test_dataloader(config: ExperimentConfig) -> DataLoader:
    """Build the held-out test DataLoader from the official LDCT-IQAC test split.

    Use only for final evaluation, never during training or model selection.
    """
    test_dataset = LDCTIQACDataset(
        image_dir=config.test_image_dir,
        json_path=config.test_json_path,
        image_size=config.image_size,
    )
    return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Study-v2 loaders: built from the frozen split manifest. The official test set is NOT reachable from here;
# it is only loaded by `ct_iqa.experiment.run.final_test_evaluation` behind explicit guards.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from ct_iqa.data.splits import partition_indices  # noqa: E402


def build_study_datasets(data_cfg, manifest: dict) -> dict[str, tuple[Subset, list[str]]]:
    """Return {partition: (Subset, filenames in subset order)} for train / validation / calibration."""
    full = LDCTIQACDataset(image_dir=data_cfg.train_image_dir, json_path=data_cfg.train_json_path,
                           image_size=data_cfg.image_size)
    names = [s.filename for s in full.samples]
    out = {}
    for part in ("train", "validation", "calibration"):
        idx = partition_indices(names, manifest, part)
        out[part] = (Subset(full, idx), [names[i] for i in idx])
    return out


def build_study_loaders(data_cfg, train_cfg, manifest: dict) -> tuple[dict[str, DataLoader], dict[str, list[str]]]:
    """Seeded loaders: the train loader shuffles with its own generator so batch order depends only on `train.seed`."""
    parts = build_study_datasets(data_cfg, manifest)
    generator = torch.Generator().manual_seed(train_cfg.seed)
    loaders = {
        "train": DataLoader(parts["train"][0], batch_size=train_cfg.batch_size, shuffle=True,
                            drop_last=False, generator=generator),
        "validation": DataLoader(parts["validation"][0], batch_size=train_cfg.batch_size, shuffle=False),
        "calibration": DataLoader(parts["calibration"][0], batch_size=train_cfg.batch_size, shuffle=False),
    }
    return loaders, {k: v[1] for k, v in parts.items()}
