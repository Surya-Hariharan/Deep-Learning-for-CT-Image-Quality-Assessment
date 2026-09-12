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
