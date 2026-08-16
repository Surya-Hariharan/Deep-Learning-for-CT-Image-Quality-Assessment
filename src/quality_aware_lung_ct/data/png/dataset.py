"""PNG dataset and dataloader construction.

Migrated from the author's `utils/dataloader.py` (class `DatasetNG3T`,
renamed `PNGDataset`). Behavior is unchanged; paths are resolved through
`metadata.resolve_series_path` instead of naive `os.path.join`, so the
manifest's `.nii.gz`-vs-`.nii` mismatch (see docs/datasets.md) is handled
without modifying the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader, Dataset

from ...ngpnet.config import NGPNetConfig
from . import transforms as T
from .metadata import get_split, load_manifest, resolve_series_path


class PNGDataset(Dataset):
    ''' Nodule Growth Dataset with 3 Time Points (PNG) '''

    def __init__(self,
                 data_root: Union[str, Path],
                 data_list: list,
                 transforms: object):
        ''' Args:
        * `data_root`: root of the PNG dataset.
        * `data_list`: manifest entries in this split/fold.
        * `transforms`: transform pipeline (see `.transforms.get_transforms`).
        '''
        super().__init__()
        self.data_root = data_root
        self.data_list = data_list
        self.transforms = transforms

    def __getitem__(self, index: int):
        ''' get images, masks, intervals and info by index '''
        info = self.data_list[index]["info"]
        itvs = [int(info["Months1"]), int(info["Months2"])]
        images, labels = [], []
        for series in self.data_list[index]["series"]:
            images.append(str(resolve_series_path(self.data_root, series["image"])))
            labels.append(str(resolve_series_path(self.data_root, series["label"])))
        images, labels = self.transforms(images, labels)
        return *images, *labels, *itvs, info

    def __len__(self):
        return len(self.data_list)


def split_data_list(data_list: list, num_folds: int, fold: int = 0):
    ''' Split a manifest split into (train, val) by index modulo `num_folds`. '''
    if not fold < num_folds:
        raise ValueError("`fold` should be less than `num_folds`.")
    tra_data, val_data = [], []
    for idx, item in enumerate(data_list):
        if idx % num_folds == fold:
            val_data.append(item)
        else:
            tra_data.append(item)
    return tra_data, val_data


def build_dataloaders(config: NGPNetConfig, is_test: bool = False):
    ''' Build the train/val loaders, or the test loader, per `config`.

    Mirrors the official train/test split (`manifest["training"]` /
    `manifest["test"]`) vs. the patient-grouped k-fold split derived from
    it -- see docs/datasets.md for the documented split-overlap caveat.
    '''
    manifest = load_manifest(config.data_root, config.json_name)
    transforms = T.get_transforms(config, is_test)

    if not is_test:
        data_list = get_split(manifest, "training")
        tra_data, val_data = split_data_list(data_list, config.num_folds, config.fold)

        train_dataset = PNGDataset(config.data_root, tra_data, transforms[0])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)

        val_dataset = PNGDataset(config.data_root, val_data, transforms[1])
        val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=config.num_workers)
        return train_loader, val_loader
    else:
        test_data = get_split(manifest, "test")
        test_dataset = PNGDataset(config.data_root, test_data, transforms)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False, num_workers=config.num_workers)
        return test_loader
