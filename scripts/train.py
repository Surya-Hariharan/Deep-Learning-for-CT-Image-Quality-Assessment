"""Train NGP-Net on the PNG dataset.

    python scripts/train.py --data-root data/external/png
    python scripts/train.py --data-root data/external/png --cross-validate

Requires the training extras: `pip install -e ".[train]"`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_aware_lung_ct.common import set_seed
from quality_aware_lung_ct.data.png import build_dataloaders
from quality_aware_lung_ct.ngpnet.config import config_from_args
from quality_aware_lung_ct.training import NGPNetTrainer


def main(argv=None):
    config = config_from_args(argv)
    set_seed(config.seed)
    trainer = NGPNetTrainer(config)

    if getattr(config, "cross_validate", False):
        results = []
        for fold in range(config.num_folds):
            config.fold = fold
            train_loader, val_loader = build_dataloaders(config, is_test=False)
            results.append(trainer.fit(train_loader, val_loader, fold=fold))
        mean_ssim = sum(r["ssim"] for r in results) / len(results)
        print(f"【Cross-validation finished】 Mean best SSIM: {mean_ssim:.4f}")
    else:
        train_loader, val_loader = build_dataloaders(config, is_test=False)
        trainer.fit(train_loader, val_loader, fold=None)


if __name__ == "__main__":
    main()
