"""Evaluate a trained NGP-Net checkpoint on the PNG test split.

    python scripts/evaluate.py --data-root data/external/png --trained-model best_model.pth

Requires a checkpoint produced by scripts/train.py; there is no shipped
pretrained checkpoint (see docs/reproduction.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_aware_lung_ct.data.png import build_dataloaders
from quality_aware_lung_ct.evaluation.evaluator import build_metrics, evaluate
from quality_aware_lung_ct.ngpnet.config import config_from_args
from quality_aware_lung_ct.training.trainer import build_model


def main(argv=None):
    config = config_from_args(argv)
    test_loader = build_dataloaders(config, is_test=True)
    model = build_model(config, pretrained=True)
    metrics = build_metrics(config)

    results = evaluate(model, test_loader, config.resolve_device(), metrics,
                       config=config, save_predictions_to_disk=True)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    return results


if __name__ == "__main__":
    main()
