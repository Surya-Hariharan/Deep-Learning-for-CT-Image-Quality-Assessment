"""Evaluation metric calculations, using synthetic tensors."""

import math

import torch

from quality_aware_lung_ct.evaluation.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    MSEMetric,
    PSNRMetric,
    SSIMMetric,
)


def test_mse_identical_inputs_is_zero():
    x = torch.rand(1, 1, 8, 8, 8)
    assert MSEMetric()(x, x) == 0.0


def test_psnr_identical_inputs_is_infinite():
    x = torch.rand(1, 1, 8, 8, 8)
    psnr = PSNRMetric(max_val=1.0)(x, x)
    assert math.isinf(psnr)


def test_ssim_identical_inputs_is_one():
    x = torch.rand(1, 1, 16, 16, 16)
    ssim = SSIMMetric(max_val=1.0)(x, x)
    assert abs(ssim - 1.0) < 1e-4


def test_dice_identical_masks_is_one():
    mask = torch.randint(0, 2, (1, 1, 8, 8, 8))
    dice = DiceMetric(n_classes=2, ignore_bg=True)(mask, mask)
    assert abs(dice - 1.0) < 1e-4


def test_dice_disjoint_masks_is_zero():
    pred = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    true = torch.ones(1, 1, 4, 4, 4, dtype=torch.long)
    dice = DiceMetric(n_classes=2, ignore_bg=True)(pred, true)
    assert dice < 1e-4  # exactly 0 plus the metric's smoothing epsilon


def test_confusion_matrix_perfect_prediction():
    mask = torch.randint(0, 2, (1, 1, 8, 8, 8))
    metric = ConfusionMatrixMetric(n_classes=2, ignore_bg=True)
    matrix = metric(mask, mask)
    assert abs(metric.compute("tpr", matrix)[0] - 1.0) < 1e-4
    assert abs(metric.compute("fpr", matrix)[0] - 0.0) < 1e-4
