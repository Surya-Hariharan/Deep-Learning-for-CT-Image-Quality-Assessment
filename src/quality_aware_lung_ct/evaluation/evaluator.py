"""NGP-Net test-time evaluation.

Consolidates the author's `test.py` and `cross_test.py`, which duplicated
the same per-batch evaluation loop (single model vs. k-fold models). The
computed metrics (DSC, sensitivity/TPR, specificity/TNR, PPV, PSNR, SSIM,
MSE_ROI, MSE_PN) and their formulas are unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from ..ngpnet.config import NGPNetConfig
from ..ngpnet.model import NGPNet
from ..preprocessing.volumes import normalize_ct, save_as_file
from .metrics import ConfusionMatrixMetric, DiceMetric, MaskedMetric, MSEMetric, PSNRMetric, SSIMMetric


def save_predictions(pred_save_dir: str, pred_model: str, info: dict,
                     im2, mk2, im2_pred, mk2_pred, scope, range_) -> None:
    ''' Write source/predicted image+mask pairs as NIfTI, matching the
    author's `test.py:save_predictions`. `scope`/`range_` are the CT
    windowing bounds used to invert the model's normalized intensity range
    back to Hounsfield units. '''
    pid = info["PatientId"][0]
    nid = "%02d" % int(info["NoduleId"][0])
    date0, date1, date2 = info["t0"][0], info["t1"][0], info["t2"][0]
    save_dir = os.path.join(pred_save_dir, pred_model, str(pid), nid)
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"{date0}+{date1}={date2}")

    src_img = normalize_ct(im2.cpu().numpy(), scope=range_, range=scope)
    save_as_file(f"{fname}_SrcImg.nii.gz", src_img, dtype="float32")
    save_as_file(f"{fname}_SrcMsk.nii.gz", mk2.cpu().numpy(), dtype="uint8")

    pred_img = normalize_ct(im2_pred.cpu().numpy(), scope=range_, range=scope)
    save_as_file(f"{fname}_PredImg.nii.gz", pred_img, dtype="float32")
    save_as_file(f"{fname}_PredMsk.nii.gz", mk2_pred.cpu().numpy(), dtype="uint8")


def build_metrics(config: NGPNetConfig) -> dict:
    ''' Construct the NGP-Net evaluation metrics per `config`. '''
    mse = MSEMetric()
    return {
        "dice": DiceMetric(n_classes=2, ignore_bg=True),
        "PSNR": PSNRMetric(max_val=config.r_max - config.r_min),
        "SSIM": SSIMMetric(max_val=config.r_max - config.r_min),
        "MSE-ROI": mse,
        "MSE-PN": MaskedMetric(func=mse, w_out=0, expand=1.1),
        "matrix": ConfusionMatrixMetric(n_classes=2, ignore_bg=True),
    }


@torch.no_grad()
def evaluate(model: NGPNet, loader, device: str,
            metrics: Optional[dict] = None, fold: Optional[int] = None,
            config: Optional[NGPNetConfig] = None, save_predictions_to_disk: bool = False) -> dict:
    ''' Run the test-time evaluation loop over `loader`. Returns averaged metrics.

    If `save_predictions_to_disk` is set, writes source/predicted NIfTI
    pairs under `config.pred_save_dir` (requires `config`).
    '''
    if metrics is None:
        metrics = build_metrics(NGPNetConfig())
    if save_predictions_to_disk and config is None:
        raise ValueError("`config` is required when `save_predictions_to_disk=True`.")

    model.eval()
    psnr_list, ssim_list, mse_roi_list, mse_pn_list = [], [], [], []
    tpr_list, tnr_list, fnr_list, fpr_list = [], [], [], []
    dsc_list, ppv_list = [], []

    for bid, batch in enumerate(loader):
        batch = list(batch)
        for idx in range(len(batch) - 1):
            batch[idx] = batch[idx].to(device)
        (im0, im1, im2, mk0, mk1, mk2, tm0, tm1, info) = batch

        im2_pred, mk2_pred = model(im0, im1, tm0, tm1)

        mse_roi = metrics["MSE-ROI"](im2_pred, im2)
        mse_pn = metrics["MSE-PN"](im2_pred, im2, mk2)
        psnr = metrics["PSNR"](im2_pred, im2)
        ssim = metrics["SSIM"](im2_pred, im2)
        dice = metrics["dice"](mk2_pred, mk2)

        matrix = metrics["matrix"](mk2_pred, mk2)
        tpr = metrics["matrix"].compute("tpr", matrix)
        tnr = metrics["matrix"].compute("tnr", matrix)
        fpr = metrics["matrix"].compute("fpr", matrix)
        fnr = metrics["matrix"].compute("fnr", matrix)
        ppv = metrics["matrix"].compute("ppv", matrix)

        psnr_list.append(np.mean(psnr))
        ssim_list.append(np.mean(ssim))
        mse_roi_list.append(np.mean(mse_roi))
        mse_pn_list.append(np.mean(mse_pn))
        dsc_list.append(np.mean(dice))
        ppv_list.append(np.mean(ppv))
        tpr_list.append(np.mean(tpr))
        fpr_list.append(np.mean(fpr))
        tnr_list.append(np.mean(tnr))
        fnr_list.append(np.mean(fnr))

        fold_tag = f"Fold: {fold}, " if fold is not None else ""
        print(f"【Test】{fold_tag}Step: {bid}, DSC: {dsc_list[-1]*100:.2f}%, "
              f"PSNR: {psnr_list[-1]:.2f}, SSIM: {ssim_list[-1]*100:.2f}%")

        if save_predictions_to_disk:
            save_predictions(config.pred_save_dir, config.pred_model, info,
                             im2, mk2, im2_pred, mk2_pred,
                             scope=(config.s_min, config.s_max),
                             range_=(config.r_min, config.r_max))

    return {
        "PSNR": float(np.mean(psnr_list)), "SSIM": float(np.mean(ssim_list)),
        "MSE-ROI": float(np.mean(mse_roi_list)), "MSE-PN": float(np.mean(mse_pn_list)),
        "DSC": float(np.mean(dsc_list)), "PPV": float(np.mean(ppv_list)),
        "TPR": float(np.mean(tpr_list)), "TNR": float(np.mean(tnr_list)),
        "FPR": float(np.mean(fpr_list)), "FNR": float(np.mean(fnr_list)),
    }
