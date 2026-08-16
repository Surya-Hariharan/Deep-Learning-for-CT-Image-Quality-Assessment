"""NGP-Net training loop.

Consolidates the author's `train.py` and `cross_val.py`, which were
duplicate implementations of the same epoch loop (single split vs. k-fold).
`NGPNetTrainer.fit` covers both by taking a `fold` label used only for
logging/checkpoint naming; the training math is unchanged from the author's
scripts. Hyperparameter defaults are inherited from `NGPNetConfig`
unchanged -- see docs/reproduction.md for what's been verified vs. not.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
from torch.nn import Module
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..ngpnet.config import NGPNetConfig
from ..ngpnet.losses import DiceCELoss, GradLoss, MaskedLoss, SSIMLoss
from ..ngpnet.model import NGPNet
from ..evaluation.metrics import DiceMetric, PSNRMetric, SSIMMetric
from ..common.logging import LogWriter
from .checkpoints import load_checkpoint, save_checkpoint


def build_model(config: NGPNetConfig, pretrained: bool = False) -> NGPNet:
    ''' Construct NGP-Net per `config`, optionally loading a checkpoint. '''
    model_name = config.pred_model.lower()
    if model_name != "ngpnet":
        raise ValueError(f"Growth Prediction Model `{config.pred_model}` is not supported!")

    model = NGPNet(img_size=config.in_size,
                  feat_dim=config.feat_dim,
                  depths=config.depths)
    model = model.to(config.resolve_device())

    if pretrained:
        path = os.path.join(config.model_save_dir, config.pred_model, config.trained_model)
        load_checkpoint(model, path, map_location=config.resolve_device())
    return model


def build_optimizer(model: Module, config: NGPNetConfig):
    ''' Construct the optimizer and (optional) LR scheduler per `config`. '''
    optim_name = config.optim_name.lower()
    lrschedule = config.lrschedule.lower()

    if optim_name == "sgd":
        optim = SGD(model.parameters(), momentum=0.9,
                   lr=config.optim_lr, weight_decay=config.reg_weight)
    elif optim_name == "adam":
        optim = Adam(model.parameters(),
                    lr=config.optim_lr, weight_decay=config.reg_weight)
    else:
        optim = AdamW(model.parameters(),
                     lr=config.optim_lr, weight_decay=config.reg_weight)

    if lrschedule == "cosine":
        scheduler = CosineAnnealingLR(optim, T_max=config.max_epochs)
    elif lrschedule == "warmupcosine":
        # Training-only dependency -- see requirements.txt group 3. Not
        # imported unless this schedule is actually selected.
        from monai.optimizers.lr_scheduler import WarmupCosineSchedule
        scheduler = WarmupCosineSchedule(optim, t_total=config.max_epochs,
                                         warmup_steps=config.warmup_steps)
    else:
        scheduler = None
    return optim, scheduler


def build_losses(config: NGPNetConfig) -> dict:
    ''' Construct the NGP-Net training losses per `config`. '''
    data_range = config.r_max - config.r_min
    ssim = MaskedLoss(func=SSIMLoss(max_val=data_range),
                      w_out=config.w_out, expand=config.expand)
    grad = GradLoss(square=True, reduction="mean")
    dice = DiceCELoss(n_classes=2, include_bg=True, ce_lambda=2.0, dice_lambda=1.0)
    return {
        "sim": {"f": ssim, "w": config.w0_sim},
        "seg": {"f": dice, "w": config.w1_seg},
        "reg": {"f": ssim, "w": config.w2_reg},
        "smooth": {"f": grad, "w": config.w3_smooth},
    }


class NGPNetTrainer:
    ''' Runs the NGP-Net train/validate epoch loop. '''

    def __init__(self, config: NGPNetConfig):
        self.config = config
        self.device = config.resolve_device()

    def train_epoch(self, model: NGPNet, loader, optim: Optimizer, losses: dict,
                    epoch: int, fold: Optional[int] = None) -> float:
        model.train()
        loss_list = []
        for bid, batch in enumerate(loader):
            batch = list(batch)
            for idx in range(len(batch) - 1):   # moves to target device; last item is `info`
                batch[idx] = batch[idx].to(self.device)
            (im0, im1, im2, mk0, mk1, mk2, tm0, tm1, _) = batch

            t_zero = torch.zeros_like(tm1).to(tm1)
            im1_pred0, mk1_pred, _ = model.ngpnet(im0, im0, t_zero, tm1)
            im1_pred1, mk1_pred, _ = model.ngpnet(im0, im1, tm0, t_zero)
            im2_pred, mk2_pred, field2 = model.ngpnet(im0, im1, tm0, tm1)

            Lsim = losses["sim"]["f"](im2_pred, im2, mk1) * losses["sim"]["w"]
            Lreg = (losses["reg"]["f"](im1_pred1, im1, mk1) +
                    losses["reg"]["f"](im1_pred0, im1, mk0)) * losses["reg"]["w"]
            Lseg = losses["seg"]["f"](mk2_pred, mk2) * losses["seg"]["w"]
            Lsmooth = losses["smooth"]["f"](field2) * losses["smooth"]["w"]
            loss = Lsim + Lreg + Lseg + Lsmooth
            loss_list.extend([loss.item()] * len(tm0))

            fold_tag = f"Fold: {fold}, " if fold is not None else ""
            print(f"【Train】{fold_tag}Epoch: {epoch}, Batch: {bid}, Loss: {loss_list[-1]:.4f}")

            optim.zero_grad()
            loss.backward()
            optim.step()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return float(np.mean(loss_list))

    @torch.no_grad()
    def validate_epoch(self, model: NGPNet, loader, epoch: int,
                       fold: Optional[int] = None) -> dict:
        model.eval()
        psnr_metric, ssim_metric, dice_metric = PSNRMetric(), SSIMMetric(), DiceMetric(n_classes=2, ignore_bg=True)
        ssim_list, psnr_list, dice_list = [], [], []
        for bid, batch in enumerate(loader):
            batch = list(batch)
            for idx in range(len(batch) - 1):
                batch[idx] = batch[idx].to(self.device)
            (im0, im1, im2, mk0, mk1, mk2, tm0, tm1, _) = batch

            im2_pred, mk2_pred = model(im0, im1, tm0, tm1)
            psnr_list.append(psnr_metric(im2_pred, im2))
            ssim_list.append(ssim_metric(im2_pred, im2))
            dice_list.append(dice_metric(mk2_pred, mk2))

        return {
            "dice": float(np.mean(dice_list)),
            "psnr": float(np.mean(psnr_list)),
            "ssim": float(np.mean(ssim_list)),
        }

    def fit(self, train_loader, val_loader, fold: Optional[int] = None) -> dict:
        ''' Run the full training loop for one split (or one fold).

        Returns the best validation metrics observed (selection metric: SSIM,
        matching the author's scripts).
        '''
        config = self.config
        model = build_model(config, pretrained=False)
        optim, scheduler = build_optimizer(model, config)
        losses = build_losses(config)
        fold_suffix = f"_f{fold}" if fold is not None else ""
        writer = LogWriter(config.log_save_dir, prefix=f"{config.pred_model}{fold_suffix}")

        best = {"dice": 0.0, "psnr": 0.0, "ssim": 0.0}
        start = time.time()
        for epoch in range(config.max_epochs):
            train_loss = self.train_epoch(model, train_loader, optim, losses, epoch, fold)
            if scheduler is not None:
                scheduler.step()
            if (epoch + 1) % config.val_freq == 0:
                val_metrics = self.validate_epoch(model, val_loader, epoch, fold)
                writer.add_row({"fold": fold, "epoch": epoch + 1, "train_loss": train_loss,
                                "val_dice": val_metrics["dice"], "val_psnr": val_metrics["psnr"],
                                "val_ssim": val_metrics["ssim"]})
                writer.save()
                if val_metrics["ssim"] > best["ssim"]:
                    best_name = f"best_model{fold_suffix}.pth"
                    save_checkpoint(model, optim, os.path.join(config.model_save_dir, config.pred_model), best_name)
                best = {k: max(best[k], val_metrics[k]) for k in best}
            save_checkpoint(model, optim, os.path.join(config.model_save_dir, config.pred_model), "latest_model.pth")

        print(f"【FINISHED】 Fold: {fold}, Best SSIM: {best['ssim']:.4f}, "
              f"Time: {(time.time() - start) / 3600:.2f}h")
        return best
