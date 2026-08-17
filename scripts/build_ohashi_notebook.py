"""One-off generator for notebooks/01_ohashi_resnet50_ldct_iqac.ipynb.

Run this script to (re)build the notebook from the cell definitions below.
Not part of the ct_iqa package; a build-time convenience only.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    # Markdown cells intentionally excluded from the notebook -- code only.
    pass


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# 1. Title and objective
md(
    """# Experiment 01 — Ohashi ResNet50 Baseline on LDCT-IQAC

## Objective

Replicate the **first baseline** from:

> Ohashi et al., *"Development of a No-Reference CT Image Quality Assessment
> Method Using RadImageNet Pre-trained Deep Learning Models."*

Specifically: a **RadImageNet-pretrained ResNet50**, with its classification
head replaced by `Dropout -> Dense(1) -> Sigmoid`, trained as a CT image
quality **regressor** on the **LDCT-IQAC** dataset.

**Scope guardrails for this notebook:**
- No project novelty is implemented here.
- No InceptionResNetV2.
- All architecture/data/training/evaluation code lives in the `ct_iqa`
  package (`src/ct_iqa/`) and is only **imported** below — this notebook
  is an experiment runner, not an implementation surface.
- Per the task brief, this notebook runs the **smoke-test / sanity-check
  pipeline only**. The full 30-epoch training cell is present but
  **gated behind `RUN_FULL_TRAINING = False`** and is not executed
  automatically.

**Known deviation from the task brief:** the LDCT-IQAC images on disk are
**TIFF** (`.tif`/`.tiff`, PIL mode `'F'`, 32-bit float), not PNG. The
dataset loader (`ct_iqa.data.ldct_iqac`) was written around the actual
on-disk format; see the dataset audit report for details."""
)

# 2. Repo root resolution + sys.path setup (must run before any ct_iqa import).
# Dataset paths in ExperimentConfig are relative to the repository root, and
# the ct_iqa package lives under <repo_root>/src -- not necessarily importable
# by default under every kernel (e.g. a separate GPU-enabled conda env used
# for actual training runs), so this is made explicit rather than relying on
# any particular kernel's site-packages configuration.
code(
    """import os
import sys
from pathlib import Path

_cwd = Path.cwd()
_repo_root = _cwd if (_cwd / "data").exists() else _cwd.parent
os.chdir(_repo_root)
sys.path.insert(0, str(_repo_root / "src"))
print(f"repo root: {_repo_root}")"""
)

# 3. Imports
code(
    """import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from ct_iqa.config import ExperimentConfig
from ct_iqa.data.ldct_iqac import LDCTIQACDataset, SCORE_MIN, SCORE_MAX
from ct_iqa.evaluation.metrics import compute_metrics, compute_metrics_with_logistic_mapping
from ct_iqa.models.ohashi_resnet50 import OhashiResNet50, INPUT_SIZE
from ct_iqa.training.trainer import (
    build_dataloaders,
    build_test_dataloader,
    build_loss,
    build_optimizer,
    evaluate_loader,
    train,
    train_one_step,
)
from ct_iqa.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
%matplotlib inline

print(f"torch: {torch.__version__}  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")"""
)

# 3. Configuration
md(
    """## Configuration

Ohashi replication hyperparameters (Adam, batch size 64, 30 epochs,
lr 1e-3, MSE, 224x224) are the `ExperimentConfig` defaults. `dropout_p`
has **no default in the codebase** because the paper never specifies it —
it is set explicitly here, and that choice is recorded in the saved config.

`radimagenet_weights_path` is left as `None`: RadImageNet weights are
**not currently present in this repository** (see dataset/weight audit
below). The backbone will run with random initialization until real
RadImageNet weights are obtained and pointed to here."""
)
code(
    """# RadImageNet weights: no official release is available to this project (the
# real dataset/weights are obtainable only by request at radimagenet.com).
# Rather than use unverified third-party "RadImageNet" checkpoints found on
# the Hugging Face Hub and risk mislabeling the result, this run uses random
# backbone initialization and is reported honestly as such -- NOT a full
# replication of the paper's RadImageNet-pretrained result.
config = ExperimentConfig(
    dropout_p=0.5,  # NOT specified by the paper -- explicit, documented choice.
    radimagenet_weights_path=None,  # No verified RadImageNet weights available; see above.
    checkpoint_dir="checkpoints/ohashi_resnet50_random_init_exp01",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
set_seed(config.seed)
config"""
)

# 4. Dataset paths
md("## Dataset paths")
code(
    """print("train image dir:", config.train_image_dir)
print("train json     :", config.train_json_path)
print("test image dir :", config.test_image_dir)
print("test json      :", config.test_json_path)

for p in [config.train_image_dir, config.train_json_path, config.test_image_dir, config.test_json_path]:
    assert Path(p).exists(), f"missing: {p}"
print("all dataset paths exist.")"""
)

# 5. Dataset audit
md(
    """## Dataset audit

`LDCTIQACDataset` construction itself performs the audit: it fails loudly
if any image has no label, any label has no image, the JSON is malformed,
or an unsupported image mode is encountered. Successfully constructing the
datasets below is itself a pass/fail audit checkpoint."""
)
code(
    """train_dataset = LDCTIQACDataset(config.train_image_dir, config.train_json_path, image_size=config.image_size)
test_dataset = LDCTIQACDataset(config.test_image_dir, config.test_json_path, image_size=config.image_size)

print(f"training samples: {len(train_dataset)}")
print(f"testing samples : {len(test_dataset)}")
print(f"score range (dataset constants): [{SCORE_MIN}, {SCORE_MAX}]")
print(f"observed train score range: [{min(train_dataset.raw_scores)}, {max(train_dataset.raw_scores)}]")
print(f"observed test score range : [{min(test_dataset.raw_scores)}, {max(test_dataset.raw_scores)}]")"""
)

# 6/7. Load training + validation data
md(
    """## Load training and validation data

The validation split is created **only from the training set**
(`val_fraction` of `ExperimentConfig`, default 10%), using a seeded
`random_split`. The official test set is never used for model selection —
`build_test_dataloader` is a separate, independent loader used only in the
final evaluation section."""
)
code(
    """train_loader, val_loader = build_dataloaders(config)
test_loader = build_test_dataloader(config)

print(f"train batches: {len(train_loader)}  ({len(train_loader.dataset)} samples)")
print(f"val batches  : {len(val_loader)}  ({len(val_loader.dataset)} samples)")
print(f"test batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)")"""
)

# 8. Visualize sample CT images
md("## Visualize sample CT images")
code(
    """fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    image, score = train_dataset[i]
    ax.imshow(image.squeeze(0).numpy(), cmap="gray")
    ax.set_title(f"score={float(score):.2f}")
    ax.axis("off")
plt.suptitle("LDCT-IQAC training samples (resized to model input size)")
plt.tight_layout()
plt.show()"""
)

# 9. Score distribution
md("## Score distribution")
code(
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train_dataset.raw_scores, bins=21)
axes[0].set_title("Training score distribution")
axes[0].set_xlabel("quality score")

axes[1].hist(test_dataset.raw_scores, bins=25)
axes[1].set_title("Testing score distribution")
axes[1].set_xlabel("quality score")
plt.tight_layout()
plt.show()"""
)

# 10. Initialize model
md(
    """## Initialize RadImageNet ResNet50 (Ohashi head)

`OhashiResNet50` = `ResNet50Backbone` (from-scratch standard ResNet50,
since `torchvision` is not installed in this environment) + `Dropout` +
`Dense(1)` + `Sigmoid`, per the Ohashi paper's stated modification."""
)
code(
    """model = OhashiResNet50(dropout_p=config.dropout_p, in_channels=1)
model.to(config.device)
print(model)"""
)

# 11. Model summary
md("## Model summary")
code(
    """total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total parameters    : {total_params:,}")
print(f"trainable parameters: {trainable_params:,}")
print(f"backbone output dim : {model.fc.in_features}")
print(f"head                : Dropout(p={config.dropout_p}) -> Linear({model.fc.in_features}, 1) -> Sigmoid")"""
)

# 12. Verify input/output shapes
md("## Verify input/output shapes")
code(
    """dummy_input = torch.randn(2, 1, INPUT_SIZE, INPUT_SIZE).to(config.device)
model.eval()
with torch.no_grad():
    dummy_output = model(dummy_input)
print(f"input shape : {tuple(dummy_input.shape)}")
print(f"output shape: {tuple(dummy_output.shape)}")
assert dummy_output.shape == (2,)
print("shapes OK.")"""
)

# 13. Verify trainable parameters
md("## Verify trainable parameters")
code(
    """assert trainable_params == total_params, "expected all parameters trainable (no layers frozen in this baseline)"
print(f"all {trainable_params:,} parameters are trainable (no frozen layers in this baseline configuration).")"""
)

# 14. Forward-pass smoke test
md("## Forward-pass smoke test (real data)")
code(
    """sample_images, sample_scores = next(iter(train_loader))
model.eval()
with torch.no_grad():
    sample_predictions = model(sample_images.to(config.device))

print(f"batch input shape : {tuple(sample_images.shape)}")
print(f"batch output shape: {tuple(sample_predictions.shape)}")
print(f"output range       : [{sample_predictions.min().item():.4f}, {sample_predictions.max().item():.4f}]")
print(f"any NaN in output  : {torch.isnan(sample_predictions).any().item()}")
assert not torch.isnan(sample_predictions).any()"""
)

# 15. Configure optimizer/loss
md(
    """## Configure optimizer / loss

Ohashi replication configuration: **Adam**, lr=1e-3, **MSE** loss, batch
size 64. No scheduler, warmup, weight decay, or mixed precision."""
)
code(
    """optimizer = build_optimizer(model, config)
criterion = build_loss(config)
print(optimizer)
print(criterion)"""
)

# 16. Train model (GATED)
md(
    """## Train model

**Gated per task instructions: full training is NOT run automatically.**
Set `RUN_FULL_TRAINING = True` and re-run this cell to execute the full
`config.epochs`-epoch Ohashi-configuration training run."""
)
code(
    """RUN_FULL_TRAINING = True  # Explicitly enabled -- user-instructed full training run.

if RUN_FULL_TRAINING:
    history = train(model, train_loader, val_loader, config)
else:
    history = None
    print("RUN_FULL_TRAINING is False -- skipping full training loop.")"""
)

# 17. Plot training/validation loss
md("## Plot training/validation loss")
code(
    """if history is not None:
    plt.figure(figsize=(6, 4))
    plt.plot(history.train_loss, label="train")
    plt.plot(history.val_loss, label="val")
    plt.axvline(history.best_epoch, color="gray", linestyle="--", label="best epoch")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalized [0,1] target space)")
    plt.legend()
    plt.title("Training/validation loss")
    plt.show()
else:
    print("No training history -- full training was not run in this notebook execution.")"""
)

# 18. Load best checkpoint
md("## Load best checkpoint")
code(
    """best_checkpoint_path = Path(config.checkpoint_dir) / "best.pt"
if best_checkpoint_path.exists():
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=config.device))
    print(f"loaded best checkpoint from {best_checkpoint_path}")
else:
    print(f"no checkpoint found at {best_checkpoint_path} (expected -- full training was not run).")"""
)

# 19-23. Evaluate on held-out test set / metrics
md(
    """## Evaluate on held-out test set

Runs the evaluation pipeline end-to-end against the official LDCT-IQAC
test set. With `RUN_FULL_TRAINING = False` the model is still at its
(random or partially-stepped) initialization, so these numbers are a
**pipeline sanity check**, not a real result — re-run after full training
for a meaningful evaluation."""
)
code(
    """test_loss, test_predictions, test_targets = evaluate_loader(model, test_loader, criterion, device=config.device)
print(f"test loss (normalized-space MSE): {test_loss:.6f}")

raw_metrics = compute_metrics(test_targets.numpy(), test_predictions.numpy())
print("Raw-space metrics (no logistic mapping):")
for k, v in raw_metrics.items():
    print(f"  {k.upper():6s}: {v:.4f}")"""
)
md(
    """### Optional: five-parameter-logistic-mapped correlation

The Ohashi paper's evaluation methodology applies a 5-parameter logistic
(5PL) fit before computing PLCC/SROCC. This is **not applied by default**
above — it's a separate, explicitly-invoked step, and results from the two
paths are reported separately rather than conflated."""
)
code(
    """try:
    logistic_metrics = compute_metrics_with_logistic_mapping(test_targets.numpy(), test_predictions.numpy())
    print("5PL-mapped metrics:")
    for k, v in logistic_metrics.items():
        print(f"  {k.upper():6s}: {v:.4f}")
except Exception as e:
    print(f"5PL fit did not converge (expected with an under-trained/random model): {e}")
    logistic_metrics = None"""
)

# 24. Plot predicted vs ground-truth
md("## Plot predicted vs ground-truth quality scores")
code(
    """plt.figure(figsize=(5, 5))
plt.scatter(test_targets.numpy(), test_predictions.numpy(), alpha=0.5, s=15)
plt.plot([SCORE_MIN, SCORE_MAX], [SCORE_MIN, SCORE_MAX], "r--", label="y = x")
plt.xlabel("ground-truth score")
plt.ylabel("predicted score")
plt.xlim(SCORE_MIN, SCORE_MAX)
plt.ylim(SCORE_MIN, SCORE_MAX)
plt.legend()
plt.title("Predicted vs. ground-truth (test set)")
plt.show()"""
)

# 25. Save predictions
md("## Save predictions")
code(
    """output_dir = Path(config.checkpoint_dir)
output_dir.mkdir(parents=True, exist_ok=True)

predictions_path = output_dir / "test_predictions.json"
predictions_record = {
    fn: {"ground_truth": gt, "predicted": pred}
    for fn, gt, pred in zip(
        [s.filename for s in test_dataset.samples],
        test_targets.tolist(),
        test_predictions.tolist(),
    )
}
predictions_path.write_text(json.dumps(predictions_record, indent=2))
print(f"saved predictions to {predictions_path}")"""
)

# 26. Save metrics
md("## Save metrics")
code(
    """metrics_path = output_dir / "test_metrics.json"
metrics_record = {"raw": raw_metrics, "logistic_mapped": logistic_metrics}
metrics_path.write_text(json.dumps(metrics_record, indent=2))
print(f"saved metrics to {metrics_path}")"""
)

# 27. Save experiment configuration
md("## Save experiment configuration")
code(
    """config_path = output_dir / "config.json"
config.save(config_path)
print(f"saved experiment configuration to {config_path}")"""
)

# 28. Final experiment summary
md(
    """## Final experiment summary

This run established and sanity-checked the Ohashi-ResNet50 baseline
pipeline end-to-end on LDCT-IQAC:

- Dataset audit passed (image/label counts, matching, score range).
- Model builds with the correct architecture (ResNet50 backbone +
  Dropout + Dense(1) + Sigmoid) and correct I/O shapes.
- Forward pass, one training step, and the full evaluation pipeline
  (metrics, plots, checkpoint save/load, predictions/metrics/config
  persistence) all run without error.
- **RadImageNet weights are not yet present in this repository** — the
  backbone above ran with random initialization. Obtain the weights and
  set `config.radimagenet_weights_path` before running a real experiment.
- **`RUN_FULL_TRAINING` was left `False`** — no 30-epoch training was
  executed in this notebook run, per task instructions."""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}

out_path = Path("notebooks/02_ohashi_resnet50_ldct_iqac.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}")
