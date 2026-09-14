"""Final independent test-set evaluation of experiment 001's checkpoint.

Run this script (with the project venv) to reproduce
`results/metrics/001_resnet50_baseline_test_metrics.json`,
`results/predictions/001_resnet50_baseline_test_predictions.csv`, and the
`001_resnet50_baseline_test_*.png` diagnostic figures from scratch, given
`experiments/001_resnet50_baseline/checkpoint/best.pt`.

Why this exists as a standalone script rather than a notebook: this
evaluation used to live in `notebooks/04_evaluation/01_test_set_evaluation.ipynb`,
but that notebook was repurposed (2026-09-13) to evaluate experiment 003
instead, using experiment 001's numbers only as a fixed, hardcoded
reference for its comparison table (`EXPERIMENT_001_RAW_METRICS` in
`tools/build_evaluation_notebook.py`) -- it no longer reproduces them. This
script is the current, actual reproduction path for experiment 001's test
metrics (e.g. after the 2026-09-14 checkpoint-selection fix required
retraining and re-evaluating experiment 001 -- see
`docs/internal/decisions/README.md`). Mirrors
`notebooks/04_evaluation/01_test_set_evaluation.ipynb`'s inference/metrics
methodology for experiment 003 so both experiments' numbers come from the
same evaluation code path; performs no retraining, no fine-tuning, and no
checkpoint/model selection of any kind. **TEST SET USED ONLY FOR FINAL
EVALUATION.**

If experiment 001 is ever retrained again, re-run this script afterward
and manually update the numbers in `README.md`,
`docs/replication/final_results.md`, and
`experiments/001_resnet50_baseline/README.md` to match (they are not
regenerated automatically).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_cwd = Path.cwd()
_repo_root = _cwd if (_cwd / "data").exists() else _cwd.parent  # tools/ -> repo root
os.chdir(_repo_root)
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ct_iqa.config import ExperimentConfig
from ct_iqa.data.ldct_iqac import SCORE_MIN, SCORE_MAX, denormalize_score, normalize_score
from ct_iqa.data.loader import build_test_dataloader
from ct_iqa.evaluation.metrics import compute_metrics
from ct_iqa.models.ohashi_resnet50 import OhashiResNet50
from ct_iqa.training.checkpointing import best_checkpoint_path, load_checkpoint, load_checkpoint_metadata

EXPERIMENT_NAME = "001_resnet50_baseline"

config = ExperimentConfig.from_yaml_files(
    experiment_dir=f"experiments/{EXPERIMENT_NAME}",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
checkpoint_path = best_checkpoint_path(config.checkpoint_dir)
assert checkpoint_path.exists(), (
    f"no checkpoint found at {checkpoint_path} -- run the baseline training notebook first "
    "(notebooks/03_training/01_resnet50_baseline.ipynb)"
)

metadata = load_checkpoint_metadata(config.checkpoint_dir, map_location=config.device)
checkpoint_config = metadata.get("config", {})
print("=== CHECKPOINT SUMMARY ===")
print(f"path            : {checkpoint_path}")
print(f"epoch           : {metadata.get('epoch')}")
print(f"checkpoint_type : {metadata.get('checkpoint_type')!r}")
print(f"seed            : {metadata.get('seed')}")

print()
print("=== ARCHITECTURE CONSISTENCY CHECK (checkpoint's config vs. current configs/*.yaml) ===")
_architecture_ok = True
for key in ["dropout_p", "in_channels", "image_size"]:
    current_value = getattr(config, key)
    checkpoint_value = checkpoint_config.get(key)
    match = current_value == checkpoint_value
    _architecture_ok &= match
    print(f"  {key:<12}: checkpoint={checkpoint_value!r}  current={current_value!r}  match={match}")
assert _architecture_ok, "architecture-relevant configuration has changed since this checkpoint was trained"

model = OhashiResNet50(dropout_p=config.dropout_p, in_channels=config.in_channels)  # fresh instance
load_checkpoint(model, config.checkpoint_dir, map_location=config.device)  # loads model_state_dict only
model.to(config.device)
model.eval()

test_loader = build_test_dataloader(config)
test_dataset = test_loader.dataset
filenames = [s.filename for s in test_dataset.samples]
assert len(test_dataset) == 300, f"expected 300 test images, found {len(test_dataset)}"

records = []
with torch.no_grad():
    offset = 0
    for images, raw_scores in test_loader:
        images = images.to(config.device)
        predictions_normalized = model(images)  # Sigmoid output, [0,1]

        batch_size = images.shape[0]
        batch_filenames = filenames[offset : offset + batch_size]
        offset += batch_size

        gt_normalized = normalize_score(raw_scores)
        pred_raw_scale = denormalize_score(predictions_normalized.cpu())

        for fn, gt_r, gt_n, pred_n, pred_r in zip(
            batch_filenames, raw_scores.tolist(), gt_normalized.tolist(),
            predictions_normalized.cpu().tolist(), pred_raw_scale.tolist(),
        ):
            records.append({
                "filename": fn,
                "ground_truth_score": gt_r,
                "ground_truth_normalized": gt_n,
                "prediction_normalized": pred_n,
                "prediction_score": pred_r,
            })

assert len(records) == 300, f"expected 300 inference records, got {len(records)}"
print(f"\ninference complete: {len(records)} predictions collected.")

gt_raw_arr = np.array([r["ground_truth_score"] for r in records])
pred_raw_arr = np.array([r["prediction_score"] for r in records])
gt_norm_arr = np.array([r["ground_truth_normalized"] for r in records])
pred_norm_arr = np.array([r["prediction_normalized"] for r in records])

raw_scale_metrics = compute_metrics(gt_raw_arr, pred_raw_arr)  # PLCC/SROCC/KROCC scale-invariant; MSE in [0,4]^2 units
normalized_scale_mse = compute_metrics(gt_norm_arr, pred_norm_arr)["mse"]
mae_raw = float(np.mean(np.abs(pred_raw_arr - gt_raw_arr)))
rmse_raw = float(np.sqrt(raw_scale_metrics["mse"]))

print("=== FINAL RAW TEST METRICS (original [0,4] score scale, n=300) ===")
print(f"  PLCC : {raw_scale_metrics['plcc']:.4f}")
print(f"  SROCC: {raw_scale_metrics['srocc']:.4f}")
print(f"  KROCC: {raw_scale_metrics['krocc']:.4f}")
print(f"  MSE  : {raw_scale_metrics['mse']:.4f}   (score-unit^2, scale [0,4])")
print(f"  MAE  : {mae_raw:.4f}   (score units, scale [0,4])")
print(f"  RMSE : {rmse_raw:.4f}   (score units, scale [0,4])")


def describe(arr: np.ndarray, name: str) -> dict:
    return {
        "name": name, "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std()),
    }


pred_stats = describe(pred_raw_arr, "prediction")
gt_stats = describe(gt_raw_arr, "ground_truth")
residuals = pred_raw_arr - gt_raw_arr
residual_stats = {
    "mean": float(residuals.mean()), "median": float(np.median(residuals)),
    "std": float(residuals.std()), "min": float(residuals.min()), "max": float(residuals.max()),
}

# --- persist (nested "raw" schema -- matches the pre-existing
# results/metrics/001_resnet50_baseline_test_metrics.json shape that
# tools/build_prediction_analysis_notebook.py already reads) ---
metrics_record = {
    "experiment_name": EXPERIMENT_NAME,
    "checkpoint": str(checkpoint_path),
    "checkpoint_epoch": metadata.get("epoch"),
    "checkpoint_type": metadata.get("checkpoint_type"),
    "dataset": "LDCT-IQAC test split",
    "number_of_test_images": len(records),
    "model_description": "RadImageNet-pretrained ResNet50 -> GAP -> Dropout(0.5) -> Linear(2048,1) -> Sigmoid (Ohashi-style, adapted to LDCT-IQAC)",
    "evaluation_scale": "raw LDCT-IQAC score scale [0,4] for PLCC/SROCC/KROCC/MSE/MAE/RMSE; normalized [0,1] MSE also reported",
    "seed": metadata.get("seed"),
    "selection_metric": checkpoint_config.get("selection_metric"),
    "raw": {
        "PLCC": raw_scale_metrics["plcc"],
        "SROCC": raw_scale_metrics["srocc"],
        "KROCC": raw_scale_metrics["krocc"],
        "MSE": raw_scale_metrics["mse"],
        "MAE": mae_raw,
        "RMSE": rmse_raw,
        "MSE_normalized_0_1_scale": normalized_scale_mse,
    },
    "calibrated": None,
    "calibration_note": (
        "Calibrated final-test evaluation was not performed because no non-test "
        "calibration protocol has been established. See "
        "docs/replication/final_results.md, section I."
    ),
    "prediction_distribution": {"ground_truth": gt_stats, "prediction": pred_stats},
    "residuals": residual_stats,
}
metrics_json_path = Path(f"results/metrics/{EXPERIMENT_NAME}_test_metrics.json")
metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
metrics_json_path.write_text(json.dumps(metrics_record, indent=2))
print(f"\nsaved metrics to {metrics_json_path}")

predictions_csv_path = Path(f"results/predictions/{EXPERIMENT_NAME}_test_predictions.csv")
predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
with predictions_csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename", "ground_truth_score", "prediction_score",
        "ground_truth_normalized", "prediction_normalized",
    ])
    writer.writeheader()
    writer.writerows(records)
assert sum(1 for _ in predictions_csv_path.open(encoding="utf-8")) - 1 == 300
print(f"saved {len(records)} predictions to {predictions_csv_path}")

# --- diagnostic figures (mirrors notebooks/04_evaluation/01_test_set_evaluation.ipynb's
# plotting cells for experiment 003) ---
figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(gt_raw_arr, pred_raw_arr, alpha=0.5, s=15)
ax.plot([SCORE_MIN, SCORE_MAX], [SCORE_MIN, SCORE_MAX], "r--", label="y = x (identity)")
ax.set_xlabel("ground-truth score [0,4]")
ax.set_ylabel("predicted score [0,4]")
ax.set_xlim(SCORE_MIN, SCORE_MAX)
ax.set_ylim(SCORE_MIN, SCORE_MAX)
ax.legend()
ax.set_title(f"Experiment 001 (Baseline) -- Predicted vs. Ground Truth (test set, n={len(records)})")
plt.tight_layout()
plt.savefig(figures_dir / "001_resnet50_baseline_test_pred_vs_gt.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(residuals, bins=30, edgecolor="black")
ax.axvline(0.0, color="red", linestyle="--", label="zero error")
ax.axvline(residuals.mean(), color="gray", linestyle=":", label=f"mean residual = {residuals.mean():+.3f}")
ax.set_xlabel("prediction - ground truth [0,4] scale")
ax.set_ylabel("count")
ax.set_title(f"Experiment 001 (Baseline) -- Residual Distribution (test set, n={len(records)})")
ax.legend()
plt.tight_layout()
plt.savefig(figures_dir / "001_resnet50_baseline_test_residuals.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(gt_raw_arr, bins=25, alpha=0.5, density=True, label=f"ground truth (n={len(gt_raw_arr)})")
ax.hist(pred_raw_arr, bins=25, alpha=0.5, density=True, label=f"prediction (n={len(pred_raw_arr)})")
ax.set_xlabel("quality score [0,4]")
ax.set_ylabel("density")
ax.set_title("Experiment 001 (Baseline) -- Ground-Truth vs. Prediction Distribution (test set)")
ax.legend()
plt.tight_layout()
plt.savefig(figures_dir / "001_resnet50_baseline_test_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved 001_resnet50_baseline_test_{pred_vs_gt,residuals,score_distribution}.png")
print()
print("=== TEST SET CONTAMINATION CHECK ===")
print(f"- Test set: {len(records)} images (expected 300).")
print("- Training (experiment 001) completed before this script ran.")
print(f"- Checkpoint selected by config.selection_metric={checkpoint_config.get('selection_metric')!r} "
      "on the validation split only -- no test-set metric was involved in selecting it.")
print("- This script never constructs a training Dataset/DataLoader.")
print()
print("TEST SET USED ONLY FOR FINAL EVALUATION.")
