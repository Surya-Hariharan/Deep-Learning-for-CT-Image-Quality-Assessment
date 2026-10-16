"""Frozen-backbone ridge probe over three input conventions (rules: docs/methodology/PREPROCESSING.md, Part 1).

Uses ONLY the training partition to fit (alpha by 5-fold CV on train) and the validation partition to evaluate.
Calibration and the official test set are never read. Writes `reports/preprocessing_probe/probe_results.json`.

    python tools/preprocessing_probe.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ct_iqa.evaluation.metrics import compute_point_metrics
from ct_iqa.models.backbone import ResNet50Backbone
from ct_iqa.models.weights import load_backbone_weights
from ct_iqa.preprocessing.crop import center_crop

WEIGHTS = "weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt"
SHA = "45cba0d53f1d0ec1e1e2ae569a985b56a21da71c4a7271328b01ed5ea9a0f785"
MANIFEST = Path("experiments/study_v2/protocol/split_manifest.json")
LABELS = Path("data/labels/ldct_iqac/train.json")
IMG_DIR = Path("data/raw/ldct_iqac/train/image")
OUT = Path("reports/preprocessing_probe")
ALPHAS = np.logspace(-3, 4, 15)
N_BOOT, BOOT_SEED, CV_SEED = 2000, 0, 0
BGR_MEAN = np.array([103.939, 116.779, 123.68])
IMNET_MEAN, IMNET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def conv_p1(x):  # (N,224,224) in [0,1] -> (N,3,224,224)
    return np.repeat((2 * x - 1)[:, None], 3, axis=1)


def conv_p2(x):
    return np.stack([(x * 255 - m) / 255 for m in BGR_MEAN], axis=1)


def conv_p3(x):
    return np.stack([(x - m) / s for m, s in zip(IMNET_MEAN, IMNET_STD)], axis=1)


CONVENTIONS = {"P1_2x-1": conv_p1, "P2_caffe_style": conv_p2, "P3_imagenet_meanstd": conv_p3}


def load_split(names):
    return np.stack([center_crop(np.asarray(Image.open(IMG_DIR / n), dtype=np.float32), 224) for n in names])


@torch.no_grad()
def features(model, x, device, bs=50):
    out = []
    for i in range(0, len(x), bs):
        out.append(model(torch.from_numpy(x[i:i + bs]).float().to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def ridge_fit(X, y, alpha):
    ym = y.mean()
    A = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ (y - ym))
    return w, ym


def cv_alpha(X, y, k=5):
    idx = np.random.default_rng(CV_SEED).permutation(len(y))
    folds = np.array_split(idx, k)
    best, best_rmse, table = None, np.inf, {}
    for a in ALPHAS:
        sq = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            w, ym = ridge_fit((X[tr] - mu) / sd, y[tr], a)
            sq.append(((((X[f] - mu) / sd) @ w + ym) - y[f]) ** 2)
        rmse = float(np.sqrt(np.concatenate(sq).mean()))
        table[float(a)] = rmse
        if rmse < best_rmse:
            best, best_rmse = float(a), rmse
    return best, best_rmse, table


def plcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    tr_n, va_n = manifest["train"], manifest["validation"]
    ytr = np.array([labels[n] for n in tr_n]); yva = np.array([labels[n] for n in va_n])
    Xtr_img, Xva_img = load_split(tr_n), load_split(va_n)

    model = ResNet50Backbone(in_channels=3, bn_eps=1.001e-5)
    rep = load_backbone_weights(model, WEIGHTS, expected_sha256=SHA, strict=True)
    model.eval().to(device)
    res = {"generated_utc": datetime.now(timezone.utc).isoformat(), "device": device, "torch": torch.__version__,
           "weights_sha256": SHA, "weights_report": str(rep), "n_train": len(tr_n), "n_validation": len(va_n),
           "n_boot": N_BOOT, "conventions": {}}
    preds = {}
    for name, fn in CONVENTIONS.items():
        Ftr, Fva = features(model, fn(Xtr_img), device), features(model, fn(Xva_img), device)
        assert np.isfinite(Ftr).all() and np.isfinite(Fva).all()
        alpha, cv_rmse, table = cv_alpha(Ftr, ytr)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
        w, ym = ridge_fit((Ftr - mu) / sd, ytr, alpha)
        p = ((Fva - mu) / sd) @ w + ym
        preds[name] = p
        res["conventions"][name] = {"alpha": alpha, "cv_rmse_train": cv_rmse, "cv_table": table,
                                    "validation": compute_point_metrics(yva, p),
                                    "feature_mean": float(Ftr.mean()), "feature_std": float(Ftr.std())}
    # paired bootstrap of PLCC differences
    rng = np.random.default_rng(BOOT_SEED)
    boots = {n: [] for n in preds}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yva), len(yva))
        if np.ptp(yva[i]) == 0:
            continue
        for n, p in preds.items():
            boots[n].append(plcc(yva[i], p[i]))
    B = {n: np.array(v) for n, v in boots.items()}
    names = list(preds)
    pairs = {}
    for a in names:
        for b in names:
            if a < b:
                d = B[a] - B[b]
                lo, hi = np.percentile(d, [2.5, 97.5])
                pairs[f"{a}-{b}"] = {"plcc_diff": plcc(yva, preds[a]) - plcc(yva, preds[b]),
                                     "ci95": [float(lo), float(hi)], "half_width": float((hi - lo) / 2)}
    res["paired_bootstrap_plcc"] = pairs
    best = max(names, key=lambda n: res["conventions"][n]["validation"]["plcc"])
    others = [n for n in names if n != best]
    distinguishable = all(
        (pairs.get(f"{best}-{o}") or {"plcc_diff": -pairs[f"{o}-{best}"]["plcc_diff"],
                                      "half_width": pairs[f"{o}-{best}"]["half_width"]})["plcc_diff"]
        > (pairs.get(f"{best}-{o}") or pairs[f"{o}-{best}"])["half_width"] for o in others)
    res["highest_validation_plcc"] = best
    res["distinguishably_better_by_declared_rule"] = bool(distinguishable)
    res["project_convention_unchanged_by_probe"] = "P1_2x-1 (declared decision rule, PREPROCESSING.md Part 1 item 6)"
    (OUT / "probe_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n in names:
        v = res["conventions"][n]
        print(n, "alpha", v["alpha"], {k: round(v["validation"][k], 4) for k in ("plcc", "srocc", "krocc", "rmse")})
    print("best:", best, "distinguishable:", distinguishable)
    print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    main()
