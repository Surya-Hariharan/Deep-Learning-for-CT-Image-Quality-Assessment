"""Read-only dataset audit and EDA for the LDCT-IQAC data used by study v2.

Writes tables (CSV/JSON) and figures to `reports/eda/`. No model is involved and no data file is modified.
Test labels/images are read only for descriptive dataset facts (counts, distribution, integrity, duplicate
checks against the training set); no development decision is taken from them.

    python tools/run_eda.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ct_iqa.data.splits import manifest_hash, verify_manifest  # noqa: E402

OUT = Path("reports/eda")
TRAIN_JSON = Path("data/labels/ldct_iqac/train.json")
TEST_JSON = Path("data/labels/ldct_iqac/test.json")
TRAIN_DIR = Path("data/raw/ldct_iqac/train/image")
TEST_DIR = Path("data/raw/ldct_iqac/test/images")
MANIFEST = Path("experiments/study_v2/protocol/split_manifest.json")
EXPECTED = {"n_train": 800, "n_val": 100, "n_cal": 100, "seed": 2026, "prefix": "be5831c5", "suffix": "0a16"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(arr: np.ndarray, size: int = 8) -> int:
    im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def stats(v: np.ndarray) -> dict:
    q = np.quantile(v, [0, .05, .25, .5, .75, .95, 1])
    return {"n": int(len(v)), "min": q[0], "q05": q[1], "q25": q[2], "median": q[3], "q75": q[4], "q95": q[5],
            "max": q[6], "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "n_unique": int(len(np.unique(v)))}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def scan_images(directory: Path, names: list[str]) -> tuple[list[dict], dict]:
    rows, errors = [], {}
    for n in names:
        p = directory / n
        try:
            with Image.open(p) as im:
                mode, size = im.mode, im.size
                arr = np.asarray(im, dtype=np.float64)
            rows.append({"filename": n, "mode": mode, "width": size[0], "height": size[1],
                         "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
                         "std": float(arr.std()), "finite": bool(np.isfinite(arr).all()),
                         "sha256": sha256(p), "dhash": dhash(arr / (arr.max() or 1.0)),
                         "_thumb": np.asarray(Image.fromarray(arr.astype(np.float32)).resize((64, 64), Image.BILINEAR),
                                              dtype=np.float32).ravel()})
        except Exception as e:  # noqa: BLE001 - report every unreadable file
            errors[n] = repr(e)
    return rows, errors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_labels = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    test_labels = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script": "tools/run_eda.py"}

    # --- manifest integrity -------------------------------------------------------------------------------
    verify_manifest(manifest, train_labels)
    parts = {k: manifest[k] for k in ("train", "validation", "calibration")}
    h = manifest_hash(manifest["seed"], parts)
    emb = manifest.get("manifest_sha256")
    ok = (len(parts["train"]), len(parts["validation"]), len(parts["calibration"]), manifest["seed"]) == (
        EXPECTED["n_train"], EXPECTED["n_val"], EXPECTED["n_cal"], EXPECTED["seed"]) and \
        h.startswith(EXPECTED["prefix"]) and h.endswith(EXPECTED["suffix"]) and h == emb
    overlap = {f"{a}&{b}": len(set(parts[a]) & set(parts[b])) for a, b in
               (("train", "validation"), ("train", "calibration"), ("validation", "calibration"))}
    report["manifest"] = {"file_sha256": sha256(MANIFEST), "canonical_hash_recomputed": h, "canonical_hash_embedded": emb,
                          "matches_expected_prefix_suffix_and_embedded": bool(ok), "counts":
                          {k: len(v) for k, v in parts.items()}, "seed": manifest["seed"], "pairwise_overlap": overlap,
                          "label_file_sha256": {"train": sha256(TRAIN_JSON), "test": sha256(TEST_JSON)}}

    # --- files / labels -----------------------------------------------------------------------------------
    train_disk = sorted(p.name for p in TRAIN_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    test_disk = sorted(p.name for p in TEST_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    report["files"] = {
        "train_images_on_disk": len(train_disk), "train_labels": len(train_labels),
        "train_missing_image": sorted(set(train_labels) - set(train_disk)),
        "train_missing_label": sorted(set(train_disk) - set(train_labels)),
        "test_images_on_disk": len(test_disk), "test_labels": len(test_labels),
        "test_missing_image": sorted(set(test_labels) - set(test_disk)),
        "test_missing_label": sorted(set(test_disk) - set(test_labels))}
    for name, lab in (("train", train_labels), ("test", test_labels)):
        v = np.array(list(lab.values()), dtype=float)
        report["files"][f"{name}_labels_numeric_finite_in_0_4"] = bool(np.isfinite(v).all() and v.min() >= 0 and v.max() <= 4)

    # --- label statistics ---------------------------------------------------------------------------------
    part_labels = {k: np.array([train_labels[f] for f in v]) for k, v in parts.items()}
    part_labels["train_all_1000"] = np.array(list(train_labels.values()))
    part_labels["official_test"] = np.array(list(test_labels.values()))
    rows = [{"partition": k, **stats(v)} for k, v in part_labels.items()]
    write_csv(OUT / "label_statistics.csv", rows)
    freq_rows = []
    for k, v in part_labels.items():
        c = Counter(np.round(v, 4).tolist())
        for s, n in sorted(c.items()):
            freq_rows.append({"partition": k, "score": s, "count": n, "fraction": n / len(v)})
    write_csv(OUT / "score_frequency.csv", freq_rows)
    from scipy import stats as sps  # local import keeps module import light
    ks = {}
    for a, b in (("train", "validation"), ("train", "calibration"), ("validation", "calibration")):
        r = sps.ks_2samp(part_labels[a], part_labels[b])
        ks[f"{a}_vs_{b}"] = {"ks_stat": float(r.statistic), "p_value": float(r.pvalue)}
    r = sps.ks_2samp(part_labels["train_all_1000"], part_labels["official_test"])
    ks["train1000_vs_official_test"] = {"ks_stat": float(r.statistic), "p_value": float(r.pvalue)}
    report["partition_ks_tests_descriptive"] = ks

    # --- image properties ---------------------------------------------------------------------------------
    tr_rows, tr_err = scan_images(TRAIN_DIR, train_disk)
    te_rows, te_err = scan_images(TEST_DIR, test_disk)
    for r in tr_rows:
        r["split"] = next(k for k, v in parts.items() if r["filename"] in set(v))
        r["label"] = train_labels[r["filename"]]
    for r in te_rows:
        r["split"] = "official_test"
        r["label"] = test_labels.get(r["filename"], float("nan"))
    write_csv(OUT / "image_properties.csv", [{k: v for k, v in r.items() if k != "_thumb"} for r in tr_rows + te_rows])
    report["unreadable_files"] = {"train": tr_err, "test": te_err}
    allr = tr_rows + te_rows
    report["image_properties"] = {
        "modes": dict(Counter(r["mode"] for r in allr)),
        "sizes": {f"{w}x{h}": n for (w, h), n in Counter((r["width"], r["height"]) for r in allr).items()},
        "global_min": min(r["min"] for r in allr), "global_max": max(r["max"] for r in allr),
        "all_finite": all(r["finite"] for r in allr),
        "per_split_mean_pixel": {s: float(np.mean([r["mean"] for r in allr if r["split"] == s]))
                                 for s in sorted({r["split"] for r in allr})},
        "per_split_std_pixel": {s: float(np.mean([r["std"] for r in allr if r["split"] == s]))
                                for s in sorted({r["split"] for r in allr})}}

    # --- duplicates ---------------------------------------------------------------------------------------
    by_hash = defaultdict(list)
    for r in allr:
        by_hash[r["sha256"]].append((r["split"], r["filename"]))
    exact = [v for v in by_hash.values() if len(v) > 1]
    cross_exact = [v for v in exact if len({s for s, _ in v}) > 1]
    by_dh = defaultdict(list)
    for r in allr:
        by_dh[r["dhash"]].append((r["split"], r["filename"]))
    near = [v for v in by_dh.values() if len(v) > 1]
    cross_near = [v for v in near if len({s for s, _ in v}) > 1]
    report["duplicates"] = {"exact_groups": len(exact), "exact_cross_partition_groups": len(cross_exact),
                            "dhash8_collision_groups": len(near), "dhash8_cross_partition_groups": len(cross_near),
                            "examples_exact": exact[:10], "examples_dhash_cross": cross_near[:10],
                            "note": "dHash(8x8) collisions are a coarse screen, not proof of duplication."}
    # near-duplicate screen: Pearson correlation of 64x64 thumbnails; report cross-partition pairs above 0.995
    T = np.stack([r["_thumb"] for r in allr]).astype(np.float64)
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)
    C = T @ T.T / T.shape[1]
    sp = np.array([r["split"] for r in allr])
    iu = np.triu_indices(len(allr), 1)
    cross = sp[iu[0]] != sp[iu[1]]
    vals = C[iu][cross]
    order = np.argsort(-vals)[:10]
    ci, cj = iu[0][cross], iu[1][cross]
    report["duplicates"]["thumb_corr_cross_partition"] = {
        "max": float(vals.max()), "n_pairs_gt_0.995": int((vals > 0.995).sum()), "n_pairs_gt_0.98": int((vals > 0.98).sum()),
        "top10": [[allr[ci[k]]["filename"], allr[ci[k]]["split"], allr[cj[k]]["filename"], allr[cj[k]]["split"],
                   float(vals[k])] for k in order]}
    # same-anatomy groups: connected components of thumbnail-correlation > threshold over all 1300 images
    Cm = C.copy()
    np.fill_diagonal(Cm, 0)
    groups = {}
    for th in (0.999, 0.99):
        adj = Cm > th
        seen, comps = set(), []
        for i in range(len(allr)):
            if i in seen:
                continue
            st, comp = [i], []
            while st:
                k = st.pop()
                if k in seen:
                    continue
                seen.add(k)
                comp.append(k)
                st += list(np.where(adj[k])[0])
            comps.append(comp)
        info = {"n_groups": len(comps), "largest_sizes": sorted((len(c) for c in comps), reverse=True)[:10],
                "singletons": sum(len(c) == 1 for c in comps)}
        for part in ("validation", "calibration", "official_test"):
            idx = [i for i in range(len(allr)) if sp[i] == part]
            comp_of = {k: ci_ for ci_, c in enumerate(comps) for k in c}
            tr_groups = {comp_of[i] for i in range(len(allr)) if sp[i] == "train"}
            info[f"{part}_frac_with_train_groupmate"] = float(np.mean([comp_of[i] in tr_groups for i in idx]))
        groups[str(th)] = info
        if th == 0.999:
            write_csv(OUT / "anatomy_groups_thr0.999.csv", [
                {"group": gi, "filename": allr[k]["filename"], "split": allr[k]["split"], "label": allr[k]["label"]}
                for gi, c in enumerate(comps) for k in c])
    report["anatomy_groups"] = {"method": "connected components of 64x64-thumbnail Pearson correlation > threshold",
                                "by_threshold": groups,
                                "interpretation": "Near-identical slices at different quality levels share a group; an "
                                "image-level split places group members in different partitions (leakage risk)."}
    report["duplicates"].pop("examples_dhash_cross", None)
    report["duplicates"].pop("examples_exact", None)
    report["duplicates"]["note"] = ("dHash(8x8) is reported but uninformative for CT slices (many collisions between "
                                    "different slices); the thumbnail-correlation screen is the near-duplicate evidence.")
    for r in allr:
        r.pop("_thumb", None)

    # --- figures ------------------------------------------------------------------------------------------
    bins = np.linspace(-0.1, 4.1, 22)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(part_labels["train_all_1000"], bins=bins, color="#4c78a8")
    ax[0].set(title="Quality score, all 1000 labeled images", xlabel="score [0,4]", ylabel="count")
    for k in ("train", "validation", "calibration", "official_test"):
        ax[1].hist(part_labels[k], bins=bins, density=True, histtype="step", label=f"{k} (n={len(part_labels[k])})")
    ax[1].set(title="Density by partition", xlabel="score [0,4]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "score_distributions.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist([r["mean"] for r in tr_rows], bins=30, alpha=.7, label="train/val/cal")
    ax[0].hist([r["mean"] for r in te_rows], bins=30, alpha=.7, label="test")
    ax[0].set(title="Per-image mean pixel", xlabel="mean"); ax[0].legend()
    ax[1].scatter([r["mean"] for r in tr_rows], [r["label"] for r in tr_rows], s=6, alpha=.5)
    ax[1].set(title="Mean pixel vs score (train set)", xlabel="mean pixel", ylabel="score")
    fig.tight_layout()
    fig.savefig(OUT / "pixel_statistics.png", dpi=150)
    plt.close(fig)
    # representative images: one per score level 0,1,2,3,4 from the training partition
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax_, s in zip(axs, (0.0, 1.0, 2.0, 3.0, 4.0)):
        cand = sorted((abs(train_labels[f] - s), f) for f in parts["train"])[0][1]
        with Image.open(TRAIN_DIR / cand) as im:
            ax_.imshow(np.asarray(im, dtype=float), cmap="gray", vmin=0, vmax=1)
        ax_.set_title(f"{cand}: {train_labels[cand]:.1f}", fontsize=9)
        ax_.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "representative_images.png", dpi=150)
    plt.close(fig)

    (OUT / "dataset_audit.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("manifest", "files", "unreadable_files", "image_properties",
                                              "duplicates", "partition_ks_tests_descriptive")}, indent=1, default=str))


if __name__ == "__main__":
    main()
