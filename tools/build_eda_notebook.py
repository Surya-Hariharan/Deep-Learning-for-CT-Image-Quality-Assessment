"""Generator for notebooks/01_exploration/01_dataset_eda.ipynb.

Run this script to (re)build the notebook from the cell definitions below.
Not part of the `ct_iqa` package -- a dev-time notebook-generation tool
(see docs/internal/repository_architecture_audit.md on why these generators live
under `tools/` rather than `scripts/`).

The generated notebook is a research record: hand-editing the `.ipynb` JSON
directly will drift out of sync with this generator (as previously
happened with stale dataset paths) -- always edit this file and re-run it.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# 1. Title and objective
md(
    """# Dataset EDA -- LDCT-IQAC

## Objective

Exploratory data analysis of the **LDCT-IQAC** dataset used by this
project's Ohashi-ResNet50 baseline: file/label consistency, image
properties, duplicate-image check, score distribution, and sample image
grids.

This notebook performs **no training and defines no model architecture** --
it works directly against the raw dataset files. Any preprocessing/model
code referenced elsewhere in this project lives in `src/ct_iqa/`, not here."""
)

# 2. Imports and repo-root resolution
code(
    """import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_cwd = Path.cwd()
_repo_root = _cwd if (_cwd / "data").exists() else _cwd.parent.parent  # notebooks/01_exploration/ -> repo root
os.chdir(_repo_root)
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))  # only needed until `pip install -e .` is run
print(f"working directory: {Path.cwd()}")
%matplotlib inline

from ct_iqa.config import ExperimentConfig"""
)

# 3. Dataset paths
md(
    """## Configuration

Dataset paths, loaded via `ExperimentConfig.from_yaml_files()` -- the same
single authoritative configuration source used by the training and
evaluation notebooks (see `src/ct_iqa/config.py`, `configs/dataset.yaml`)."""
)
code(
    """config = ExperimentConfig.from_yaml_files()
TRAIN_IMAGE_DIR = Path(config.train_image_dir)
TRAIN_JSON_PATH = Path(config.train_json_path)
TEST_IMAGE_DIR = Path(config.test_image_dir)
TEST_JSON_PATH = Path(config.test_json_path)

for p in [TRAIN_IMAGE_DIR, TRAIN_JSON_PATH, TEST_IMAGE_DIR, TEST_JSON_PATH]:
    assert p.exists(), f"missing: {p}"

print("=== DATASET PATHS ===")
print(f"train images : {TRAIN_IMAGE_DIR}")
print(f"train labels : {TRAIN_JSON_PATH}")
print(f"test images  : {TEST_IMAGE_DIR}")
print(f"test labels  : {TEST_JSON_PATH}")"""
)

# 4. Data
md(
    """## Data

Load and inspect the JSON label files directly (no `ct_iqa.data.ldct_iqac`
involved yet -- this section validates the raw files themselves)."""
)
code(
    """with TRAIN_JSON_PATH.open() as f:
    train_labels = json.load(f)
with TEST_JSON_PATH.open() as f:
    test_labels = json.load(f)

print("=== JSON FILE PURPOSE ===")
print("Each JSON file is a flat {filename: quality_score} mapping -- one")
print("entry per CT image, giving that image's radiologist-derived quality score.")
print()
print(f"train.json: {len(train_labels)} entries, type={type(train_labels).__name__}")
print("first 5 entries:")
for k, v in list(train_labels.items())[:5]:
    print(f"    {k!r}: {v!r}")
print()
print(f"test.json: {len(test_labels)} entries, type={type(test_labels).__name__}")
print("first 5 entries:")
for k, v in list(test_labels.items())[:5]:
    print(f"    {k!r}: {v!r}")"""
)

code(
    """def list_files_by_extension(directory):
    counts = Counter(p.suffix.lower() for p in directory.iterdir() if p.is_file())
    return counts

train_ext_counts = list_files_by_extension(TRAIN_IMAGE_DIR)
test_ext_counts = list_files_by_extension(TEST_IMAGE_DIR)

print("=== FILE EXTENSIONS ON DISK ===")
print(f"train/image/  : {dict(train_ext_counts)}")
print(f"test/images/  : {dict(test_ext_counts)}")
print()
print("Non-image metadata files (e.g. .DS_Store) would show up here as an")
print("extension outside {.tif, .tiff} -- none were found in either directory.")"""
)

md("### Image/label consistency")
code(
    """VALID_EXTENSIONS = (".tif", ".tiff")

def image_filenames(directory):
    return {p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS}

train_image_files = image_filenames(TRAIN_IMAGE_DIR)
test_image_files = image_filenames(TEST_IMAGE_DIR)

train_images_without_label = train_image_files - train_labels.keys()
train_labels_without_image = train_labels.keys() - train_image_files
test_images_without_label = test_image_files - test_labels.keys()
test_labels_without_image = test_labels.keys() - test_image_files

print("=== IMAGE <-> LABEL CONSISTENCY ===")
print(f"training images on disk : {len(train_image_files)}")
print(f"training JSON entries   : {len(train_labels)}")
print(f"training images with no label : {len(train_images_without_label)}")
print(f"training labels with no image : {len(train_labels_without_image)}")
print()
print(f"testing images on disk : {len(test_image_files)}")
print(f"testing JSON entries   : {len(test_labels)}")
print(f"testing images with no label : {len(test_images_without_label)}")
print(f"testing labels with no image : {len(test_labels_without_image)}")
print()
assert not train_images_without_label and not train_labels_without_image
assert not test_images_without_label and not test_labels_without_image
print("Every image has exactly one label and every label has exactly one image, in both splits.")"""
)

md("### Image properties (mode, size, dtype, pixel range)")
code(
    """def inspect_images(directory, filenames, sample_size=None):
    names = sorted(filenames)
    if sample_size is not None:
        names = names[:sample_size]
    modes, sizes, dtypes = Counter(), Counter(), Counter()
    px_min, px_max = float("inf"), float("-inf")
    for name in names:
        with Image.open(directory / name) as im:
            modes[im.mode] += 1
            sizes[im.size] += 1
            arr = np.array(im)
            dtypes[str(arr.dtype)] += 1
            px_min = min(px_min, float(arr.min()))
            px_max = max(px_max, float(arr.max()))
    return modes, sizes, dtypes, px_min, px_max

train_modes, train_sizes, train_dtypes, train_px_min, train_px_max = inspect_images(TRAIN_IMAGE_DIR, train_image_files)
test_modes, test_sizes, test_dtypes, test_px_min, test_px_max = inspect_images(TEST_IMAGE_DIR, test_image_files)

print("=== IMAGE PROPERTIES ===")
print("training:")
print(f"    PIL mode(s): {dict(train_modes)}")
print(f"    size(s)    : {dict(train_sizes)}")
print(f"    dtype(s)   : {dict(train_dtypes)}")
print(f"    pixel range: [{train_px_min}, {train_px_max}]")
print("testing:")
print(f"    PIL mode(s): {dict(test_modes)}")
print(f"    size(s)    : {dict(test_sizes)}")
print(f"    dtype(s)   : {dict(test_dtypes)}")
print(f"    pixel range: [{test_px_min}, {test_px_max}]")"""
)

md("### Duplicate image check (exact pixel match)")
code(
    """import hashlib

def find_duplicate_images(directory, filenames):
    seen = {}
    duplicates = []
    for name in sorted(filenames):
        with Image.open(directory / name) as im:
            digest = hashlib.md5(np.array(im).tobytes()).hexdigest()
        if digest in seen:
            duplicates.append((name, seen[digest]))
        else:
            seen[digest] = name
    return duplicates

train_duplicates = find_duplicate_images(TRAIN_IMAGE_DIR, train_image_files)
test_duplicates = find_duplicate_images(TEST_IMAGE_DIR, test_image_files)

print("=== DUPLICATE IMAGE CHECK (exact pixel match) ===")
print(f"training duplicates: {len(train_duplicates)} {train_duplicates[:5]}")
print(f"testing duplicates : {len(test_duplicates)} {test_duplicates[:5]}")"""
)

md(
    """## Results

Score distribution statistics and visualizations."""
)
code(
    """train_scores = np.array(list(train_labels.values()))
test_scores = np.array(list(test_labels.values()))

print("=== SCORE STATISTICS ===")
for name, scores in [("train", train_scores), ("test", test_scores)]:
    print(f"{name}: n={len(scores)}  min={scores.min():.4f}  max={scores.max():.4f}  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}  unique_values={len(np.unique(scores.round(6)))}")"""
)

code(
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train_scores, bins=21, edgecolor="black")
axes[0].set_title(f"Training score distribution (n={len(train_scores)})")
axes[0].set_xlabel("quality score")
axes[0].set_ylabel("count")

axes[1].hist(test_scores, bins=25, edgecolor="black", color="orange")
axes[1].set_title(f"Testing score distribution (n={len(test_scores)})")
axes[1].set_xlabel("quality score")
plt.tight_layout()
plt.show()"""
)

code(
    """def show_sample_grid(directory, labels, title, n_rows=2, n_cols=5, seed=0):
    rng = np.random.default_rng(seed)
    filenames = rng.choice(sorted(labels.keys()), size=n_rows * n_cols, replace=False)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for ax, fn in zip(axes.flat, filenames):
        with Image.open(directory / fn) as im:
            arr = np.array(im)
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"{fn}\\nscore={labels[fn]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

show_sample_grid(TRAIN_IMAGE_DIR, train_labels, "Training samples (raw 512x512)")"""
)

code(
    """show_sample_grid(TEST_IMAGE_DIR, test_labels, "Testing samples (raw 512x512)")"""
)

code(
    """def show_score_range_examples(directory, labels, title, n_examples=6):
    sorted_items = sorted(labels.items(), key=lambda kv: kv[1])
    indices = np.linspace(0, len(sorted_items) - 1, n_examples).astype(int)
    picks = [sorted_items[i] for i in indices]

    fig, axes = plt.subplots(1, n_examples, figsize=(3 * n_examples, 3))
    for ax, (fn, score) in zip(axes, picks):
        with Image.open(directory / fn) as im:
            arr = np.array(im)
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"score={score:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title} -- low to high quality score")
    plt.tight_layout()
    plt.show()

show_score_range_examples(TRAIN_IMAGE_DIR, train_labels, "Training set")
show_score_range_examples(TEST_IMAGE_DIR, test_labels, "Testing set")"""
)

# Notes
md(
    """## Notes

- Image format: TIFF, PIL mode `'F'` (32-bit float), 512x512, pixel range
  `[0, 1]`. **This deviates from a PNG assumption in an earlier task
  brief** -- the loader (`ct_iqa.data.ldct_iqac`) is written around the
  actual on-disk TIFF format. See `docs/replication/deviations.md`.
- `train.json`/`test.json` each map filename -> quality score; no other
  fields. This is LDCT-IQAC's radiologist-assigned quality score, **not**
  the VIF metric used as the regression target in the original Ohashi
  paper -- see `docs/replication/dataset_adaptation.md`.
- No missing labels, no orphan images, no duplicate images, no non-image
  files found, in either split."""
)
code(
    """print("=== EDA SUMMARY ===")
print(f"training: {len(train_labels)} image/label pairs, score range [{train_scores.min():.2f}, {train_scores.max():.2f}]")
print(f"testing : {len(test_labels)} image/label pairs, score range [{test_scores.min():.2f}, {test_scores.max():.2f}]")"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = Path("notebooks/01_exploration/01_dataset_eda.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}")
