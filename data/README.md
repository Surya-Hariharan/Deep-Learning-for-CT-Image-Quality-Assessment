# data/

## `data/external/png/` -- the PNG dataset (not committed)

This directory is gitignored (see the repository `.gitignore`). It is not
distributed with this repository because it is patient data collected
under a hospital ethics approval (see `docs/datasets.md`).

**To obtain it:** see `docs/upstream/NGP-Net-README.md` for the
authors' Kaggle source, or `docs/datasets.md` for the dataset's measured
properties.

**Expected structure** once obtained:

```
data/external/png/
    NGP3T.json
    characteristics.csv
    patient-characteristics.csv
    images/<PatientId>/<NoduleId>/<date>_<NoduleId>_img.nii
    masks/<PatientId>/<NoduleId>/<date>_<NoduleId>_msk.nii
```

Point `--data-root` at wherever you place it if not at that default path.

**Validate it's set up correctly:**

```bash
python scripts/validate_dataset.py --data-root data/external/png
```

**Preprocessing policy:** the dataset is treated as immutable source data.
Nothing in this codebase renames, recompresses, or modifies the files
under `data/external/png/`. The manifest's `.nii.gz`-vs-`.nii` filename
mismatch (see `docs/datasets.md`) is resolved at read time in
`quality_aware_lung_ct.data.png.metadata.resolve_series_path`, not by
touching the dataset. Any future preprocessing output belongs under
`data/processed/` (also gitignored), never under `data/external/`.
