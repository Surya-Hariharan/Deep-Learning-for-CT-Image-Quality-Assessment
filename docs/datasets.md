# PNG dataset

**PNG = Pulmonary Nodule Growth** -- the dataset's name, not the image
format. Volumes are NIfTI.

Collected under approval from the Medical Ethics Committee, General
Hospital of Central Theater Command of the PLA (Grant [2020]035-1), by the
NGP-Net authors. Source: <https://kaggle.com/datasets/eb40d82b3bdd92bcdd202c50092443a8280a039f70fd96f2ad0cbc988ae57ca7>
(see `docs/upstream/NGP-Net-README.md`).

| | |
|---|---|
| Volumes | 768 images + 768 masks, all 64<sup>3</sup> at ~1 mm isotropic |
| Cohort | 103 patients, 226 nodules |
| Windows | 748 training + 150 test three-timepoint windows (`data/external/png/NGP3T.json`) |
| Intervals | 2-64 months between timepoints |

## Location

The dataset is not committed to Git (see `data/README.md`). It lives at:

```
data/external/png/
    NGP3T.json                    manifest: training/test splits, per-window info + series paths
    characteristics.csv           per-nodule measurements (volume, diameter, ...)
    patient-characteristics.csv   per-patient age/sex/first-exam-date
    images/<PatientId>/<NoduleId>/<date>_<NoduleId>_img.nii
    masks/<PatientId>/<NoduleId>/<date>_<NoduleId>_msk.nii
```

## Known manifest defect: `.nii.gz` vs `.nii`

`NGP3T.json` lists every volume's path with a `.nii.gz` extension
(e.g. `images/10165143/02/20181229_02_img.nii.gz`), but the volumes
actually shipped are **uncompressed** `.nii` files. A literal
`os.path.join(data_root, manifest_path)` therefore does not resolve.

This is handled by `quality_aware_lung_ct.data.png.metadata.resolve_series_path`,
which tries the manifest path verbatim, then its `.nii`/`.nii.gz`
counterpart, and raises `FileNotFoundError` if neither exists. **The
dataset files themselves are never renamed or modified** -- the fix lives
entirely in the reader. `scripts/validate_dataset.py` and
`tests/test_dataset.py` cover this behavior.

## Split protocol

`NGP3T.json` defines the official `training`/`test` split (748/150
windows). `quality_aware_lung_ct.data.png.dataset.split_data_list` further
divides `training` into k folds by `index % num_folds`, for
cross-validation during model selection -- it does not touch the official
test split.

**Known overlap:** the manifest's `training` and `test` splits are windowed
at the (patient, nodule, timepoint-triple) level, not the patient level.
Measured directly from `NGP3T.json`: of 663 distinct volumes referenced by
`training` and 252 referenced by `test`, **147 volumes appear in both**
(as part of different three-timepoint windows). This is the official
NGP-Net split, used unmodified here; it is not a preprocessing bug. Report
test-set numbers with this caveat rather than treating them as
patient-disjoint generalization results.
