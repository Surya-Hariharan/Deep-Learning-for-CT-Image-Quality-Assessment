"""Create the FROZEN study-v2 split manifest (800 train / 100 validation / 100 calibration).

Run once. It refuses to overwrite an existing manifest (the protocol is never silently changed).
Uses only `data/labels/ldct_iqac/train.json` and the image directory listing; the official test
labels and images are not read. No model is involved.

    python tools/create_split_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ct_iqa.data.splits import build_manifest, save_manifest, verify_manifest

SEED = 2026          # declared here before any experiment result exists; arbitrary, never re-drawn
N_VAL, N_CAL = 100, 100
OUT = Path("experiments/study_v2/protocol/split_manifest.json")
TRAIN_JSON = Path("data/labels/ldct_iqac/train.json")
TRAIN_DIR = Path("data/raw/ldct_iqac/train/image")

RATIONALE = (
    "800/100/100 keeps the historical training-set size close to the 900 used by experiments 001-003 while "
    "reserving 100 images for model selection and a further 100 for uncertainty calibration. With n=100 "
    "calibration images the split-conformal rank for 90% coverage is k=ceil(101*0.9)=91 <= 100, so the interval "
    "is finite, but the calibration estimate is noisy; this limitation is documented. The split is a single "
    "seeded random permutation (not stratified, never re-drawn) at IMAGE level because no patient identifiers "
    "are available. The same partition is used by Experiments A, B and C; only initialization/data order may vary "
    "if multiple training seeds are used later."
)


def main() -> None:
    labels = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    on_disk = {p.name for p in TRAIN_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff")}
    if on_disk != set(labels):
        raise SystemExit("image directory and train.json disagree; refusing to create the manifest")
    manifest = build_manifest(labels, N_VAL, N_CAL, SEED, RATIONALE)
    verify_manifest(manifest, labels)
    save_manifest(manifest, OUT)
    print(f"wrote {OUT}  sha256={manifest['manifest_sha256']}")
    for k, v in manifest["label_statistics_by_partition"].items():
        print(f"  {k:<11} {v}")


if __name__ == "__main__":
    main()
