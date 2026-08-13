"""Build machine-readable manifests for the datasets that are present.

Read-only with respect to dataset files; writes only to data/manifests/.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --datasets ldctiqa --hash

Datasets that are not downloaded produce an empty manifest with a header
explaining why, so that a missing dataset is an explicit recorded state rather
than an absent file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_iqa.config.loader import dataset_config, random_seed  # noqa: E402
from ct_iqa.data.manifest import ImageRecord, ManifestHeader, write_manifest  # noqa: E402
from ct_iqa.data.validation import sha256sum  # noqa: E402
from ct_iqa.utils.paths import as_relative, resolve  # noqa: E402

GENERATOR = "scripts/build_manifest.py"


def build_ldctiqa(with_hash: bool = False) -> tuple[list[ImageRecord], ManifestHeader]:
    """Manifest for the LDCT-IQAC 2023 dataset.

    Fields the source does not carry -- patient/study/series ids, pixel
    spacing, slice thickness, scanner -- are left as None. They are genuinely
    absent from the PNG distribution. ``label`` carries the untouched original
    expert_score; it is never overwritten by anything downstream.
    """
    from PIL import Image

    cfg = dataset_config("ldctiqa")
    img_dir = resolve(cfg["images_dir"])
    labels = json.loads(resolve(cfg["labels_file"]).read_text(encoding="utf-8"))
    window = cfg["labels"]["window"]
    window_str = f"{window['width']}/{window['level']}"

    records: list[ImageRecord] = []
    for path in sorted(img_dir.glob("*.png")):
        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
        records.append(
            ImageRecord(
                dataset_name="ldctiqa",
                image_id=path.stem,
                filename=path.name,
                source_path=as_relative(path),
                patient_id=None,      # not distributed with the challenge PNGs
                study_id=None,
                series_id=None,
                modality="CT",
                slice_index=None,     # slice position not distributed
                image_width=width,
                image_height=height,
                mode=mode,
                channels=len(mode) if mode not in ("1", "L", "P") else 1,
                spacing=None,         # stripped by the PNG conversion
                slice_thickness=None,
                scanner=None,
                windowing=window_str,
                preprocessing_status="raw",
                label=labels.get(path.name),
                label_kind="subjective_mos_0_4",
                split=None,           # patient-level split blocked; reference-level split
                                       # lives in the quality_iqa manifest instead
                sha256=sha256sum(path) if with_hash else None,
                notes="RGB with three identical channels (grayscale replicated)",
            )
        )

    header = ManifestHeader(
        dataset_name="ldctiqa",
        generator=GENERATOR,
        config_used="configs/datasets/ldctiqa.yaml",
        seed=random_seed(),
        source_root=cfg["raw_root"],
        notes=[
            "Labels are radiologist mean opinion scores (expert_score) on a 0-4 scale, "
            "step 0.2. Kept exactly as distributed; never overwritten.",
            "Also serves as the Ohashi reference-image source in this project "
            "(PROJECT ADAPTATION) -- see docs/dataset_adaptation.md.",
            "No patient-level split assigned: patient ids are not distributed. "
            "The reference-level 60/20/20 split lives in quality_iqa_manifest.csv.",
        ],
        unknown_fields=[
            "patient_id", "study_id", "series_id", "slice_index",
            "spacing", "slice_thickness", "scanner",
        ],
    )
    return records, header


def build_absent(name: str) -> tuple[list[ImageRecord], ManifestHeader]:
    """Empty manifest recording that a dataset has not been downloaded."""
    cfg = dataset_config(name)
    header = ManifestHeader(
        dataset_name=name,
        generator=GENERATOR,
        config_used=f"configs/datasets/{name}.yaml",
        seed=random_seed(),
        source_root=cfg["raw_root"],
        notes=[
            f"NOT DOWNLOADED as of this run: {cfg['raw_root']} contains no data files.",
            f"Intended role: {cfg.get('role')}.",
            "Re-run this script after acquiring the dataset.",
        ],
        unknown_fields=["everything - dataset absent"],
    )
    return [], header


BUILDERS = {"ldctiqa": build_ldctiqa}
ALL = ["png", "lidc_idri", "luna16", "lndb", "deeplesion", "cq500", "ldctiqa"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=ALL, choices=ALL)
    parser.add_argument("--hash", action="store_true", help="compute SHA-256 per file (slow)")
    args = parser.parse_args()

    for name in args.datasets:
        cfg = dataset_config(name)
        root = resolve(cfg["raw_root"])
        has_data = root.exists() and any(
            p.is_file() and p.name != ".gitkeep" for p in root.rglob("*")
        )

        if not has_data:
            records, header = build_absent(name)
        elif name in BUILDERS:
            records, header = BUILDERS[name](args.hash) if name == "ldctiqa" else BUILDERS[name]()
        else:
            print(f"{name:<12} present on disk but no builder implemented -- SKIPPED")
            continue

        written = write_manifest(records, header, stem=f"{name}_manifest", record_type=ImageRecord)
        print(f"{name:<12} {len(records):>6} records -> {written['csv'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
