"""Validate the PNG dataset at --data-root without modifying it.

Checks: manifest is readable, each entry has a valid PatientId/NoduleId and
three chronological timepoints, every referenced volume resolves to a file
on disk (tolerating the manifest's .nii/.nii.gz mismatch), and every
resolved volume has the expected shape.

    python scripts/validate_dataset.py --data-root data/external/png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_aware_lung_ct.data.png.metadata import get_split, load_manifest, resolve_series_path
from quality_aware_lung_ct.preprocessing.volumes import load_as_array


def validate(data_root: str, json_name: str = "NGP3T.json", expected_shape=(64, 64, 64)):
    errors = []
    manifest = load_manifest(data_root, json_name)

    for split_name in ("training", "test"):
        try:
            entries = get_split(manifest, split_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        for entry in entries:
            info = entry.get("info", {})
            for key in ("PatientId", "NoduleId", "t0", "t1", "t2", "Months1", "Months2"):
                if key not in info:
                    errors.append(f"{split_name}: entry missing `info.{key}`: {entry}")
            if info.get("t0") and info.get("t1") and info.get("t2"):
                if not (int(info["t0"]) < int(info["t1"]) < int(info["t2"])):
                    errors.append(f"{split_name}: non-chronological timepoints in {info}")

            for series in entry.get("series", []):
                for kind in ("image", "label"):
                    rel = series.get(kind)
                    if rel is None:
                        errors.append(f"{split_name}: series entry missing `{kind}`: {series}")
                        continue
                    try:
                        path = resolve_series_path(data_root, rel)
                    except FileNotFoundError as exc:
                        errors.append(str(exc))
                        continue
                    try:
                        arr = load_as_array(str(path))
                    except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
                        errors.append(f"{path}: failed to load ({exc})")
                        continue
                    if tuple(arr.shape) != expected_shape:
                        errors.append(f"{path}: shape {arr.shape} != expected {expected_shape}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/external/png")
    parser.add_argument("--json-name", type=str, default="NGP3T.json")
    args = parser.parse_args(argv)

    errors = validate(args.data_root, args.json_name)
    if errors:
        print(f"FAILED: {len(errors)} issue(s) found")
        for err in errors[:50]:
            print(f"  - {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        raise SystemExit(1)
    print("PNG dataset validation passed.")


if __name__ == "__main__":
    main()
