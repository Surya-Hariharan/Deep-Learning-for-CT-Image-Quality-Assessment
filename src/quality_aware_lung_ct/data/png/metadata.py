"""PNG dataset manifest handling.

The distributed manifest (``NGP3T.json``) references volumes as
``*.nii.gz``, but the shipped volumes are uncompressed ``*.nii``. This
module resolves that mismatch at read time without renaming or modifying
the source files, and reports it via `resolve_series_path`'s exceptions
rather than silently guessing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union


def load_manifest(data_root: Union[str, Path], json_name: str = "NGP3T.json") -> dict:
    ''' Load the PNG manifest JSON. Raises `FileNotFoundError` if missing. '''
    path = Path(data_root) / json_name
    if not path.is_file():
        raise FileNotFoundError(
            f"PNG manifest not found at {path}. Pass the correct --data-root."
        )
    with open(path, "r") as f:
        return json.load(f)


def get_split(manifest: dict, key: str) -> list:
    ''' Get a split ("training" or "test") from the manifest, validated non-empty. '''
    data = manifest.get(key)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Manifest split `{key}` is missing or empty.")
    return data


def resolve_series_path(data_root: Union[str, Path], manifest_path: str) -> Path:
    ''' Resolve a manifest-listed relative path to the file that actually
    exists on disk, tolerating the known `.nii.gz` vs `.nii` mismatch.

    Does not modify or rename anything under `data_root`.
    '''
    root = Path(data_root)
    candidate = root / manifest_path
    if candidate.is_file():
        return candidate

    if manifest_path.endswith(".nii.gz"):
        alt = root / manifest_path[: -len(".gz")]
    elif manifest_path.endswith(".nii"):
        alt = root / (manifest_path + ".gz")
    else:
        alt = None

    if alt is not None and alt.is_file():
        return alt

    raise FileNotFoundError(
        f"Neither `{manifest_path}` nor its .nii/.nii.gz counterpart exists "
        f"under {root}. The PNG dataset may be incomplete or `--data-root` "
        f"is wrong."
    )


def load_characteristics(data_root: Union[str, Path],
                         filename: str = "characteristics.csv"):
    ''' Load per-nodule measurement characteristics as a pandas DataFrame. '''
    import pandas as pd
    path = Path(data_root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)


def load_patient_characteristics(data_root: Union[str, Path],
                                  filename: str = "patient-characteristics.csv"):
    ''' Load per-patient characteristics (age, sex, first exam) as a DataFrame. '''
    import pandas as pd
    path = Path(data_root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)
