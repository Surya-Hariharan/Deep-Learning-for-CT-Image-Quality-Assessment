"""Small dataset loader: resolves the manifest for a configured dataset and
exposes its records without ever hardcoding a machine-specific path.

The actual dataset location is resolved through
``ct_iqa.utils.paths.resolve``, which honours the ``LDCT_IQA_DATA_ROOT``
environment variable (see Section 19 of the project brief / docs/research_decisions.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ct_iqa.config.loader import dataset_config, manifests_dir
from ct_iqa.data.manifest import read_manifest_csv
from ct_iqa.utils.paths import resolve


@dataclass(frozen=True)
class DatasetHandle:
    """A resolved, on-disk view of one configured dataset."""

    name: str
    raw_root: Path
    manifest_path: Path
    available: bool


def resolve_dataset(name: str) -> DatasetHandle:
    """Locate a dataset's raw root and manifest, without reading either yet."""
    cfg = dataset_config(name)
    raw_root = resolve(cfg["raw_root"])
    manifest_path = manifests_dir() / f"{name}_manifest.csv"
    available = raw_root.is_dir() and any(
        p.is_file() and p.name != ".gitkeep" for p in raw_root.rglob("*")
    )
    return DatasetHandle(name=name, raw_root=raw_root, manifest_path=manifest_path, available=available)


def load_records(name: str) -> list[dict[str, str]]:
    """Load a dataset's manifest rows.

    Returns an empty list (not an error) if the manifest has not been built
    yet -- callers decide whether that is fatal for their purpose.
    """
    handle = resolve_dataset(name)
    if not handle.manifest_path.exists():
        return []
    return read_manifest_csv(handle.manifest_path)
