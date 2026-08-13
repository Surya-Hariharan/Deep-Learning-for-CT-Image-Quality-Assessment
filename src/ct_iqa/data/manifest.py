"""Machine-readable dataset manifests.

A manifest is a flat table of per-image records plus a small provenance header.
Records are written as CSV (human-diffable) and Parquet (typed, compact); the
header goes to JSON alongside them.

Missing values are written as ``None``/empty, never as a plausible substitute.
An unknown pixel spacing stays unknown.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ct_iqa.config.loader import manifests_dir
from ct_iqa.utils.paths import as_relative

SCHEMA_VERSION = "1.0.0"


@dataclass
class ImageRecord:
    """One image in one dataset.

    Fields that a given dataset does not carry stay ``None`` -- that is a
    recorded fact about the dataset, not a gap to be filled in later by
    guessing. The original ``expert_score`` (for LDCT-IQAC: the 5-radiologist
    mean-opinion score) lives in ``label`` here and is never overwritten.
    """

    dataset_name: str
    image_id: str
    filename: str                         # e.g. "0000.png"
    source_path: str                      # project-relative, forward slashes ("relative_path")
    patient_id: str | None = None
    study_id: str | None = None
    series_id: str | None = None
    modality: str | None = None
    slice_index: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    mode: str | None = None               # PIL mode, e.g. "RGB"
    channels: int | None = None
    spacing: str | None = None            # "row,col" in mm
    slice_thickness: float | None = None
    scanner: str | None = None
    windowing: str | None = None          # "width/level"
    preprocessing_status: str = "raw"     # raw | interim | processed
    label: float | None = None            # the untouched original expert_score
    label_kind: str | None = None
    split: str | None = None
    sha256: str | None = None
    notes: str | None = None


@dataclass
class DegradedRecord:
    """One image in the Ohashi-method synthetic degradation dataset.

    Two label fields are kept deliberately separate and must never be
    conflated (PROJECT ADAPTATION, see docs/dataset_adaptation.md):

    * ``vif_score`` -- the OHASHI-SPECIFIED synthetic training target,
      Full-Reference VIF of this degraded image against its own clean
      reference. ``None`` at generation time; filled by the VIF labelling pass.
    * ``expert_score`` -- the LDCT-IQAC radiologist mean-opinion score
      (0-4) of the *clean reference image itself*, carried through for the
      later subjective/real-image evaluation stage only. It is NEVER used as
      a training target for the synthetic stage and is constant across all
      169 degraded variants of one reference.
    """

    reference_id: str
    degraded_id: str
    degradation_type: str                 # clean | noise | blur | noise_blur
    noise_sigma: float | None = None
    blur_sigma: float | None = None
    combined_degradation: bool = False
    vif_score: float | None = None        # synthetic training target (Ohashi)
    expert_score: float | None = None     # LDCT-IQAC subjective score of the reference (kept separate)
    predicted_score: float | None = None  # raw model output, filled after training
    calibrated_score: float | None = None # after five-parameter logistic calibration
    split: str | None = None
    reference_source: str | None = None   # dataset the reference came from
    reference_path: str | None = None
    degraded_path: str | None = None


@dataclass
class ManifestHeader:
    """Provenance for a manifest file."""

    dataset_name: str
    schema_version: str = SCHEMA_VERSION
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    generator: str | None = None
    config_used: str | None = None
    seed: int | None = None
    n_records: int = 0
    source_root: str | None = None
    notes: list[str] = field(default_factory=list)
    unknown_fields: list[str] = field(default_factory=list)


def _record_columns(record_type: type) -> list[str]:
    return [f.name for f in fields(record_type)]


def write_manifest(
    records: Sequence[Any],
    header: ManifestHeader,
    stem: str,
    out_dir: Path | None = None,
    record_type: type | None = None,
) -> dict[str, Path]:
    """Write ``<stem>.csv``, ``<stem>.parquet`` and ``<stem>.header.json``.

    Returns the paths written. Parquet is skipped (with a note in the header)
    if pyarrow is unavailable, rather than failing the whole run.
    """
    import csv

    out_dir = out_dir or manifests_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in records]
    if record_type is None:
        record_type = type(records[0]) if records else ImageRecord
    columns = _record_columns(record_type)

    header.n_records = len(rows)
    written: dict[str, Path] = {}

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    written["csv"] = csv_path

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows) if rows else pa.table({c: [] for c in columns})
        parquet_path = out_dir / f"{stem}.parquet"
        pq.write_table(table, parquet_path)
        written["parquet"] = parquet_path
    except ImportError:
        header.notes.append("pyarrow unavailable: parquet manifest not written")

    header_path = out_dir / f"{stem}.header.json"
    header_path.write_text(json.dumps(asdict(header), indent=2), encoding="utf-8")
    written["header"] = header_path

    return written


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    """Read a manifest CSV back as raw string dicts."""
    import csv

    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def relative_source_path(path: Path) -> str:
    """Convenience re-export so manifest builders never store absolute paths."""
    return as_relative(path)
