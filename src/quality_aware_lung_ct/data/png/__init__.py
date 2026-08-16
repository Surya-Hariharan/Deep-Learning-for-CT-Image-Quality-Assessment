"""PNG (Pulmonary Nodule Growth) dataset access: manifest, splits, loader.

PNG is the dataset's name (Pulmonary Nodule Growth), not the image format —
volumes are NIfTI. See docs/datasets.md for the measured dataset properties
and the manifest's `.nii.gz`-vs-`.nii` filename mismatch, which
`metadata.resolve_series_path` resolves without modifying the source data.
"""

from .dataset import PNGDataset, build_dataloaders
from .metadata import load_manifest, resolve_series_path

__all__ = [
    "PNGDataset",
    "build_dataloaders",
    "load_manifest",
    "resolve_series_path",
]
