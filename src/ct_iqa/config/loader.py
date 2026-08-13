"""YAML config loading, layered on ``ct_iqa.utils.paths``."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from ct_iqa.utils.paths import project_root, resolve

_PATHS_CONFIG = "configs/paths.yaml"


@lru_cache(maxsize=None)
def load_config(relative_path: str) -> dict[str, Any]:
    """Load any YAML config by project-relative path. Cached: configs are
    read-only for the lifetime of a process."""
    with resolve(relative_path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def paths_config() -> dict[str, Any]:
    return load_config(_PATHS_CONFIG)


def random_seed() -> int:
    """The single global seed. All sampling and splitting must consume this."""
    return int(paths_config()["random_seed"])


def dataset_config(name: str) -> dict[str, Any]:
    return load_config(f"configs/datasets/{name}.yaml")


def ldctiqac_config() -> dict[str, Any]:
    return dataset_config("ldctiqa")


def ohashi_degradation_config(active: bool = True) -> dict[str, Any]:
    """The Ohashi degradation spec.

    Args:
        active: ``True`` (default) loads the ACTIVE config for this project --
            ``configs/quality/ohashi_ldctiqac.yaml``, the LDCT-IQAC adaptation.
            ``False`` loads ``configs/quality/ohashi_degradation.yaml``, the
            original 105-reference DeepLesion+CQ500 specification, retained
            only as documentation of Ohashi's exact original configuration.
    """
    name = "ohashi_ldctiqac.yaml" if active else "ohashi_degradation.yaml"
    return load_config(f"configs/quality/{name}")


def resnet50_config() -> dict[str, Any]:
    return load_config("configs/quality/resnet50_vif.yaml")


def manifests_dir():
    return resolve(paths_config()["data"]["manifests"])


def reports_dir():
    return resolve(paths_config()["reports"])


__all__ = [
    "load_config",
    "paths_config",
    "random_seed",
    "dataset_config",
    "ldctiqac_config",
    "ohashi_degradation_config",
    "resnet50_config",
    "manifests_dir",
    "reports_dir",
    "project_root",
]
