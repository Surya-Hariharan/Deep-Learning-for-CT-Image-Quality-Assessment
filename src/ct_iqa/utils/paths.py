"""Project-root-relative path resolution, with a machine-local data-root override.

Rule: never hardcode a machine-specific absolute path. Code paths resolve from
the repository root (found by walking up from this file until
``configs/paths.yaml`` is seen). Dataset paths additionally honour the
``LDCT_IQA_DATA_ROOT`` environment variable, so the actual dataset location can
live anywhere on a given machine without that location ever being written into
a config file or committed to Git.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ROOT_MARKER = Path("configs") / "paths.yaml"
_DATA_ROOT_ENV_VAR = "LDCT_IQA_DATA_ROOT"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root directory."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate project root: no ancestor of {here} contains {_ROOT_MARKER}"
    )


def data_root() -> Path:
    """Return the dataset root directory.

    Defaults to ``<project_root>/data``. If the ``LDCT_IQA_DATA_ROOT``
    environment variable is set, that path is used instead -- this is how a
    dataset stored outside the repository (a different drive, a shared mount)
    is wired in without ever hardcoding its location in a config file.
    """
    override = os.environ.get(_DATA_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "data"


def resolve(relative: str | Path) -> Path:
    """Resolve a project-relative path to an absolute one.

    Paths under ``data/`` are rebased onto :func:`data_root`, so configs can
    keep writing plain ``"data/raw/ldctiqa"`` while the actual data lives
    wherever ``LDCT_IQA_DATA_ROOT`` points. Every other path resolves against
    the repository root as usual.
    """
    rel = Path(relative)
    parts = rel.parts
    if parts and parts[0] == "data":
        return (data_root() / Path(*parts[1:])).resolve() if len(parts) > 1 else data_root()
    return (project_root() / rel).resolve()


def as_relative(path: Path | str) -> str:
    """Render a path relative to the project root, with forward slashes.

    Manifests must never contain machine-specific absolute paths, so every
    ``source_path`` written to disk goes through this function. Paths under
    the (possibly relocated) data root are rendered relative to it and
    re-prefixed with ``data/``, so the manifest stays portable even when
    ``LDCT_IQA_DATA_ROOT`` differs between machines.
    """
    p = Path(path).resolve()
    try:
        return ("data" / p.relative_to(data_root())).as_posix()
    except ValueError:
        pass
    try:
        return p.relative_to(project_root()).as_posix()
    except ValueError:
        return p.as_posix()
