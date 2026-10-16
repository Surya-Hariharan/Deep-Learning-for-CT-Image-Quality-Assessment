"""Run provenance: git state, environment, timestamps, hashes."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git(args: list[str], cwd: str | Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo_root: str | Path = ".") -> dict:
    sha = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    return {
        "commit": sha,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "dirty": None if status is None else bool(status),
    }


def environment_info(device: str) -> dict:
    import numpy
    import scipy
    import torch

    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
        "device_requested": device, "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def config_hash(config_dict: dict) -> str:
    return sha256_bytes(json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8"))
