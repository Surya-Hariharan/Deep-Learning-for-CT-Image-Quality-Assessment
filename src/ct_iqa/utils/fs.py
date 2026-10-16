"""Filesystem helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(src: str | Path, dst: str | Path, attempts: int = 10, delay: float = 0.5) -> None:
    """`os.replace` that retries on `PermissionError`.

    On Windows a file under a synced folder (OneDrive) or being scanned can be briefly locked, which makes the
    atomic temp-file rename fail intermittently (observed once in a 30-epoch run). The rename is retried with a
    short delay; the final failure is re-raised unchanged, so a persistent problem still fails the run loudly.
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
