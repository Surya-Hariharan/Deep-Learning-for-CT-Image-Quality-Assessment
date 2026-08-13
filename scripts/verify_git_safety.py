"""Audit `git ls-files` for accidentally-tracked dataset/binary content.

Never deletes or modifies anything -- reports only. Prints SAFE or
POTENTIAL DATA LEAK, and lists every suspicious tracked path.

    python scripts/verify_git_safety.py

Exit code is 0 for SAFE, 1 for POTENTIAL DATA LEAK (or if git is unusable),
so it can gate a commit in CI or a pre-commit hook.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_iqa.utils.paths import project_root  # noqa: E402

# Suffixes that should never be tracked: raw imagery, medical-image formats,
# archives, and common model/checkpoint artefacts.
SUSPICIOUS_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".dcm", ".dicom", ".nii", ".nii.gz", ".mhd", ".raw",
    ".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".7z", ".rar",
    ".h5", ".hdf5", ".ckpt", ".pt", ".pth", ".pb", ".onnx", ".safetensors",
    ".npy", ".npz",
}

# Explicit allow-list: small, intentionally-committed reference/documentation
# assets are not a leak. Extend this set deliberately, never broadly.
ALLOWED_PREFIXES = (
    "tests/fixtures/",
    "docs/",
)


def _has_suspicious_suffix(path: str) -> bool:
    lower = path.lower()
    if lower.endswith(".nii.gz") or lower.endswith(".tar.gz"):
        return True
    return Path(lower).suffix in SUSPICIOUS_SUFFIXES


def list_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=project_root(), capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def find_suspicious(tracked: list[str]) -> list[str]:
    suspicious = []
    for path in tracked:
        if not _has_suspicious_suffix(path):
            continue
        if path.startswith(ALLOWED_PREFIXES):
            continue
        suspicious.append(path)
    return suspicious


def find_large_files(tracked: list[str], threshold_mb: float = 5.0) -> list[tuple[str, float]]:
    """Flag any tracked file over ``threshold_mb`` -- catches accidental large
    commits that don't match a suspicious suffix (e.g. a stray CSV export)."""
    large = []
    root = project_root()
    for path in tracked:
        p = root / path
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            if size_mb > threshold_mb:
                large.append((path, round(size_mb, 2)))
    return large


def main() -> int:
    try:
        tracked = list_tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Could not run git ls-files: {exc}")
        return 1

    suspicious = find_suspicious(tracked)
    large = find_large_files(tracked)

    print(f"Tracked files: {len(tracked)}")

    if not suspicious and not large:
        print("\nSAFE: no dataset/binary/checkpoint files are tracked in Git.")
        return 0

    print("\nPOTENTIAL DATA LEAK")
    if suspicious:
        print(f"\n{len(suspicious)} suspicious file(s) by extension:")
        for path in suspicious:
            print(f"  - {path}")
        print(
            "\nTo remove from the Git index WITHOUT deleting the local file:\n"
            "  git rm --cached <path>"
        )
    if large:
        print(f"\n{len(large)} large tracked file(s) (>5MB):")
        for path, size in large:
            print(f"  - {path} ({size} MB)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
