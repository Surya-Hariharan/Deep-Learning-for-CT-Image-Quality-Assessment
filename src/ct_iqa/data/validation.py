"""File-level integrity helpers: hashing, readability, format sniffing,
directory scanning, and the checks used to validate a dataset.

Nothing in this module writes to or mutates the data it inspects.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Magic-number signatures used to identify a file by content rather than suffix,
# so a mislabelled or truncated download is caught rather than trusted.
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip_empty"),
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!\x1a\x07", "rar"),
    (b"II\x2a\x00", "tiff"),
    (b"MM\x00\x2a", "tiff"),
]

ARCHIVE_KINDS = {"zip", "zip_empty", "gzip", "bzip2", "7z", "rar", "tar"}


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_format(path: Path) -> str:
    """Identify a file's real format from its leading bytes.

    Returns a lowercase format token, or ``"unknown"``. DICOM is detected via
    the 'DICM' marker at byte offset 128; NIfTI via its 348-byte header field.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return "unreadable"

    for sig, kind in _SIGNATURES:
        if head.startswith(sig):
            return kind

    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    # NIfTI-1: int32 sizeof_hdr == 348 at offset 0, either endianness.
    if len(head) >= 4 and head[:4] in (b"\x5c\x01\x00\x00", b"\x00\x00\x01\x5c"):
        return "nifti"
    if head[:10] == b"ObjectType":
        return "metaimage"  # .mhd header used by LUNA16 / LNDb
    if len(head) >= 257 + 5:
        with path.open("rb") as fh:
            fh.seek(257)
            if fh.read(5) == b"ustar":
                return "tar"

    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def is_archive_intact(path: Path) -> tuple[bool, str]:
    """Cheap corruption check for archive formats we can open with stdlib.

    Returns ``(ok, detail)``. Formats we cannot verify return ``(True, "unverified:<kind>")``
    so that "not checked" is never silently reported as "verified good".
    """
    kind = sniff_format(path)
    if kind.startswith("zip"):
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
            return (bad is None, "ok" if bad is None else f"corrupt member: {bad}")
        except zipfile.BadZipFile as exc:
            return False, f"bad zip: {exc}"
        except OSError as exc:
            return False, f"unreadable: {exc}"
    if kind in ARCHIVE_KINDS:
        return True, f"unverified:{kind}"
    return True, "not-an-archive"


@dataclass
class FileRecord:
    """One inspected file."""

    path: str
    size_bytes: int
    detected_format: str
    suffix: str
    sha256: str | None = None


@dataclass
class ScanResult:
    """Aggregate of a directory scan. Purely descriptive; no side effects."""

    root: str
    exists: bool
    n_files: int = 0
    n_dirs: int = 0
    total_bytes: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    archives: list[str] = field(default_factory=list)
    corrupt: list[tuple[str, str]] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    sample_files: list[str] = field(default_factory=list)

    @property
    def total_mb(self) -> float:
        return round(self.total_bytes / 1e6, 2)


def scan_directory(root: Path, sample_limit: int = 10, verify_archives: bool = True) -> ScanResult:
    """Recursively describe a directory without modifying anything in it."""
    result = ScanResult(root=str(root), exists=root.exists())
    if not result.exists:
        return result

    for entry in root.rglob("*"):
        if entry.name == ".gitkeep":
            continue
        if entry.is_dir():
            result.n_dirs += 1
            continue
        if not entry.is_file():
            continue
        try:
            size = entry.stat().st_size
        except OSError as exc:
            result.unreadable.append(f"{entry}: {exc}")
            continue

        result.n_files += 1
        result.total_bytes += size
        kind = sniff_format(entry)
        result.formats[kind] = result.formats.get(kind, 0) + 1
        if len(result.sample_files) < sample_limit:
            result.sample_files.append(entry.name)
        if kind in ARCHIVE_KINDS:
            result.archives.append(str(entry))
            if verify_archives:
                ok, detail = is_archive_intact(entry)
                if not ok:
                    result.corrupt.append((str(entry), detail))
        if size == 0:
            result.corrupt.append((str(entry), "zero-byte file"))

    return result


def find_duplicate_filenames(paths: list[Path]) -> dict[str, list[Path]]:
    """Group paths by filename; return only groups with more than one member."""
    by_name: dict[str, list[Path]] = {}
    for p in paths:
        by_name.setdefault(p.name, []).append(p)
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


def find_duplicate_content(paths: list[Path]) -> dict[str, list[Path]]:
    """Group paths by SHA-256; return only groups with more than one member."""
    by_hash: dict[str, list[Path]] = {}
    for p in paths:
        by_hash.setdefault(sha256sum(p), []).append(p)
    return {h: paths for h, paths in by_hash.items() if len(paths) > 1}
