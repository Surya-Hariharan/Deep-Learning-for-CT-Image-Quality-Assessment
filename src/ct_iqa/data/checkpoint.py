"""Reference-level checkpoint/resume support for the production dataset
generator (``scripts/quality/generate_production_dataset.py``).

Design (per docs/research_decisions.md, "Production Generation Decision"):

    reference completed -> checkpoint -> next reference

rather than a single giant in-memory record list. One reference (169
records) is generated, written to its own manifest shard file, and only then
is that reference marked complete in a small, fast-to-scan completion log.
Restarting the generator (process kill, machine reboot, anything) replays
that log, skips every reference already marked complete, and resumes at the
next incomplete one -- it never regenerates a completed reference and never
needs the full 168,000/169,000-record set in memory at once.

Every write that could be observed half-written (the completion log line,
the progress-manifest JSON, a manifest shard) goes through
:func:`atomic_write_bytes` / :func:`atomic_write_text` -- write to a
temp file in the same directory, then ``os.replace`` (atomic on both POSIX
and Windows NTFS), so a crash mid-write can never leave a corrupt file that
looks valid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_append_line(path: Path, line: str) -> None:
    """Append one line to a log file.

    Not itself atomic in the temp-file sense (that would rewrite the whole
    log on every reference, which gets expensive at 1,000 references) --
    instead relies on a single ``write`` of a newline-terminated line being
    effectively atomic at the OS level for small writes, plus an explicit
    ``fsync`` so the line survives a crash immediately after this call
    returns. :func:`CheckpointManager.load_completed` is tolerant of a
    trailing partial line regardless (see its docstring), so even the
    theoretical worst case degrades to "redo one reference," never to a
    corrupt read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def config_hash(*config_paths: Path) -> str:
    """SHA-256 over the concatenated bytes of every config file, in order.

    Recorded in the progress manifest so a resumed run can detect (and
    refuse to silently continue under) a changed configuration.
    """
    digest = hashlib.sha256()
    for p in config_paths:
        digest.update(p.read_bytes())
    return digest.hexdigest()


def code_commit_hash(repo_root: Path) -> str | None:
    """Best-effort ``git rev-parse HEAD``; ``None`` if git is unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


@dataclass
class ReferenceCompletion:
    reference_id: str
    n_records: int
    shard_sha256: str
    completed_at: str
    worker: str | None = None


@dataclass
class ProgressManifest:
    """Small, whole-run summary -- rewritten atomically after every
    reference, never appended to (unlike the completion log). Kept
    deliberately tiny (no per-record data) so rewriting it every reference
    is cheap."""

    run_id: str
    config_hash: str
    code_commit_hash: str | None
    seed: int
    total_references: int
    started_at: str
    updated_at: str = ""
    n_references_completed: int = 0
    n_records_written: int = 0
    status: str = "in_progress"  # in_progress | completed | failed
    notes: list[str] = field(default_factory=list)


class ChecksumMismatchError(RuntimeError):
    """Raised when a manifest shard's on-disk checksum does not match the
    checksum recorded at completion time -- signals disk corruption or a
    partial/torn write that must be regenerated, not silently accepted."""


class CheckpointManager:
    """Owns the checkpoint directory: completion log, progress manifest, and
    per-reference manifest shards.

    All state needed to resume lives under ``checkpoint_dir``:

        completed_references.jsonl   -- append-only, one JSON line per
                                         completed reference (source of truth
                                         for "is this reference done?")
        progress_manifest.json       -- small whole-run summary, rewritten
                                         atomically after every reference
        manifest_shards/<ref_id>.csv -- that reference's 169 records, written
                                         once, atomically, then never touched
                                         again
    """

    SHARD_FIELDNAMES = [
        "reference_id", "source_filename", "expert_score", "degraded_id",
        "degradation_type", "noise_sigma", "blur_sigma", "vif_score",
        "diagnostic_status", "max_channel_gain", "covariance_condition_number",
        "split",
    ]

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        config_hash_value: str,
        code_commit_hash_value: str | None,
        seed: int,
        total_references: int,
        run_id: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.shard_dir = self.checkpoint_dir / "manifest_shards"
        self.completed_log_path = self.checkpoint_dir / "completed_references.jsonl"
        self.progress_manifest_path = self.checkpoint_dir / "progress_manifest.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self._config_hash = config_hash_value
        self._code_commit_hash = code_commit_hash_value
        self._seed = seed
        self._total_references = total_references
        self._run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")

        self._completed_cache: dict[str, ReferenceCompletion] | None = None

    # -- completion log ----------------------------------------------------

    def load_completed(self) -> dict[str, ReferenceCompletion]:
        """Return ``{reference_id: ReferenceCompletion}`` from the log.

        Tolerant of a truncated final line (the one crash scenario an
        `fsync`-then-append design can still produce, e.g. power loss mid
        `write`): that single line is skipped with a warning rather than
        raising, so a resumed run treats that one reference as incomplete
        and simply regenerates it -- never crashes on its own checkpoint.
        """
        if self._completed_cache is not None:
            return self._completed_cache

        completed: dict[str, ReferenceCompletion] = {}
        if self.completed_log_path.exists():
            with self.completed_log_path.open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"WARNING: checkpoint log line {line_no} is truncated/corrupt, "
                            "skipping -- that reference will be regenerated."
                        )
                        continue
                    completed[obj["reference_id"]] = ReferenceCompletion(**obj)
        self._completed_cache = completed
        return completed

    def is_reference_complete(self, reference_id: str, expected_n_records: int) -> bool:
        """A reference counts as complete only if ALL of: it is in the
        completion log, its record count matches what was expected at
        completion time, its shard file exists on disk, and that shard's
        content still hashes to the value recorded at completion time.

        This is the "detect incomplete/corrupt output records" requirement:
        a shard deleted, truncated, or edited out-of-band after being marked
        complete is detected here and treated as incomplete, not trusted.
        """
        completed = self.load_completed()
        record = completed.get(reference_id)
        if record is None:
            return False
        if record.n_records != expected_n_records:
            return False
        shard_path = self.shard_dir / f"{reference_id}.csv"
        if not shard_path.exists():
            return False
        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        return actual_hash == record.shard_sha256

    def read_shard(self, reference_id: str) -> list[dict[str, str]]:
        shard_path = self.shard_dir / f"{reference_id}.csv"
        with shard_path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def write_shard_and_mark_complete(
        self, reference_id: str, records: Sequence[dict[str, Any]]
    ) -> ReferenceCompletion:
        """Write one reference's records to its shard file atomically, then
        append a completion record. Order matters: the shard is durable on
        disk (via ``atomic_write_bytes``, which itself renames only after an
        `fsync`) *before* the completion log is told about it, so a crash
        between the two steps leaves the reference correctly marked
        incomplete rather than falsely marked complete with a missing shard.
        """
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.SHARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
        payload = buf.getvalue().encode("utf-8")

        shard_path = self.shard_dir / f"{reference_id}.csv"
        atomic_write_bytes(shard_path, payload)

        completion = ReferenceCompletion(
            reference_id=reference_id,
            n_records=len(records),
            shard_sha256=hashlib.sha256(payload).hexdigest(),
            completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        atomic_append_line(self.completed_log_path, json.dumps(completion.__dict__))

        if self._completed_cache is not None:
            self._completed_cache[reference_id] = completion
        return completion

    # -- progress manifest ---------------------------------------------------

    def load_progress(self) -> ProgressManifest | None:
        if not self.progress_manifest_path.exists():
            return None
        obj = json.loads(self.progress_manifest_path.read_text(encoding="utf-8"))
        return ProgressManifest(**obj)

    def start_or_resume(self) -> ProgressManifest:
        """Load an existing progress manifest if the config/seed still
        match, otherwise start a fresh one. Never silently continues a run
        whose configuration changed underneath it."""
        existing = self.load_progress()
        if existing is not None:
            if existing.config_hash != self._config_hash:
                raise ValueError(
                    "Checkpoint config_hash mismatch: the configuration changed since this "
                    "run started. Refusing to resume with a different config -- start a new "
                    "run (new --checkpoint-dir) or intentionally clear this checkpoint first."
                )
            if existing.seed != self._seed:
                raise ValueError(
                    f"Checkpoint seed mismatch: checkpoint has seed={existing.seed}, "
                    f"current config has seed={self._seed}. Refusing to resume."
                )
            return existing

        progress = ProgressManifest(
            run_id=self._run_id,
            config_hash=self._config_hash,
            code_commit_hash=self._code_commit_hash,
            seed=self._seed,
            total_references=self._total_references,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._save_progress(progress)
        return progress

    def checkpoint_progress(self, progress: ProgressManifest) -> None:
        """Rewrite the progress manifest atomically. Call after every
        completed reference (per checkpoint.checkpoint_every_n_references
        in configs/degradation.yaml, default 1 -- every reference)."""
        progress.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save_progress(progress)

    def _save_progress(self, progress: ProgressManifest) -> None:
        atomic_write_text(
            self.progress_manifest_path,
            json.dumps(progress.__dict__, indent=2),
        )

    # -- final assembly ---------------------------------------------------

    def iter_all_shard_records(self) -> Any:
        """Yield every record from every completed shard, one shard's rows
        at a time -- for assembling the final manifest without holding all
        169,000 records in memory simultaneously."""
        completed = self.load_completed()
        for reference_id in completed:
            yield from self.read_shard(reference_id)
