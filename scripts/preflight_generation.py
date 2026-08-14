"""Pre-flight check for the production Ohashi/LDCT-IQAC dataset generation run.

Run before every ``--execute`` invocation of
``scripts/quality/generate_production_dataset.py`` (and always before the
first one). Prints READY / NOT READY and exits non-zero if any CRITICAL
check fails -- it FAILS rather than proceeding on an unresolved problem, per
``docs/research_decisions.md``, "Production Generation Decision".

Checks performed:

    - the LDCT-IQAC dataset exists on disk
    - exactly 1,000 references are available
    - all required expert scores (LDCT-IQAC MOS) exist
    - no corrupted reference images (every PNG opens and verifies)
    - configs/degradation.yaml + configs/quality/ohashi_ldctiqac.yaml are
      valid and every former blocker (U-D01/U-D02/vif_variant) is resolved
    - the VIF implementation imports and computes correctly
    - the output directory has sufficient free disk space
    - the checkpoint directory is writable
    - Git safety rules are active (data/raw, data/interim, data/processed,
      the checkpoint dir are all Git-ignored; scripts/verify_git_safety.py passes)
    - no existing production run conflicts with the current configuration
    - the expected image/manifest-row count is correct and stated explicitly

Usage:
    python scripts/preflight_generation.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ct_iqa.config.loader import load_config  # noqa: E402
from ct_iqa.data.checkpoint import CheckpointManager, code_commit_hash, config_hash  # noqa: E402
from ct_iqa.data.manifest import read_manifest_csv  # noqa: E402
from ct_iqa.utils.paths import resolve  # noqa: E402

DEGRADATION_CONFIG = "configs/degradation.yaml"
DATASET_CONFIG = "configs/quality/ohashi_ldctiqac.yaml"

EXPECTED_N_REFERENCES = 1000
EXPECTED_PER_REFERENCE = 169
EXPECTED_DEGRADED_ONLY_TOTAL = EXPECTED_N_REFERENCES * (EXPECTED_PER_REFERENCE - 1)   # 168,000
EXPECTED_WITH_CLEAN_TOTAL = EXPECTED_N_REFERENCES * EXPECTED_PER_REFERENCE            # 169,000


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str, bool]] = []

    def add(self, label: str, ok: bool, detail: str = "", critical: bool = True) -> bool:
        self.checks.append((label, ok, detail, critical))
        return ok

    @property
    def ready(self) -> bool:
        return all(ok for _, ok, _, critical in self.checks if critical)

    def print_report(self) -> None:
        for label, ok, detail, critical in self.checks:
            tag = "CRITICAL" if critical else "advisory"
            status = "OK" if ok else "FAIL"
            line = f"  [{status}] ({tag}) {label}"
            if detail:
                line += f" -- {detail}"
            print(line)


def check_dataset_and_references(report: Report) -> list[dict[str, str]]:
    data_root = resolve("data/raw/ldctiqa")
    report.add("LDCT-IQAC dataset directory exists", data_root.exists(), str(data_root))

    manifest_path = resolve("data/manifests/ldctiqa_manifest.csv")
    references = read_manifest_csv(manifest_path) if manifest_path.exists() else []
    report.add("LDCT-IQAC reference manifest present", manifest_path.exists(), str(manifest_path))
    report.add(
        f"exactly {EXPECTED_N_REFERENCES} references available",
        len(references) == EXPECTED_N_REFERENCES,
        f"found {len(references)}",
    )

    missing_scores = [r["image_id"] for r in references if r.get("label") in (None, "")]
    report.add(
        "all required expert scores (LDCT-IQAC MOS) present",
        len(missing_scores) == 0,
        f"{len(missing_scores)} reference(s) missing a score" if missing_scores else "",
    )
    return references


def check_no_corrupted_references(report: Report, references: list[dict[str, str]]) -> None:
    from PIL import Image, UnidentifiedImageError

    corrupt: list[str] = []
    for row in references:
        path = resolve(row["source_path"])
        try:
            with Image.open(path) as im:
                im.verify()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            corrupt.append(f"{row.get('image_id', '?')}: {exc}")
    report.add(
        "no corrupted reference images",
        len(corrupt) == 0,
        f"{len(corrupt)} corrupt (first: {corrupt[0]})" if corrupt else f"{len(references)} verified",
    )


def check_degradation_config(report: Report) -> tuple[dict, dict]:
    try:
        deg_cfg = load_config(DEGRADATION_CONFIG)
        ds_cfg = load_config(DATASET_CONFIG)
        report.add("degradation configuration loads", True, f"{DEGRADATION_CONFIG}, {DATASET_CONFIG}")
    except Exception as exc:  # noqa: BLE001
        report.add("degradation configuration loads", False, repr(exc))
        return {}, {}

    required = {
        "noise.sigma_units": deg_cfg.get("noise", {}).get("sigma_units"),
        "combination.order": deg_cfg.get("combination", {}).get("order"),
        "vif.variant": deg_cfg.get("vif", {}).get("variant"),
        "vif.profile": deg_cfg.get("vif", {}).get("profile"),
        "checkpoint.dir": deg_cfg.get("checkpoint", {}).get("dir"),
        "output.root": deg_cfg.get("output", {}).get("root"),
    }
    unresolved = [k for k, v in required.items() if v in (None, "")]
    report.add(
        "no unresolved (null) production parameters in configs/degradation.yaml",
        len(unresolved) == 0,
        f"unresolved: {unresolved}" if unresolved else "sigma_units/order/vif_variant/profile all set",
    )

    counts_ok = (
        deg_cfg.get("counts", {}).get("degraded_only_total") == EXPECTED_DEGRADED_ONLY_TOTAL
        and deg_cfg.get("counts", {}).get("with_clean_total") == EXPECTED_WITH_CLEAN_TOTAL
        and ds_cfg.get("counts", {}).get("dataset_total") == EXPECTED_WITH_CLEAN_TOTAL
    )
    report.add(
        "degradation/dataset config counts match expected arithmetic",
        counts_ok,
        f"expected degraded_only={EXPECTED_DEGRADED_ONLY_TOTAL}, with_clean={EXPECTED_WITH_CLEAN_TOTAL}",
    )
    return deg_cfg, ds_cfg


def check_vif_implementation(report: Report, deg_cfg: dict) -> None:
    try:
        from ct_iqa.vif.wavelet import vif_wavelet

        profile = deg_cfg.get("vif", {}).get("profile", "project")
        probe = np.random.default_rng(0).normal(128, 20, size=(512, 512))
        score = vif_wavelet(probe, probe, profile=profile)
        report.add(
            "VIF implementation (vif_wavelet) imports and computes correctly",
            abs(score - 1.0) < 0.05,
            f"identity probe score={score:.4f} (expect ~1.0), profile={profile!r}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("VIF implementation (vif_wavelet) imports and computes correctly", False, repr(exc))


def check_disk_and_checkpoint_dir(report: Report, deg_cfg: dict) -> None:
    out_root = resolve(deg_cfg.get("output", {}).get("root", "data/processed/quality_iqa"))
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(out_root)
        free_gb = usage.free / 1e9
        projected_gb = deg_cfg.get("projections", {}).get("projected_disk_usage_gb", 28.3)
        report.add(
            "output directory has sufficient free disk space",
            free_gb > projected_gb * 1.1,
            f"{free_gb:.1f} GB free, need ~{projected_gb} GB (+10% margin)",
        )
    except OSError as exc:
        report.add("output directory has sufficient free disk space", False, repr(exc))

    checkpoint_dir = resolve(deg_cfg.get("checkpoint", {}).get("dir", "data/interim/quality_iqa_checkpoint"))
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        probe = checkpoint_dir / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.add("checkpoint directory is writable", True, str(checkpoint_dir))
    except OSError as exc:
        report.add("checkpoint directory is writable", False, repr(exc))


def check_git_safety(report: Report) -> None:
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "verify_git_safety.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        report.add(
            "scripts/verify_git_safety.py passes (no tracked dataset/binary files)",
            result.returncode == 0,
            result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("scripts/verify_git_safety.py passes (no tracked dataset/binary files)", False, repr(exc))

    probes = [
        "data/raw/ldctiqa/probe.png",
        "data/interim/quality_iqa_checkpoint/probe.json",
        "data/processed/quality_iqa/degraded/noise/probe.png",
    ]
    not_ignored = []
    for rel in probes:
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", rel], cwd=REPO_ROOT, timeout=15,
            )
            if result.returncode != 0:
                not_ignored.append(rel)
        except Exception:  # noqa: BLE001
            not_ignored.append(rel)
    report.add(
        "raw/interim/processed/checkpoint paths are all Git-ignored",
        len(not_ignored) == 0,
        f"NOT ignored: {not_ignored}" if not_ignored else "all 3 representative paths ignored",
    )


def check_no_conflicting_run(report: Report, deg_cfg: dict, ds_cfg: dict) -> None:
    from ct_iqa.config.loader import random_seed

    checkpoint_dir = resolve(deg_cfg.get("checkpoint", {}).get("dir", "data/interim/quality_iqa_checkpoint"))
    progress_path = checkpoint_dir / "progress_manifest.json"
    if not progress_path.exists():
        report.add("no existing production run conflicts", True, "no prior checkpoint -- fresh start")
        return

    try:
        cfg_hash = config_hash(resolve(DEGRADATION_CONFIG), resolve(DATASET_CONFIG))
        seed = random_seed()
        mgr = CheckpointManager(
            checkpoint_dir, config_hash_value=cfg_hash, code_commit_hash_value=None,
            seed=seed, total_references=ds_cfg.get("references", {}).get("n_total", EXPECTED_N_REFERENCES),
        )
        progress = mgr.load_progress()
        if progress is None:
            report.add("no existing production run conflicts", True, "no prior checkpoint -- fresh start")
            return
        conflict = progress.config_hash != cfg_hash or progress.seed != seed
        report.add(
            "no existing production run conflicts",
            not conflict,
            (
                f"checkpoint config_hash/seed mismatch (checkpoint seed={progress.seed}, current seed={seed})"
                if conflict
                else f"existing checkpoint compatible: status={progress.status}, "
                f"{progress.n_references_completed}/{progress.total_references} references complete -- will resume"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        report.add("no existing production run conflicts", False, repr(exc))


def main() -> int:
    report = Report()
    print("=== Pre-flight: production Ohashi/LDCT-IQAC dataset generation ===\n")

    references = check_dataset_and_references(report)
    if references:
        check_no_corrupted_references(report, references)
    else:
        report.add("no corrupted reference images", False, "skipped -- no references to check", critical=True)

    deg_cfg, ds_cfg = check_degradation_config(report)
    if deg_cfg:
        check_vif_implementation(report, deg_cfg)
        check_disk_and_checkpoint_dir(report, deg_cfg)
        check_no_conflicting_run(report, deg_cfg, ds_cfg)
    else:
        for label in (
            "VIF implementation (vif_wavelet) imports and computes correctly",
            "output directory has sufficient free disk space",
            "checkpoint directory is writable",
            "no existing production run conflicts",
        ):
            report.add(label, False, "skipped -- configuration failed to load")

    check_git_safety(report)

    print()
    report.print_report()
    print()
    print(
        f"Expected: {EXPECTED_N_REFERENCES} references x {EXPECTED_PER_REFERENCE} conditions = "
        f"{EXPECTED_WITH_CLEAN_TOTAL} manifest rows "
        f"({EXPECTED_DEGRADED_ONLY_TOTAL} degraded images + {EXPECTED_N_REFERENCES} reference-copy "
        f"images = {EXPECTED_WITH_CLEAN_TOTAL} total image files on disk)."
    )
    print()
    print("READY" if report.ready else "NOT READY")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
