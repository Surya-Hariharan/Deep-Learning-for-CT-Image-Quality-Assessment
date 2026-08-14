"""Production Ohashi-method synthetic degradation dataset generator.

Reads every degradation/generation parameter from ``configs/degradation.yaml``
(the authoritative production config, resolved 2026-08-14 after the
100-reference full-grid pilot -- see ``docs/research_decisions.md``,
"Production Generation Decision") and ``configs/quality/ohashi_ldctiqac.yaml``
(reference/split spec). No production parameter is hardcoded here.

Design, per the checkpointing requirement in the same decision record:

    reference completed -> checkpoint -> next reference

One reference (169 records: 1 clean-reference row + 168 computed degraded
records) is generated fully, written to a manifest shard, and marked
complete (``ct_iqa.data.checkpoint.CheckpointManager``) before the next
reference starts. Records for a reference are held in memory only while
that reference is being processed (~169 small dicts, released once the
shard is written) -- the full ~169,000-record set is never held in memory
at once; the final manifest is assembled by streaming shard files back out.

Restarting this script (kill, crash, machine reboot) resumes automatically:
every already-checkpointed reference is skipped, never regenerated.

Usage:
    python scripts/quality/generate_production_dataset.py --dry-run
    python scripts/quality/generate_production_dataset.py --execute

``--execute`` is guarded: it refuses to run unless the reference manifest is
present and matches the expected count, and unless the reference-level
split has no leakage. It does NOT independently re-run every check
``scripts/preflight_generation.py`` performs -- run that first.

**This script does not run itself.** Per the instruction that authorized
building this pipeline, the full 1,000-reference / 169,000-image generation
run requires a separate, explicit human decision to invoke ``--execute``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ct_iqa.config.loader import load_config, manifests_dir, random_seed  # noqa: E402
from ct_iqa.data.checkpoint import (  # noqa: E402
    CheckpointManager,
    code_commit_hash,
    config_hash,
)
from ct_iqa.data.manifest import ManifestHeader, ProductionManifestRecord, read_manifest_csv  # noqa: E402
from ct_iqa.data.splitting import assign_splits, check_group_leakage  # noqa: E402
from ct_iqa.degradation import BLUR_SIGMAS, Condition, NOISE_SIGMAS, apply_condition, condition_rng  # noqa: E402
from ct_iqa.utils.paths import as_relative, resolve  # noqa: E402
from ct_iqa.vif.labeling import compute_vif_with_diagnostics  # noqa: E402

DEGRADATION_CONFIG = "configs/degradation.yaml"
DATASET_CONFIG = "configs/quality/ohashi_ldctiqac.yaml"

RECORDS_PER_REFERENCE = 169  # 1 clean + 12 noise + 12 blur + 144 combined


def build_conditions() -> list[Condition]:
    conditions: list[Condition] = [Condition("clean")]
    conditions += [Condition("noise", noise_sigma=float(s)) for s in NOISE_SIGMAS]
    conditions += [Condition("blur", blur_sigma=float(s)) for s in BLUR_SIGMAS]
    conditions += [
        Condition("noise_blur", noise_sigma=float(n), blur_sigma=float(b))
        for n in NOISE_SIGMAS
        for b in BLUR_SIGMAS
    ]
    assert len(conditions) == RECORDS_PER_REFERENCE
    return conditions


def load_reference_image(source_path: Path) -> np.ndarray:
    """Read-only load. The source PNG is never written back to -- dataset
    immutability, see docs/deviations_from_ohashi.md DEV-02."""
    from PIL import Image

    with Image.open(source_path) as im:
        arr = np.array(im, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def save_png(array: np.ndarray, path: Path, clip_range: tuple[float, float]) -> None:
    """Encode as 8-bit PNG, written atomically (temp file + rename)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(array, *clip_range).astype(np.uint8)
    tmp_path = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp_path, format="PNG")
    tmp_path.replace(path)


def process_reference(
    ref_row: dict[str, str],
    *,
    output_root: str,
    references_dir: str,
    degraded_dirs: dict[str, str],
    order: str,
    sigma_scale: float,
    clip_after: bool,
    blur_truncate: float,
    blur_mode: str,
    output_clip_range: tuple[float, float],
    vif_variant: str,
    vif_profile: str,
    clean_reference_label: float,
    seed: int,
    split: str,
    conditions: list[Condition],
) -> list[dict]:
    """Generate every record for one reference. Writes pixels to disk as it
    goes; returns the 169 manifest rows for that reference (held in memory
    only for the duration of this call)."""
    ref_id = ref_row["image_id"]
    expert_score_raw = ref_row.get("label")
    expert_score = float(expert_score_raw) if expert_score_raw not in (None, "") else None
    source_filename = Path(ref_row.get("filename", f"{ref_id}.png")).name
    reference_img = load_reference_image(resolve(ref_row["source_path"]))

    ref_png_path = resolve(f"{output_root}/{references_dir}/{ref_id}.png")
    if not ref_png_path.exists():
        save_png(reference_img, ref_png_path, output_clip_range)

    records: list[dict] = []
    for cond in conditions:
        if cond.degradation_type == "clean":
            records.append(
                {
                    "reference_id": ref_id, "source_filename": source_filename,
                    "expert_score": expert_score, "degraded_id": f"{ref_id}_clean",
                    "degradation_type": "clean", "noise_sigma": None, "blur_sigma": None,
                    "vif_score": clean_reference_label,
                    "diagnostic_status": "clean_reference_identity_not_computed",
                    "max_channel_gain": None, "covariance_condition_number": None, "split": split,
                }
            )
            continue

        rng = condition_rng(seed, ref_id, cond)
        degraded = apply_condition(
            reference_img, cond, order=order, sigma_scale=sigma_scale, rng=rng,
            clip_range=output_clip_range if clip_after else None,
            blur_truncate=blur_truncate, blur_mode=blur_mode,
        )
        degraded_id = f"{ref_id}_{cond.suffix()}"
        degraded_path = resolve(f"{output_root}/{degraded_dirs[cond.degradation_type]}/{degraded_id}.png")
        save_png(degraded, degraded_path, output_clip_range)

        result = compute_vif_with_diagnostics(reference_img, degraded, variant=vif_variant, profile=vif_profile)
        records.append(
            {
                "reference_id": ref_id, "source_filename": source_filename,
                "expert_score": expert_score, "degraded_id": degraded_id,
                "degradation_type": cond.degradation_type,
                "noise_sigma": cond.noise_sigma, "blur_sigma": cond.blur_sigma,
                "vif_score": result.vif_score, "diagnostic_status": result.diagnostic_status,
                "max_channel_gain": result.max_channel_gain,
                "covariance_condition_number": result.covariance_condition_number,
                "split": split,
            }
        )
    return records


def _cast_row(row: dict[str, str]) -> dict:
    def _f(x):
        return float(x) if x not in (None, "") else None

    return {
        "reference_id": row["reference_id"], "source_filename": row["source_filename"],
        "expert_score": _f(row["expert_score"]), "degraded_id": row["degraded_id"],
        "degradation_type": row["degradation_type"], "noise_sigma": _f(row["noise_sigma"]),
        "blur_sigma": _f(row["blur_sigma"]), "vif_score": _f(row["vif_score"]),
        "diagnostic_status": row["diagnostic_status"], "max_channel_gain": _f(row["max_channel_gain"]),
        "covariance_condition_number": _f(row["covariance_condition_number"]), "split": row["split"],
    }


def assemble_final_manifest(
    mgr: CheckpointManager, deg_cfg: dict, seed: int, cfg_hash: str, commit_hash: str | None
) -> int:
    """Stream every completed shard into the final manifest -- never holds
    all records in memory at once, unlike ``ct_iqa.data.manifest.write_manifest``
    (fine for planning-stage manifests, not for a ~169,000-record production one)."""
    out_dir = manifests_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields(ProductionManifestRecord)]

    csv_path = out_dir / "quality_iqa_manifest.csv"
    tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
    n = 0
    with tmp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in mgr.iter_all_shard_records():
            writer.writerow(_cast_row(row))
            n += 1
    tmp_csv.replace(csv_path)

    parquet_path = out_dir / "quality_iqa_manifest.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        tmp_parquet = parquet_path.with_name(parquet_path.name + ".tmp")
        pq_writer = None
        batch: list[dict] = []
        batch_size = 5000
        for row in mgr.iter_all_shard_records():
            batch.append(_cast_row(row))
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch)
                pq_writer = pq_writer or pq.ParquetWriter(tmp_parquet, table.schema)
                pq_writer.write_table(table)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch)
            pq_writer = pq_writer or pq.ParquetWriter(tmp_parquet, table.schema)
            pq_writer.write_table(table)
        if pq_writer is not None:
            pq_writer.close()
            tmp_parquet.replace(parquet_path)
        parquet_note = "written"
    except ImportError:
        parquet_note = "SKIPPED: pyarrow unavailable"

    header = ManifestHeader(
        dataset_name="quality_iqa",
        generator="scripts/quality/generate_production_dataset.py --execute",
        config_used=f"{DEGRADATION_CONFIG}, {DATASET_CONFIG}",
        seed=seed,
        n_records=n,
        source_root=deg_cfg["output"]["root"],
        notes=[
            f"config_hash={cfg_hash}",
            f"code_commit_hash={commit_hash}",
            f"parquet: {parquet_note}",
            "vif_score/diagnostic_status/max_channel_gain/covariance_condition_number retained "
            "verbatim for every record -- flag-and-retain policy, decision A-19, docs/research_decisions.md.",
            "Assembled by streaming completed reference shards; see "
            "scripts/quality/generate_production_dataset.py:assemble_final_manifest.",
        ],
    )
    (out_dir / "quality_iqa_manifest.header.json").write_text(
        json.dumps(asdict(header), indent=2), encoding="utf-8"
    )
    print(f"Assembled final manifest: {n} records -> {csv_path}")
    return n


def dry_run(deg_cfg: dict, ds_cfg: dict, references: list[dict[str, str]]) -> bool:
    """Validate readiness without writing any production pixels or manifest.

    Everything this touches on disk lives under a throwaway subdirectory of
    the checkpoint dir, cleaned up before returning.
    """
    print("=== DRY RUN: configs/degradation.yaml + configs/quality/ohashi_ldctiqac.yaml ===\n")
    ready = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal ready
        if not ok:
            ready = False
        print(f"  [{'OK' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    check("configuration loaded", True, f"{DEGRADATION_CONFIG}, {DATASET_CONFIG}")
    check(
        "dataset inspected: reference manifest present and correctly sized",
        len(references) == ds_cfg["references"]["n_total"],
        f"{len(references)} / {ds_cfg['references']['n_total']} references found",
    )

    n_refs = len(references) or ds_cfg["references"]["n_total"]
    degraded_only = n_refs * (RECORDS_PER_REFERENCE - 1)
    with_clean_images = degraded_only + n_refs
    print(
        f"  Expected output: {degraded_only} degraded images + {n_refs} reference-copy images "
        f"= {with_clean_images} total image files ({n_refs} x {RECORDS_PER_REFERENCE} = "
        f"{n_refs * RECORDS_PER_REFERENCE} manifest rows, clean rows included)."
    )

    out_root = resolve(deg_cfg["output"]["root"])
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        check("output root path is valid and creatable", True, str(out_root))
    except OSError as exc:
        check("output root path is valid and creatable", False, str(exc))

    checkpoint_dir = resolve(deg_cfg["checkpoint"]["dir"])
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        probe = checkpoint_dir / ".dry_run_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check("checkpoint directory is writable", True, str(checkpoint_dir))
    except OSError as exc:
        check("checkpoint directory is writable", False, str(exc))

    projected_gb = deg_cfg["projections"]["projected_disk_usage_gb"]
    try:
        usage = shutil.disk_usage(out_root)
        free_gb = usage.free / 1e9
        check(
            "sufficient free disk space",
            free_gb > projected_gb * 1.1,
            f"{free_gb:.1f} GB free at {out_root.anchor}, need ~{projected_gb} GB (+10% margin)",
        )
    except OSError as exc:
        check("sufficient free disk space", False, str(exc))

    try:
        from ct_iqa.vif.wavelet import vif_wavelet

        probe_img = np.random.default_rng(0).normal(128, 20, size=(512, 512))
        score = vif_wavelet(probe_img, probe_img, profile=deg_cfg["vif"]["profile"])
        check(
            "VIF implementation imports and computes",
            abs(score - 1.0) < 0.05,
            f"identity probe score={score:.4f} (expect ~1.0)",
        )
    except Exception as exc:  # noqa: BLE001 -- report any failure, don't crash the dry run
        check("VIF implementation imports and computes", False, repr(exc))

    dry_ckpt_dir = checkpoint_dir / "_dry_run_probe"
    try:
        if dry_ckpt_dir.exists():
            shutil.rmtree(dry_ckpt_dir)
        mgr = CheckpointManager(
            dry_ckpt_dir, config_hash_value="dry-run-probe",
            code_commit_hash_value=None, seed=0, total_references=1,
        )
        mgr.start_or_resume()
        probe_records = [
            {
                "reference_id": "PROBE", "source_filename": "probe.png", "expert_score": "2.0",
                "degraded_id": "PROBE_noise_1", "degradation_type": "noise", "noise_sigma": "1.0",
                "blur_sigma": "", "vif_score": "0.9", "diagnostic_status": "ok",
                "max_channel_gain": "1.0", "covariance_condition_number": "10.0", "split": "train",
            }
        ]
        mgr.write_shard_and_mark_complete("PROBE", probe_records)
        resumed_ok = CheckpointManager(
            dry_ckpt_dir, config_hash_value="dry-run-probe",
            code_commit_hash_value=None, seed=0, total_references=1,
        ).is_reference_complete("PROBE", expected_n_records=1)
        check("checkpoint write/resume self-test", resumed_ok)
    except Exception as exc:  # noqa: BLE001
        check("checkpoint write/resume self-test", False, repr(exc))
    finally:
        shutil.rmtree(dry_ckpt_dir, ignore_errors=True)

    if references:
        probe_out_dir = checkpoint_dir / "_dry_run_probe_output"
        try:
            if probe_out_dir.exists():
                shutil.rmtree(probe_out_dir)
            conditions = build_conditions()
            records = process_reference(
                references[0],
                output_root=as_relative(probe_out_dir), references_dir="references",
                degraded_dirs=deg_cfg["output"]["degraded_dirs"],
                order=deg_cfg["combination"]["order"], sigma_scale=deg_cfg["noise"]["sigma_scale"],
                clip_after=deg_cfg["noise"]["clip_after"], blur_truncate=deg_cfg["blur"]["truncate"],
                blur_mode=deg_cfg["blur"]["boundary_mode"],
                output_clip_range=tuple(deg_cfg["image_handling"]["output_clip_range"]),
                vif_variant=deg_cfg["vif"]["variant"], vif_profile=deg_cfg["vif"]["profile"],
                clean_reference_label=deg_cfg["vif"]["clean_reference_label"],
                seed=random_seed(), split="train", conditions=conditions,
            )
            check(
                "single-reference full pipeline probe (fixture output, not production paths)",
                len(records) == RECORDS_PER_REFERENCE,
                f"{len(records)} records generated (expected {RECORDS_PER_REFERENCE})",
            )
        except Exception as exc:  # noqa: BLE001
            check("single-reference full pipeline probe (fixture output, not production paths)", False, repr(exc))
        finally:
            shutil.rmtree(probe_out_dir, ignore_errors=True)
    else:
        check("single-reference full pipeline probe", False, "no references available to probe")

    print()
    print("READY" if ready else "NOT READY")
    print(
        "No production dataset was written by this dry run: nothing under "
        "data/processed/quality_iqa was created outside the throwaway probe directories above "
        "(now deleted), and data/manifests/quality_iqa_manifest.* was not touched."
    )
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degradation-config", default=DEGRADATION_CONFIG)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="validate readiness; writes nothing production-facing")
    parser.add_argument("--execute", action="store_true", help="run production generation (resumable)")
    parser.add_argument(
        "--max-references", type=int, default=None,
        help="process at most this many references this invocation (testing/staged rollout only, "
        "not for the authorized full run)",
    )
    args = parser.parse_args()

    deg_cfg = load_config(args.degradation_config)
    ds_cfg = load_config(args.dataset_config)
    seed = random_seed()

    manifest_path = resolve(ds_cfg["references"]["manifest"])
    references = read_manifest_csv(manifest_path) if manifest_path.exists() else []

    if args.dry_run:
        return 0 if dry_run(deg_cfg, ds_cfg, references) else 1

    if not args.execute:
        print("Nothing to do: pass --dry-run to validate readiness, or --execute to run production generation.")
        print("Run scripts/preflight_generation.py before --execute.")
        return 0

    if len(references) != ds_cfg["references"]["n_total"]:
        print(
            f"REFUSING: expected {ds_cfg['references']['n_total']} references, "
            f"found {len(references)}. Run scripts/preflight_generation.py for the full diagnostic."
        )
        return 2

    ref_ids = [r["image_id"] for r in references]
    split_map = assign_splits(
        ref_ids, ratios=ds_cfg["splits"]["ratios"], seed=seed, counts=ds_cfg["splits"]["reference_counts"]
    )
    leakage = check_group_leakage(ref_ids, [split_map[i] for i in ref_ids])
    if not leakage.ok:
        print(f"REFUSING: reference split leakage detected ({len(leakage.violations)} references).")
        return 2

    conditions = build_conditions()
    cfg_hash = config_hash(resolve(args.degradation_config), resolve(args.dataset_config))
    commit_hash = code_commit_hash(REPO_ROOT)

    checkpoint_dir = resolve(deg_cfg["checkpoint"]["dir"])
    mgr = CheckpointManager(
        checkpoint_dir, config_hash_value=cfg_hash, code_commit_hash_value=commit_hash,
        seed=seed, total_references=len(references),
    )
    progress = mgr.start_or_resume()

    queue = references if args.max_references is None else references[: args.max_references]
    print(
        f"Production generation: run_id={progress.run_id}, "
        f"{len(mgr.load_completed())}/{len(references)} references already complete, "
        f"{len(queue)} queued this invocation."
    )

    log_every_n = deg_cfg["logging"]["log_every_n_references"]
    log_every_s = deg_cfg["logging"]["log_every_n_seconds"]
    output_cfg = deg_cfg["output"]

    start_time = time.perf_counter()
    last_log_time = start_time
    n_processed = 0

    for idx, ref_row in enumerate(queue):
        ref_id = ref_row["image_id"]
        if mgr.is_reference_complete(ref_id, expected_n_records=RECORDS_PER_REFERENCE):
            continue

        records = process_reference(
            ref_row,
            output_root=output_cfg["root"], references_dir=output_cfg["references_dir"],
            degraded_dirs=output_cfg["degraded_dirs"],
            order=deg_cfg["combination"]["order"], sigma_scale=deg_cfg["noise"]["sigma_scale"],
            clip_after=deg_cfg["noise"]["clip_after"], blur_truncate=deg_cfg["blur"]["truncate"],
            blur_mode=deg_cfg["blur"]["boundary_mode"],
            output_clip_range=tuple(deg_cfg["image_handling"]["output_clip_range"]),
            vif_variant=deg_cfg["vif"]["variant"], vif_profile=deg_cfg["vif"]["profile"],
            clean_reference_label=deg_cfg["vif"]["clean_reference_label"],
            seed=seed, split=split_map[ref_id], conditions=conditions,
        )
        mgr.write_shard_and_mark_complete(ref_id, records)

        n_completed_total = len(mgr.load_completed())
        progress.n_references_completed = n_completed_total
        progress.n_records_written = n_completed_total * RECORDS_PER_REFERENCE
        mgr.checkpoint_progress(progress)  # checkpoint after EVERY reference

        n_processed += 1
        now = time.perf_counter()
        if n_processed % log_every_n == 0 or (now - last_log_time) >= log_every_s:
            elapsed = now - start_time
            rate = n_processed / elapsed if elapsed > 0 else 0.0
            remaining = len(references) - n_completed_total
            eta_h = (remaining / rate / 3600) if rate > 0 else float("nan")
            print(
                f"  [{idx + 1}/{len(queue)} this run] ref={ref_id} done -- "
                f"{n_completed_total}/{len(references)} total complete, "
                f"{rate:.3f} refs/s, ETA {eta_h:.2f}h"
            )
            last_log_time = now

    n_completed_total = len(mgr.load_completed())
    if n_completed_total >= len(references):
        progress.status = "completed"
        mgr.checkpoint_progress(progress)
        assemble_final_manifest(mgr, deg_cfg, seed, cfg_hash, commit_hash)
        print("Production generation complete.")
    else:
        print(
            f"Invocation finished: {n_completed_total}/{len(references)} references complete overall. "
            "Re-run with --execute to resume; already-completed references will be skipped."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
