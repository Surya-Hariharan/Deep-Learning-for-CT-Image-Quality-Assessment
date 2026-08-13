"""Generate the Ohashi-method synthetic degradation dataset and its manifest,
using LDCT-IQAC as the reference-image source (PROJECT ADAPTATION replacing
Ohashi's original DeepLesion + CQ500 set).

SKELETON. Runs in --dry-run mode by default and REFUSES to write pixels while
any blocking prerequisite is unresolved, because generating 169,000 images
under guessed parameters would produce a dataset that looks right and is not.

    python scripts/quality/build_ohashi_dataset.py                 # plan + preflight
    python scripts/quality/build_ohashi_dataset.py --manifest-only # write the planned manifest
    python scripts/quality/build_ohashi_dataset.py --execute       # blocked until preflight clears

Pipeline (raw -> interim -> processed, each step reproducible from config):

    data/raw/ldctiqa (1,000 real reference images, present + verified)
        -> reference-level split 60/20/20 (OHASHI-SPECIFIED, Section 8)     [IMPLEMENTED]
        -> degradation engine (src/ct_iqa/degradation/)                     [IMPLEMENTED]
        -> data/processed/quality_iqa/degraded/{noise,blur,noise_blur}
        -> VIF labelling (src/ct_iqa/vif/)                                  [variant NOT SPECIFIED BY OHASHI]
        -> data/processed/quality_iqa/labels
        -> data/manifests/quality_iqa_manifest.{csv,parquet}

expert_score (LDCT-IQAC MOS of each reference) is carried into every record of
that reference but is never treated as the synthetic training target -- see
src/ct_iqa/data/manifest.py:DegradedRecord.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ct_iqa.config.loader import load_config, manifests_dir, random_seed  # noqa: E402
from ct_iqa.data.manifest import (  # noqa: E402
    DegradedRecord,
    ManifestHeader,
    read_manifest_csv,
    write_manifest,
)
from ct_iqa.data.splitting import assign_splits, check_group_leakage  # noqa: E402
from ct_iqa.degradation import build_grid  # noqa: E402
from ct_iqa.utils.paths import resolve  # noqa: E402

CONFIG = "configs/quality/ohashi_ldctiqac.yaml"


def load_references(cfg: dict) -> list[dict[str, str]]:
    """Load the real reference images from the LDCT-IQAC manifest.

    Unlike the original 105-reference Ohashi config (which points at datasets
    not present on disk and so has to use placeholder ids), this dataset is
    present, so the plan uses its actual image_id and expert_score values.
    """
    manifest_path = resolve(cfg["references"]["manifest"])
    if not manifest_path.exists():
        return []
    return read_manifest_csv(manifest_path)


def preflight(cfg: dict, references: list[dict[str, str]]) -> list[str]:
    """Return the list of blockers preventing real pixel generation.

    Each blocker is something that would otherwise have to be guessed. An
    empty list means every input is resolved and generation may proceed.
    """
    blockers: list[str] = []

    if not references:
        blockers.append(
            f"DATA: {cfg['references']['manifest']} not found or empty -- "
            "run scripts/dataset/build_manifests.py first"
        )
    elif len(references) != cfg["references"]["n_total"]:
        blockers.append(
            f"DATA: expected {cfg['references']['n_total']} references, "
            f"manifest has {len(references)}"
        )

    if cfg["noise"]["sigma_units"] is None:
        blockers.append("NOT SPECIFIED BY OHASHI: noise sigma units (HU vs 8-bit grey levels)")
    if cfg["combination"]["order"] is None:
        blockers.append("NOT SPECIFIED BY OHASHI: combined-degradation order (blur-then-noise vs noise-then-blur)")
    if cfg["labels"]["vif_variant"] is None:
        blockers.append("NOT SPECIFIED BY OHASHI: VIF variant (pixel-domain VIFp vs wavelet-domain VIF)")

    return blockers


def plan_records(cfg: dict, references: list[dict[str, str]]) -> list[DegradedRecord]:
    """Enumerate every record the dataset will contain, without touching pixels."""
    conditions = build_grid(
        noise_sigmas=cfg["noise"]["sigma"],
        blur_sigmas=cfg["blur"]["sigma"],
        include_clean=cfg["counts"]["per_reference"]["clean"] > 0,
    )

    ref_ids = [r["image_id"] for r in references]
    expert_scores = {r["image_id"]: r.get("label") for r in references}

    split_map = assign_splits(
        ref_ids,
        ratios=cfg["splits"]["ratios"],
        seed=random_seed(),
        counts=cfg["splits"]["reference_counts"],
    )

    out_root = cfg["output"]["root"]
    dirs = cfg["output"]["degraded_dirs"]

    records: list[DegradedRecord] = []
    for ref_id in ref_ids:
        expert_score = expert_scores[ref_id]
        expert_score_f = float(expert_score) if expert_score not in (None, "") else None
        for cond in conditions:
            degraded_id = f"{ref_id}_{cond.suffix()}"
            if cond.degradation_type == "clean":
                rel = f"{out_root}/{cfg['output']['references_dir']}/{ref_id}.png"
            else:
                rel = f"{out_root}/{dirs[cond.degradation_type]}/{degraded_id}.png"
            records.append(
                DegradedRecord(
                    reference_id=ref_id,
                    degraded_id=degraded_id,
                    degradation_type=cond.degradation_type,
                    noise_sigma=cond.noise_sigma,
                    blur_sigma=cond.blur_sigma,
                    combined_degradation=cond.combined,
                    vif_score=None,           # filled by the separate VIF labelling pass
                    expert_score=expert_score_f,  # carried through, never a training target
                    predicted_score=None,
                    calibrated_score=None,
                    split=split_map[ref_id],
                    reference_source="ldctiqa",
                    reference_path=f"{out_root}/{cfg['output']['references_dir']}/{ref_id}.png",
                    degraded_path=rel,
                )
            )
    return records


def verify_plan(cfg: dict, records: list[DegradedRecord]) -> list[str]:
    """Check the plan against the configured counts and the leakage rule."""
    problems: list[str] = []

    expected_total = cfg["counts"]["dataset_total"]
    if len(records) != expected_total:
        problems.append(f"total {len(records)} != expected {expected_total}")

    by_split = Counter(r.split for r in records)
    for name, expected in cfg["splits"]["image_counts"].items():
        if by_split.get(name, 0) != expected:
            problems.append(f"split {name}: {by_split.get(name, 0)} != {expected}")

    report = check_group_leakage([r.reference_id for r in records], [r.split for r in records])
    if not report.ok:
        problems.append(f"reference leakage: {len(report.violations)} references span partitions")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--manifest-only", action="store_true", help="write the planned manifest")
    parser.add_argument("--execute", action="store_true", help="generate images (requires clear preflight)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    references = load_references(cfg)
    blockers = preflight(cfg, references)

    print("Ohashi-method synthetic dataset (LDCT-IQAC adaptation) - preflight")
    print(f"  config: {args.config}")
    print(f"  seed:   {random_seed()}")
    print(f"  references found: {len(references)} / {cfg['references']['n_total']}")
    if blockers:
        print(f"\n  {len(blockers)} BLOCKER(S):")
        for b in blockers:
            print(f"    - {b}")
    else:
        print("  no blockers")

    if not references:
        print("\nCannot plan without a reference manifest. Run scripts/dataset/build_manifests.py first.")
        return 2

    records = plan_records(cfg, references)
    problems = verify_plan(cfg, records)

    print(f"\nPlanned dataset: {len(records)} images from {len({r.reference_id for r in records})} references")
    print(f"  by type:  {dict(Counter(r.degradation_type for r in records))}")
    print(f"  by split: {dict(Counter(r.split for r in records))}")
    print(f"  expert_score carried through: {sum(1 for r in records if r.expert_score is not None)}/{len(records)}")
    print(f"  plan verification: {'OK' if not problems else problems}")

    if args.manifest_only:
        header = ManifestHeader(
            dataset_name="quality_iqa",
            generator="scripts/quality/build_ohashi_dataset.py --manifest-only",
            config_used=args.config,
            seed=random_seed(),
            source_root=cfg["output"]["root"],
            notes=[
                "PLANNED manifest: reference ids are real LDCT-IQAC image_ids; no degraded "
                "image files exist yet -- pixel generation is a separate, blocked step.",
                "vif_score is null by design; filled by the VIF labelling pass.",
                "expert_score is the LDCT-IQAC MOS of the reference, carried through for "
                "subjective evaluation only -- never the synthetic training target.",
                "PROJECT ADAPTATION: reference source is LDCT-IQAC (1000 images), replacing "
                "Ohashi's original DeepLesion+CQ500 (105 images). See docs/dataset_adaptation.md.",
            ] + [f"BLOCKER: {b}" for b in blockers],
            unknown_fields=["noise sigma units", "combination order", "vif variant"],
        )
        written = write_manifest(records, header, stem="quality_iqa_manifest", record_type=DegradedRecord)
        print(f"\nWrote planned manifest: {written['csv'].name}")

    if args.execute:
        if blockers:
            print("\nREFUSING TO GENERATE: resolve the blockers above first.")
            print("Guessing them would yield a dataset that is reproducible but wrong.")
            return 2
        print("\nGeneration not implemented yet: preflight clear, implement the pixel writer next.")
        return 3

    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
