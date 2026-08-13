"""Validate dataset integrity, metadata and split safety.

Exits non-zero if any check fails, so it can gate every downstream stage:
nothing gets trained until this passes.

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --strict   # warnings also fail

Checks:
  missing files, corrupt/zero-byte files, duplicate identifiers, duplicate file
  content, invalid metadata, unexpected image dimensions, invalid spacing,
  missing annotations/labels, patient leakage across splits, and reference
  leakage in the IQA split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_iqa.config.loader import dataset_config, load_config, manifests_dir  # noqa: E402
from ct_iqa.data.manifest import read_manifest_csv  # noqa: E402
from ct_iqa.data.splitting import check_group_leakage  # noqa: E402
from ct_iqa.utils.paths import resolve  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    items: list[str] = field(default_factory=list)


class Validator:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def record(self, name: str, status: str, detail: str = "", items: list[str] | None = None) -> None:
        self.results.append(CheckResult(name, status, detail, items or []))

    # ---------------------------------------------------------------- checks

    def check_manifest_present(self, dataset: str) -> list[dict[str, str]] | None:
        path = manifests_dir() / f"{dataset}_manifest.csv"
        if not path.exists():
            self.record(f"{dataset}: manifest exists", FAIL, f"missing {path.name}; run build_manifest.py")
            return None
        rows = read_manifest_csv(path)
        self.record(f"{dataset}: manifest exists", PASS, f"{len(rows)} records")
        return rows

    def check_files_exist(self, dataset: str, rows: list[dict[str, str]]) -> None:
        missing = [r["source_path"] for r in rows if not resolve(r["source_path"]).is_file()]
        if missing:
            self.record(f"{dataset}: files on disk", FAIL, f"{len(missing)} manifest paths missing", missing[:20])
        else:
            self.record(f"{dataset}: files on disk", PASS, f"all {len(rows)} present")

    def check_zero_byte(self, dataset: str, rows: list[dict[str, str]]) -> None:
        empty = [r["source_path"] for r in rows
                 if resolve(r["source_path"]).is_file() and resolve(r["source_path"]).stat().st_size == 0]
        self.record(
            f"{dataset}: non-empty files",
            FAIL if empty else PASS,
            f"{len(empty)} zero-byte files" if empty else "no zero-byte files",
            empty[:20],
        )

    def check_duplicate_ids(self, dataset: str, rows: list[dict[str, str]]) -> None:
        counts = Counter(r["image_id"] for r in rows)
        dups = [k for k, v in counts.items() if v > 1]
        self.record(
            f"{dataset}: unique image_id",
            FAIL if dups else PASS,
            f"{len(dups)} duplicated ids" if dups else f"{len(counts)} unique ids",
            dups[:20],
        )

    def check_duplicate_content(self, dataset: str, rows: list[dict[str, str]]) -> None:
        """Identical file content under different ids. Reported, never deleted."""
        by_hash: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            if r.get("sha256"):
                by_hash[r["sha256"]].append(r["image_id"])
        if not by_hash:
            self.record(f"{dataset}: duplicate content", WARN, "no hashes in manifest; re-run with --hash")
            return
        dups = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
        self.record(
            f"{dataset}: duplicate content",
            WARN if dups else PASS,
            f"{len(dups)} duplicated-content groups (reported, not removed)" if dups
            else f"{len(by_hash)} distinct files",
            [",".join(v) for v in list(dups.values())[:10]],
        )

    def check_dimensions(self, dataset: str, rows: list[dict[str, str]]) -> None:
        expected = {tuple(s) for s in load_config("configs/preprocessing/ct_common.yaml")["integrity"]["expected_slice_sizes"]}
        odd = []
        for r in rows:
            w, h = r.get("image_width"), r.get("image_height")
            if not w or not h:
                continue
            if (int(h), int(w)) not in expected and (int(w), int(h)) not in expected:
                odd.append(f"{r['image_id']}:{w}x{h}")
        self.record(
            f"{dataset}: image dimensions",
            WARN if odd else PASS,
            f"{len(odd)} images outside {sorted(expected)}" if odd else f"all match {sorted(expected)}",
            odd[:20],
        )

    def check_spacing(self, dataset: str, rows: list[dict[str, str]]) -> None:
        limits = load_config("configs/preprocessing/ct_common.yaml")["integrity"]
        lo, hi = limits["min_pixel_spacing_mm"], limits["max_pixel_spacing_mm"]
        present = [r for r in rows if r.get("spacing")]
        if not present:
            self.record(f"{dataset}: pixel spacing", WARN, "spacing absent for all records (recorded as unknown)")
            return
        bad = []
        for r in present:
            try:
                vals = [float(v) for v in r["spacing"].split(",")]
            except ValueError:
                bad.append(f"{r['image_id']}:unparseable({r['spacing']})")
                continue
            if any(not (lo <= v <= hi) for v in vals):
                bad.append(f"{r['image_id']}:{r['spacing']}")
        self.record(
            f"{dataset}: pixel spacing",
            FAIL if bad else PASS,
            f"{len(bad)} outside [{lo},{hi}] mm" if bad else f"{len(present)} within range",
            bad[:20],
        )

    def check_labels(self, dataset: str, rows: list[dict[str, str]]) -> None:
        cfg = dataset_config(dataset)
        labelled = [r for r in rows if r.get("label") not in (None, "")]
        if not labelled:
            self.record(f"{dataset}: annotations", WARN, "no labels in manifest")
            return
        missing = [r["image_id"] for r in rows if r.get("label") in (None, "")]
        scale = (cfg.get("labels") or {}).get("scale")
        out_of_range = []
        if scale:
            lo, hi = scale
            out_of_range = [r["image_id"] for r in labelled if not (lo <= float(r["label"]) <= hi)]
        status = FAIL if (missing or out_of_range) else PASS
        self.record(
            f"{dataset}: annotations",
            status,
            f"{len(missing)} unlabelled, {len(out_of_range)} out of range {scale}"
            if status == FAIL else f"all {len(labelled)} labelled and in range {scale}",
            (missing + out_of_range)[:20],
        )

    def check_split_leakage(self, dataset: str, rows: list[dict[str, str]]) -> None:
        """Patient-level leakage: all scans of a patient must share a partition."""
        split_rows = [r for r in rows if r.get("split")]
        if not split_rows:
            self.record(f"{dataset}: split leakage", PASS, "no splits assigned yet - nothing to leak")
            return
        group_key = "patient_id" if any(r.get("patient_id") for r in split_rows) else None
        if group_key is None:
            self.record(
                f"{dataset}: split leakage",
                FAIL,
                "splits assigned but no patient_id available - patient-level safety cannot be proven",
            )
            return
        report = check_group_leakage(
            [r[group_key] for r in split_rows], [r["split"] for r in split_rows]
        )
        self.record(
            f"{dataset}: split leakage",
            PASS if report.ok else FAIL,
            f"{report.n_groups} patients, {len(report.violations)} spanning partitions",
            list(report.violations)[:20],
        )

    def check_iqa_split(self) -> None:
        """Reference-level leakage in the Ohashi degradation dataset."""
        path = manifests_dir() / "quality_iqa_manifest.csv"
        if not path.exists():
            self.record("quality_iqa: reference leakage", PASS, "IQA dataset not generated yet")
            return
        rows = read_manifest_csv(path)
        split_rows = [r for r in rows if r.get("split")]
        if not split_rows:
            self.record("quality_iqa: reference leakage", WARN, "manifest has no split column populated")
            return
        report = check_group_leakage(
            [r["reference_id"] for r in split_rows], [r["split"] for r in split_rows]
        )
        self.record(
            "quality_iqa: reference leakage",
            PASS if report.ok else FAIL,
            f"{report.n_groups} references, {len(report.violations)} spanning partitions",
            list(report.violations)[:20],
        )

        counts = Counter(r["split"] for r in split_rows)
        expected = load_config("configs/quality/ohashi_ldctiqac.yaml")["splits"]["image_counts"]
        mismatch = {k: (counts.get(k, 0), v) for k, v in expected.items() if counts.get(k, 0) != v}
        self.record(
            "quality_iqa: split sizes",
            WARN if mismatch else PASS,
            f"actual vs expected mismatch: {mismatch}" if mismatch else f"match target {expected}",
        )

    def check_cross_dataset_overlap(self) -> None:
        """LUNA16 is derived from LIDC-IDRI; overlapping patients would leak."""
        lidc = manifests_dir() / "lidc_idri_manifest.csv"
        luna = manifests_dir() / "luna16_manifest.csv"
        if not (lidc.exists() and luna.exists()):
            self.record("lidc/luna16 overlap", PASS, "one or both manifests absent")
            return
        a = {r["patient_id"] for r in read_manifest_csv(lidc) if r.get("patient_id")}
        b = {r["patient_id"] for r in read_manifest_csv(luna) if r.get("patient_id")}
        if not a or not b:
            self.record("lidc/luna16 overlap", WARN, "patient ids not populated in one of the manifests")
            return
        shared = sorted(a & b)
        self.record(
            "lidc/luna16 overlap",
            WARN if shared else PASS,
            f"{len(shared)} patients appear in both - exclude before using LUNA16 for evaluation"
            if shared else "no shared patients",
            shared[:20],
        )

    # ----------------------------------------------------------------- driver

    def run(self, datasets: list[str]) -> None:
        for name in datasets:
            rows = self.check_manifest_present(name)
            if rows is None:
                continue
            if not rows:
                self.record(f"{name}: content", WARN, "manifest empty - dataset not downloaded")
                continue
            self.check_files_exist(name, rows)
            self.check_zero_byte(name, rows)
            self.check_duplicate_ids(name, rows)
            self.check_duplicate_content(name, rows)
            self.check_dimensions(name, rows)
            self.check_spacing(name, rows)
            self.check_labels(name, rows)
            self.check_split_leakage(name, rows)
        self.check_iqa_split()
        self.check_cross_dataset_overlap()

    def report(self, strict: bool) -> int:
        width = max(len(r.name) for r in self.results) + 2
        for r in self.results:
            print(f"[{r.status:<4}] {r.name:<{width}} {r.detail}")
            for item in r.items[:5]:
                print(f"         - {item}")

        n_fail = sum(r.status == FAIL for r in self.results)
        n_warn = sum(r.status == WARN for r in self.results)
        print(f"\n{len(self.results)} checks: {len(self.results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail")

        out = resolve("reports/validation_report.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([r.__dict__ for r in self.results], indent=2), encoding="utf-8"
        )
        print("Wrote reports/validation_report.json")
        return 1 if (n_fail or (strict and n_warn)) else 0


ALL = ["png", "lidc_idri", "luna16", "lndb", "deeplesion", "cq500", "ldctiqa"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=ALL, choices=ALL)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()

    validator = Validator()
    validator.run(args.datasets)
    return validator.report(args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
