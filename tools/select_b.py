"""Apply the declared Experiment B selection rule and write the (never-overwritten) selection record.

    python tools/select_b.py --stage 1 --a-run <A seed-42 run dir>
    python tools/select_b.py --stage 2 --a-run <A seed-42 run dir>      # all 7 configurations

Reads validation artifacts of completed runs only. Seed 42 is the declared selection seed.
"""

from __future__ import annotations

import argparse
import json

from ct_iqa.experiment.grid import select_final, select_stage1, write_selection_record

SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--a-run", required=True)
    a = ap.parse_args()
    rec = select_stage1(a.a_run, SEED) if a.stage == 1 else select_final(a.a_run, SEED)
    path = write_selection_record(rec)
    print(f"wrote {path}")
    for r in rec["candidates"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "run_id"})
    print("reference A:", rec["reference_run"], "rmse", round(rec["reference_rmse"], 4), "mean_corr", round(rec["reference_mean_corr"], 4))
    print("outcome:", rec["outcome"])
    print("selected:", None if rec["selected"] is None else {k: rec["selected"][k] for k in ("lambda_rank", "margin", "min_score_diff")})
    print("bootstrap:", json.dumps(rec.get("bootstrap_vs_A")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
