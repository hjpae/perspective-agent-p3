#cear_pilot/scripts/summarize_phase3_diagnostics.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize counterfactual diagnostics produced by diagnose_body_field_conative.py.

Expected per-run files:
  <base>/<cohort>/s<seed>/diagnostics/bodydecoder_counterfactual_summary.csv

Example:
  PYTHONPATH=. python cear_pilot/scripts/summarize_phase3_diagnostics.py \
    --base outputs/p3_v3_cohorts --seed_start 0 --seed_end 29
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_COHORTS = ["full", "no_conative", "no_body_in_g"]


def mean_se(x) -> str:
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "NA"
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{m:.4f} ± {se:.4f}"


def read_diag(path: Path, cohort: str, seed: int) -> list[dict]:
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = {"cohort": cohort, "seed": seed, "valence_name": r.get("valence_name", r.get("zone", "unknown"))}
        for c in df.columns:
            if c == "valence_name":
                continue
            v = r[c]
            try:
                row[c] = float(v)
            except Exception:
                row[c] = v
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="outputs/p3_v3_cohorts")
    ap.add_argument("--cohorts", type=str, default=",".join(DEFAULT_COHORTS))
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=29)
    args = ap.parse_args()

    base = Path(args.base)
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    rows = []
    for cohort in cohorts:
        for seed in range(args.seed_start, args.seed_end + 1):
            p = base / cohort / f"s{seed}" / "diagnostics" / "bodydecoder_counterfactual_summary.csv"
            if not p.exists():
                print(f"[missing] {p}")
                continue
            rows.extend(read_diag(p, cohort, seed))

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No diagnostic summaries found.")

    out_csv = base / "cohort_counterfactual_summary.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

    # Candidate metrics from current diagnose script.
    metrics = [
        "pred_UP_minus_DOWN", "true_UP_minus_DOWN",
        "pred_trend_UP_minus_DOWN", "true_trend_UP_minus_DOWN",
        "abs_err_UP", "abs_err_DOWN", "abs_err_STAY",
        "abs_err_trend_UP", "abs_err_trend_DOWN",
        "body_state", "body_u",
    ]

    print("\n=== COUNTERFACTUAL DIAGNOSTIC SUMMARY ===")
    for cohort, g0 in res.groupby("cohort"):
        print(f"\n[{cohort}] seeds={g0['seed'].nunique()}")
        for val in ["top", "mid", "bottom"]:
            g = g0[g0["valence_name"] == val]
            if g.empty:
                continue
            print(f"  ({val}) n={len(g)}")
            for m in metrics:
                if m in g.columns:
                    print(f"    {m:30s} {mean_se(g[m])}")

    # Wide quick table for the core claim.
    core = [m for m in ["pred_trend_UP_minus_DOWN", "true_trend_UP_minus_DOWN"] if m in res.columns]
    if core:
        pivot = res.pivot_table(
            index=["cohort"],
            columns="valence_name",
            values=core,
            aggfunc="mean",
        )
        print("\n=== CORE TREND DIRECTION MEANS ===")
        print(pivot.round(4))

    print("\nInterpretation:")
    print("  no_conative should still show positive pred_trend_UP_minus_DOWN if the field is learned.")
    print("  full/no_body_in_g behavior may differ little, so use g-intervention/probe assay for embodied perspective differences.")


if __name__ == "__main__":
    main()
