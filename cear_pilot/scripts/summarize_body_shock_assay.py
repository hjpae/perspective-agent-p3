#cear_pilot/scripts/summarize_body_shock_assay.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize Phase 3 body-shock perturbation assay.

Reads assay_manifest.csv and each run's assay_summary.json, then computes:
  shock_minus_control within each cohort/seed for pre/shock/recovery phases
  residue_diff = (body_shock.recovery - body_shock.pre) - (control.recovery - control.pre)

Run from repo root:
  PYTHONPATH=. python cear_pilot/scripts/summarize_body_shock_assay.py \
    --manifest outputs/p3_v3_assay/assay_manifest.csv

Or after copying this file:
  PYTHONPATH=. python summarize_body_shock_assay.py --manifest outputs/p3_v3_assay/assay_manifest.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PHASES = ["pre", "shock", "recovery"]
METRICS = [
    "top_occ", "TR_occ", "bottom_occ",
    "body_state_mean", "body_u_mean",
    "obs_pe_mean",
    "q_UP_minus_DOWN_mean",
    "pred_trend_UP_minus_DOWN_mean",
    "alpha_mean", "g_norm_mean", "g_delta_norm_mean",
]


def load_summary(outdir: Path) -> dict:
    p = outdir / "assay_summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_summary(cohort: str, seed: int, condition: str, outdir: Path, s: dict) -> dict:
    row = {"cohort": cohort, "seed": int(seed), "condition": condition, "outdir": str(outdir)}
    for phase in PHASES:
        for m in METRICS:
            row[f"{phase}.{m}"] = s.get(phase, {}).get(m, np.nan)
    # Preserve existing residue fields if present.
    res = s.get("residue_recovery_minus_pre", {})
    for m in METRICS:
        row[f"residue.{m}"] = res.get(m, np.nan)
    return row


def mean_se(xs: pd.Series) -> str:
    xs = xs.dropna().astype(float)
    if len(xs) == 0:
        return "NA"
    m = xs.mean()
    se = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return f"{m:.4f} ± {se:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--control", type=str, default="control")
    ap.add_argument("--shock", type=str, default="body_shock")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--diff_csv", type=str, default="")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    base = manifest.parent
    man = pd.read_csv(manifest)
    rows = []
    missing = []
    for _, r in man.iterrows():
        if str(r.get("status", "done")) != "done":
            continue
        outdir = Path(str(r["outdir"]))
        # If relative path does not resolve from current cwd, try relative to manifest parent.
        if not (outdir / "assay_summary.json").exists():
            alt = base / outdir.relative_to(base) if False else outdir
            # Leave as-is; most users run from repo root where outputs/... resolves.
        try:
            s = load_summary(outdir)
            rows.append(flatten_summary(str(r["cohort"]), int(r["seed"]), str(r["condition"]), outdir, s))
        except FileNotFoundError:
            missing.append(str(outdir / "assay_summary.json"))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No assay_summary.json files found. Run from repo root, or check manifest outdir paths.")

    out_csv = Path(args.out_csv) if args.out_csv else base / "assay_summary_flat.csv"
    df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} ({len(df)} rows)")
    if missing:
        print(f"[warn] missing {len(missing)} summaries; first few:")
        for m in missing[:5]:
            print("  ", m)

    # Pair control and shock within each cohort/seed.
    pairs = []
    for (cohort, seed), g in df.groupby(["cohort", "seed"]):
        if args.control not in set(g["condition"]) or args.shock not in set(g["condition"]):
            continue
        c = g[g["condition"] == args.control].iloc[0]
        sh = g[g["condition"] == args.shock].iloc[0]
        row = {"cohort": cohort, "seed": int(seed)}
        for phase in PHASES:
            for m in METRICS:
                col = f"{phase}.{m}"
                row[f"{phase}.shock_minus_control.{m}"] = sh[col] - c[col]
        # Difference-in-differences residue:
        # (shock recovery - shock pre) - (control recovery - control pre)
        for m in METRICS:
            row[f"did_residue.{m}"] = (
                (sh[f"recovery.{m}"] - sh[f"pre.{m}"])
                - (c[f"recovery.{m}"] - c[f"pre.{m}"])
            )
        pairs.append(row)

    diff = pd.DataFrame(pairs)
    diff_csv = Path(args.diff_csv) if args.diff_csv else base / "body_shock_vs_control_diff.csv"
    diff.to_csv(diff_csv, index=False)
    print(f"[save] {diff_csv} ({len(diff)} paired rows)")

    key_cols = [
        "recovery.shock_minus_control.body_state_mean",
        "recovery.shock_minus_control.body_u_mean",
        "recovery.shock_minus_control.q_UP_minus_DOWN_mean",
        "recovery.shock_minus_control.pred_trend_UP_minus_DOWN_mean",
        "recovery.shock_minus_control.alpha_mean",
        "recovery.shock_minus_control.g_delta_norm_mean",
        "did_residue.body_state_mean",
        "did_residue.body_u_mean",
        "did_residue.q_UP_minus_DOWN_mean",
        "did_residue.pred_trend_UP_minus_DOWN_mean",
        "did_residue.alpha_mean",
        "did_residue.g_delta_norm_mean",
    ]

    print("\n=== BODY SHOCK - CONTROL: cohort means ===")
    for cohort, g in diff.groupby("cohort"):
        print(f"\n[{cohort}] n={len(g)}")
        for col in key_cols:
            if col in g:
                print(f"  {col:58s} {mean_se(g[col])}")

    print("\nInterpretation guide:")
    print("  recovery.shock_minus_control.* compares body_shock vs control during recovery.")
    print("  did_residue.* subtracts each condition's pre→recovery drift, so it is the cleaner residue estimate.")
    print("  If full has larger did_residue.alpha/g_delta than no_body_in_g, body perturbation is entering perspective dynamics more strongly when body_in_g is enabled.")

if __name__ == "__main__":
    main()
