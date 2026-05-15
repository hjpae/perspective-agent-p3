# -*- coding: utf-8 -*-
"""
Spyder-friendly Phase 3 cohort summarizer.

Purpose
-------
Create cohort_summary_v2_last30.csv (or any final-N episode summary) from
per-run traj.parquet files, without using command-line arguments.

How to use in Spyder
--------------------
1. Put this file in your project root, or leave it anywhere and set BASE below.
2. Edit CONFIG.
3. Run this file in Spyder (F5).
4. Use the generated CSV as INPUT_BEHAVIOR / INPUT_Q in the combined Fig. 2 script.

Expected files
--------------
BASE/full/s0/traj.parquet
BASE/full/s1/traj.parquet
...
BASE/no_conative/s0/traj.parquet
BASE/no_body_in_g/s29/traj.parquet

Optional diagnostic aggregation writes:
BASE/cohort_counterfactual_summary.csv
from per-run diagnostics/bodydecoder_counterfactual_summary.csv.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG — edit here in Spyder
# =============================================================================

BASE = Path("outputs/p3_v3_cohorts")
COHORTS = ["full", "no_conative", "no_body_in_g"]
SEED_START = 0
SEED_END = 29

# Main setting: final-N episode window for behavior/readiness summaries.
LAST_EPS = 30

# If START_EP is not None, it overrides LAST_EPS and uses episode >= START_EP.
# Usually leave as None.
START_EP = None

# Also regenerate cohort_counterfactual_summary.csv from diagnostics?
# This is not needed for last30 behavior, but harmless if diagnostics exist.
RUN_DIAGNOSTICS = False


# =============================================================================
# Helpers copied from summarize_phase3_cohorts.py, made Spyder-friendly
# =============================================================================


def zone3(x: int, y: int) -> str:
    xz = "L" if x <= 4 else ("M" if x <= 9 else "R")
    yz = "T" if y <= 4 else ("M" if y <= 9 else "B")
    return yz + xz


def valence_name(v: int) -> str:
    return {0: "top", 1: "mid", 2: "bottom"}.get(int(v), str(v))


def mean_se(x) -> str:
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "NA"
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{m:.3f} ± {se:.3f}"


def mean_se4(x) -> str:
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "NA"
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{m:.4f} ± {se:.4f}"


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize_run(
    traj_path: Path,
    cohort: str,
    seed: int,
    start_ep: int | None,
    last_eps: int,
) -> dict | None:
    df = pd.read_parquet(traj_path)
    if df.empty:
        return None

    max_ep = int(df["episode"].max())
    if start_ep is not None:
        d = df[df["episode"] >= start_ep].copy()
        window = f"ep{start_ep}-{max_ep}"
    else:
        d = df[df["episode"] >= max_ep - last_eps + 1].copy()
        window = f"last{last_eps}"
    if d.empty:
        return None

    d["zone3"] = [zone3(x, y) for x, y in zip(d["x"], d["y"])]
    d["valence_name"] = d["valence_zone_id"].map(valence_name)

    if "body_u" in d.columns:
        d["body_u_delta"] = d.groupby("episode")["body_u"].diff()
    if "body_state" in d.columns:
        d["body_delta"] = d.groupby("episode")["body_state"].diff()

    out: dict[str, float | int | str] = {
        "cohort": cohort,
        "seed": seed,
        "window": window,
        "n_steps": int(len(d)),
        "max_episode": max_ep,
        "top_occ": float((d["valence_name"] == "top").mean()),
        "mid_occ": float((d["valence_name"] == "mid").mean()),
        "bottom_occ": float((d["valence_name"] == "bottom").mean()),
        "TR_occ": float((d["zone3"] == "TR").mean()),
        "MR_occ": float((d["zone3"] == "MR").mean()),
        "BR_occ": float((d["zone3"] == "BR").mean()),
    }

    if "body_state" in d.columns:
        out["body_state_mean"] = float(d["body_state"].mean())
        out["body_state_last"] = float(d["body_state"].iloc[-1])

    for c in [
        "body_u", "body_delta", "body_u_delta", "obs_pe", "body_pe", "body_loss",
        "body_loss_action", "body_loss_tendency", "body_loss_state_aux",
        "alpha", "g_norm", "trend_target", "tendency_pred", "tendency_target",
        "conative_loss", "conative_target_entropy",
    ]:
        if c in d.columns:
            out[f"{c}_mean"] = float(d[c].mean())
            out[f"{c}_last"] = float(d[c].iloc[-1])

    if "tendency_pred" in d.columns and "tendency_target" in d.columns:
        out["trend_abs_err"] = float((d["tendency_pred"] - d["tendency_target"]).abs().mean())

    q_cols = {
        "UP": _first_existing_col(d, ["conative_pref_UP", "q_UP", "conative_q_UP"]),
        "DOWN": _first_existing_col(d, ["conative_pref_DOWN", "q_DOWN", "conative_q_DOWN"]),
        "LEFT": _first_existing_col(d, ["conative_pref_LEFT", "q_LEFT", "conative_q_LEFT"]),
        "RIGHT": _first_existing_col(d, ["conative_pref_RIGHT", "q_RIGHT", "conative_q_RIGHT"]),
        "STAY": _first_existing_col(d, ["conative_pref_STAY", "q_STAY", "conative_q_STAY"]),
    }
    for a, col in q_cols.items():
        if col is not None:
            out[f"q_{a}_mean"] = float(d[col].mean())
    if q_cols["UP"] and q_cols["DOWN"]:
        out["q_UP_minus_DOWN"] = float((d[q_cols["UP"]] - d[q_cols["DOWN"]]).mean())
    if q_cols["RIGHT"] and q_cols["LEFT"]:
        out["q_RIGHT_minus_LEFT"] = float((d[q_cols["RIGHT"]] - d[q_cols["LEFT"]]).mean())

    eig_cols = [c for c in d.columns if c.startswith("M_g_eig_")]
    if eig_cols:
        sample = d[eig_cols].sample(n=min(1000, len(d)), random_state=0)
        vals = sample.to_numpy(dtype=float)
        vals = np.clip(vals, 1e-12, None)
        p = vals / vals.sum(axis=1, keepdims=True)
        ent = -(p * np.log(p + 1e-12)).sum(axis=1)
        cond = vals.max(axis=1) / vals.min(axis=1)
        out["metric_eig_entropy_mean"] = float(ent.mean())
        out["metric_cond_mean"] = float(cond.mean())
        out["metric_cond_p95"] = float(np.percentile(cond, 95))

    g_cols = sorted(
        [c for c in d.columns if c.startswith("g_") and c[2:].isdigit()],
        key=lambda s: int(s.split("_")[1]),
    )
    if g_cols:
        cents = {}
        for vn in ["top", "mid", "bottom"]:
            sub = d[d["valence_name"] == vn]
            if len(sub) > 20:
                cents[vn] = sub[g_cols].mean().to_numpy(dtype=float)
        if "top" in cents and "bottom" in cents:
            out["g_top_bottom_dist"] = float(np.linalg.norm(cents["top"] - cents["bottom"]))
        if "top" in cents and "mid" in cents and "bottom" in cents:
            out["g_centroid_spread"] = float(np.mean([
                np.linalg.norm(cents["top"] - cents["mid"]),
                np.linalg.norm(cents["mid"] - cents["bottom"]),
                np.linalg.norm(cents["top"] - cents["bottom"]),
            ]))

    return out


def summarize_cohorts() -> Path:
    rows = []
    for cohort in COHORTS:
        for seed in range(SEED_START, SEED_END + 1):
            traj = BASE / cohort / f"s{seed}" / "traj.parquet"
            if not traj.exists():
                print(f"[missing] {traj}")
                continue
            row = summarize_run(
                traj,
                cohort,
                seed,
                start_ep=START_EP,
                last_eps=LAST_EPS,
            )
            if row is not None:
                rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No runs found. Check BASE and traj.parquet paths.")

    window = str(res["window"].iloc[0])
    out_csv = BASE / f"cohort_summary_v2_{window}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

    metrics = [
        "top_occ", "TR_occ", "bottom_occ",
        "body_state_mean", "body_u_mean", "body_u_delta_mean",
        "obs_pe_mean", "body_pe_mean", "trend_abs_err",
        "q_UP_mean", "q_DOWN_mean", "q_UP_minus_DOWN", "q_RIGHT_minus_LEFT",
        "conative_loss_mean", "conative_target_entropy_mean",
        "alpha_mean", "g_norm_mean",
        "g_top_bottom_dist", "g_centroid_spread",
        "metric_eig_entropy_mean", "metric_cond_mean",
    ]

    print("\n=== COHORT SUMMARY V2 ===")
    for cohort, g in res.groupby("cohort"):
        print(f"\n[{cohort}] n={len(g)}")
        for m in metrics:
            if m in g.columns:
                print(f"  {m:30s} {mean_se(g[m])}")

    print("\nExpected quick check:")
    print("  full/no_body_in_g should have positive q_UP_minus_DOWN and high top/TR.")
    print("  no_conative should have q fields uniform or zeroed, and weaker top/TR.")
    print("  g/metric differences should be interpreted cautiously because occupancy confounds g centroids.")

    return out_csv


# =============================================================================
# Optional diagnostics aggregation from summarize_phase3_diagnostics.py
# =============================================================================


def read_diag(path: Path, cohort: str, seed: int) -> list[dict]:
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = {
            "cohort": cohort,
            "seed": seed,
            "valence_name": r.get("valence_name", r.get("zone", "unknown")),
        }
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


def summarize_diagnostics() -> Path:
    rows = []
    for cohort in COHORTS:
        for seed in range(SEED_START, SEED_END + 1):
            p = BASE / cohort / f"s{seed}" / "diagnostics" / "bodydecoder_counterfactual_summary.csv"
            if not p.exists():
                print(f"[missing] {p}")
                continue
            rows.extend(read_diag(p, cohort, seed))

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No diagnostic summaries found.")

    out_csv = BASE / "cohort_counterfactual_summary.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

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
                    print(f"    {m:30s} {mean_se4(g[m])}")

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

    return out_csv


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=== Spyder Phase 3 summary ===")
    print(f"BASE       = {BASE}")
    print(f"COHORTS    = {COHORTS}")
    print(f"SEEDS      = {SEED_START}--{SEED_END}")
    print(f"LAST_EPS   = {LAST_EPS}")
    print(f"START_EP   = {START_EP}")

    summary_csv = summarize_cohorts()

    if RUN_DIAGNOSTICS:
        diag_csv = summarize_diagnostics()
    else:
        diag_csv = None

    print("\nDone.")
    print(f"Behavior/readiness summary: {summary_csv}")
    if diag_csv is not None:
        print(f"Counterfactual diagnostics: {diag_csv}")
    print("\nNext: set INPUT_BEHAVIOR and INPUT_Q in the figure script to:")
    print(f"  {summary_csv}")
