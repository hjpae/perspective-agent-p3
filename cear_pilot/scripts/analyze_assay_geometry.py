#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze Phase 3 perturbation-assay geometry.

Purpose
-------
This script is for the first, trajectory-level geometry analysis:

  1) Collect assay_traj.parquet from outputs/p3_v3_assay/<cohort>/s<seed>/<condition>/
  2) Extract g_0..g_{D-1} trajectories and M_g_eig_* spectra.
  3) Fit PCA on g trajectories, save PC coordinates.
  4) Summarize phase-wise and condition-wise geometry:
       - PCA centroid shifts
       - body_shock-control recovery shifts
       - pre→shock / pre→recovery residue
       - metric eigen entropy / condition number / spectral shift

This script deliberately does not claim that g_delta_norm alone is meaningful.
It focuses on shape/centroid/spectral structure of the induced geometry.

Example
-------
PYTHONPATH=. python cear_pilot/scripts/analyze_assay_geometry.py \
  --base outputs/p3_v3_assay \
  --seed_start 0 --seed_end 29 \
  --conditions control body_shock \
  --outdir outputs/p3_v3_assay_geometry

Optional figures require matplotlib. If unavailable or --no_plots, CSVs are still saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_COHORTS = ["full", "no_body_in_g", "no_conative"]
DEFAULT_CONDITIONS = ["control", "body_shock"]


def phase_from_t(t: int, shock_start: int, shock_duration: int) -> str:
    if t < shock_start:
        return "pre"
    if t < shock_start + shock_duration:
        return "shock"
    return "recovery"


def sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return columns like g_0, g_1, ... or M_g_eig_0, ...
    Excludes non-index columns such as g_norm, g_delta_norm.
    """
    cols = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        suffix = c.split("_")[-1]
        if suffix.isdigit():
            cols.append(c)
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def compute_metric_stats(d: pd.DataFrame) -> pd.DataFrame:
    eig_cols = sorted_prefixed_cols(d, "M_g_eig_")
    if not eig_cols:
        return d
    vals = d[eig_cols].to_numpy(dtype=float)
    vals = np.clip(vals, 1e-12, None)
    p = vals / vals.sum(axis=1, keepdims=True)
    d = d.copy()
    d["metric_eig_entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1)
    d["metric_cond"] = vals.max(axis=1) / vals.min(axis=1)
    d["metric_top1_mass"] = vals.max(axis=1) / vals.sum(axis=1)
    d["metric_trace"] = vals.sum(axis=1)
    return d


def load_one_run(
    base: Path,
    cohort: str,
    seed: int,
    condition: str,
    shock_start: int,
    shock_duration: int,
) -> Optional[pd.DataFrame]:
    traj_path = base / cohort / f"s{seed}" / condition / "assay_traj.parquet"
    if not traj_path.exists():
        print(f"[missing] {traj_path}")
        return None
    d = pd.read_parquet(traj_path)
    d = d.copy()
    d["cohort"] = cohort
    d["seed"] = seed
    d["condition"] = condition
    d["phase"] = [phase_from_t(int(t), shock_start, shock_duration) for t in d["t"]]
    d = compute_metric_stats(d)
    return d


def fit_pca(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA by SVD. Returns coords, components, mean, explained variance ratio."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    comps = Vt[:k]
    coords = Xc @ comps.T
    ev = (S ** 2) / max(1, X.shape[0] - 1)
    evr = ev / np.clip(ev.sum(), 1e-12, None)
    return coords, comps, mean.squeeze(0), evr[:k]


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def spectral_l2(mean_a: pd.Series, mean_b: pd.Series, eig_cols: List[str]) -> float:
    if not eig_cols:
        return float("nan")
    va = mean_a[eig_cols].to_numpy(dtype=float)
    vb = mean_b[eig_cols].to_numpy(dtype=float)
    return float(np.linalg.norm(va - vb))


def phase_summary(df: pd.DataFrame, eig_cols: List[str], g_cols: List[str]) -> pd.DataFrame:
    agg_dict = {
        "body_state": "mean",
        "body_u": "mean",
        "obs_pe": "mean",
        "alpha": "mean",
        "g_norm": "mean",
        "g_delta_norm": "mean",
        "pred_trend_UP_minus_DOWN": "mean" if "pred_trend_UP_minus_DOWN" in df.columns else "mean",
    }
    # Remove missing fields.
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    for c in ["q_UP_minus_DOWN", "conative_pref_UP", "conative_pref_DOWN", "metric_eig_entropy", "metric_cond", "metric_top1_mass", "metric_trace"]:
        if c in df.columns:
            agg_dict[c] = "mean"
    for c in eig_cols:
        agg_dict[c] = "mean"
    for c in ["PC1", "PC2", "PC3"]:
        if c in df.columns:
            agg_dict[c] = "mean"
    for c in g_cols:
        agg_dict[c] = "mean"

    return (
        df.groupby(["cohort", "seed", "condition", "phase"], as_index=False)
        .agg(agg_dict)
    )


def build_pairwise_diff(summary: pd.DataFrame, eig_cols: List[str], g_cols: List[str]) -> pd.DataFrame:
    rows = []
    key_cols = ["cohort", "seed", "phase"]
    value_cols = [c for c in summary.columns if c not in ["cohort", "seed", "condition", "phase"]]
    for (cohort, seed, phase), grp in summary.groupby(key_cols):
        if set(grp["condition"]) < {"control", "body_shock"}:
            continue
        ctrl = grp[grp["condition"] == "control"].iloc[0]
        shock = grp[grp["condition"] == "body_shock"].iloc[0]
        row = {"cohort": cohort, "seed": seed, "phase": phase}
        for c in value_cols:
            if c in shock.index and c in ctrl.index:
                row[f"shock_minus_control.{c}"] = float(shock[c] - ctrl[c])
        # PCA centroid shift and raw g centroid shift.
        pc_cols = [c for c in ["PC1", "PC2", "PC3"] if c in summary.columns]
        if pc_cols:
            row["shock_control_PC_centroid_dist"] = centroid_distance(shock[pc_cols].to_numpy(float), ctrl[pc_cols].to_numpy(float))
        if g_cols:
            row["shock_control_g_centroid_dist"] = centroid_distance(shock[g_cols].to_numpy(float), ctrl[g_cols].to_numpy(float))
        if eig_cols:
            row["shock_control_metric_spectral_l2"] = spectral_l2(shock, ctrl, eig_cols)
        rows.append(row)
    return pd.DataFrame(rows)


def build_residue_diff(summary: pd.DataFrame, eig_cols: List[str], g_cols: List[str]) -> pd.DataFrame:
    """Difference-in-difference: (shock recovery-pre) - (control recovery-pre)."""
    rows = []
    value_cols = [c for c in summary.columns if c not in ["cohort", "seed", "condition", "phase"]]
    for (cohort, seed), grp in summary.groupby(["cohort", "seed"]):
        need = [("control", "pre"), ("control", "recovery"), ("body_shock", "pre"), ("body_shock", "recovery")]
        lookup = {}
        ok = True
        for cond, phase in need:
            sub = grp[(grp["condition"] == cond) & (grp["phase"] == phase)]
            if len(sub) == 0:
                ok = False
                break
            lookup[(cond, phase)] = sub.iloc[0]
        if not ok:
            continue
        row = {"cohort": cohort, "seed": seed}
        for c in value_cols:
            shock_res = lookup[("body_shock", "recovery")][c] - lookup[("body_shock", "pre")][c]
            ctrl_res = lookup[("control", "recovery")][c] - lookup[("control", "pre")][c]
            row[f"did_residue.{c}"] = float(shock_res - ctrl_res)
        pc_cols = [c for c in ["PC1", "PC2", "PC3"] if c in summary.columns]
        if pc_cols:
            shock_vec = (lookup[("body_shock", "recovery")][pc_cols] - lookup[("body_shock", "pre")][pc_cols]).to_numpy(float)
            ctrl_vec = (lookup[("control", "recovery")][pc_cols] - lookup[("control", "pre")][pc_cols]).to_numpy(float)
            row["did_residue_PC_vector_dist"] = centroid_distance(shock_vec, ctrl_vec)
        if g_cols:
            shock_vec = (lookup[("body_shock", "recovery")][g_cols] - lookup[("body_shock", "pre")][g_cols]).to_numpy(float)
            ctrl_vec = (lookup[("control", "recovery")][g_cols] - lookup[("control", "pre")][g_cols]).to_numpy(float)
            row["did_residue_g_vector_dist"] = centroid_distance(shock_vec, ctrl_vec)
        if eig_cols:
            shock_vec = (lookup[("body_shock", "recovery")][eig_cols] - lookup[("body_shock", "pre")][eig_cols]).to_numpy(float)
            ctrl_vec = (lookup[("control", "recovery")][eig_cols] - lookup[("control", "pre")][eig_cols]).to_numpy(float)
            row["did_residue_metric_spectral_l2"] = centroid_distance(shock_vec, ctrl_vec)
        rows.append(row)
    return pd.DataFrame(rows)


def print_mean_se_table(df: pd.DataFrame, cols: List[str], group_col: str = "cohort", title: str = "SUMMARY") -> None:
    print(f"\n=== {title} ===")
    for cohort, g in df.groupby(group_col):
        print(f"\n[{cohort}] n={len(g)}")
        for c in cols:
            if c not in g.columns:
                continue
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(x) == 0:
                continue
            se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            print(f"  {c:44s} {x.mean(): .4f} ± {se:.4f}")


def make_plots(df_pc: pd.DataFrame, summary: pd.DataFrame, outdir: Path, no_plots: bool) -> None:
    if no_plots:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Plot PC trajectories collapsed by cohort/condition/phase centroids.
    phase_colors = {"pre": "C0", "shock": "C1", "recovery": "C2"}
    for cohort in sorted(df_pc["cohort"].unique()):
        for condition in sorted(df_pc["condition"].unique()):
            sub = df_pc[(df_pc["cohort"] == cohort) & (df_pc["condition"] == condition)]
            if sub.empty:
                continue
            plt.figure(figsize=(6, 5))
            for phase, ph in sub.groupby("phase"):
                plt.scatter(ph["PC1"], ph["PC2"], s=4, alpha=0.2, label=phase, c=phase_colors.get(phase, None))
            cent = sub.groupby("phase")[["PC1", "PC2"]].mean()
            for phase, row in cent.iterrows():
                plt.scatter([row["PC1"]], [row["PC2"]], s=120, marker="x", c=phase_colors.get(phase, "k"))
                plt.text(row["PC1"], row["PC2"], phase)
            plt.title(f"g PCA: {cohort} / {condition}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend(markerscale=4)
            plt.tight_layout()
            plt.savefig(figdir / f"g_pca_{cohort}_{condition}.png", dpi=180)
            plt.close()

    # Metric entropy by cohort/condition/phase.
    if "metric_eig_entropy" in summary.columns:
        piv = summary.groupby(["cohort", "condition", "phase"])["metric_eig_entropy"].mean().reset_index()
        for cohort in sorted(piv["cohort"].unique()):
            sub = piv[piv["cohort"] == cohort]
            plt.figure(figsize=(6, 4))
            for condition, cd in sub.groupby("condition"):
                order = ["pre", "shock", "recovery"]
                vals = [cd[cd["phase"] == ph]["metric_eig_entropy"].mean() for ph in order]
                plt.plot(order, vals, marker="o", label=condition)
            plt.title(f"metric eig entropy: {cohort}")
            plt.ylabel("entropy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figdir / f"metric_entropy_{cohort}.png", dpi=180)
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="outputs/p3_v3_assay")
    ap.add_argument("--cohorts", nargs="+", default=DEFAULT_COHORTS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=29)
    ap.add_argument("--shock_start", type=int, default=60)
    ap.add_argument("--shock_duration", type=int, default=20)
    ap.add_argument("--pca_sample_per_run", type=int, default=160,
                    help="Max rows per run used to fit PCA and save PC csv. Use -1 for all.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir) if args.outdir else base / "geometry_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    runs: List[pd.DataFrame] = []
    for cohort in args.cohorts:
        for seed in range(args.seed_start, args.seed_end + 1):
            for cond in args.conditions:
                d = load_one_run(base, cohort, seed, cond, args.shock_start, args.shock_duration)
                if d is not None:
                    if args.pca_sample_per_run is not None and args.pca_sample_per_run > 0 and len(d) > args.pca_sample_per_run:
                        d_pca = d.sample(args.pca_sample_per_run, random_state=seed).copy()
                    else:
                        d_pca = d.copy()
                    runs.append(d_pca)
    if not runs:
        raise SystemExit("No assay trajectories found.")

    df = pd.concat(runs, ignore_index=True)
    g_cols = sorted_prefixed_cols(df, "g_")
    g_cols = [c for c in g_cols if c.split("_")[-1].isdigit()]
    eig_cols = sorted_prefixed_cols(df, "M_g_eig_")
    if not g_cols:
        raise SystemExit("No g_i columns found in assay trajectories.")

    coords, comps, mean, evr = fit_pca(df[g_cols].to_numpy(float), n_components=3)
    for i in range(coords.shape[1]):
        df[f"PC{i+1}"] = coords[:, i]
    pca_meta = {
        "g_cols": g_cols,
        "explained_variance_ratio": evr.tolist(),
        "mean": mean.tolist(),
        "components": comps.tolist(),
    }
    (outdir / "pca_meta.json").write_text(json.dumps(pca_meta, indent=2), encoding="utf-8")
    df.to_parquet(outdir / "assay_g_pca_points.parquet", index=False)
    df[["cohort", "seed", "condition", "phase", "t", "PC1", "PC2", "PC3"] + [c for c in ["body_state", "body_u", "alpha", "g_norm", "g_delta_norm"] if c in df.columns]].to_csv(outdir / "assay_g_pca_points.csv", index=False)

    summ = phase_summary(df, eig_cols=eig_cols, g_cols=g_cols)
    summ.to_csv(outdir / "assay_phase_geometry_summary.csv", index=False)

    pair = build_pairwise_diff(summ, eig_cols=eig_cols, g_cols=g_cols)
    pair.to_csv(outdir / "body_shock_control_phase_diff.csv", index=False)

    did = build_residue_diff(summ, eig_cols=eig_cols, g_cols=g_cols)
    did.to_csv(outdir / "body_shock_control_residue_did.csv", index=False)

    print(f"[save] {outdir / 'assay_g_pca_points.parquet'}")
    print(f"[save] {outdir / 'assay_phase_geometry_summary.csv'}")
    print(f"[save] {outdir / 'body_shock_control_phase_diff.csv'}")
    print(f"[save] {outdir / 'body_shock_control_residue_did.csv'}")
    print(f"[PCA] explained variance ratio: {evr}")

    # Focus on recovery pairwise differences and DID residues.
    rec = pair[pair["phase"] == "recovery"].copy()
    print_mean_se_table(
        rec,
        cols=[
            "shock_minus_control.body_state",
            "shock_minus_control.body_u",
            "shock_minus_control.alpha",
            "shock_minus_control.g_delta_norm",
            "shock_control_PC_centroid_dist",
            "shock_control_g_centroid_dist",
            "shock_control_metric_spectral_l2",
            "shock_minus_control.metric_eig_entropy",
            "shock_minus_control.metric_cond",
            "shock_minus_control.metric_top1_mass",
        ],
        title="RECOVERY: body_shock - control",
    )
    print_mean_se_table(
        did,
        cols=[
            "did_residue.body_state",
            "did_residue.body_u",
            "did_residue.alpha",
            "did_residue.g_delta_norm",
            "did_residue_PC_vector_dist",
            "did_residue_g_vector_dist",
            "did_residue_metric_spectral_l2",
            "did_residue.metric_eig_entropy",
            "did_residue.metric_cond",
            "did_residue.metric_top1_mass",
        ],
        title="DID RESIDUE: (shock recovery-pre) - (control recovery-pre)",
    )

    make_plots(df, summ, outdir, no_plots=args.no_plots)
    print("\nInterpretation hints:")
    print("  Use PCA/centroid/spectral shifts, not g_delta_norm alone, as geometry evidence.")
    print("  If full shows larger metric spectral or PCA residue than no_body_in_g, body shock is entering perspective geometry.")
    print("  If not, move to same-state history-g probe; trajectory geometry may be confounded by behavior/position.")


if __name__ == "__main__":
    main()
