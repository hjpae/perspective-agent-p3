# cear_pilot/experiments/aggregate_multiseed.py
# -*- coding: utf-8 -*-
"""
Aggregate multi-seed phase 3 results.

For each (architecture, seed) run, extract:
  - Training-side metrics (alpha behavior, body saturation, visit pattern,
    body coupling, FiLM/salience/M_g learning indicators)
  - Probe-side metrics (Q1 g_to_probe_ratio, Q2(a)/(b)/(c)/(d) results)

Then:
  - Per-architecture: mean ± std across seeds for each metric.
  - Cross-architecture: paired comparison (sigmoid vs metric_c1, same seed pair).
    Reports paired-t and Wilcoxon for robustness.

Outputs:
  - multiseed_summary.parquet (one row per (arch, seed))
  - multiseed_summary.txt (human-readable summary)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-run metric extraction
# ---------------------------------------------------------------------------

def extract_train_metrics(traj_path: Path) -> Dict[str, float]:
    """From traj.parquet, compute training-side metrics matching the analyses
    we've been doing manually."""
    df = pd.read_parquet(traj_path)
    out: Dict[str, float] = {}
    out["n_episodes"] = int(df["episode"].nunique())
    out["n_steps"] = int(len(df))

    late = df[df["episode"] >= 100]

    # Alpha behavior
    out["alpha_mean_late"] = float(late["alpha"].mean())
    out["alpha_std_late"] = float(late["alpha"].std())
    out["alpha_q90_late"] = float(late["alpha"].quantile(0.9))
    out["alpha_ceiling_frac"] = float((late["alpha"] >= 0.29).mean())

    # Alpha differential response
    on = late[late["perturbation_active"] == 1]["alpha"]
    off = late[late["perturbation_active"] == 0]["alpha"]
    out["alpha_delta_perturb"] = float(on.mean() - off.mean()) if len(on) > 0 and len(off) > 0 else float("nan")
    sat = late[(late["body_state"] <= 0.001) | (late["body_state"] >= 0.999)]
    nonsat = late[(late["body_state"] > 0.001) & (late["body_state"] < 0.999)]
    out["alpha_delta_sat"] = (float(sat["alpha"].mean() - nonsat["alpha"].mean())
                              if len(sat) > 100 and len(nonsat) > 100 else float("nan"))

    # Body saturation pattern (last 50 episodes)
    end = df[df["episode"] >= df["episode"].max() - 50]
    out["body_sat_low_late"]  = float((end["body_state"] <= 0.001).mean())
    out["body_sat_high_late"] = float((end["body_state"] >= 0.999).mean())
    out["body_middle_late"]   = float(((end["body_state"] > 0.1) &
                                       (end["body_state"] < 0.9)).mean())

    # Visit pattern: vertical (top/mid/bot via valence_zone_id) and
    # horizontal (left/mid/right via zone_id, the noise-gradient axis).
    # AAAI's classic Z2 preference emerges along the noise gradient.
    early = df[df["episode"] < 50]
    end_eps = df[df["episode"] >= df["episode"].max() - 50]
    for label, sub in (("early", early), ("late", end_eps)):
        # vertical (valence)
        for vz_id, vz_name in ((0, "top"), (1, "mid"), (2, "bot")):
            out[f"visit_{vz_name}_{label}"] = float(
                (sub["valence_zone_id"] == vz_id).mean()
            )
        # horizontal (predictability gradient)
        # zone_id 0 = left (high noise), 2 = right (low noise)
        for cz_id, cz_name in ((0, "left"), (1, "midcol"), (2, "right")):
            out[f"visit_{cz_name}_{label}"] = float(
                (sub["zone_id"] == cz_id).mean()
            )
    # Late-training mean x position — directly captures right-side bias
    out["mean_x_late"] = float(end_eps["x"].mean())
    out["std_x_late"] = float(end_eps["x"].std())
    out["mean_y_late"] = float(end_eps["y"].mean())
    out["std_y_late"] = float(end_eps["y"].std())
    # Spatial preference shift: right - left fraction (late)
    out["right_minus_left_late"] = (
        out["visit_right_late"] - out["visit_left_late"]
    )
    out["top_minus_bot_late"] = (
        out["visit_top_late"] - out["visit_bot_late"]
    )

    # Body coupling
    df_v = df.copy()
    df_v["body_state_next"] = df_v.groupby("episode")["body_state"].shift(-1)
    df_v = df_v.dropna(subset=["body_state_next"])
    late_v = df_v[df_v["episode"] >= 100]
    if len(late_v) > 100:
        out["corr_body_pred_next"] = float(
            np.corrcoef(late_v["body_pred"], late_v["body_state_next"])[0, 1]
        )
    else:
        out["corr_body_pred_next"] = float("nan")

    nonsat_v = late_v[(late_v["body_state"] > 0.001) &
                      (late_v["body_state"] < 0.999)]
    if len(nonsat_v) > 100:
        out["corr_affordance_body_pe"] = float(
            np.corrcoef(nonsat_v["affordance_here"],
                        nonsat_v["body_pe"])[0, 1]
        )
    else:
        out["corr_affordance_body_pe"] = float("nan")

    # FiLM / salience / metric learning indicators
    z_dim = sum(1 for c in df.columns if c.startswith("gamma_") and c[6:].isdigit())
    if z_dim > 0:
        gamma_cols = [c for c in df.columns if c.startswith("gamma_") and c[6:].isdigit()]
        beta_cols  = [c for c in df.columns if c.startswith("beta_")  and c[5:].isdigit()]
        sub_late = late[gamma_cols + beta_cols]
        out["gamma_abs_mean_late"] = float(sub_late[gamma_cols].abs().values.mean())
        out["beta_abs_mean_late"]  = float(sub_late[beta_cols].abs().values.mean())

    sal_cols = [c for c in df.columns if c.startswith("salience_") and c[9:].isdigit()]
    if sal_cols:
        s_late = late[sal_cols].values
        out["salience_mean_late"] = float(s_late.mean())
        out["salience_std_late"]  = float(s_late.std())
        out["salience_sharpness_late"] = float(2 * np.abs(s_late - 0.5).mean())

    eig_cols = [c for c in df.columns
                if c.startswith("M_g_eig_") and c[8:].isdigit()]
    if eig_cols:
        eig_cols = sorted(eig_cols, key=lambda s: int(s.split("_")[-1]))
        eigs_late = late[eig_cols].values  # (n_steps_late, z_dim) ascending
        out["m_g_max_eig_mean_late"] = float(eigs_late[:, -1].mean())
        out["m_g_min_eig_mean_late"] = float(eigs_late[:, 0].mean())
        cond_late = eigs_late[:, -1] / np.maximum(eigs_late[:, 0], 1e-9)
        out["m_g_cond_mean_late"] = float(cond_late.mean())
        out["m_g_max_eig_std_late"] = float(eigs_late[:, -1].std())

    return out


def extract_probe_metrics(probe_results_path: Path,
                          probe_zquad_path: Optional[Path] = None) -> Dict[str, float]:
    """Recompute Q1 + Q2 metrics from probe_results.parquet.

    For metric_c1 mode, requires probe_zquad.parquet for z_quad-based metrics.
    """
    pr = pd.read_parquet(probe_results_path)
    out: Dict[str, float] = {}

    # mode detection
    mode = str(pr["mode"].iloc[0]) if "mode" in pr.columns else _infer_mode(pr)
    out["mode"] = mode

    n_g = pr["g_id"].nunique()
    n_p = pr["probe_id"].nunique()

    # Build target tensor (n_g, n_p, dim)
    if mode == "metric_c1" and probe_zquad_path is not None:
        zq = pd.read_parquet(probe_zquad_path)
        zq_cols = sorted([c for c in zq.columns if c.startswith("zq_")],
                         key=lambda s: int(s.split("_")[1]))
        target = (zq.sort_values(["g_id", "probe_id"])[zq_cols].values
                  .reshape(n_g, n_p, len(zq_cols)).astype(np.float32))
    else:
        z_cols = sorted([c for c in pr.columns
                         if re.fullmatch(r"z_\d+", c)],
                        key=lambda s: int(s.split("_")[1]))
        target = (pr.sort_values(["g_id", "probe_id"])[z_cols].values
                  .reshape(n_g, n_p, len(z_cols)).astype(np.float32))

    # Build z_raw tensor for sanity check
    zraw_cols = sorted([c for c in pr.columns
                        if re.fullmatch(r"z_raw_\d+", c)],
                       key=lambda s: int(s.rsplit("_", 1)[1]))
    if zraw_cols:
        z_raw_arr = (pr.sort_values(["g_id", "probe_id"])[zraw_cols].values
                     .reshape(n_g, n_p, len(zraw_cols)).astype(np.float32))
    else:
        z_raw_arr = None

    # ---- Q1 ----
    out["q1_var_target_across_g"] = float(target.var(axis=0).mean())
    out["q1_var_target_across_probes"] = float(target.var(axis=1).mean())
    out["q1_g_to_probe_ratio"] = (out["q1_var_target_across_g"]
                                  / max(out["q1_var_target_across_probes"], 1e-9))
    if z_raw_arr is not None:
        out["q1_var_zraw_across_g_sanity"] = float(z_raw_arr.var(axis=0).mean())

    # ---- Q2(a) consistency ----
    pairwise = []
    for ga in range(n_g):
        for gb in range(ga + 1, n_g):
            d = target[ga] - target[gb]
            norms = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
            unit = d / norms
            cos = unit @ unit.T
            mask = ~np.eye(n_p, dtype=bool)
            pairwise.append(float(cos[mask].mean()))
    pw = np.array(pairwise)
    out["q2a_consistency_mean"] = float(pw.mean())
    out["q2a_consistency_min"] = float(pw.min())
    out["q2a_consistency_max"] = float(pw.max())
    out["q2a_consistency_std"] = float(pw.std())

    # ---- Q2(b) linear R^2 ----
    r2 = []
    for ga in range(n_g):
        for gb in range(ga + 1, n_g):
            X = target[ga]
            Y = target[gb]
            X_aug = np.concatenate([X, np.ones((n_p, 1))], axis=1)
            sol, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
            Y_hat = X_aug @ sol
            ss_res = ((Y - Y_hat) ** 2).sum()
            ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
            r2.append(1.0 - ss_res / max(ss_tot, 1e-9))
    out["q2b_linear_r2_mean"] = float(np.mean(r2))
    out["q2b_linear_r2_min"] = float(np.min(r2))

    # ---- Q2(c) silhouette by candidate labels ----
    g_desc = target.mean(axis=1)
    g_meta = (pr.sort_values(["g_id", "probe_id"])
                 .drop_duplicates("g_id")
                 [["g_id", "g_body_state", "g_perturbation",
                   "g_valence_zone"]]
                 .reset_index(drop=True))
    body = g_meta["g_body_state"].values
    body_lbl = pd.cut(body, bins=[-0.01, 0.33, 0.67, 1.01],
                      labels=[0, 1, 2])
    body_lbl = np.asarray(body_lbl.astype(int))
    out["q2c_silhouette_body"] = float(_silhouette(g_desc, body_lbl))

    pert_lbl = g_meta["g_perturbation"].values.astype(int)
    if len(np.unique(pert_lbl)) > 1:
        out["q2c_silhouette_perturb"] = float(_silhouette(g_desc, pert_lbl))
    else:
        out["q2c_silhouette_perturb"] = float("nan")

    vz_lbl = g_meta["g_valence_zone"].values.astype(int)
    if len(np.unique(vz_lbl)) > 1:
        out["q2c_silhouette_vz"] = float(_silhouette(g_desc, vz_lbl))
    else:
        out["q2c_silhouette_vz"] = float("nan")

    # ---- Q2(d) probe-dep spread CoV ----
    spread_per_probe = target.var(axis=0).sum(axis=-1)
    out["q2d_spread_mean"] = float(spread_per_probe.mean())
    out["q2d_spread_cov"]  = float(spread_per_probe.std()
                                   / max(spread_per_probe.mean(), 1e-9))

    # ---- Vis 1 PCA structure ----
    Xc = g_desc - g_desc.mean(axis=0, keepdims=True)
    if len(g_desc) >= 2 and Xc.shape[1] >= 2:
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2) / max(len(g_desc) - 1, 1)
        var_total = var.sum()
        out["pca_pc1_var_ratio"] = float(var[0] / max(var_total, 1e-9))
        if len(var) >= 2:
            out["pca_pc2_var_ratio"] = float(var[1] / max(var_total, 1e-9))
        # Body/PC correlation
        Y2 = (Xc @ np.linalg.svd(Xc, full_matrices=False)[2].T)[:, :min(5, len(var))]
        body_corrs = [float(np.corrcoef(Y2[:, k], body)[0, 1])
                      for k in range(Y2.shape[1])]
        out["pca_max_body_corr"] = float(max(abs(c) for c in body_corrs))

    # ---- Vis 6 probe-discrimination ----
    probe_disc = target.var(axis=1).sum(axis=-1)
    out["disc_max"] = float(probe_disc.max())
    out["disc_min"] = float(probe_disc.min())
    out["disc_max_min_ratio"] = float(probe_disc.max() / max(probe_disc.min(), 1e-9))
    out["disc_mean"] = float(probe_disc.mean())
    out["disc_corr_body"] = float(np.corrcoef(probe_disc, body)[0, 1])

    return out


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    n = len(X)
    if n < 4:
        return float("nan")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    sil = []
    for i in range(n):
        same = (labels == labels[i]) & (np.arange(n) != i)
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b_per = []
        for L in uniq:
            if L == labels[i]:
                continue
            mask = labels == L
            if mask.sum() > 0:
                b_per.append(D[i, mask].mean())
        if not b_per:
            continue
        b = min(b_per)
        sil.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(sil)) if sil else float("nan")


def _infer_mode(pr: pd.DataFrame) -> str:
    cols = set(pr.columns)
    if "M_trace" in cols or any(c.startswith("z_metric_") for c in cols):
        return "metric_c1"
    sal_cols = [c for c in cols if c.startswith("salience_")]
    gam_cols = [c for c in cols if c.startswith("gamma_")]
    sal_nz = sal_cols and pr[sal_cols].abs().sum().sum() > 1e-6
    gam_nz = gam_cols and pr[gam_cols].abs().sum().sum() > 1e-6
    if sal_nz and not gam_nz: return "salience"
    if gam_nz and not sal_nz: return "film"
    if sal_nz: return "salience"
    return "film"


# ---------------------------------------------------------------------------
# Walk runs and aggregate
# ---------------------------------------------------------------------------

def collect_all(root: Path) -> pd.DataFrame:
    rows = []
    for arch_dir in sorted(p for p in root.iterdir() if p.is_dir()
                           and p.name in ("salience", "metric_c1", "film", "both")):
        for seed_dir in sorted(arch_dir.iterdir()):
            m = re.match(r"seed_(\d+)$", seed_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            traj = seed_dir / "traj.parquet"
            probe = seed_dir / "probe" / "probe_results.parquet"
            zquad = seed_dir / "probe" / "probe_zquad.parquet"
            if not traj.exists() or not probe.exists():
                print(f"  [skip] {arch_dir.name}/seed_{seed:02d} — missing files")
                continue

            row: Dict[str, object] = {"arch": arch_dir.name, "seed": seed}
            try:
                row.update(extract_train_metrics(traj))
            except Exception as e:
                print(f"  [warn] train extraction failed for "
                      f"{arch_dir.name}/seed_{seed:02d}: {e}")
            try:
                row.update(extract_probe_metrics(
                    probe, zquad if zquad.exists() else None,
                ))
            except Exception as e:
                print(f"  [warn] probe extraction failed for "
                      f"{arch_dir.name}/seed_{seed:02d}: {e}")
            rows.append(row)
            print(f"  [ok] {arch_dir.name}/seed_{seed:02d}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

PRIMARY_METRICS = [
    # alpha
    "alpha_mean_late", "alpha_ceiling_frac",
    "alpha_delta_perturb", "alpha_delta_sat",
    # body
    "body_sat_low_late", "body_middle_late",
    "corr_body_pred_next",
    # visit pattern (vertical: valence)
    "visit_top_late", "visit_mid_late", "visit_bot_late",
    "top_minus_bot_late",
    # visit pattern (horizontal: predictability)
    "visit_left_late", "visit_midcol_late", "visit_right_late",
    "right_minus_left_late",
    # spatial summary
    "mean_x_late", "mean_y_late",
    # probe Q1
    "q1_g_to_probe_ratio",
    # probe Q2
    "q2a_consistency_mean", "q2a_consistency_min",
    "q2c_silhouette_body", "q2c_silhouette_vz",
    "q2d_spread_cov",
    # PCA
    "pca_pc1_var_ratio", "pca_max_body_corr",
    # discrimination
    "disc_max_min_ratio", "disc_corr_body",
]


def summarize(df: pd.DataFrame, outdir: Path) -> str:
    archs = sorted(df["arch"].unique())
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("MULTI-SEED SUMMARY")
    lines.append("=" * 90)
    for arch in archs:
        sub = df[df["arch"] == arch]
        lines.append(f"\n--- {arch} (n_seeds = {len(sub)}) ---")
        lines.append(f"{'metric':<35} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
        for k in PRIMARY_METRICS:
            if k not in sub.columns:
                continue
            v = sub[k].dropna()
            if len(v) == 0:
                continue
            lines.append(f"{k:<35} {v.mean():>10.4f} {v.std():>10.4f} "
                         f"{v.min():>10.4f} {v.max():>10.4f}")

    if len(archs) >= 2 and "salience" in archs and "metric_c1" in archs:
        lines.append("\n" + "=" * 90)
        lines.append("PAIRED COMPARISON: salience vs metric_c1 (per seed)")
        lines.append("=" * 90)
        a = df[df["arch"] == "salience"].set_index("seed")
        b = df[df["arch"] == "metric_c1"].set_index("seed")
        common_seeds = sorted(set(a.index) & set(b.index))
        lines.append(f"common seeds: {common_seeds}  (n={len(common_seeds)})")
        if len(common_seeds) >= 3:
            lines.append(f"\n{'metric':<35} {'sal mean':>10} {'met mean':>10} "
                         f"{'Δ (m-s)':>10} {'paired-t':>10} {'p_two':>10}")
            from math import sqrt
            for k in PRIMARY_METRICS:
                if k not in a.columns or k not in b.columns:
                    continue
                a_v = a.loc[common_seeds, k].astype(float).values
                b_v = b.loc[common_seeds, k].astype(float).values
                mask = ~(np.isnan(a_v) | np.isnan(b_v))
                a_v = a_v[mask]; b_v = b_v[mask]
                if len(a_v) < 3:
                    continue
                d = b_v - a_v
                d_mean = d.mean()
                d_std = d.std(ddof=1)
                if d_std < 1e-12:
                    t_stat = float("inf") if d_mean != 0 else 0.0
                    p_val = 0.0 if d_mean != 0 else 1.0
                else:
                    t_stat = d_mean / (d_std / sqrt(len(d)))
                    # rough two-sided p using normal approx (n>=10 OK)
                    from math import erf
                    p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
                lines.append(f"{k:<35} {a_v.mean():>10.4f} {b_v.mean():>10.4f} "
                             f"{d_mean:>+10.4f} {t_stat:>10.3f} {p_val:>10.4f}")
            lines.append("\n  (p-value via normal approx of paired-t; treat as approximate)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/multiseed")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[walk] {root}")
    df = collect_all(root)
    if len(df) == 0:
        print("no runs found")
        return

    df.to_parquet(outdir / "multiseed_summary.parquet", index=False)
    print(f"[save] {outdir / 'multiseed_summary.parquet'}  ({len(df)} rows)")

    summary = summarize(df, outdir)
    print(summary)
    (outdir / "multiseed_summary.txt").write_text(summary)
    print(f"[save] {outdir / 'multiseed_summary.txt'}")


if __name__ == "__main__":
    main()