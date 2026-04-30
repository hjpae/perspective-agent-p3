# cear_pilot/experiments/visualize_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 visualization. Currently implements Vis 1 (g state PCA in target
space). Designed to be extended with Vis 4/5/6 later.

Vis 1: g state PCA
  - For metric_c1 mode: PCA on z_quad descriptors (256-dim).
  - For film/salience modes: PCA on z descriptors (16-dim).
  - For each g state, average target representation over probes → one descriptor.
  - PCA → 2D, color/marker by candidate organizers (body, valence, perturb).

  This visualization addresses the silhouette-paradox question: even when
  the silhouette score in full target space is near zero or negative,
  PCA can reveal whether the organization is real but lives in a low-dim
  subspace.

Inputs:
  --probe_results   probe_results.parquet (from probe_phase3.py)
  --probe_zquad     probe_zquad.parquet (only for metric_c1 mode)
  --outdir          where to save figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------------
# Load: build (n_g, n_p, dim) target tensor + g meta
# ---------------------------------------------------------------------------

def load_probe_data(
    probe_results_path: str,
    probe_zquad_path: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """
    Returns:
      target: (n_g, n_p, dim) — z_quad if metric_c1, else z
      g_meta: per-g metadata dataframe
      mode: gating mode string
    """
    pr = pd.read_parquet(probe_results_path)
    mode = str(pr["mode"].iloc[0])
    n_g = pr["g_id"].nunique()
    n_p = pr["probe_id"].nunique()

    if mode == "metric_c1":
        if probe_zquad_path is None:
            raise ValueError("metric_c1 mode requires --probe_zquad path")
        zq = pd.read_parquet(probe_zquad_path)
        zq_cols = sorted([c for c in zq.columns if c.startswith("zq_")],
                         key=lambda s: int(s.split("_")[1]))
        target = (zq.sort_values(["g_id", "probe_id"])[zq_cols].values
                  .reshape(n_g, n_p, len(zq_cols)).astype(np.float32))
    else:
        z_cols = sorted(
            [c for c in pr.columns if c.startswith("z_") and c[2:].isdigit()],
            key=lambda s: int(s.split("_")[1]),
        )
        target = (pr.sort_values(["g_id", "probe_id"])[z_cols].values
                  .reshape(n_g, n_p, len(z_cols)).astype(np.float32))

    # per-g metadata (all rows for given g_id share same g meta, take first)
    g_meta = (pr.sort_values(["g_id", "probe_id"])
                .drop_duplicates(subset="g_id")
                [["g_id", "g_episode", "g_t", "g_body_state",
                  "g_perturbation", "g_valence_zone"]]
                .reset_index(drop=True))

    print(f"[load] mode={mode}  target shape={target.shape}  g_meta rows={len(g_meta)}")
    return target, g_meta, mode


# ---------------------------------------------------------------------------
# PCA via numpy
# ---------------------------------------------------------------------------

def pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Centered PCA via SVD. Returns (Y, explained_var_ratio)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # use SVD on centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = (U[:, :n_components] * S[:n_components])
    var = (S ** 2) / max(len(X) - 1, 1)
    var_ratio = var / var.sum()
    return Y, var_ratio[:n_components]


# ---------------------------------------------------------------------------
# Vis 1: g state PCA scatter
# ---------------------------------------------------------------------------

def vis1_g_pca(
    target: np.ndarray,
    g_meta: pd.DataFrame,
    mode: str,
    outdir: Path,
) -> Dict[str, float]:
    """
    g descriptor = mean target over probes (one vector per g state).
    PCA → 2D. Plot with three encodings:
      - color   = body_state (continuous viridis)
      - marker  = valence_zone (▲ top, ● mid, ▼ bot)
      - edge    = perturb_active (black = on, none = off)

    Saves vis1_g_pca.png and prints summary statistics.
    """
    g_desc = target.mean(axis=1)  # (n_g, dim)
    Y, var_ratio = pca(g_desc, n_components=2)

    body = g_meta["g_body_state"].values
    vz = g_meta["g_valence_zone"].values.astype(int)
    pert = g_meta["g_perturbation"].values.astype(int)

    # Pearson correlation: body_state with PC1, PC2
    corr_pc1 = float(np.corrcoef(Y[:, 0], body)[0, 1]) if len(Y) > 2 else float("nan")
    corr_pc2 = float(np.corrcoef(Y[:, 1], body)[0, 1]) if len(Y) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(8.5, 7))

    marker_map = {0: "^", 1: "o", 2: "v"}     # vz: top=▲ mid=● bot=▼
    label_map = {0: "vz top (positive)", 1: "vz mid", 2: "vz bot (negative)"}

    for vz_id in (0, 1, 2):
        for p_active in (0, 1):
            sel = (vz == vz_id) & (pert == p_active)
            if sel.sum() == 0:
                continue
            edge = "black" if p_active == 1 else "none"
            label = label_map[vz_id] + (" + perturb" if p_active else "")
            sc = ax.scatter(
                Y[sel, 0], Y[sel, 1],
                c=body[sel], cmap="viridis",
                vmin=0.0, vmax=1.0,
                marker=marker_map[vz_id],
                s=180,
                edgecolors=edge, linewidths=1.5,
                label=label, alpha=0.95,
            )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("body_state (0 = depleted ↘ 1 = saturated)", fontsize=11)

    # annotate g_id for traceability
    for i in range(len(Y)):
        ax.annotate(f"  g{i}", (Y[i, 0], Y[i, 1]), fontsize=8, alpha=0.7)

    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% variance)", fontsize=12)
    target_name = "z_quad" if mode == "metric_c1" else "z"
    ax.set_title(
        f"Vis 1: g state PCA in {target_name} space  (mode = {mode})\n"
        f"corr(body, PC1) = {corr_pc1:+.3f}  |  corr(body, PC2) = {corr_pc2:+.3f}",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    out_path = outdir / "vis1_g_pca.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    # Also: best linear axis correlation across first 5 PCs
    Y5, var5 = pca(g_desc, n_components=min(5, g_desc.shape[0] - 1))
    pc_correlations = {
        "body": [float(np.corrcoef(Y5[:, k], body)[0, 1]) for k in range(Y5.shape[1])],
    }
    if len(np.unique(vz)) > 1:
        # for categorical, compute per-PC eta-squared (variance explained by category)
        pc_correlations["valence_zone_eta2"] = []
        for k in range(Y5.shape[1]):
            ss_total = float(np.var(Y5[:, k]) * len(Y5))
            ss_between = 0.0
            grand_mean = Y5[:, k].mean()
            for c in np.unique(vz):
                grp = Y5[vz == c, k]
                ss_between += len(grp) * (grp.mean() - grand_mean) ** 2
            eta2 = ss_between / max(ss_total, 1e-9)
            pc_correlations["valence_zone_eta2"].append(float(eta2))
    if len(np.unique(pert)) > 1:
        pc_correlations["perturb_eta2"] = []
        for k in range(Y5.shape[1]):
            ss_total = float(np.var(Y5[:, k]) * len(Y5))
            ss_between = 0.0
            grand_mean = Y5[:, k].mean()
            for c in np.unique(pert):
                grp = Y5[pert == c, k]
                ss_between += len(grp) * (grp.mean() - grand_mean) ** 2
            eta2 = ss_between / max(ss_total, 1e-9)
            pc_correlations["perturb_eta2"].append(float(eta2))

    print(f"\n  PCA explained variance ratios (first 5 PCs): "
          f"{[round(float(v), 3) for v in var5]}")
    print(f"  cumulative: {[round(float(v), 3) for v in np.cumsum(var5)]}")
    print(f"\n  Body correlation per PC: "
          f"{[round(c, 3) for c in pc_correlations['body']]}")
    if "valence_zone_eta2" in pc_correlations:
        print(f"  Valence zone eta^2 per PC: "
              f"{[round(e, 3) for e in pc_correlations['valence_zone_eta2']]}")
    if "perturb_eta2" in pc_correlations:
        print(f"  Perturb eta^2 per PC: "
              f"{[round(e, 3) for e in pc_correlations['perturb_eta2']]}")

    # Honest interpretation hint
    max_body_corr = max(abs(c) for c in pc_correlations["body"])
    print(f"\n  Strongest body|PC correlation across first 5 PCs: {max_body_corr:.3f}")
    if max_body_corr > 0.5:
        print(f"  → Body organizes a meaningful PC axis (despite silhouette).")
    elif max_body_corr > 0.3:
        print(f"  → Body has moderate PC alignment.")
    else:
        print(f"  → No PC axis strongly aligns with body. silhouette result reflects "
              f"true lack of body-organization.")

    return {
        "pc1_var_ratio": float(var_ratio[0]),
        "pc2_var_ratio": float(var_ratio[1]),
        "corr_body_pc1": corr_pc1,
        "corr_body_pc2": corr_pc2,
        "max_body_pc_corr": max_body_corr,
        "n_g": int(len(Y)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_results", type=str, required=True)
    ap.add_argument("--probe_zquad", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs/visualize_phase3")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target, g_meta, mode = load_probe_data(args.probe_results, args.probe_zquad)

    print(f"\n=== Vis 1: g state PCA ===")
    summary = vis1_g_pca(target, g_meta, mode, outdir)
    print(f"\nSummary: {summary}")
    print(f"\n[done] figures in {outdir}/")


if __name__ == "__main__":
    main()
