# -*- coding: utf-8 -*-
"""
Paper Figure: Body perturbation leaves perspective-geometric residue.
Combined 6-panel version for SAB/LNCS paper.

Panels
------
(a-c) PCA displacement fans in recovery-phase perspective space:
      shock-history centroid minus control-history centroid, per seed.
(d)   3D PC centroid distance, summarizing panels (a-c).
(e)   Perturbation-assay metric spectrum distance, computed from the
      Fisher-style M_g eigenvalue spectra in control vs body-shock rollouts.
(f)   Same-state history-g metric spectrum distance: identical probe inputs
      are processed with control-history vs shock-history g. This controls
      trajectory/input differences and tests whether the history carried by g
      reorganizes the same input.

Inputs expected under project root:
  outputs/p3_v3_assay_geometry/assay_g_pca_points.csv
  outputs/p3_v3_assay_geometry/assay_phase_geometry_summary.csv
  outputs/p3_v3_assay_geometry/body_shock_control_phase_diff.csv
  outputs/p3_v3_same_state_probe/same_state_history_probe_shock_control_diff_recovery_mean.csv

Spyder usage
------------
1. Put this file at project root or run it with project root as working dir.
2. Edit CONFIG if paths differ.
3. Run File (F5). Statistics print to console; figure is saved to OUTDIR.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_PCA = "outputs/p3_v3_assay_geometry/assay_g_pca_points.csv"
INPUT_PHASE_GEOMETRY = "outputs/p3_v3_assay_geometry/assay_phase_geometry_summary.csv"
INPUT_PHASE_DIFF = "outputs/p3_v3_assay_geometry/body_shock_control_phase_diff.csv"
INPUT_SAME_STATE = "outputs/p3_v3_same_state_probe/same_state_history_probe_shock_control_diff_recovery_mean.csv"

OUTDIR = "./figures"
NAME = "figure_geometry_body_to_g_combined_6panel"
FORMATS = ["pdf", "png"]

PHASE = "recovery"
N_EIG = 20

# Visual axis ranges; adjust if your local figure needs a little more room.
PCA_XLIM = (-2.0, 2.0)
PCA_YLIM = (-2.0, 2.0)
CENTROID_YLIM = (0.0, 0.85)
PERTURB_SPEC_YLIM = (0.0, 3.0)
SAME_STATE_SPEC_YLIM = (0.003, 30.0)  # log scale


# %% ─── STYLE ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
    "font.size": 9,
    "axes.labelsize": 8.8,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

COLORS = {
    "full": "#2c5282",
    "no_conative": "#c0392b",
    "no_body_in_g": "#8b7355",
}

COHORT_ORDER = ["full", "no_conative", "no_body_in_g"]

COHORT_TITLES = {
    "full": "Full",
    "no_conative": "No conation",
    "no_body_in_g": "No body→g",
}

COHORT_LABELS_2L = {
    "full": "Full",
    "no_conative": "No\nconation",
    "no_body_in_g": "No\nbody→g",
}


# %% ─── UTILS ───────────────────────────────────────────────────────────────

def resolve_path(path_str: str) -> Path:
    """Resolve paths for project-root, /mnt/data, or flat exported CSV usage."""
    candidates = [
        Path(path_str),
        Path("/mnt/data") / path_str,
        Path("/mnt/data") / Path(path_str).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {path_str}. Tried: " + ", ".join(str(p) for p in candidates)
    )


def bh_fdr(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, returned in original order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj_sorted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running = min(running, val)
        adj_sorted[i] = running
    adj = np.empty(n, dtype=float)
    adj[order] = np.clip(adj_sorted, 0, 1)
    return adj


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    # Same convention as previous scripts: r = 1 - 2U/(n1*n2).
    return float(1.0 - (2.0 * u) / (n1 * n2))


def print_pairwise_mwu(df: pd.DataFrame, col: str, title: str) -> None:
    """Print cohort summary and Mann-Whitney U pairwise tests with BH-FDR."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("  Test: Mann-Whitney U (two-sided), with BH-FDR correction.")
    print("  Effect size: rank-biserial r = 1 - 2U/(n1*n2).")
    print(f"\n  Metric: {col}")
    print(f"    {'cohort':14s} {'n':>4s} {'median':>10s} {'IQR':>23s}")
    for cohort in COHORT_ORDER:
        vals = df[df["cohort"] == cohort][col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            print(f"    {COHORT_TITLES[cohort]:14s} {0:4d} {'NA':>10s} {'NA':>23s}")
            continue
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        print(f"    {COHORT_TITLES[cohort]:14s} {len(vals):4d} {np.median(vals):10.4f} [{q25:7.4f}, {q75:7.4f}]")

    rows = []
    pvals = []
    for c1, c2 in combinations(COHORT_ORDER, 2):
        x = df[df["cohort"] == c1][col].dropna().to_numpy(dtype=float)
        y = df[df["cohort"] == c2][col].dropna().to_numpy(dtype=float)
        if len(x) == 0 or len(y) == 0:
            u, p, r = np.nan, np.nan, np.nan
        else:
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            r = rank_biserial_from_u(float(u), len(x), len(y))
        rows.append((c1, c2, u, p, r))
        pvals.append(p if np.isfinite(p) else 1.0)
    padj = bh_fdr(pvals)

    print(f"\n    {'pair':33s} {'U':>8s} {'p_raw':>7s} {'p_adj':>7s} {'r_rb':>7s} {'sig':>4s}")
    for (c1, c2, u, p, r), pa in zip(rows, padj):
        pair = f"{COHORT_TITLES[c1]} vs {COHORT_TITLES[c2]}"
        print(f"    {pair:33s} {u:8.1f} {fmt_p(p):>7s} {fmt_p(pa):>7s} {r:+7.2f} {sig_marker(pa):>4s}")


# %% ─── LOAD DATA ───────────────────────────────────────────────────────────

path_pca = resolve_path(INPUT_PCA)
path_geom = resolve_path(INPUT_PHASE_GEOMETRY)
path_diff = resolve_path(INPUT_PHASE_DIFF)
path_same = resolve_path(INPUT_SAME_STATE)

pca = pd.read_csv(path_pca)
pca = pca[pca["phase"] == PHASE].copy()

geom = pd.read_csv(path_geom)
geom = geom[geom["phase"] == PHASE].copy()

phase_diff = pd.read_csv(path_diff)
phase_diff = phase_diff[phase_diff["phase"] == PHASE].copy()

same_raw = pd.read_csv(path_same)

print(f"Loaded PCA points       : {len(pca)} rows from {path_pca}")
print(f"Loaded phase geometry   : {len(geom)} rows from {path_geom}")
print(f"Loaded phase diff       : {len(phase_diff)} rows from {path_diff}")
print(f"Loaded same-state probe : {len(same_raw)} rows from {path_same}")
print(f"PCA cohorts             : {pca['cohort'].value_counts().to_dict()}")
print(f"Same-state seeds/cohort : {same_raw.groupby('cohort')['seed'].nunique().to_dict()}")


# %% ─── DERIVE METRICS ──────────────────────────────────────────────────────

def per_seed_pc2_displacement(df_pca: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, seed), g in df_pca.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][["PC1", "PC2"]].mean().to_numpy(dtype=float)
        shock = g[g["condition"] == "body_shock"][["PC1", "PC2"]].mean().to_numpy(dtype=float)
        if np.any(~np.isfinite(ctrl)) or np.any(~np.isfinite(shock)):
            continue
        rows.append({
            "cohort": cohort, "seed": seed,
            "dPC1": shock[0] - ctrl[0],
            "dPC2": shock[1] - ctrl[1],
            "pc2_vector_norm": float(np.linalg.norm(shock - ctrl)),
        })
    return pd.DataFrame(rows)


def per_seed_pc_centroid_dist(df_pca: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["PC1", "PC2", "PC3"]
    for (cohort, seed), g in df_pca.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][cols].mean().to_numpy(dtype=float)
        shock = g[g["condition"] == "body_shock"][cols].mean().to_numpy(dtype=float)
        if np.any(~np.isfinite(ctrl)) or np.any(~np.isfinite(shock)):
            continue
        rows.append({
            "cohort": cohort, "seed": seed,
            "centroid_dist_3d": float(np.linalg.norm(shock - ctrl)),
        })
    return pd.DataFrame(rows)


def per_seed_perturb_metric_spectral_l2(df_geom: pd.DataFrame, n_eig: int = N_EIG) -> pd.DataFrame:
    eig_cols = [f"M_g_eig_{i}" for i in range(n_eig)]
    missing = [c for c in eig_cols if c not in df_geom.columns]
    if missing:
        raise KeyError(f"Missing eigen columns in phase geometry: {missing[:5]}...")
    rows = []
    for (cohort, seed), g in df_geom.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][eig_cols].mean().to_numpy(dtype=float)
        shock = g[g["condition"] == "body_shock"][eig_cols].mean().to_numpy(dtype=float)
        if np.any(~np.isfinite(ctrl)) or np.any(~np.isfinite(shock)):
            continue
        rows.append({
            "cohort": cohort, "seed": seed,
            "perturb_metric_spectral_l2": float(np.linalg.norm(np.sort(shock)[::-1] - np.sort(ctrl)[::-1])),
        })
    return pd.DataFrame(rows)


def per_seed_same_state_metric_spectrum(df_same: pd.DataFrame, n_eig: int = N_EIG) -> pd.DataFrame:
    # Current same-state output stores shock-minus-control eigenvalue difference vector.
    eig_cols = [f"shock_minus_control.metric_eig_{i}" for i in range(n_eig)]
    missing = [c for c in eig_cols if c not in df_same.columns]
    if missing:
        raise KeyError(f"Missing same-state eig diff columns: {missing[:5]}...")
    d = df_same.copy()
    d["same_state_metric_spectrum_distance"] = np.sqrt((d[eig_cols].to_numpy(dtype=float) ** 2).sum(axis=1))
    out = d.groupby(["cohort", "seed"], as_index=False)["same_state_metric_spectrum_distance"].mean()
    return out


disp_df = per_seed_pc2_displacement(pca)
centroid_df = per_seed_pc_centroid_dist(pca)
perturb_spec_df = per_seed_perturb_metric_spectral_l2(geom)
same_spec_df = per_seed_same_state_metric_spectrum(same_raw)

# Optional sanity check: shock magnitude is matched across cohorts.
if "shock_minus_control.body_u" in phase_diff.columns:
    shock_body_df = phase_diff[["cohort", "seed", "shock_minus_control.body_u"]].dropna().copy()
else:
    shock_body_df = pd.DataFrame()


# %% ─── DRAW FUNCTIONS ──────────────────────────────────────────────────────

def draw_displacement_fan(
    ax,
    disp: pd.DataFrame,
    cohort: str,
    panel_id: str,
    xlim=PCA_XLIM,
    ylim=PCA_YLIM,
    ref_radii=(0.5, 1.0, 1.5),
    show_radius_labels=False,
    arrow_alpha=0.55,
    arrow_lw=1.05,
    tip_size=18,
    mean_arrow_lw=2.5,
    mean_arrow_mut=18,
):
    for r in ref_radii:
        circ = plt.Circle((0, 0), r, fill=False, color="gray",
                          linewidth=0.5, linestyle=":", alpha=0.5)
        ax.add_patch(circ)
    if show_radius_labels:
        for r in ref_radii:
            ax.text(r, -0.08, f"{r}", fontsize=6.5, color="gray",
                    ha="center", va="top", alpha=0.75)

    sub = disp[disp["cohort"] == cohort]
    arr = sub[["dPC1", "dPC2"]].to_numpy(dtype=float)
    for dx, dy in arr:
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=COLORS[cohort],
                                    alpha=arrow_alpha, lw=arrow_lw,
                                    shrinkA=0, shrinkB=0))
        ax.scatter(dx, dy, s=tip_size, color=COLORS[cohort], alpha=0.75,
                   edgecolor="white", linewidth=0.4, marker="X", zorder=4)

    if len(arr) > 0:
        mean = arr.mean(axis=0)
        ax.annotate("", xy=tuple(mean), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=mean_arrow_lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=mean_arrow_mut), zorder=10)
    ax.scatter(0, 0, s=45, color="white", edgecolor="black",
               linewidth=1.2, marker="o", zorder=11)

    ax.set_title(COHORT_TITLES[cohort], fontsize=10, pad=4)
    ax.set_xlabel(r"$\Delta$PC1", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.text(-0.10, 1.06, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def draw_summary_panel(
    ax,
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    panel_id: str,
    ylim=None,
    log_y=False,
    zero_ref=False,
    box_width=0.46,
    dot_jitter=0.13,
    dot_size=10,
    dot_alpha=0.45,
    box_alpha=0.22,
    median_lw=2.35,
    jitter_seed=42,
    panel_x=-0.20,
):
    rng = np.random.default_rng(jitter_seed)
    for i, cohort in enumerate(COHORT_ORDER):
        vals = df[df["cohort"] == cohort][col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        med = np.median(vals)
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        ax.bar(i, q75 - q25, bottom=q25, width=box_width,
               color=COLORS[cohort], alpha=box_alpha,
               edgecolor=COLORS[cohort], linewidth=1.0, zorder=2)
        ax.hlines(med, i - box_width / 2, i + box_width / 2,
                  colors=COLORS[cohort], linewidth=median_lw, zorder=4)
        xs = i + rng.uniform(-dot_jitter, dot_jitter, size=len(vals))
        ax.scatter(xs, vals, s=dot_size, color=COLORS[cohort],
                   alpha=dot_alpha, edgecolor="none", zorder=3)

    if zero_ref:
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels([COHORT_LABELS_2L[c] for c in COHORT_ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5, pad=4)
    ax.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.5, which="both")
    ax.set_axisbelow(True)
    ax.text(panel_x, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


# %% ─── FIGURE ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(8.5, 5.7))
gs = fig.add_gridspec(
    2, 3,
    hspace=0.45,
    wspace=0.34,
    height_ratios=[1.22, 1.0],
)

# Top row: PCA displacement fans.
for j, cohort in enumerate(COHORT_ORDER):
    ax = fig.add_subplot(gs[0, j])
    draw_displacement_fan(ax, disp_df, cohort, f"({chr(ord('a') + j)})",
                          show_radius_labels=(j == 0))
    if j == 0:
        ax.set_ylabel(r"$\Delta$PC2", fontsize=8.5)

# Bottom row: scalar summaries.
ax_d = fig.add_subplot(gs[1, 0])
draw_summary_panel(
    ax_d, centroid_df, "centroid_dist_3d",
    ylabel=r"PC centroid distance",
    title="Recovery displacement",
    panel_id="(d)",
    ylim=CENTROID_YLIM,
    zero_ref=False,
)

ax_e = fig.add_subplot(gs[1, 1])
draw_summary_panel(
    ax_e, perturb_spec_df, "perturb_metric_spectral_l2",
    ylabel=r"Metric-spectrum distance",
    title="Perturbation metric shift",
    panel_id="(e)",
    ylim=PERTURB_SPEC_YLIM,
    zero_ref=False,
)

ax_f = fig.add_subplot(gs[1, 2])
draw_summary_panel(
    ax_f, same_spec_df, "same_state_metric_spectrum_distance",
    ylabel=r"Metric-spectrum distance",
    title="Same-state history-$g$ probe",
    panel_id="(f)",
    ylim=SAME_STATE_SPEC_YLIM,
    log_y=True,
    zero_ref=False,
)

plt.show()


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)
for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=300 if fmt == "png" else None)
    print(f"  saved: {out}")


# %% ─── STATISTICS ──────────────────────────────────────────────────────────

print("\n" + "#" * 70)
print("COMBINED GEOMETRY FIGURE STATISTICS")
print("#" * 70)

# Top-row displacement in 2D PC space; useful for panel (a-c) qualitative read.
print_pairwise_mwu(
    disp_df,
    "pc2_vector_norm",
    "Panels 2a-c — PCA displacement vector norm in PC1-PC2",
)

print_pairwise_mwu(
    centroid_df,
    "centroid_dist_3d",
    "Panel 2d — 3D PC centroid distance, shock vs control",
)

print_pairwise_mwu(
    perturb_spec_df,
    "perturb_metric_spectral_l2",
    "Panel 2e — Perturbation-assay metric-spectrum distance",
)

print_pairwise_mwu(
    same_spec_df,
    "same_state_metric_spectrum_distance",
    "Panel 2f — Same-state history-g metric-spectrum distance",
)

if not shock_body_df.empty:
    print_pairwise_mwu(
        shock_body_df,
        "shock_minus_control.body_u",
        "Sanity check — matched latent body shock magnitude",
    )

# Save the derived per-seed tables for reproducibility.
centroid_out = outdir / f"{NAME}_panel_d_centroid_dist.csv"
perturb_out = outdir / f"{NAME}_panel_e_perturb_metric_spectral_l2.csv"
same_out = outdir / f"{NAME}_panel_f_same_state_metric_spectrum.csv"
disp_out = outdir / f"{NAME}_panels_abc_pc2_displacement.csv"

disp_df.to_csv(disp_out, index=False)
centroid_df.to_csv(centroid_out, index=False)
perturb_spec_df.to_csv(perturb_out, index=False)
same_spec_df.to_csv(same_out, index=False)
print(f"\n  saved: {disp_out}")
print(f"  saved: {centroid_out}")
print(f"  saved: {perturb_out}")
print(f"  saved: {same_out}")
