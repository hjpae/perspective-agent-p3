# -*- coding: utf-8 -*-
"""
Paper Figure: Body perturbation leaves perspective-geometric residue.
Non-redundant 3x2 version for SAB/LNCS paper.

Panels
------
(a-c) PCA displacement fans in recovery-phase perspective space:
      shock-history centroid minus control-history centroid, per seed.
      No mean arrow is drawn; each arrow is one seed.
(d)   Metric-geometry separation in the same-state history-g probe,
      measured as metric-spectrum distance under identical probe inputs.
(e)   Seed-level coupling between recovery-phase PCA displacement and
      same-state metric-geometry separation.
(f)   Time-resolved recovery trajectory of shock-control separation in g
      after the perturbation window has ceased.

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
NAME = "figure_geometry_body_to_g_nonredundant_6panel"
FORMATS = ["pdf", "png"]

PHASE = "recovery"
RECOVERY_START_T = 80
N_EIG = 20

PCA_XLIM = (-2.0, 2.0)
PCA_YLIM = (-2.0, 2.0)
TRAJ_YLIM = (0.0, 0.8)
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
    "legend.fontsize": 7.5,
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


def print_spearman(x: np.ndarray, y: np.ndarray, label: str) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    if len(x) < 3:
        print("  Not enough points for Spearman correlation.")
        return
    rho, p = stats.spearmanr(x, y)
    print(f"  Spearman rho = {rho:+.3f}, p = {fmt_p(float(p))}, n = {len(x)}")


# %% ─── LOAD DATA ───────────────────────────────────────────────────────────

path_pca = resolve_path(INPUT_PCA)
path_geom = resolve_path(INPUT_PHASE_GEOMETRY)
path_diff = resolve_path(INPUT_PHASE_DIFF)
path_same = resolve_path(INPUT_SAME_STATE)

pca_all = pd.read_csv(path_pca)
pca = pca_all[pca_all["phase"] == PHASE].copy()

geom = pd.read_csv(path_geom)
geom = geom[geom["phase"] == PHASE].copy()

phase_diff = pd.read_csv(path_diff)
phase_diff = phase_diff[phase_diff["phase"] == PHASE].copy()

same_raw = pd.read_csv(path_same)

print(f"Loaded PCA points       : {len(pca_all)} rows from {path_pca}")
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


def per_seed_recovery_trajectory(df_pca_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return time-resolved shock-control distance and per-seed summary.

    Uses PC1-PC3 distance at matched recovery timesteps. The per-seed summary
    is the mean recovery distance across recovery timesteps, used for stats.
    """
    d = df_pca_all[df_pca_all["phase"] == PHASE].copy()
    cols = ["PC1", "PC2", "PC3"]
    ctrl = d[d["condition"] == "control"][["cohort", "seed", "t"] + cols].copy()
    shock = d[d["condition"] == "body_shock"][["cohort", "seed", "t"] + cols].copy()
    m = pd.merge(ctrl, shock, on=["cohort", "seed", "t"], suffixes=("_ctrl", "_shock"))
    if m.empty:
        raise ValueError("No matched control/body_shock PCA rows found for recovery trajectory.")
    diff = np.column_stack([
        m[f"{c}_shock"].to_numpy(dtype=float) - m[f"{c}_ctrl"].to_numpy(dtype=float)
        for c in cols
    ])
    m["pc3_shock_control_dist"] = np.sqrt((diff ** 2).sum(axis=1))
    m["t_rel"] = m["t"] - int(RECOVERY_START_T)
    auc = m.groupby(["cohort", "seed"], as_index=False)["pc3_shock_control_dist"].mean()
    auc = auc.rename(columns={"pc3_shock_control_dist": "recovery_mean_pc3_distance"})
    return m, auc


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
traj_df, traj_seed_df = per_seed_recovery_trajectory(pca_all)
same_spec_df = per_seed_same_state_metric_spectrum(same_raw)

# Scatter coupling table.
coupling_df = pd.merge(
    disp_df[["cohort", "seed", "pc2_vector_norm"]],
    same_spec_df[["cohort", "seed", "same_state_metric_spectrum_distance"]],
    on=["cohort", "seed"],
    how="inner",
)

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

    # No mean arrow: each arrow is one seed-level displacement vector.
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


def draw_recovery_trajectory(ax, traj: pd.DataFrame, panel_id: str):
    for cohort in COHORT_ORDER:
        sub = traj[traj["cohort"] == cohort].copy()
        q = sub.groupby("t_rel")["pc3_shock_control_dist"].quantile([0.25, 0.5, 0.75]).unstack()
        q = q.sort_index()
        x = q.index.to_numpy(dtype=float)
        med = q[0.5].to_numpy(dtype=float)
        lo = q[0.25].to_numpy(dtype=float)
        hi = q[0.75].to_numpy(dtype=float)
        ax.plot(x, med, color=COLORS[cohort], lw=1.9)
        ax.fill_between(x, lo, hi, color=COLORS[cohort], alpha=0.12, linewidth=0)
    ax.set_title("Recovery trajectory", fontsize=9.0, pad=4)
    ax.set_xlabel("Recovery timestep")
    ax.set_ylabel(r"Shock--control distance in $g$")
    ax.set_ylim(*TRAJ_YLIM)
    ax.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.5)
    ax.text(-0.17, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def draw_seed_bar_panel(
    ax,
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    panel_id: str,
    ylim=None,
    log_y=False,
    jitter_seed=42,
):
    """Fig-3 style bar panel: IQR box + median line + seed dots."""
    rng = np.random.default_rng(jitter_seed)
    bar_w = 0.48
    x_centers = np.arange(len(COHORT_ORDER), dtype=float)

    for i, cohort in enumerate(COHORT_ORDER):
        vals = df[df["cohort"] == cohort][col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        med = np.median(vals)
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        x_off = x_centers[i]

        # IQR box.
        ax.bar(x_off, q75 - q25, bottom=q25, width=bar_w,
               color=COLORS[cohort], alpha=0.28,
               edgecolor=COLORS[cohort], linewidth=0.9, zorder=2)
        # Median line.
        ax.hlines(med, x_off - bar_w * 0.48, x_off + bar_w * 0.48,
                  colors=COLORS[cohort], linewidth=2.0, zorder=4)
        # Jittered seed points.
        xs = x_off + rng.uniform(-bar_w * 0.28, bar_w * 0.28, size=len(vals))
        ax.scatter(xs, vals, s=10, color=COLORS[cohort], alpha=0.36,
                   edgecolor='none', zorder=3)

    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([COHORT_LABELS_2L[c] for c in COHORT_ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.8, pad=4)
    ax.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.5, which="both")
    ax.set_axisbelow(True)
    ax.text(-0.17, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

def draw_coupling_scatter(ax, df: pd.DataFrame, panel_id: str):
    stat_lines = []

    for cohort in COHORT_ORDER:
        sub = df[df["cohort"] == cohort].copy()

        x = sub["pc2_vector_norm"].to_numpy(dtype=float)
        y = sub["same_state_metric_spectrum_distance"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x = x[mask]
        y = y[mask]

        # Scatter points
        ax.scatter(
            x,
            y,
            s=24,
            color=COLORS[cohort],
            alpha=0.42,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

        # Within-cohort Spearman correlation, shown compactly
        if len(x) >= 3:
            rho, p = stats.spearmanr(x, y)
            stat_lines.append(
                f"{COHORT_TITLES[cohort]} "
                rf"$\rho_s$={rho:+.2f}{sig_marker(float(p))}"
            )

        # Cohort-wise trend line in log-y space
        if len(x) >= 3 and np.nanstd(x) > 0:
            logy = np.log10(y)

            X = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(X, logy, rcond=None)

            xg = np.linspace(np.min(x), np.max(x), 100)
            Xg = np.column_stack([np.ones_like(xg), xg])
            yhat = 10 ** (Xg @ beta)

            ax.plot(
                xg,
                yhat,
                color=COLORS[cohort],
                linewidth=1.55,
                alpha=0.92,
                zorder=2,
            )

    # Small compact inset: rho + significance stars only
    if stat_lines:
        ax.text(
            0.96,
            0.04,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            linespacing=1.15,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
            ),
        )

    ax.set_title("PCA-metric correlation", fontsize=8.8, pad=4)
    ax.set_xlabel(r"PCA displacement norm")
    ax.set_ylabel(r"Metric-spectrum distance")
    ax.set_yscale("log")
    ax.set_ylim(*SAME_STATE_SPEC_YLIM)
    ax.grid(axis="both", alpha=0.18, linestyle=":", linewidth=0.5, which="both")
    ax.set_axisbelow(True)
    ax.text(
        -0.17,
        1.05,
        panel_id,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

# %% ─── FIGURE ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(8.5, 5.85))
gs = fig.add_gridspec(
    2, 3,
    hspace=0.45,
    wspace=0.34,
    height_ratios=[1.12, 0.95],
)

# Top row: PCA displacement fans.
for j, cohort in enumerate(COHORT_ORDER):
    ax = fig.add_subplot(gs[0, j])
    draw_displacement_fan(ax, disp_df, cohort, f"({chr(ord('a') + j)})",
                          show_radius_labels=(j == 0))
    if j == 0:
        ax.set_ylabel(r"$\Delta$PC2", fontsize=8.5)

# Bottom row: same-state geometry, cross-metric coupling, and temporal recovery.
# Visual order is unchanged; only panel labels are corrected to (d) - (e) - (f).

ax_e = fig.add_subplot(gs[1, 0])
draw_seed_bar_panel(
    ax_e,
    same_spec_df,
    "same_state_metric_spectrum_distance",
    ylabel=r"Metric-spectrum distance",
    title="Metric-geometry separation",
    panel_id="(d)",
    ylim=SAME_STATE_SPEC_YLIM,
    log_y=True,
)

ax_f = fig.add_subplot(gs[1, 1])
draw_coupling_scatter(ax_f, coupling_df, "(e)")

ax_d = fig.add_subplot(gs[1, 2])
draw_recovery_trajectory(ax_d, traj_df, "(f)")

legend_handles = [
    mpl.lines.Line2D([0], [0], color=COLORS[c], lw=2.2, label=COHORT_TITLES[c])
    for c in COHORT_ORDER
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.90),
    columnspacing=1.4,
    handlelength=2.0,
    borderaxespad=0.2,
)

fig.subplots_adjust(top=0.82)

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
print("NON-REDUNDANT GEOMETRY FIGURE STATISTICS")
print("#" * 70)

print_pairwise_mwu(
    disp_df,
    "pc2_vector_norm",
    "Panels 3a-c — PCA displacement vector norm in PC1-PC2",
)

print_pairwise_mwu(
    traj_seed_df,
    "recovery_mean_pc3_distance",
    "Panel 3f — Mean recovery shock-control distance in g",
)

print_pairwise_mwu(
    same_spec_df,
    "same_state_metric_spectrum_distance",
    "Panel 3d — Metric-geometry separation",
)

print_spearman(
    coupling_df["pc2_vector_norm"].to_numpy(dtype=float),
    coupling_df["same_state_metric_spectrum_distance"].to_numpy(dtype=float),
    "Panel 3e — Coupling between PCA displacement and metric-geometry separation (all seeds)",
)

body_to_g = coupling_df[coupling_df["cohort"].isin(["full", "no_conative"])]
print_spearman(
    body_to_g["pc2_vector_norm"].to_numpy(dtype=float),
    body_to_g["same_state_metric_spectrum_distance"].to_numpy(dtype=float),
    "Panel 3e — Coupling within body→g cohorts only",
)

if not shock_body_df.empty:
    print_pairwise_mwu(
        shock_body_df,
        "shock_minus_control.body_u",
        "Sanity check — matched latent body shock magnitude",
    )

# Save the derived per-seed tables for reproducibility.
disp_out = outdir / f"{NAME}_panels_abc_pc2_displacement.csv"
same_out = outdir / f"{NAME}_panel_d_metric_geometry_separation.csv"
coupling_out = outdir / f"{NAME}_panel_e_coupling.csv"
traj_out = outdir / f"{NAME}_panel_f_recovery_trajectory.csv"
traj_seed_out = outdir / f"{NAME}_panel_f_recovery_mean_distance.csv"

disp_df.to_csv(disp_out, index=False)
traj_df.to_csv(traj_out, index=False)
traj_seed_df.to_csv(traj_seed_out, index=False)
same_spec_df.to_csv(same_out, index=False)
coupling_df.to_csv(coupling_out, index=False)
print(f"\n  saved: {disp_out}")
print(f"  saved: {traj_out}")
print(f"  saved: {traj_seed_out}")
print(f"  saved: {same_out}")
print(f"  saved: {coupling_out}")
