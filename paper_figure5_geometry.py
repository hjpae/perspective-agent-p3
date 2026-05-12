# -*- coding: utf-8 -*-
"""
Paper Figure 5: Body perturbation reshapes perspective geometry — but only
when body PE is routed into the perspective latent.

Procedure
---------
Frozen trained agents (no gradient updates) undergo a paired assay:
  control     : ordinary online dynamics for 160 steps
  body_shock  : same trajectory, but with body_u perturbation of -0.08 per
                step injected during a shock window (t=40..70)
Recovery phase (t=80..159) is analyzed: how does the perspective state
relax after the perturbation has ended?

Message
-------
- (a, b, c) Top row: per-seed perspective displacement (shock centroid −
  control centroid) in the first two principal components of g, anchored
  at the origin. Each arrow is one seed's displacement; the thick black
  arrow is the cohort-mean. Dotted reference circles at radii 0.5/1.0/1.5.
    Full         : arrows fan broadly, some past r=1.5.
    No conation  : arrows concentrate inside r≈1.0.
    No body→g    : arrows collapse to a small ball inside r≈0.5 — same
                    input under different perturbation histories yields
                    nearly the same g, because body PE is not routed into
                    the perspective latent.

- (d) Shock magnitude on body_u is identical across cohorts (≈ -1.25).
  Clean control: downstream divergence is not driven by uneven perturbation.
- (e) PC centroid distance summarizes the top row.
- (f) Metric spectral L2 (L2 distance between sorted eigenvalue spectra of
  the Fisher-style metric M_g, control vs shock): full and no_conative both
  show clear spectral reshaping in magnitude; no_body_in_g does not.
  (The direction of reshape is examined in Figure 6.)

Data
----
Inputs:
  assay_g_pca_points.csv             — time-resolved (PC1, PC2, PC3)
  body_shock_control_phase_diff.csv  — phase-level shock − control diffs
  assay_phase_geometry_summary.csv   — phase-level M_g eigenvalues

Spyder usage
------------
1. Edit CONFIG below.
2. Run File (F5), or run cell by cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_PCA            = "outputs/p3_v3_assay_geometry/assay_g_pca_points.csv"
INPUT_PHASE_DIFF     = "outputs/p3_v3_assay_geometry/body_shock_control_phase_diff.csv"
INPUT_PHASE_GEOMETRY = "outputs/p3_v3_assay_geometry/assay_phase_geometry_summary.csv"
OUTDIR  = "./figures"
NAME    = "figure5_geometry"
FORMATS = ["pdf", "png"]

PHASE   = "recovery"


# %% ─── STYLE ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Liberation Serif', 'Times New Roman'],
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
})

COLORS = {
    'full':         '#2c5282',
    'no_conative':  '#c0392b',
    'no_body_in_g': '#8b7355',
}

COHORT_TITLES = {
    'full':         'Full',
    'no_conative':  'No conation',
    'no_body_in_g': 'No body→g',
}

COHORT_LABELS_2L = {
    'full':         'Full',
    'no_conative':  'No\nconation',
    'no_body_in_g': 'No\nbody→g',
}

COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']


# %% ─── LOAD ────────────────────────────────────────────────────────────────

df_pca = pd.read_csv(INPUT_PCA)
df_pca = df_pca[df_pca["phase"] == PHASE].copy()

df_diff = pd.read_csv(INPUT_PHASE_DIFF)
df_diff = df_diff[df_diff["phase"] == PHASE].copy()

df_geom = pd.read_csv(INPUT_PHASE_GEOMETRY)
df_geom = df_geom[df_geom["phase"] == PHASE].copy()

print(f"PCA points  : {len(df_pca)} rows ({PHASE} phase)")
print(f"Phase diffs : {len(df_diff)} rows")
print(f"Geometry    : {len(df_geom)} rows")


# %% ─── DERIVED METRICS ─────────────────────────────────────────────────────

def per_seed_pc_centroid_dist(df_pca_in):
    """For each (cohort, seed), Euclidean distance between control and shock
    centroids in (PC1, PC2, PC3) space."""
    rows = []
    for (cohort, seed), g in df_pca_in.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][["PC1", "PC2", "PC3"]].mean()
        shock = g[g["condition"] == "body_shock"][["PC1", "PC2", "PC3"]].mean()
        rows.append({"cohort": cohort, "seed": seed,
                     "centroid_dist_3d": np.linalg.norm(shock.values - ctrl.values)})
    return pd.DataFrame(rows)


def per_seed_metric_spectral_l2(df_geom_in, n_eig=20):
    """For each (cohort, seed), L2 distance between sorted eigenvalue
    spectra of the Fisher metric M_g, control vs shock."""
    eig_cols = [f"M_g_eig_{i}" for i in range(n_eig)]
    rows = []
    for (cohort, seed), g in df_geom_in.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][eig_cols].mean().values
        shock = g[g["condition"] == "body_shock"][eig_cols].mean().values
        spec_l2 = np.linalg.norm(np.sort(shock)[::-1] - np.sort(ctrl)[::-1])
        rows.append({"cohort": cohort, "seed": seed, "spectral_l2": spec_l2})
    return pd.DataFrame(rows)


def per_seed_pc2_displacement(df_pca_in):
    """For each (cohort, seed), the (ΔPC1, ΔPC2) displacement vector from
    control centroid to shock centroid. Used by the displacement-fan panel."""
    rows = []
    for (cohort, seed), g in df_pca_in.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][["PC1", "PC2"]].mean().values
        shock = g[g["condition"] == "body_shock"][["PC1", "PC2"]].mean().values
        rows.append({"cohort": cohort, "seed": seed,
                     "dPC1": shock[0] - ctrl[0],
                     "dPC2": shock[1] - ctrl[1]})
    return pd.DataFrame(rows)


centroid_df = per_seed_pc_centroid_dist(df_pca)
spec_df     = per_seed_metric_spectral_l2(df_geom)
disp_df     = per_seed_pc2_displacement(df_pca)


# %% ─── DRAW: DISPLACEMENT FAN PANEL ────────────────────────────────────────

def draw_displacement_fan(ax, disp_df, cohort, panel_id,
                          xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
                          ref_radii=(0.5, 1.0, 1.5),
                          show_radius_labels=False,
                          arrow_alpha=0.55, arrow_lw=1.1,
                          tip_size=18,
                          mean_arrow_lw=2.5, mean_arrow_mut=18):
    """Per-seed (ΔPC1, ΔPC2) anchored at origin, plus cohort-mean arrow.

    Visual knobs:
      ref_radii          — tuple of reference circle radii
      arrow_alpha,_lw    — per-seed arrow style
      tip_size           — endpoint X marker
      mean_arrow_lw,_mut — cohort-mean arrow style
    """
    # Reference circles
    for r in ref_radii:
        circ = plt.Circle((0, 0), r, fill=False, color='gray',
                          linewidth=0.5, linestyle=':', alpha=0.5)
        ax.add_patch(circ)
    if show_radius_labels:
        for r in ref_radii:
            ax.text(r, -0.08, f"{r}", fontsize=6.5, color='gray',
                    ha='center', va='top', alpha=0.7)

    # Per-seed arrows
    sub = disp_df[disp_df["cohort"] == cohort]
    disps = sub[["dPC1", "dPC2"]].to_numpy()
    for dx, dy in disps:
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color=COLORS[cohort],
                                    alpha=arrow_alpha, lw=arrow_lw,
                                    shrinkA=0, shrinkB=0))
        ax.scatter(dx, dy, s=tip_size, color=COLORS[cohort], alpha=0.75,
                   edgecolor='white', linewidth=0.4, marker='X', zorder=4)

    # Cohort-mean arrow (thick black)
    mean_disp = disps.mean(axis=0)
    ax.annotate("", xy=tuple(mean_disp), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>",
                                color="black", lw=mean_arrow_lw,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=mean_arrow_mut), zorder=10)
    ax.scatter(0, 0, s=50, color='white', edgecolor='black',
               linewidth=1.5, marker='o', zorder=11)

    ax.set_title(COHORT_TITLES[cohort], fontsize=10)
    ax.set_xlabel(r"$\Delta$PC1", fontsize=8.5)
    ax.set_aspect('equal')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axhline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax.text(-0.10, 1.06, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── DRAW: SUMMARY BAR PANEL ─────────────────────────────────────────────

def draw_summary_panel(ax, df, col, ylabel, ylim, panel_id, title,
                       zero_ref=True,
                       box_width=0.46, dot_jitter=0.13, dot_size=10,
                       dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
                       jitter_seed=42, panel_x=-0.20):
    rng = np.random.default_rng(jitter_seed)
    for i, cohort in enumerate(COHORT_ORDER):
        g = df[df["cohort"] == cohort][col].to_numpy()
        med = np.median(g)
        q25, q75 = np.quantile(g, [0.25, 0.75])

        ax.bar(i, q75 - q25, bottom=q25, width=box_width,
               color=COLORS[cohort], alpha=box_alpha,
               edgecolor=COLORS[cohort], linewidth=1.0, zorder=2)
        ax.hlines(med, i - box_width/2, i + box_width/2,
                  colors=COLORS[cohort], linewidth=median_lw, zorder=4)
        xs = i + rng.uniform(-dot_jitter, dot_jitter, size=len(g))
        ax.scatter(xs, g, s=dot_size, color=COLORS[cohort], alpha=dot_alpha,
                   edgecolor='none', zorder=3)

    if zero_ref:
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--',
                   alpha=0.5, zorder=1)
    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels([COHORT_LABELS_2L[c] for c in COHORT_ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5, pad=4)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(panel_x, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(8.5, 5.6))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                      height_ratios=[1.3, 1.0])

# Top row: displacement fan
for j, cohort in enumerate(COHORT_ORDER):
    ax = fig.add_subplot(gs[0, j])
    draw_displacement_fan(ax, disp_df, cohort, f"({chr(ord('a')+j)})",
                          show_radius_labels=(j == 0))
    if j == 0:
        ax.set_ylabel(r"$\Delta$PC2", fontsize=8.5)

# Bottom row: summaries
ax_d = fig.add_subplot(gs[1, 0])
draw_summary_panel(ax_d, df_diff, "shock_minus_control.body_u",
                   r"$\Delta\, u^{\mathrm{body}}$",
                   (-1.6, 0.1), '(d)', "Shock magnitude (body)")

ax_e = fig.add_subplot(gs[1, 1])
draw_summary_panel(ax_e, centroid_df, "centroid_dist_3d",
                   r"$\Vert \bar g^{\mathrm{shock}}_{1:3} - \bar g^{\mathrm{ctrl}}_{1:3} \Vert$",
                   (0, 0.8), '(e)', "PC centroid distance",
                   zero_ref=False)

ax_f = fig.add_subplot(gs[1, 2])
draw_summary_panel(ax_f, spec_df, "spectral_l2",
                   r"$\Vert \lambda^{\mathrm{shock}} - \lambda^{\mathrm{ctrl}} \Vert_2$",
                   (0, 3.0), '(f)', "Metric spectral L2",
                   zero_ref=False)

plt.show()


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")