# -*- coding: utf-8 -*-
"""
Paper Figure 5: Body perturbation leaves residue in perspective geometry.

Message
-------
- Top row: per-seed displacement vectors in PCA space. Each arrow is one
  seed's recovery-phase shock-minus-control displacement. No cohort-mean
  arrow is drawn, because it visually occludes the individual trajectories.
- Bottom row: compact summaries in full perspective state space and in the
  stance-dependent metric geometry.
- Main comparison: Full and No conation retain geometry-level residue; No
  body→g collapses toward zero, showing that body prediction error must enter
  the perspective latent for bodily perturbation history to reshape geometry.

Spyder usage
------------
1. Edit CONFIG below.
2. Run File (F5), or run cell by cell (Ctrl+Enter in each # %% block).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from itertools import combinations


# %% ─── CONFIG ──────────────────────────────────────────────────────────────
# Edit these and re-run.

INPUT_PCA            = "outputs/p3_v3_assay_geometry/assay_g_pca_points.csv"
INPUT_PHASE_DIFF     = "outputs/p3_v3_assay_geometry/body_shock_control_phase_diff.csv"
INPUT_PHASE_GEOMETRY = "outputs/p3_v3_assay_geometry/assay_phase_geometry_summary.csv"
OUTDIR  = "./figures"
NAME    = "figure5_geometry_v4"
FORMATS = ["pdf", "png"]

PHASE   = "recovery"

# Sandbox fallback only; harmless in Spyder if your relative paths exist.
if not Path(INPUT_PCA).exists() and Path("/mnt/data/assay_g_pca_points.csv").exists():
    INPUT_PCA            = "/mnt/data/assay_g_pca_points.csv"
    INPUT_PHASE_DIFF     = "/mnt/data/body_shock_control_phase_diff.csv"
    INPUT_PHASE_GEOMETRY = "/mnt/data/assay_phase_geometry_summary.csv"
    OUTDIR               = "/mnt/data/figures"


# %% ─── STYLE (LNCS-friendly serif, no top/right spines) ────────────────────

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Liberation Serif', 'Times New Roman'],
    'font.size':         9,
    'axes.labelsize':    8.5,
    'axes.titlesize':    9.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
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

def per_seed_pc2_displacement(df_pca_in):
    """For each (cohort, seed), the (ΔPC1, ΔPC2) displacement vector from
    control centroid to body-shock centroid."""
    rows = []
    for (cohort, seed), g in df_pca_in.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"][["PC1", "PC2"]].mean().values
        shock = g[g["condition"] == "body_shock"][["PC1", "PC2"]].mean().values
        rows.append({"cohort": cohort, "seed": seed,
                     "dPC1": shock[0] - ctrl[0],
                     "dPC2": shock[1] - ctrl[1]})
    return pd.DataFrame(rows)


disp_df = per_seed_pc2_displacement(df_pca)

# Use phase-level distances from the analysis pipeline, plus one signed
# metric-scale summary. Keep labels short in the plot; define details here.
summary_df = df_diff[[
    "cohort", "seed",
    "shock_control_g_centroid_dist",
    "shock_control_metric_spectral_l2",
]].copy()
summary_df = summary_df.rename(columns={
    "shock_control_g_centroid_dist": "g_distance",
    "shock_control_metric_spectral_l2": "metric_spectrum_distance",
})

# Metric anisotropy distance.  This uses the condition number of the
# perspective-induced metric M_g: anisotropy(M) = log(kappa(M)).
# The plotted value is |log(kappa_shock) - log(kappa_control)|, so it
# captures shock-control changes in directional selectivity, not total scale.
def metric_anisotropy_distance(df_geom_in, eps=1e-8):
    rows = []
    for (cohort, seed), g in df_geom_in.groupby(["cohort", "seed"]):
        ctrl = g[g["condition"] == "control"]
        shock = g[g["condition"] == "body_shock"]
        if len(ctrl) == 0 or len(shock) == 0:
            continue
        cond_ctrl = float(ctrl["metric_cond"].iloc[0])
        cond_shock = float(shock["metric_cond"].iloc[0])
        anis_dist = abs(np.log(cond_shock + eps) - np.log(cond_ctrl + eps))
        rows.append({
            "cohort": cohort,
            "seed": seed,
            "metric_anisotropy_distance": anis_dist,
        })
    return pd.DataFrame(rows)

aniso_df = metric_anisotropy_distance(df_geom)
summary_df = summary_df.merge(aniso_df, on=["cohort", "seed"], how="left")


# %% ─── DRAW: DISPLACEMENT FAN PANEL ────────────────────────────────────────

def draw_displacement_fan(ax, disp_df, cohort, panel_id,
                          xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
                          ref_radii=(0.5, 1.0, 1.5),
                          show_radius_labels=False,
                          arrow_alpha=0.52, arrow_lw=1.0,
                          tip_size=16):
    """Per-seed (ΔPC1, ΔPC2) anchored at origin. No mean arrow."""
    # Reference circles
    for r in ref_radii:
        circ = plt.Circle((0, 0), r, fill=False, color='gray',
                          linewidth=0.5, linestyle=':', alpha=0.5)
        ax.add_patch(circ)
    if show_radius_labels:
        for r in ref_radii:
            ax.text(r, -0.08, f"{r}", fontsize=6.5, color='gray',
                    ha='center', va='top', alpha=0.7)

    sub = disp_df[disp_df["cohort"] == cohort]
    disps = sub[["dPC1", "dPC2"]].to_numpy()
    for dx, dy in disps:
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color=COLORS[cohort],
                                    alpha=arrow_alpha, lw=arrow_lw,
                                    shrinkA=0, shrinkB=0))
        ax.scatter(dx, dy, s=tip_size, color=COLORS[cohort], alpha=0.74,
                   edgecolor='white', linewidth=0.35, marker='X', zorder=4)

    # Quiet origin marker only.
    ax.scatter(0, 0, s=22, color='white', edgecolor='black',
               linewidth=0.8, marker='o', zorder=5)

    ax.set_title(COHORT_TITLES[cohort], fontsize=10)
    ax.set_xlabel(r"$\Delta$PC1", fontsize=8.5)
    ax.set_aspect('equal')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axhline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax.text(-0.10, 1.06, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── DRAW: SUMMARY PANEL ─────────────────────────────────────────────────

def draw_summary_panel(ax, df, col, ylabel, ylim, panel_id, title,
                       zero_ref=True,
                       box_width=0.46, dot_jitter=0.13, dot_size=10,
                       dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
                       jitter_seed=42, panel_x=-0.15, ylabel_pad=1):
    """Median + IQR box + scatter of per-seed values."""
    rng = np.random.default_rng(jitter_seed)
    for i, cohort in enumerate(COHORT_ORDER):
        g = df[df["cohort"] == cohort][col].dropna().to_numpy()
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
    ax.set_ylabel(ylabel, labelpad=ylabel_pad)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5, pad=4)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(panel_x, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


def build_cohort_legend(fig, loc='upper center'):
    handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=COLORS[c], markeredgecolor='white',
               markeredgewidth=0.7, markersize=6,
               label=COHORT_TITLES[c])
        for c in COHORT_ORDER
    ]
    return fig.legend(handles=handles, loc=loc, ncol=len(COHORT_ORDER),
                      frameon=False, bbox_to_anchor=(0.5, 0.992),
                      handletextpad=0.45, columnspacing=1.25,
                      borderaxespad=0.0)


# %% ─── STATISTICS ──────────────────────────────────────────────────────────
# Mann-Whitney U (two-sided, non-parametric) with rank-biserial effect size
# and BH-FDR adjustment over the three cohort pairs, computed per metric.
# Expected pattern for Figure 5:
#   Full vs No conation       : n.s.  (both retain residue)
#   Full vs No body→g         : ***   (body→g routing is required)
#   No conation vs No body→g  : ***

COHORT_DISPLAY = {'full': 'Full', 'no_conative': 'No conation',
                  'no_body_in_g': 'No body→g'}


def _mannwhitney_with_effect(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return dict(U=np.nan, p=np.nan, r=np.nan, n1=n1, n2=n2)
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 1.0 - 2.0 * U / (n1 * n2)
    return dict(U=U, p=p, r=r, n1=n1, n2=n2)


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.argsort(order) + 1
    p_sorted = p[order]
    p_adj_sorted = np.minimum.accumulate(
        (p_sorted * n / np.arange(1, n + 1))[::-1])[::-1]
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)
    return p_adj_sorted[ranks - 1]


def _stars(p):
    if np.isnan(p):  return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "n.s."


def _fmt_p(p):
    if np.isnan(p): return "  n/a"
    if p < 0.001:   return "<.001"
    return f"{p:.3f}"


def print_metric_stats(df, value_col, label):
    print(f"\n  [{label}]  ({value_col})")
    print(f"    {'cohort':<14s} {'n':>3s}  {'median':>9s}  {'IQR':>20s}")
    for c in COHORT_ORDER:
        v = df[df['cohort'] == c][value_col].dropna().to_numpy()
        med = np.median(v); q25, q75 = np.quantile(v, [0.25, 0.75])
        iqr_str = f"[{q25:.3f}, {q75:.3f}]"
        print(f"    {COHORT_DISPLAY[c]:<14s} {len(v):>3d}  {med:>9.3f}  {iqr_str:>20s}")
    pairs = list(combinations(COHORT_ORDER, 2))
    results = []
    for a, b in pairs:
        ga = df[df['cohort'] == a][value_col].to_numpy()
        gb = df[df['cohort'] == b][value_col].to_numpy()
        res = _mannwhitney_with_effect(ga, gb)
        res['pair'] = (a, b)
        results.append(res)
    p_adj = _bh_fdr([r['p'] for r in results])
    print(f"    {'pair':<32s} {'U':>7s}  {'p_raw':>6s}  {'p_adj':>6s}  "
          f"{'r_rb':>6s}  sig")
    for r, padj in zip(results, p_adj):
        a, b = r['pair']
        pair_str = f"{COHORT_DISPLAY[a]} vs {COHORT_DISPLAY[b]}"
        sig = _stars(padj)
        print(f"    {pair_str:<32s} {r['U']:>7.1f}  "
              f"{_fmt_p(r['p']):>6s}  {_fmt_p(padj):>6s}  "
              f"{r['r']:>+6.2f}  {sig}")


def print_stats_report(df, metrics, title="Statistics report"):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("  Test: Mann-Whitney U (two-sided), with BH-FDR correction.")
    print("  Effect size: rank-biserial r = 1 − 2U/(n1·n2), range [−1, +1].")
    print("  Sign of r: positive when the first cohort tends to exceed the second.")
    for col, label in metrics:
        print_metric_stats(df, col, label)
    print()


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(8.8, 5.45))
gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.34,
                      height_ratios=[1.28, 1.0])

# Top row: displacement fan
for j, cohort in enumerate(COHORT_ORDER):
    ax = fig.add_subplot(gs[0, j])
    draw_displacement_fan(ax, disp_df, cohort, f"({chr(ord('a')+j)})",
                          show_radius_labels=(j == 0))
    if j == 0:
        ax.set_ylabel(r"$\Delta$PC2", fontsize=8.5, labelpad=1)

# Bottom row: geometry summaries
ax_d = fig.add_subplot(gs[1, 0])
draw_summary_panel(ax_d, summary_df, "g_distance",
                   "g distance", (0, 2.6), '(d)',
                   "Perspective displacement", zero_ref=False,
                   panel_x=-0.13)

ax_e = fig.add_subplot(gs[1, 1])
draw_summary_panel(ax_e, summary_df, "metric_spectrum_distance",
                   "spectrum distance", (0, 3.0), '(e)',
                   "Metric spectrum distance", zero_ref=False,
                   panel_x=-0.13)

ax_f = fig.add_subplot(gs[1, 2])
draw_summary_panel(ax_f, summary_df, "metric_anisotropy_distance",
                   "anisotropy distance", (0, 2.2), '(f)',
                   "Metric anisotropy", zero_ref=False,
                   panel_x=-0.13)

build_cohort_legend(fig)

plt.tight_layout()
plt.subplots_adjust(top=0.88, left=0.065, right=0.99)
plt.show()

# Statistics — printed to console, not on figure.
print_stats_report(summary_df, [
    ("g_distance",                 "Perspective g distance (3D centroid)"),
    ("metric_spectrum_distance",   "Metric spectrum distance (eigenvalue L2)"),
    ("metric_anisotropy_distance", "Metric anisotropy distance"),
], title="Figure 5 — Body perturbation geometry")


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")