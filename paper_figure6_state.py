# -*- coding: utf-8 -*-
"""
Paper Figure 6 v4: Same-state history probe — two-panel version.

Message
-------
For identical probe inputs, recovery-phase shock-history g and control-history
g induce separable downstream organization. This version keeps only the two
core readouts:
  (a) policy-facing state separation
  (b) perspective-induced metric-geometry separation

The previous readiness-shift panel is intentionally removed because it was a
noisier downstream echo and did not add a distinct claim beyond the state and
metric-geometry readouts.

Design
------
- Removed the schematic panel. Use a separate TikZ/vector schematic if needed.
- Removed legend; cohort identity is given directly by x-axis labels.
- Distance panels use log y-scale to show the No body→g collapse without
  hiding heavy-tailed seed-level effects.

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
from scipy import stats
from itertools import combinations


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_CSV = "outputs/p3_v3_same_state_probe/same_state_history_probe_shock_control_diff_recovery_mean.csv"
OUTDIR    = "./figures"
NAME      = "figure6_same_state_2panel"
FORMATS   = ["pdf", "png"]

# Sandbox fallback. Leave this block as-is; it does not affect project-root runs.
if not Path(INPUT_CSV).exists():
    INPUT_CSV = "/mnt/data/same_state_history_probe_shock_control_diff_recovery_mean.csv"
    OUTDIR = "/mnt/data/figures"


# %% ─── STYLE ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Liberation Serif', 'Times New Roman'],
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9.5,
    'xtick.labelsize':   8.5,
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

COHORT_LABELS_2L = {
    'full':         'Full',
    'no_conative':  'No\nconation',
    'no_body_in_g': 'No\nbody→g',
}

COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']


# %% ─── LOAD & DERIVE ───────────────────────────────────────────────────────

df_raw = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df_raw)} rows from {INPUT_CSV}")
print(f"Cohorts: {df_raw['cohort'].value_counts().to_dict()}")
print(f"Seeds per cohort: {df_raw.groupby('cohort')['seed'].nunique().to_dict()}")

eig_cols = [f"shock_minus_control.metric_eig_{i}" for i in range(20)]

# Spectrum distance: L2 norm of the shock-control eigenvalue-difference vector.
df_raw['metric_spectrum_distance'] = np.sqrt((df_raw[eig_cols] ** 2).sum(axis=1))


metric_cols = [
    's_distance_shock_control',
    'metric_spectrum_distance',
]
# Seed-level mean across identical probe specifications.
df = df_raw.groupby(['cohort', 'seed'], as_index=False)[metric_cols].mean()


# %% ─── PANEL DRAW ──────────────────────────────────────────────────────────

def draw_panel(ax, df, col, title, ylabel, ylim, panel_id,
               log_y=False, panel_x=-0.20,
               box_width=0.46, dot_jitter=0.13, dot_size=10,
               dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
               jitter_seed=42):
    """Median + IQR box + seed dots, same grammar as other figures."""
    rng = np.random.default_rng(jitter_seed)

    for i, cohort in enumerate(COHORT_ORDER):
        g = df[df['cohort'] == cohort][col].dropna().to_numpy()
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

    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels([COHORT_LABELS_2L[c] for c in COHORT_ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5, pad=4)
    if log_y:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5, which='both')
    ax.set_axisbelow(True)
    ax.text(panel_x, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── PANEL SPECS ─────────────────────────────────────────────────────────
# (col, title, ylabel, ylim, panel_id, log_y)

PANELS = [
    ('s_distance_shock_control',
     'Policy-state separation',
     'Policy-state distance',
     (0.008, 3.2), '(a)', True),
    ('metric_spectrum_distance',
     'Metric-geometry separation',
     'Metric-spectrum distance',
     (0.003, 30.0), '(b)', True),
]


# %% ─── STATISTICS ──────────────────────────────────────────────────────────
# Seed-level Mann-Whitney U (two-sided, non-parametric) with rank-biserial
# effect size and BH-FDR correction over the three cohort pairs.
# Expected pattern, mirroring Figure 5:
#   Full vs No conation       : n.s.  (both retain residue under identical
#                                       inputs once body→g routing is present)
#   Full vs No body→g         : ***
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
    # rank-biserial r = 1 − 2U/(n1·n2). Sign: positive when a > b in rank.
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
    """Per-cohort summary + pairwise Mann-Whitney with BH-FDR.

    The unit of analysis is the seed (one value per cohort × seed, obtained
    by averaging across the identical probe specifications).
    """
    print(f"\n  [{label}]  ({value_col})")
    print(f"    {'cohort':<14s} {'n':>3s}  {'median':>9s}  {'IQR':>20s}")
    for c in COHORT_ORDER:
        v = df[df['cohort'] == c][value_col].dropna().to_numpy()
        med = np.median(v)
        q25, q75 = np.quantile(v, [0.25, 0.75])
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
    print("  Unit of analysis: seed/run (probe specifications averaged within seed).")
    print("  Test: Mann-Whitney U (two-sided), with BH-FDR correction over the")
    print("  three cohort pairs, computed independently for each metric.")
    print("  Effect size: rank-biserial r = 1 − 2U/(n1·n2), range [−1, +1].")
    print("  Sign of r: positive when the first cohort tends to exceed the second.")
    for col, label in metrics:
        print_metric_stats(df, col, label)
    print()


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))
for ax, args in zip(axes, PANELS):
    draw_panel(ax, df, *args)

plt.tight_layout()
plt.subplots_adjust(top=0.86, wspace=0.38)
plt.show()

# Statistics — printed to console, not on figure.
print_stats_report(df, [
    ("s_distance_shock_control",  "Policy-state distance"),
    ("metric_spectrum_distance",  "Metric-spectrum distance"),
], title="Figure 6 — Same-state history probe")


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")