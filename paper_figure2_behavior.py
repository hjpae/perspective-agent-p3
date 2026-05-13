# -*- coding: utf-8 -*-
"""
Paper Figure 2: Behavioral comparison across cohorts (training, last 50 episodes).

Message
-------
- Full ≈ No body→g across ordinary spatial behavior.
- No conation: weaker top/TR occupancy and elevated bottom occupancy,
  including a rare severe failure run.
- Panel (c) uses a broken y-axis so the outlier is visible without flattening
  the main distribution.

Spyder usage
------------
1. Edit CONFIG section below (input path, output dir, layout).
2. Run File (F5), or run cell by cell (Ctrl+Enter in each # %% block).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from scipy import stats
from itertools import combinations


# %% ─── CONFIG ──────────────────────────────────────────────────────────────
# Edit these and re-run.

INPUT_CSV = "outputs/p3_v3_cohorts/cohort_summary_v2_last50.csv"   # 30 seeds × 3 cohorts
OUTDIR    = "./figures"
NAME      = "figure2_behavior_broken"
LAYOUT    = "1x3_wider"   # one of: "1x3_compact", "1x3_wider"
FORMATS   = ["pdf", "png"]

# Panel (c) broken-axis ranges. Values are proportions; axis labels show %.
BOTTOM_MAIN_YLIM = (0.0, 0.35)
BOTTOM_TOP_YLIM  = (0.68, 0.80)


# %% ─── STYLE (LNCS-friendly serif, no top/right spines) ────────────────────

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
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Cohort colors — locked across all paper figures
COLORS = {
    'full':         '#2c5282',  # deep blue
    'no_conative':  '#c0392b',  # red-orange (strongest ablation)
    'no_body_in_g': '#8b7355',  # warm gray-brown
}

COHORT_LABELS = {
    'full':         'Full',
    'no_conative':  'No\nconation',
    'no_body_in_g': 'No\nbody→g',
}

# Keep this order consistent across all paper figures.
COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']


# %% ─── PANEL DRAW (median + IQR box + jittered seed dots) ──────────────────

def draw_panel(ax, df, col, title, ylim, panel_id=None, ylabel=None,
               box_width=0.46, dot_jitter=0.13, dot_size=10,
               dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
               jitter_seed=42, show_xticklabels=True):
    """Median + IQR box + scatter of per-seed values, for one metric column.

    Adjust visual knobs here:
      box_width  — IQR box width (0.4 narrow, 0.55 wide)
      dot_size   — seed dot size in pts
      dot_alpha  — dot transparency (lower if dots get busy)
      box_alpha  — IQR box fill alpha
      median_lw  — median line thickness
      dot_jitter — horizontal spread of dots (smaller → more vertical)
    """
    rng = np.random.default_rng(jitter_seed)

    for i, cohort in enumerate(COHORT_ORDER):
        g = df[df["cohort"] == cohort][col].to_numpy()
        med = np.median(g)
        q25, q75 = np.quantile(g, [0.25, 0.75])

        # IQR box
        ax.bar(i, q75 - q25, bottom=q25, width=box_width,
               color=COLORS[cohort], alpha=box_alpha,
               edgecolor=COLORS[cohort], linewidth=1.0, zorder=2)

        # Median line
        ax.hlines(med, i - box_width/2, i + box_width/2,
                  colors=COLORS[cohort], linewidth=median_lw, zorder=4)

        # Per-seed dots
        xs = i + rng.uniform(-dot_jitter, dot_jitter, size=len(g))
        ax.scatter(xs, g, s=dot_size, color=COLORS[cohort], alpha=dot_alpha,
                   edgecolor='none', zorder=3)

    ax.set_xticks(range(len(COHORT_ORDER)))
    if show_xticklabels:
        ax.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if panel_id is not None:
        ax.text(-0.18, 1.06, panel_id, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='left')


def add_axis_break_marks(ax_top, ax_bottom, d=0.010, lw=0.8):
    """Add quiet diagonal break marks for a broken y-axis.

    d controls the slash size in axes coordinates. Smaller is subtler.
    """
    kwargs_top = dict(transform=ax_top.transAxes, color='k', clip_on=False,
                      linewidth=lw)
    kwargs_bottom = dict(transform=ax_bottom.transAxes, color='k', clip_on=False,
                         linewidth=lw)

    # bottom edge of top axis
    ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)

    # top edge of bottom axis
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)


# %% ─── PANEL SPECS ─────────────────────────────────────────────────────────
# Spatial behavior panels. Panel (c) is drawn with a broken y-axis.

PANELS = [
    ('top_occ',    'Top-zone dwell',    (0.0, 1.0), '(a)'),
    ('TR_occ',     'Top-right dwell',   (0.0, 1.0), '(b)'),
    ('bottom_occ', 'Bottom-zone dwell', None,       '(c)'),
]


# %% ─── LAYOUT ──────────────────────────────────────────────────────────────

def make_figure(df, layout='1x3_wider'):
    """Layouts:
      '1x3_compact' : 6.7 x 2.8 in — compact 2-column wide
      '1x3_wider'   : 7.4 x 2.9 in — recommended for LNCS figure*
    """
    if layout == '1x3_compact':
        fig = plt.figure(figsize=(6.7, 2.8))
        width_ratios = [1.0, 1.0, 1.0]
        wspace = 0.36
    elif layout == '1x3_wider':
        fig = plt.figure(figsize=(7.4, 2.9))
        width_ratios = [1.0, 1.0, 1.0]
        wspace = 0.38
    else:
        raise ValueError(f"Unknown layout: {layout}")

    outer = fig.add_gridspec(
        1, 3,
        width_ratios=width_ratios,
        wspace=wspace,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    # Broken-axis panel: top strip for outlier, bottom panel for main mass.
    c_grid = outer[0, 2].subgridspec(
        2, 1,
        height_ratios=[0.34, 1.0],
        hspace=0.06,
    )
    ax_c_top = fig.add_subplot(c_grid[0, 0])
    ax_c_bot = fig.add_subplot(c_grid[1, 0], sharex=ax_c_top)

    # (a), (b)
    draw_panel(ax_a, df, *PANELS[0], ylabel='Occupancy (%)')
    draw_panel(ax_b, df, *PANELS[1], ylabel='Occupancy (%)')

    # (c) — draw same data on both axes, with different y ranges.
    col, title, _, panel_id = PANELS[2]
    draw_panel(ax_c_top, df, col, title, BOTTOM_TOP_YLIM, panel_id=panel_id,
               ylabel=None, show_xticklabels=False)
    draw_panel(ax_c_bot, df, col, '', BOTTOM_MAIN_YLIM, panel_id=None,
               ylabel='Occupancy (%)', show_xticklabels=True)

    # Broken-axis cosmetics.
    ax_c_top.spines['bottom'].set_visible(False)
    ax_c_bot.spines['top'].set_visible(False)
    ax_c_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_c_bot.tick_params(axis='x', which='both', top=False)
    add_axis_break_marks(ax_c_top, ax_c_bot, d=0.010, lw=0.8)

    # Ticks: keep the broken panel quiet.
    ax_c_top.set_yticks([0.75])
    ax_c_bot.set_yticks([0.0, 0.10, 0.20, 0.30])

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig


# %% ─── STATISTICS ──────────────────────────────────────────────────────────
# Mann-Whitney U (two-sided, non-parametric) with rank-biserial effect size
# and BH-FDR adjustment over the three cohort pairs.

COHORT_DISPLAY = {'full': 'Full', 'no_conative': 'No conation',
                  'no_body_in_g': 'No body→g'}


def _mannwhitney_with_effect(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return dict(U=np.nan, p=np.nan, r=np.nan, n1=n1, n2=n2)
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    # rank-biserial r = 1 - 2U/(n1·n2). Sign: positive when a > b in rank.
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
    """Per-cohort summary + pairwise Mann-Whitney with BH-FDR."""
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
    print(f"    {'pair':<32s} {'U':>7s}  {'p_raw':>6s}  {'p_adj':>6s}  {'r_rb':>6s}  sig")
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

# Allows the same script to run both locally and in this sandbox.
input_path = Path(INPUT_CSV)
if not input_path.exists():
    sandbox_path = Path('/mnt/data') / INPUT_CSV
    if sandbox_path.exists():
        input_path = sandbox_path


df = pd.read_csv(input_path)
print(f"Loaded {len(df)} rows from {input_path}")
print(f"Cohorts: {df['cohort'].value_counts().to_dict()}")

fig = make_figure(df, layout=LAYOUT)
plt.show()

# Stats report — message echoed by the figure:
#   Full ≈ No body→g (n.s.), Full ≠ No conation, No body→g ≠ No conation.
print_stats_report(df, [
    ("top_occ",    "Top-zone occupancy"),
    ("TR_occ",     "Top-right occupancy"),
    ("bottom_occ", "Bottom-zone occupancy"),
], title="Figure 2 — Behavior")


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
if str(outdir).startswith('./') or not outdir.is_absolute():
    # In local Spyder this saves to ./figures. In sandbox, use /mnt/data/figures.
    sandbox_outdir = Path('/mnt/data') / str(outdir).lstrip('./')
    if Path('/mnt/data').exists():
        outdir = sandbox_outdir
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}_{LAYOUT}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")