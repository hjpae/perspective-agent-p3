# -*- coding: utf-8 -*-
"""
Paper Figure 2: Behavioral comparison across cohorts (training, last 50 episodes).

Message
-------
- Full ≈ No body→g across all behavioral metrics.
- No conation: dramatically weaker top/TR occupancy, elevated bottom occupancy,
  much higher seed-level variance — the conative bridge is required to translate
  the learned interoceptive field into stable approach/avoidance behavior.

Spyder usage
------------
1. Edit CONFIG section below (input path, output dir, layout).
2. Run File (F5), or run cell by cell (Ctrl+Enter in each # %% block).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# %% ─── CONFIG ──────────────────────────────────────────────────────────────
# Edit these and re-run.

INPUT_CSV = "outputs/p3_v3_cohorts/cohort_summary_v2_last50.csv"   # 30 seeds × 3 cohorts
OUTDIR    = "./figures"
NAME      = "figure2_behavior"
LAYOUT    = "1x4_wider"   # one of: "1x4_compact", "1x4_wider", "2x2"
FORMATS   = ["pdf", "png"]


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

COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']


# %% ─── PANEL DRAW (median + IQR box + jittered seed dots) ──────────────────

def draw_panel(ax, df, col, title, ylim, panel_id,
               box_width=0.46, dot_jitter=0.13, dot_size=10,
               dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
               jitter_seed=42):
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
    ax.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.18, 1.06, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── PANEL SPECS ─────────────────────────────────────────────────────────
# (column, title, ylim, panel_id)
# To crop the bottom_occ outlier: change ylim to (0.0, 0.4).

PANELS = [
    ('top_occ',         'Top zone occupancy',     (0.0, 1.0), '(a)'),
    ('TR_occ',          'Top-right occupancy',    (0.0, 1.0), '(b)'),
    ('bottom_occ',      'Bottom zone occupancy',  (0.0, 0.8), '(c)'),
    ('body_state_mean', 'Mean body state',        (0.2, 0.8), '(d)'),
]


# %% ─── LAYOUT ──────────────────────────────────────────────────────────────

def make_figure(df, layout='1x4_wider'):
    """Layouts:
      '1x4_compact' : 7.2 x 2.7 in — tight 2-column wide
      '1x4_wider'   : 8.4 x 2.8 in — recommended for LNCS \\begin{figure*}
      '2x2'         : 4.6 x 4.4 in — single-column block
    """
    if layout == '1x4_compact':
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7))
        adjust = dict(top=0.88, wspace=0.42)
    elif layout == '1x4_wider':
        fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.8))
        adjust = dict(top=0.88, wspace=0.40)
    elif layout == '2x2':
        fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.4))
        adjust = dict(hspace=0.55, wspace=0.42)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    axes_flat = axes.flat if hasattr(axes, 'flat') else axes

    for ax, panel_args in zip(axes_flat, PANELS):
        draw_panel(ax, df, *panel_args)

    plt.tight_layout()
    plt.subplots_adjust(**adjust)
    return fig


# %% ─── RUN ─────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")
print(f"Cohorts: {df['cohort'].value_counts().to_dict()}")

fig = make_figure(df, layout=LAYOUT)
plt.show()


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}_{LAYOUT}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")