# -*- coding: utf-8 -*-
"""
Paper Figure 3 (v1, single panel): Action-conditioned interoceptive tendency
field is learned across all cohorts.

Message
-------
- All cohorts (full, no_conative, no_body_in_g) learn a positive UP–DOWN
  viability tendency in all three valence zones.
- No conation: visibly larger seed-level variance, but median still positive.
- BodyDecoder predictions (colored boxes) track the env-computed ground
  truth (black 'x'), confirming the tendency model is well-calibrated.

This figure is paired with Figure 2: the field is learned even when conation
is ablated, but it is not translated into stable behavior without conation.

Data
----
Input: cohort_counterfactual_summary.csv
  - Produced by summarize_phase3_diagnostics.py
  - Per-seed, per-zone (top/mid/bottom) means of:
      pred_trend_UP_minus_DOWN  : BodyDecoder τ̂(UP) − τ̂(DOWN)
      true_trend_UP_minus_DOWN  : env-computed τ(UP) − τ(DOWN) over k=5 steps

Spyder usage
------------
1. Edit CONFIG section below.
2. Run File (F5), or run cell by cell (Ctrl+Enter).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_CSV = "outputs/p3_v3_cohorts/cohort_counterfactual_summary.csv"
OUTDIR    = "./figures"
NAME      = "figure3_tendency_v1"
FORMATS   = ["pdf", "png"]


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
})

COLORS = {
    'full':         '#2c5282',
    'no_conative':  '#c0392b',
    'no_body_in_g': '#8b7355',
}

COHORT_LABELS = {
    'full':         'Full',
    'no_conative':  'No conation',
    'no_body_in_g': 'No body→g',
}

COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']
ZONES = ["top", "mid", "bottom"]


# %% ─── LOAD & AGGREGATE TO SEED LEVEL ──────────────────────────────────────

df_raw = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df_raw)} rows from {INPUT_CSV}")
print(f"Cohorts: {df_raw['cohort'].value_counts().to_dict()}")

# Mean across probe samples per (cohort, seed, zone)
df = df_raw.groupby(["cohort", "seed", "valence_name"], as_index=False)[
    ["pred_trend_UP_minus_DOWN", "true_trend_UP_minus_DOWN"]
].mean()


# %% ─── DRAW ────────────────────────────────────────────────────────────────

def draw_tendency_panel(ax, df, zones=ZONES,
                        bar_w=0.25, dot_size=6, dot_alpha=0.4,
                        box_alpha=0.25, median_lw=2.0,
                        gt_marker_size=24, gt_marker_lw=1.4,
                        jitter_seed=42, ylim=(-0.02, 0.18)):
    """One panel: grouped bars by zone × cohort, with GT 'x' overlay.

    Visual knobs:
      bar_w           — width of each cohort's bar within a zone group
      dot_size, dot_alpha — per-seed scatter
      box_alpha       — IQR box fill
      median_lw       — median line thickness
      gt_marker_size, gt_marker_lw — ground truth 'x' marker
      ylim            — y axis limits
    """
    rng = np.random.default_rng(jitter_seed)
    group_gap = 1.0
    x_centers = np.arange(len(zones)) * group_gap

    for j, zone in enumerate(zones):
        for k, cohort in enumerate(COHORT_ORDER):
            g = df[(df["cohort"] == cohort) & (df["valence_name"] == zone)]
            pred = g["pred_trend_UP_minus_DOWN"].to_numpy()
            true = g["true_trend_UP_minus_DOWN"].to_numpy()

            x_off = x_centers[j] + (k - 1) * bar_w

            # IQR box (predicted)
            med = np.median(pred)
            q25, q75 = np.quantile(pred, [0.25, 0.75])
            ax.bar(x_off, q75 - q25, bottom=q25, width=bar_w * 0.85,
                   color=COLORS[cohort], alpha=box_alpha,
                   edgecolor=COLORS[cohort], linewidth=0.9, zorder=2)
            # Median line
            ax.hlines(med, x_off - bar_w*0.42, x_off + bar_w*0.42,
                      colors=COLORS[cohort], linewidth=median_lw, zorder=4)
            # Per-seed dots
            xs = x_off + rng.uniform(-bar_w*0.32, bar_w*0.32, size=len(pred))
            ax.scatter(xs, pred, s=dot_size, color=COLORS[cohort],
                       alpha=dot_alpha, edgecolor='none', zorder=3)
            # Ground truth overlay
            true_med = np.median(true)
            ax.scatter(x_off, true_med, s=gt_marker_size, marker='x',
                       color='black', linewidth=gt_marker_lw, zorder=5)

    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([z.capitalize() for z in zones])
    ax.set_xlabel("Valence zone")
    ax.set_ylabel("Tendency field   $\\hat\\tau_{UP} - \\hat\\tau_{DOWN}$")
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)


def build_legend(ax, loc='upper right'):
    handles = [Patch(facecolor=COLORS[c], alpha=0.4,
                     edgecolor=COLORS[c], label=COHORT_LABELS[c])
               for c in COHORT_ORDER]
    handles.append(Line2D([0], [0], marker='x', color='black', linestyle='none',
                          markersize=6, markeredgewidth=1.4,
                          label='Ground truth (median)'))
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=7.5,
              handletextpad=0.5, borderaxespad=0.3)


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.0, 3.2))
draw_tendency_panel(ax, df)
build_legend(ax)
plt.tight_layout()
plt.show()


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")