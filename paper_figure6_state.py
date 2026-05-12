# -*- coding: utf-8 -*-
"""
Paper Figure 6 (MAIN, 1x3): Same-state history-g probe — the core CEAR result.

Procedure
---------
For each cohort × seed:
  1. Run a body-shock and a control assay (frozen agent), and collect the
     recovery-phase g trajectories.
  2. Construct identical probe inputs (27 grid positions × body levels).
  3. Inject control-history g vs shock-history g into the SAME probe input,
     keeping z, body_state, action context identical.
  4. Measure the difference in (a) g representation, (b) policy state s,
     (c) Fisher-style metric M_g induced by g.

Message
-------
Same input + different bodily perturbation history →
  - No body→g: no change (g is the same regardless of history → trivial).
  - No conation: g and s shift, but metric structure does not consistently
    reshape (Δ tr(M_g) ≈ 0; cohort-level reversal).
  - Full: g and s shift AND metric tensor reshapes in a consistent direction
    (Δ tr(M_g) > 0).

This is the paper's strongest dissociation: behavioral metrics alone cannot
distinguish Full from No body→g (Figure 2), but here, on identical inputs,
the two diverge sharply — bodily perturbation history has been incorporated
into perspective geometry only when body PE is routed into g.

Data
----
Input: same_state_history_probe_shock_control_diff_recovery_mean.csv
  - 30 seeds × 3 cohorts × 27 probe specs = 2,430 rows
  - Aggregate to seed level (mean across 27 specs) before plotting.

Spyder usage
------------
1. Edit CONFIG below.
2. Run File (F5), or run cell by cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_CSV = "outputs/p3_v3_same_state_probe/same_state_history_probe_shock_control_diff_recovery_mean.csv"
OUTDIR    = "./figures"
NAME      = "figure6_same_state"
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

COHORT_LABELS_2L = {
    'full':         'Full',
    'no_conative':  'No\nconation',
    'no_body_in_g': 'No\nbody→g',
}

COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']


# %% ─── LOAD & AGGREGATE TO SEED LEVEL ──────────────────────────────────────

df_raw = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df_raw)} rows from {INPUT_CSV}")
print(f"Cohorts: {df_raw['cohort'].value_counts().to_dict()}")
print(f"Seeds per cohort: {df_raw.groupby('cohort')['seed'].nunique().to_dict()}")

metric_cols = [
    "shock_minus_control.metric_trace",
    "s_distance_shock_control",
    "g_distance_shock_control",
]
df = df_raw.groupby(["cohort", "seed"], as_index=False)[metric_cols].mean()


# %% ─── PANEL DRAW ──────────────────────────────────────────────────────────

def draw_panel(ax, df, col, title, ylabel, ylim, panel_id,
               zero_ref=True, panel_x=-0.20,
               box_width=0.46, dot_jitter=0.13, dot_size=10,
               dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
               jitter_seed=42):
    """Standard cohort comparison panel. Same visual grammar as Figs 2, 4."""
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
    ax.set_title(title, fontsize=9.5, pad=4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(panel_x, 1.05, panel_id, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


def build_legend(ax, loc='upper right'):
    handles = [Patch(facecolor=COLORS[c], alpha=0.4,
                     edgecolor=COLORS[c], label=COHORT_LABELS[c])
               for c in COHORT_ORDER]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=7.5,
              handletextpad=0.5, borderaxespad=0.3)


# %% ─── PANEL SPECS ─────────────────────────────────────────────────────────
# (col, title, ylabel, ylim, panel_id)

PANELS = [
    ("g_distance_shock_control",
     "Perspective representation",
     r"$\Vert g^{\mathrm{shock}} - g^{\mathrm{ctrl}} \Vert$",
     (0, 1.6), '(a)'),
    ("s_distance_shock_control",
     "Policy state",
     r"$\Vert s^{\mathrm{shock}} - s^{\mathrm{ctrl}} \Vert$",
     (0, 0.9), '(b)'),
    ("shock_minus_control.metric_trace",
     "Metric trace",
     r"$\Delta\, \mathrm{tr}(M_g)$",
     (-1, 3), '(c)'),
]


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
for ax, args in zip(axes, PANELS):
    draw_panel(ax, df, *args)
build_legend(axes[0])
plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.45)
plt.show()


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")