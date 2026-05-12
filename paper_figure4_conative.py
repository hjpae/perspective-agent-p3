# -*- coding: utf-8 -*-
"""
Paper Figure 4: Conative attunement produces selective upward action-readiness
from a detached interoceptive viability field.

Message
-------
- (a) Action-wise conative target distribution q(a). Full and no_body_in_g
      sharply differentiate UP (~0.255) from DOWN (~0.128) while keeping
      LEFT/RIGHT/STAY near the uniform 0.2 baseline — the conative bridge
      attunes specifically to the bodily-relevant axis (vertical valence),
      not to the orthogonal exteroceptive axis.
- (b) Summary: q_UP − q_DOWN. Full ≈ no_body_in_g ≈ 0.125, no_conative ≡ 0
      (placeholder uniform when conative loss is disabled).

This figure shows that conation is a *selective* readiness, not a generic
preference broadcast. It is paired with Figure 2 (behavioral consequence)
and Figure 3 (the field that conation attunes to).

Data
----
Input: cohort_summary_v2_last50.csv
  - q_{UP,DOWN,LEFT,RIGHT,STAY}_mean : per-seed mean of softmaxed conative
                                        target distribution over last 50 ep
  - q_UP_minus_DOWN                  : derived differential

Spyder usage
------------
1. Edit CONFIG below.
2. Run File (F5), or run cell by cell (Ctrl+Enter).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_CSV = "outputs/p3_v3_cohorts/cohort_summary_v2_last50.csv"
OUTDIR    = "./figures"
NAME      = "figure4_conative"
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
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


# %% ─── LOAD ────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")
print(f"Cohorts: {df['cohort'].value_counts().to_dict()}")


# %% ─── PANEL (a): action-wise q distribution ───────────────────────────────

def draw_action_panel(ax, df, actions=ACTIONS,
                      bar_w=0.25, dot_size=6, dot_alpha=0.4,
                      box_alpha=0.25, median_lw=2.0,
                      jitter_seed=42, ylim=(0.10, 0.28)):
    """Grouped bars by action × cohort for q(a)."""
    rng = np.random.default_rng(jitter_seed)
    group_gap = 1.0
    x_centers = np.arange(len(actions)) * group_gap

    for j, act in enumerate(actions):
        col = f"q_{act}_mean"
        for k, cohort in enumerate(COHORT_ORDER):
            g = df[df["cohort"] == cohort][col].to_numpy()
            x_off = x_centers[j] + (k - 1) * bar_w

            med = np.median(g)
            q25, q75 = np.quantile(g, [0.25, 0.75])

            ax.bar(x_off, q75 - q25, bottom=q25, width=bar_w * 0.85,
                   color=COLORS[cohort], alpha=box_alpha,
                   edgecolor=COLORS[cohort], linewidth=0.9, zorder=2)
            ax.hlines(med, x_off - bar_w*0.42, x_off + bar_w*0.42,
                      colors=COLORS[cohort], linewidth=median_lw, zorder=4)
            xs = x_off + rng.uniform(-bar_w*0.32, bar_w*0.32, size=len(g))
            ax.scatter(xs, g, s=dot_size, color=COLORS[cohort],
                       alpha=dot_alpha, edgecolor='none', zorder=3)

    # Uniform reference line at 1/|A| = 0.2
    ax.axhline(0.2, color='gray', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
    # Place 'uniform' label inside the axes (top-right area), not after last tick
    ax.text(0.98, 0.51, 'uniform', fontsize=7, color='gray',
            transform=ax.transAxes, va='center', ha='right')

    ax.set_xticks(x_centers)
    ax.set_xticklabels(actions)
    ax.set_xlabel("Action")
    ax.set_ylabel("Conative target $q(a)$")
    ax.set_ylim(*ylim)
    ax.set_title("Action-wise conative distribution", fontsize=9.5, pad=4)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── PANEL (b): UP - DOWN differential ───────────────────────────────────

def draw_diff_panel(ax, df, metric="q_UP_minus_DOWN",
                    box_width=0.46, dot_jitter=0.13, dot_size=10,
                    dot_alpha=0.45, box_alpha=0.22, median_lw=2.4,
                    jitter_seed=42, ylim=(-0.02, 0.18)):
    rng = np.random.default_rng(jitter_seed)
    for i, cohort in enumerate(COHORT_ORDER):
        g = df[df["cohort"] == cohort][metric].to_numpy()
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

    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels([COHORT_LABELS_2L[c] for c in COHORT_ORDER], fontsize=8)
    ax.set_ylabel("$q_{UP} - q_{DOWN}$")
    ax.set_ylim(*ylim)
    ax.set_title("Upward action-readiness", fontsize=9.5, pad=4)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.20, 1.05, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


def build_legend(ax, loc='upper right'):
    handles = [Patch(facecolor=COLORS[c], alpha=0.4,
                     edgecolor=COLORS[c], label=COHORT_LABELS[c])
               for c in COHORT_ORDER]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=7.5,
              handletextpad=0.5, borderaxespad=0.3)


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                          gridspec_kw={'width_ratios': [1.6, 1.0]})
draw_action_panel(axes[0], df)
draw_diff_panel(axes[1], df)
build_legend(axes[0])

plt.tight_layout()
plt.subplots_adjust(wspace=0.30)
plt.show()


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
outdir.mkdir(parents=True, exist_ok=True)

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")