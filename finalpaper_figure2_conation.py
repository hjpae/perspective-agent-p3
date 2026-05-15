# -*- coding: utf-8 -*-
"""
Paper Figure 2: Conation links learned bodily tendency to behavior.

Combined 2x2 figure from the previous behavior and field/readiness figures.

Panels
------
(a) Top-right dwell (%) — behavior requiring both bodily-favorable and
    predictable regions.
(b) Bottom-zone dwell (%) — bodily unfavorable visits; broken y-axis keeps the
    rare severe No conation failure visible without flattening the main mass.
(c) BodyDecoder tendency calibration — predicted UP−DOWN tendency tracks the
    environment-computed ground truth across cohorts.
(d) Conative target q(a) — Full and No body→g form selective UP/DOWN readiness;
    No conation remains at the uniform baseline.

Message
-------
The counterfactual bodily tendency field is learned across all cohorts, but it
becomes behaviorally consequential only when conative alignment is active.

Spyder usage
------------
1. Edit CONFIG section below.
2. Run File (F5), or run cell by cell (Ctrl+Enter in each # %% block).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from itertools import combinations


# %% ─── CONFIG ──────────────────────────────────────────────────────────────
# Edit these and re-run.

INPUT_BEHAVIOR = "outputs/p3_v3_cohorts/cohort_summary_v2_last50.csv"
INPUT_TENDENCY = "outputs/p3_v3_cohorts/cohort_counterfactual_summary.csv"
OUTDIR         = "./figures"
NAME           = "figure2_conation_behavior_field_readiness"
FORMATS        = ["pdf", "png"]

# Broken-axis ranges for panel (b). Values are proportions; labels show %.
BOTTOM_MAIN_YLIM = (0.0, 0.35)
BOTTOM_TOP_YLIM  = (0.68, 0.80)

# Layout: suitable for LNCS two-column figure*.
FIGSIZE = (7.45, 5.55)


# %% ─── STYLE ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Liberation Serif', 'Times New Roman'],
    'font.size':         8.6,
    'axes.labelsize':    8.6,
    'axes.titlesize':    9.2,
    'xtick.labelsize':   7.8,
    'ytick.labelsize':   7.8,
    'legend.fontsize':   7.2,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Cohort colors — locked across all paper figures.
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

# Keep this order consistent across all paper figures.
COHORT_ORDER = ['full', 'no_conative', 'no_body_in_g']
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

COHORT_DISPLAY = {
    'full': 'Full',
    'no_conative': 'No conation',
    'no_body_in_g': 'No body→g',
}


# %% ─── PATH HELPERS / LOAD ─────────────────────────────────────────────────

def resolve_path(path_str):
    """Resolve path for project-root, current Spyder working dir, or sandbox."""
    p = Path(path_str)
    if p.exists():
        return p
    p2 = Path.cwd() / path_str
    if p2.exists():
        return p2
    p3 = Path('/mnt/data') / path_str
    if p3.exists():
        return p3
    p4 = Path('/mnt/data') / Path(path_str).name
    if p4.exists():
        return p4
    raise FileNotFoundError(f"Could not find {path_str}")


path_b = resolve_path(INPUT_BEHAVIOR)
path_t = resolve_path(INPUT_TENDENCY)

df_b = pd.read_csv(path_b)
df_t = pd.read_csv(path_t)

print(f"Loaded behavior : {len(df_b)} rows from {path_b}")
print(f"Loaded tendency : {len(df_t)} rows from {path_t}")
print(f"Behavior cohorts: {df_b['cohort'].value_counts().to_dict()}")
print(f"Tendency cohorts: {df_t['cohort'].value_counts().to_dict()}")


# %% ─── PANEL HELPERS ───────────────────────────────────────────────────────

def draw_behavior_panel(ax, df, col, title, ylim, panel_id=None, ylabel=None,
                        box_width=0.46, dot_jitter=0.13, dot_size=10,
                        dot_alpha=0.45, box_alpha=0.22, median_lw=2.2,
                        jitter_seed=42, show_xticklabels=True):
    """Median + IQR box + jittered seed dots for one behavior metric."""
    rng = np.random.default_rng(jitter_seed)

    for i, cohort in enumerate(COHORT_ORDER):
        vals = df[df['cohort'] == cohort][col].dropna().to_numpy(dtype=float)
        med = np.median(vals)
        q25, q75 = np.quantile(vals, [0.25, 0.75])

        ax.bar(i, q75 - q25, bottom=q25, width=box_width,
               color=COLORS[cohort], alpha=box_alpha,
               edgecolor=COLORS[cohort], linewidth=1.0, zorder=2)
        ax.hlines(med, i - box_width/2, i + box_width/2,
                  colors=COLORS[cohort], linewidth=median_lw, zorder=4)
        xs = i + rng.uniform(-dot_jitter, dot_jitter, size=len(vals))
        ax.scatter(xs, vals, s=dot_size, color=COLORS[cohort], alpha=dot_alpha,
                   edgecolor='none', zorder=3)

    ax.set_xticks(range(len(COHORT_ORDER)))
    if show_xticklabels:
        ax.set_xticklabels([COHORT_LABELS_2L[c] for c in COHORT_ORDER])
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    ax.set_title(title, pad=4)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if panel_id is not None:
        ax.text(-0.16, 1.05, panel_id, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='right')


def add_axis_break_marks(ax_top, ax_bottom, d=0.010, lw=0.8):
    """Add diagonal break marks for a broken y-axis."""
    kwargs_top = dict(transform=ax_top.transAxes, color='k', clip_on=False,
                      linewidth=lw)
    kwargs_bottom = dict(transform=ax_bottom.transAxes, color='k', clip_on=False,
                         linewidth=lw)
    ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)


def _pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def make_true_tendency_bins(df, x_col, n_bins=6):
    """Global quantile bins over the true tendency axis."""
    d = df.copy()
    d['true_bin'] = pd.qcut(d[x_col], q=n_bins, duplicates='drop')
    return d


def _binned_summary(g, x_col, y_col):
    rows = []
    for _, b in g.groupby('true_bin', observed=True):
        x = b[x_col].to_numpy(dtype=float)
        y = b[y_col].to_numpy(dtype=float)
        if len(b) == 0:
            continue
        rows.append({
            'x_med': np.median(x),
            'x_q25': np.quantile(x, 0.25),
            'x_q75': np.quantile(x, 0.75),
            'y_med': np.median(y),
            'y_q25': np.quantile(y, 0.25),
            'y_q75': np.quantile(y, 0.75),
            'n': len(b),
        })
    return pd.DataFrame(rows).sort_values('x_med')




def _fig_bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values for compact in-panel stats."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj_sorted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adj_sorted[i] = running
    adj = np.empty(n, dtype=float)
    adj[order] = np.clip(adj_sorted, 0, 1)
    return adj


def _fig_sig_marker(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _seed_spearman_summary(df, x_col, y_col):
    """Return compact median seed-level Spearman rho and BH-adjusted p."""
    rows = []
    raw_p = []
    for cohort in COHORT_ORDER:
        rhos = []
        g_cohort = df[df['cohort'] == cohort]
        if 'seed' not in g_cohort.columns:
            continue
        for _, g_seed in g_cohort.groupby('seed'):
            x = g_seed[x_col].to_numpy(dtype=float)
            y = g_seed[y_col].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if len(x) >= 3 and np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
                rho, _ = stats.spearmanr(x, y)
                if np.isfinite(rho):
                    rhos.append(float(rho))
        rhos = np.asarray(rhos, dtype=float)
        if len(rhos) == 0:
            med, p = np.nan, np.nan
        elif np.allclose(rhos, 0.0):
            med, p = float(np.median(rhos)), 1.0
        else:
            med = float(np.median(rhos))
            try:
                p = float(stats.wilcoxon(rhos, alternative='two-sided', zero_method='wilcox').pvalue)
            except ValueError:
                p = np.nan
        rows.append({'cohort': cohort, 'rho_med': med, 'p_raw': p})
        raw_p.append(p if np.isfinite(p) else 1.0)
    p_adj = _fig_bh_fdr(raw_p)
    for row, pa in zip(rows, p_adj):
        row['p_adj'] = float(pa)
    return rows

def draw_tendency_calibration_panel(
    ax,
    df,
    x_col='true_trend_UP_minus_DOWN',
    y_col='pred_trend_UP_minus_DOWN',
    n_bins=6,
    xlim=(0.02, 0.15),
    ylim=(0.00, 0.155),
):
    """Binned predicted-vs-ground-truth tendency calibration panel."""
    d = make_true_tendency_bins(df, x_col=x_col, n_bins=n_bins)
    stat_rows = _seed_spearman_summary(df, x_col=x_col, y_col=y_col)

    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=0.9,
            alpha=0.75, zorder=1)

    for cohort in COHORT_ORDER:
        g = d[d['cohort'] == cohort]
        summ = _binned_summary(g, x_col=x_col, y_col=y_col)
        if len(summ) == 0:
            continue
        x = summ['x_med'].to_numpy()
        y = summ['y_med'].to_numpy()
        yerr = np.vstack([
            y - summ['y_q25'].to_numpy(),
            summ['y_q75'].to_numpy() - y,
        ])
        ax.errorbar(
            x, y, yerr=yerr,
            color=COLORS[cohort], marker='o', markersize=4.0,
            linewidth=1.25, elinewidth=0.85, capsize=2.2,
            zorder=3,
        )

    stat_lines = []
    for row in stat_rows:
        cohort = row['cohort']
        rho = row['rho_med']
        pa = row['p_adj']
        if np.isfinite(rho):
            stat_lines.append(
                f"{COHORT_LABELS[cohort]} "
                rf"$\rho_s$={rho:.2f}{_fig_sig_marker(pa)}"
            )
    if stat_lines:
        ax.text(
            0.05, 0.96,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=5.9, linespacing=1.12,
            bbox=dict(
                boxstyle='round,pad=0.16',
                facecolor='white',
                edgecolor='none',
                alpha=0.74,
            ),
            zorder=10,
        )

    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.45, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.45, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Ground-truth UP−DOWN tendency')
    ax.set_ylabel('Predicted UP−DOWN tendency')
    ax.set_title('Learned bodily tendency field', pad=4)
    ax.grid(alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    # Keep the calibration reference legend. Cohort identity is handled by
    # the global legend, and cohort-level calibration statistics are shown
    # in the compact inset above.
    yx_handle = Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        linewidth=0.9,
        label='y=x',
    )
    ax.legend(
        handles=[yx_handle],
        loc='lower right',
        frameon=False,
        fontsize=6.7,
        handlelength=2.4,
        handletextpad=0.35,
        borderaxespad=0.25,
    )

    ax.text(-0.13, 1.05, '(c)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


def draw_action_panel(ax, df_q, actions=ACTIONS,
                      bar_w=0.25, dot_size=5.6, dot_alpha=0.38,
                      box_alpha=0.25, median_lw=1.9,
                      jitter_seed=42, ylim=(0.10, 0.28)):
    """Grouped IQR boxes by action × cohort for conative q(a)."""
    rng = np.random.default_rng(jitter_seed)
    x_centers = np.arange(len(actions))

    for j, act in enumerate(actions):
        col = f'q_{act}_mean'
        for k, cohort in enumerate(COHORT_ORDER):
            vals = df_q[df_q['cohort'] == cohort][col].dropna().to_numpy(dtype=float)
            x_off = x_centers[j] + (k - 1) * bar_w
            med = np.median(vals)
            q25, q75 = np.quantile(vals, [0.25, 0.75])

            ax.bar(x_off, q75 - q25, bottom=q25, width=bar_w * 0.85,
                   color=COLORS[cohort], alpha=box_alpha,
                   edgecolor=COLORS[cohort], linewidth=0.9, zorder=2)
            ax.hlines(med, x_off - bar_w*0.42, x_off + bar_w*0.42,
                      colors=COLORS[cohort], linewidth=median_lw, zorder=4)
            xs = x_off + rng.uniform(-bar_w*0.32, bar_w*0.32, size=len(vals))
            ax.scatter(xs, vals, s=dot_size, color=COLORS[cohort],
                       alpha=dot_alpha, edgecolor='none', zorder=3)

    ax.axhline(0.2, color='gray', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
    ax.text(0.98, 0.51, 'uniform', fontsize=7, color='gray',
            transform=ax.transAxes, va='center', ha='right')
    ax.set_xticks(x_centers)
    ax.set_xticklabels(actions)
    ax.set_xlabel('Action')
    ax.set_ylabel('Conative target $q(a)$')
    ax.set_ylim(*ylim)
    ax.set_title('Selective conative readiness', pad=4)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.05, '(d)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


def add_compact_legend(fig, y=0.985):
    handles = [Patch(facecolor=COLORS[c], alpha=0.35,
                     edgecolor=COLORS[c], label=COHORT_LABELS[c])
               for c in COHORT_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, y), frameon=False,
               handletextpad=0.45, columnspacing=1.2, borderaxespad=0.0)


# %% ─── MAKE FIGURE ─────────────────────────────────────────────────────────

def make_figure(df_b, df_t):
    """2x2 combined figure.

    Top row: behavior. Bottom row: field calibration and readiness.
    Panel (b) uses an internal broken y-axis.
    """
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0, 1.28],
        hspace=0.46,
        wspace=0.36,
    )

    ax_a = fig.add_subplot(outer[0, 0])

    # Broken-axis panel (b), inside top-right cell.
    b_grid = outer[0, 1].subgridspec(
        2, 1, height_ratios=[0.34, 1.0], hspace=0.055,
    )
    ax_b_top = fig.add_subplot(b_grid[0, 0])
    ax_b_bot = fig.add_subplot(b_grid[1, 0], sharex=ax_b_top)

    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    # (a) Top-right dwell, retained from Fig. 2(b)
    draw_behavior_panel(
        ax_a, df_b, 'TR_occ', 'Top-right dwell', (0.0, 1.0),
        panel_id='(a)', ylabel='Occupancy (%)', jitter_seed=43,
    )

    # (b) Bottom-zone dwell, retained from Fig. 2(c)
    draw_behavior_panel(
        ax_b_top, df_b, 'bottom_occ', 'Bottom-zone dwell', BOTTOM_TOP_YLIM,
        panel_id='(b)', ylabel=None, show_xticklabels=False, jitter_seed=44,
    )
    draw_behavior_panel(
        ax_b_bot, df_b, 'bottom_occ', '', BOTTOM_MAIN_YLIM,
        panel_id=None, ylabel='Occupancy (%)', show_xticklabels=True,
        jitter_seed=44,
    )
    ax_b_top.spines['bottom'].set_visible(False)
    ax_b_bot.spines['top'].set_visible(False)
    ax_b_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_b_bot.tick_params(axis='x', which='both', top=False)
    add_axis_break_marks(ax_b_top, ax_b_bot, d=0.010, lw=0.8)
    ax_b_top.set_yticks([0.75])
    ax_b_bot.set_yticks([0.0, 0.10, 0.20, 0.30])

    # (c), (d) from Fig. 3
    draw_tendency_calibration_panel(ax_c, df_t)
    draw_action_panel(ax_d, df_b)

    # Panel (c) carries compact seed-level calibration statistics in-panel.
    add_compact_legend(fig, y=0.994)
    plt.subplots_adjust(top=0.925, bottom=0.095, left=0.075, right=0.985)
    #plt.subplots_adjust(top=0.955, bottom=0.095, left=0.075, right=0.985)
    return fig


fig = make_figure(df_b, df_t)
plt.show()


# %% ─── STATISTICS ──────────────────────────────────────────────────────────
# Behavior panels: Mann-Whitney U with rank-biserial effect size and BH-FDR.
# Tendency calibration: seed-level Spearman/Pearson; Wilcoxon vs zero.
# Readiness: action-wise Mann-Whitney U with BH-FDR.

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
    if np.isnan(p):  return ''
    if p < 0.001:    return '***'
    if p < 0.01:     return '**'
    if p < 0.05:     return '*'
    return 'n.s.'


def _fmt_p(p):
    if np.isnan(p): return '  n/a'
    if p < 0.001:   return '<.001'
    return f'{p:.3f}'


def print_metric_stats(df, value_col, label):
    """Per-cohort summary + pairwise Mann-Whitney with BH-FDR."""
    print(f"\n  [{label}]  ({value_col})")
    print(f"    {'cohort':<14s} {'n':>3s}  {'median':>9s}  {'IQR':>20s}")
    for c in COHORT_ORDER:
        v = df[df['cohort'] == c][value_col].dropna().to_numpy(dtype=float)
        med = np.median(v)
        q25, q75 = np.quantile(v, [0.25, 0.75])
        iqr_str = f'[{q25:.3f}, {q75:.3f}]'
        print(f"    {COHORT_DISPLAY[c]:<14s} {len(v):>3d}  {med:>9.3f}  {iqr_str:>20s}")

    pairs = list(combinations(COHORT_ORDER, 2))
    results = []
    for a, b in pairs:
        ga = df[df['cohort'] == a][value_col].to_numpy(dtype=float)
        gb = df[df['cohort'] == b][value_col].to_numpy(dtype=float)
        res = _mannwhitney_with_effect(ga, gb)
        res['pair'] = (a, b)
        results.append(res)
    p_adj = _bh_fdr([r['p'] for r in results])
    print(f"    {'pair':<32s} {'U':>7s}  {'p_raw':>6s}  {'p_adj':>6s}  {'r_rb':>6s}  sig")
    for r, padj in zip(results, p_adj):
        a, b = r['pair']
        pair_str = f"{COHORT_DISPLAY[a]} vs {COHORT_DISPLAY[b]}"
        print(f"    {pair_str:<32s} {r['U']:>7.1f}  "
              f"{_fmt_p(r['p']):>6s}  {_fmt_p(padj):>6s}  "
              f"{r['r']:>+6.2f}  {_stars(padj)}")


def print_behavior_stats(df_b):
    print('\n' + '=' * 70)
    print('Figure 2a,b — Behavior panels')
    print('=' * 70)
    print('  Test: Mann-Whitney U (two-sided), with BH-FDR correction.')
    print('  Effect size: rank-biserial r = 1 − 2U/(n1·n2), range [−1, +1].')
    print_metric_stats(df_b, 'TR_occ', 'Top-right occupancy')
    print_metric_stats(df_b, 'bottom_occ', 'Bottom-zone occupancy')
    print()


def _safe_corrs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, len(x)
    rho, _ = stats.spearmanr(x, y)
    r, _ = stats.pearsonr(x, y)
    return float(rho), float(r), len(x)


def _wilcoxon_vs_zero(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative='two-sided',
                                    zero_method='wilcox').pvalue)
    except ValueError:
        return np.nan


def compute_seed_calibration_stats(df, x_col='true_trend_UP_minus_DOWN',
                                   y_col='pred_trend_UP_minus_DOWN'):
    """One calibration coefficient per independent seed."""
    if 'seed' not in df.columns:
        raise ValueError("Tendency statistics require a 'seed' column.")

    rows = []
    for (cohort, seed), g in df.groupby(['cohort', 'seed']):
        rho, r, n_probe = _safe_corrs(g[x_col].to_numpy(), g[y_col].to_numpy())
        rows.append({
            'cohort': cohort,
            'seed': int(seed),
            'n_probe_rows': int(n_probe),
            'spearman_rho': rho,
            'pearson_r': r,
        })
    seed_corr = pd.DataFrame(rows)

    stat_rows = []
    raw_p = []
    for cohort in COHORT_ORDER:
        g = seed_corr[seed_corr['cohort'] == cohort]
        rho = g['spearman_rho'].dropna().to_numpy(dtype=float)
        pear = g['pearson_r'].dropna().to_numpy(dtype=float)
        p_rho = _wilcoxon_vs_zero(rho)
        raw_p.append(p_rho)
        stat_rows.append({
            'cohort': cohort,
            'n_seeds': int(len(rho)),
            'spearman_median': float(np.median(rho)) if len(rho) else np.nan,
            'spearman_q25': float(np.quantile(rho, 0.25)) if len(rho) else np.nan,
            'spearman_q75': float(np.quantile(rho, 0.75)) if len(rho) else np.nan,
            'pearson_median': float(np.median(pear)) if len(pear) else np.nan,
            'pearson_q25': float(np.quantile(pear, 0.25)) if len(pear) else np.nan,
            'pearson_q75': float(np.quantile(pear, 0.75)) if len(pear) else np.nan,
            'wilcoxon_p_spearman_vs_zero_raw': p_rho,
        })
    cohort_stats = pd.DataFrame(stat_rows)
    cohort_stats['wilcoxon_p_spearman_vs_zero_bh'] = _bh_fdr(raw_p)
    return seed_corr, cohort_stats


def print_calibration_stats(df_t):
    seed_corr, cohort_stats = compute_seed_calibration_stats(df_t)
    print('\n' + '=' * 70)
    print('Figure 2c — Tendency calibration (seed-level statistics)')
    print('=' * 70)
    print('  Probe/zone rows estimate one calibration coefficient per seed.')
    print('  Test: Wilcoxon signed-rank test of seed-level Spearman ρ against zero;')
    print('  p-values BH-FDR adjusted across the three cohorts.')
    print(f"    {'cohort':<14s} {'n_seed':>6s}  {'Spearman ρ median [IQR]':>27s}  "
          f"{'Pearson r median [IQR]':>26s}  {'p_adj':>7s}  sig")
    for _, row in cohort_stats.iterrows():
        c = row['cohort']
        rho_iqr = (f"{row['spearman_median']:+.3f} "
                   f"[{row['spearman_q25']:+.3f}, {row['spearman_q75']:+.3f}]")
        pear_iqr = (f"{row['pearson_median']:+.3f} "
                    f"[{row['pearson_q25']:+.3f}, {row['pearson_q75']:+.3f}]")
        padj = row['wilcoxon_p_spearman_vs_zero_bh']
        print(f"    {COHORT_DISPLAY[c]:<14s} {int(row['n_seeds']):>6d}  "
              f"{rho_iqr:>27s}  {pear_iqr:>26s}  {_fmt_p(padj):>7s}  {_stars(padj)}")
    print()
    return seed_corr, cohort_stats


def print_q_stats(df_q):
    print('=' * 70)
    print('Figure 2d — Conative target distribution: pairwise tests')
    print('=' * 70)
    print('  Test: Mann-Whitney U (two-sided), with BH-FDR correction per action.')
    print('  Effect size: rank-biserial r = 1 − 2U/(n1·n2), range [−1, +1].')
    for act in ACTIONS:
        col = f'q_{act}_mean'
        print(f"\n  [q({act})]  ({col})")
        print(f"    {'cohort':<14s} {'n':>3s}  {'median':>9s}  {'IQR':>20s}")
        for c in COHORT_ORDER:
            v = df_q[df_q['cohort'] == c][col].dropna().to_numpy(dtype=float)
            med = np.median(v)
            q25, q75 = np.quantile(v, [0.25, 0.75])
            iqr_str = f'[{q25:.3f}, {q75:.3f}]'
            print(f"    {COHORT_DISPLAY[c]:<14s} {len(v):>3d}  {med:>9.3f}  {iqr_str:>20s}")

        pairs = list(combinations(COHORT_ORDER, 2))
        results = []
        for a, b in pairs:
            ga = df_q[df_q['cohort'] == a][col].to_numpy(dtype=float)
            gb = df_q[df_q['cohort'] == b][col].to_numpy(dtype=float)
            res = _mannwhitney_with_effect(ga, gb)
            res['pair'] = (a, b)
            results.append(res)
        p_adj = _bh_fdr([r['p'] for r in results])
        print(f"    {'pair':<32s} {'U':>7s}  {'p_raw':>6s}  {'p_adj':>6s}  "
              f"{'r_rb':>6s}  sig")
        for r, padj in zip(results, p_adj):
            a, b = r['pair']
            pair_str = f"{COHORT_DISPLAY[a]} vs {COHORT_DISPLAY[b]}"
            print(f"    {pair_str:<32s} {r['U']:>7.1f}  "
                  f"{_fmt_p(r['p']):>6s}  {_fmt_p(padj):>6s}  "
                  f"{r['r']:>+6.2f}  {_stars(padj)}")
    print()


print_behavior_stats(df_b)
seed_calib_stats, cohort_calib_stats = print_calibration_stats(df_t)
print_q_stats(df_b)


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
if str(outdir).startswith('./') or not outdir.is_absolute():
    # Local Spyder: ./figures. Sandbox: /mnt/data/figures.
    if Path('/mnt/data').exists():
        outdir = Path('/mnt/data') / str(outdir).lstrip('./')
outdir.mkdir(parents=True, exist_ok=True)

seed_csv = outdir / f'{NAME}_panel_c_seed_calibration.csv'
cohort_csv = outdir / f'{NAME}_panel_c_cohort_calibration.csv'
seed_calib_stats.to_csv(seed_csv, index=False)
cohort_calib_stats.to_csv(cohort_csv, index=False)
print(f'  saved: {seed_csv}')
print(f'  saved: {cohort_csv}')

for fmt in FORMATS:
    out = outdir / f'{NAME}.{fmt}'
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f'  saved: {out}')
