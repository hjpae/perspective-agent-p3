# -*- coding: utf-8 -*-
"""
Paper Figure 3 (combined v4): Binned field calibration and conative readiness.

Message
-------
(a) Field learning calibration. For each counterfactual probe sample, the
    BodyDecoder-predicted UP-DOWN bodily tendency is plotted against the
    environment-computed ground-truth UP-DOWN tendency. Alignment to the
    y=x line indicates that the interoceptive tendency field is learned
    across cohorts, including No conation.

(b) Conative target distribution q(a). Full and No body->g differentiate
    UP from DOWN while keeping LEFT/RIGHT/STAY near the uniform baseline.
    No conation remains uniform because the conative loss is disabled.

Together with Figure 2, this establishes the dissociation:
    the counterfactual interoceptive field is learned even without conation,
    but it becomes selective action-readiness only when conation is present.

Data
----
Inputs:
  cohort_counterfactual_summary.csv
  cohort_summary_v2_last50.csv

Spyder usage
------------
1. Edit CONFIG below if needed.
2. Run File (F5), or run cell by cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from itertools import combinations


# %% ─── CONFIG ──────────────────────────────────────────────────────────────

INPUT_TENDENCY = "outputs/p3_v3_cohorts/cohort_counterfactual_summary.csv"
INPUT_Q        = "outputs/p3_v3_cohorts/cohort_summary_v2_last50.csv"
OUTDIR         = "./figures"
NAME           = "figure3_field_and_readiness_v4"
FORMATS        = ["pdf", "png"]


# %% ─── STYLE ───────────────────────────────────────────────────────────────
# Same style block as the existing paper figure scripts.

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
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


# %% ─── LOAD ────────────────────────────────────────────────────────────────

def resolve_path(path_str):
    """Resolve path for both project-root and /mnt/data execution."""
    p = Path(path_str)
    if p.exists():
        return p
    p2 = Path("/mnt/data") / path_str
    if p2.exists():
        return p2
    p3 = Path("/mnt/data") / Path(path_str).name
    if p3.exists():
        return p3
    raise FileNotFoundError(f"Could not find {path_str}")

path_t = resolve_path(INPUT_TENDENCY)
path_q = resolve_path(INPUT_Q)

df_t_raw = pd.read_csv(path_t)
df_q = pd.read_csv(path_q)

print(f"Loaded tendency: {len(df_t_raw)} rows from {path_t}")
print(f"Loaded q-dist   : {len(df_q)} rows from {path_q}")
print(f"Tendency cohorts: {df_t_raw['cohort'].value_counts().to_dict()}")
print(f"q cohorts       : {df_q['cohort'].value_counts().to_dict()}")


# %% ─── PANEL (a): BINNED PREDICTED VS GROUND-TRUTH CALIBRATION ────────

def _pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def make_true_tendency_bins(df, x_col, n_bins=6):
    """Global quantile bins over the true tendency axis.

    Binning removes the distracting probe-level scatter while preserving the
    actual calibration question: do predicted tendencies increase with the
    environment-computed ground-truth tendency?
    """
    d = df.copy()
    d["true_bin"] = pd.qcut(d[x_col], q=n_bins, duplicates="drop")
    return d


def _binned_summary(g, x_col, y_col):
    rows = []
    for bin_id, b in g.groupby("true_bin", observed=True):
        x = b[x_col].to_numpy(dtype=float)
        y = b[y_col].to_numpy(dtype=float)
        if len(b) == 0:
            continue
        rows.append({
            "x_med": np.median(x),
            "x_q25": np.quantile(x, 0.25),
            "x_q75": np.quantile(x, 0.75),
            "y_med": np.median(y),
            "y_q25": np.quantile(y, 0.25),
            "y_q75": np.quantile(y, 0.75),
            "n": len(b),
        })
    out = pd.DataFrame(rows).sort_values("x_med")
    return out


def draw_tendency_calibration_panel(
    ax,
    df,
    x_col="true_trend_UP_minus_DOWN",
    y_col="pred_trend_UP_minus_DOWN",
    n_bins=6,
    xlim=(0.02, 0.15),
    ylim=(0.00, 0.155),
):
    """Binned calibration plot for BodyDecoder tendency predictions.

    Each marker is a bin median along the ground-truth tendency axis; vertical
    error bars show the IQR of predicted tendency within that bin. This is less
    noisy than raw scatter and directly communicates field calibration.
    """
    d = make_true_tendency_bins(df, x_col=x_col, n_bins=n_bins)

    # y=x reference
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=0.9,
            alpha=0.75, zorder=1)

    for cohort in COHORT_ORDER:
        g = d[d["cohort"] == cohort]
        r = _pearsonr(g[x_col], g[y_col])
        summ = _binned_summary(g, x_col=x_col, y_col=y_col)
        x = summ["x_med"].to_numpy()
        y = summ["y_med"].to_numpy()
        yerr = np.vstack([
            y - summ["y_q25"].to_numpy(),
            summ["y_q75"].to_numpy() - y,
        ])

        ax.errorbar(
            x, y, yerr=yerr,
            color=COLORS[cohort],
            marker='o', markersize=4.2,
            linewidth=1.35,
            elinewidth=0.9,
            capsize=2.4,
            label=f"{COHORT_LABELS[cohort]}  r={r:.2f}",
            zorder=3,
        )

    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.45, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.45, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Ground-truth UP−DOWN tendency")
    ax.set_ylabel("Predicted UP−DOWN tendency")
    ax.set_title("Tendency field tracks ground truth", fontsize=9.5, pad=4)
    ax.grid(alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')

# %% ─── PANEL (b): ACTION-WISE CONATIVE DISTRIBUTION ────────────────────────

def draw_action_panel(ax, df_q, actions=ACTIONS,
                      bar_w=0.25, dot_size=6, dot_alpha=0.4,
                      box_alpha=0.25, median_lw=2.0,
                      jitter_seed=42, ylim=(0.10, 0.28)):
    """Grouped IQR boxes by action × cohort for q(a)."""
    rng = np.random.default_rng(jitter_seed)
    x_centers = np.arange(len(actions))

    for j, act in enumerate(actions):
        col = f"q_{act}_mean"
        for k, cohort in enumerate(COHORT_ORDER):
            vals = df_q[df_q["cohort"] == cohort][col].to_numpy()
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
    ax.set_xlabel("Action")
    ax.set_ylabel("Conative target $q(a)$")
    ax.set_ylim(*ylim)
    ax.set_title("Selective conative readiness", fontsize=9.5, pad=4)
    ax.grid(axis='y', alpha=0.22, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.05, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


# %% ─── LEGENDS ─────────────────────────────────────────────────────────────

def build_calibration_legend(ax, loc='lower right'):
    # Rebuild handles so the y=x line appears last and the labels stay compact.
    handles = []
    for cohort in COHORT_ORDER:
        handles.append(Line2D([0], [0], marker='o', linestyle='None',
                              color=COLORS[cohort], markersize=4.5,
                              label=f"{COHORT_LABELS[cohort]}"))
    handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=0.9,
                          label='y=x'))
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=7.1,
              handletextpad=0.45, borderaxespad=0.35, labelspacing=0.33)


def build_action_legend(ax, loc='lower right'):
    """Cohort legend for panel (b), matching the IQR box colors."""
    handles = [Patch(facecolor=COLORS[c], alpha=0.35,
                     edgecolor=COLORS[c], label=COHORT_LABELS[c])
               for c in COHORT_ORDER]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=7.3,
              handletextpad=0.45, borderaxespad=0.35, labelspacing=0.35)


# %% ─── STATISTICS ──────────────────────────────────────────────────────────
# Two readouts:
#   (a) seed-level calibration: for each seed, compute Spearman/Pearson
#       correlation between predicted and ground-truth tendency across its
#       probe/zone rows. Cohort-level inference is across seeds, avoiding
#       pseudo-replication from treating probe rows as independent.
#   (b) pairwise Mann-Whitney U for each q(a) action; expects:
#         Full vs No body→g  : n.s.
#         Full vs No conation: ***
#         No body→g vs No conation: ***

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


def _safe_corrs(x, y):
    """Return Spearman rho and Pearson r for one seed/run.

    The cohort_counterfactual_summary input usually has multiple probe/zone
    rows per seed.  These rows are useful for estimating each seed's
    calibration relation, but they are not independent cohort-level samples.
    """
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
    """Two-sided Wilcoxon signed-rank test against zero, seed-level."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative='two-sided', zero_method='wilcox').pvalue)
    except ValueError:
        return np.nan


def compute_seed_calibration_stats(df, x_col="true_trend_UP_minus_DOWN",
                                   y_col="pred_trend_UP_minus_DOWN"):
    """Compute one calibration coefficient per independent seed.

    Returns
    -------
    seed_corr : DataFrame
        One row per cohort × seed, with Spearman rho and Pearson r.
    cohort_stats : DataFrame
        Cohort-level summaries and Wilcoxon tests across seeds.
    """
    if 'seed' not in df.columns:
        raise ValueError("Fig. 3a seed-level statistics require a 'seed' column.")

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


def print_calibration_stats(df, x_col="true_trend_UP_minus_DOWN",
                             y_col="pred_trend_UP_minus_DOWN"):
    """Fig. 3a calibration statistics using seed as the unit of analysis."""
    seed_corr, cohort_stats = compute_seed_calibration_stats(df, x_col=x_col, y_col=y_col)

    print("\n" + "=" * 70)
    print("Figure 3a — Tendency calibration (seed-level statistics)")
    print("=" * 70)
    print("  Probe/zone rows are used only to estimate one calibration coefficient")
    print("  per seed. Cohort-level inference treats seed/run as the independent unit.")
    print("  Test: Wilcoxon signed-rank test of seed-level Spearman ρ against zero;")
    print("  p-values BH-FDR adjusted across the three cohorts.")
    print(f"    {'cohort':<14s} {'n_seed':>6s}  {'Spearman ρ median [IQR]':>27s}  "
          f"{'Pearson r median [IQR]':>26s}  {'p_adj':>7s}  sig")

    for _, row in cohort_stats.iterrows():
        c = row['cohort']
        rho_iqr = f"{row['spearman_median']:+.3f} [{row['spearman_q25']:+.3f}, {row['spearman_q75']:+.3f}]"
        pear_iqr = f"{row['pearson_median']:+.3f} [{row['pearson_q25']:+.3f}, {row['pearson_q75']:+.3f}]"
        padj = row['wilcoxon_p_spearman_vs_zero_bh']
        print(f"    {COHORT_DISPLAY[c]:<14s} {int(row['n_seeds']):>6d}  "
              f"{rho_iqr:>27s}  {pear_iqr:>26s}  {_fmt_p(padj):>7s}  {_stars(padj)}")
    print()
    return seed_corr, cohort_stats


def print_q_stats(df_q):
    """Pairwise Mann-Whitney U for each q(a) action."""
    print("=" * 70)
    print("Figure 3b — Conative target distribution: pairwise tests")
    print("=" * 70)
    print("  Test: Mann-Whitney U (two-sided), with BH-FDR correction.")
    print("  Effect size: rank-biserial r = 1 − 2U/(n1·n2), range [−1, +1].")
    print("  Sign of r: positive when the first cohort tends to exceed the second.")
    for act in ACTIONS:
        col = f"q_{act}_mean"
        print(f"\n  [q({act})]  ({col})")
        print(f"    {'cohort':<14s} {'n':>3s}  {'median':>9s}  {'IQR':>20s}")
        for c in COHORT_ORDER:
            v = df_q[df_q['cohort'] == c][col].dropna().to_numpy()
            med = np.median(v); q25, q75 = np.quantile(v, [0.25, 0.75])
            iqr_str = f"[{q25:.3f}, {q75:.3f}]"
            print(f"    {COHORT_DISPLAY[c]:<14s} {len(v):>3d}  {med:>9.3f}  {iqr_str:>20s}")
        pairs = list(combinations(COHORT_ORDER, 2))
        results = []
        for a, b in pairs:
            ga = df_q[df_q['cohort'] == a][col].to_numpy()
            gb = df_q[df_q['cohort'] == b][col].to_numpy()
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
    print()


# %% ─── RUN ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.15),
                         gridspec_kw={'width_ratios': [1.06, 1.45]})

draw_tendency_calibration_panel(axes[0], df_t_raw)
draw_action_panel(axes[1], df_q)
build_calibration_legend(axes[0], loc='lower right')
build_action_legend(axes[1], loc='lower right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.34)
plt.show()

# Statistics — printed to console, not on figure.
seed_calib_stats, cohort_calib_stats = print_calibration_stats(df_t_raw)
print_q_stats(df_q)


# %% ─── SAVE ────────────────────────────────────────────────────────────────

outdir = Path(OUTDIR)
if not outdir.is_absolute():
    outdir = Path.cwd() / outdir
outdir.mkdir(parents=True, exist_ok=True)

# Save seed-level calibration statistics used for Fig. 3a.
seed_calib_stats.to_csv(outdir / f"{NAME}_fig3a_seed_calibration.csv", index=False)
cohort_calib_stats.to_csv(outdir / f"{NAME}_fig3a_cohort_calibration.csv", index=False)
print(f"  saved: {outdir / f'{NAME}_fig3a_seed_calibration.csv'}")
print(f"  saved: {outdir / f'{NAME}_fig3a_cohort_calibration.csv'}")

for fmt in FORMATS:
    out = outdir / f"{NAME}.{fmt}"
    fig.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"  saved: {out}")