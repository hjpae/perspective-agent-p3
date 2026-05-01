# cear_pilot/experiments/visualize_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 visualization. Currently implements Vis 1 (g state PCA in target
space). Designed to be extended with Vis 4/5/6 later.

Vis 1: g state PCA
  - For metric_c1 mode: PCA on z_quad descriptors (256-dim).
  - For film/salience modes: PCA on z descriptors (16-dim).
  - For each g state, average target representation over probes → one descriptor.
  - PCA → 2D, color/marker by candidate organizers (body, valence, perturb).

  This visualization addresses the silhouette-paradox question: even when
  the silhouette score in full target space is near zero or negative,
  PCA can reveal whether the organization is real but lives in a low-dim
  subspace.

Inputs:
  --probe_results   probe_results.parquet (from probe_phase3.py)
  --probe_zquad     probe_zquad.parquet (only for metric_c1 mode)
  --outdir          where to save figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------------
# Load: build (n_g, n_p, dim) target tensor + g meta
# ---------------------------------------------------------------------------

def _infer_mode(pr: pd.DataFrame) -> str:
    """Infer mode from columns + actual values when 'mode' column missing.

    Strategy: check which gating-param columns have nonzero values. The
    architecture used at training time should leave nonzero values in its
    own gating parameters and zeros elsewhere (or no entries at all).
    """
    cols = set(pr.columns)
    # metric_c1: distinctive cols are M_trace / M_eig_* / z_metric_*
    has_metric_cols = ("M_trace" in cols
                       or any(c.startswith("z_metric_") for c in cols)
                       or any(c.startswith("M_eig_") for c in cols))
    if has_metric_cols:
        return "metric_c1"

    # film vs salience: check which has nonzero values.
    sal_cols = [c for c in cols if c.startswith("salience_")]
    gamma_cols = [c for c in cols if c.startswith("gamma_")]
    beta_cols = [c for c in cols if c.startswith("beta_")]

    sal_nonzero = (len(sal_cols) > 0
                   and pr[sal_cols].abs().sum().sum() > 1e-6)
    gam_nonzero = (len(gamma_cols) > 0
                   and pr[gamma_cols].abs().sum().sum() > 1e-6)
    bet_nonzero = (len(beta_cols) > 0
                   and pr[beta_cols].abs().sum().sum() > 1e-6)

    # Salience init at zero gives sigmoid(0)=0.5 — *but the saved values are
    # the linear-layer raw outputs* in some old code paths. After training,
    # learned salience MLP has nonzero outputs.
    if sal_nonzero and not (gam_nonzero or bet_nonzero):
        return "salience"
    if (gam_nonzero or bet_nonzero) and not sal_nonzero:
        return "film"
    if sal_nonzero and (gam_nonzero or bet_nonzero):
        # both have signal: prefer salience if its magnitude is much larger
        sal_mag = pr[sal_cols].abs().mean().mean()
        film_mag = (pr[gamma_cols + beta_cols].abs().mean().mean()
                    if (gamma_cols or beta_cols) else 0.0)
        # Salience values are sigmoid outputs ∈ (0, 1), often near 0.5.
        # FiLM gamma/beta init at 0; trained magnitudes typically O(0.5).
        # Heuristic: if both nonzero, default to salience (it's the stronger
        # phenomenological commitment in our pipeline).
        return "salience"
    # neither — default to film
    return "film"


def load_probe_data(
    probe_results_path: str,
    probe_zquad_path: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """
    Returns:
      target: (n_g, n_p, dim) — z_quad if metric_c1, else z
      g_meta: per-g metadata dataframe
      mode: gating mode string
    """
    pr = pd.read_parquet(probe_results_path)
    if "mode" in pr.columns:
        mode = str(pr["mode"].iloc[0])
    else:
        mode = _infer_mode(pr)
        print(f"[load] 'mode' column missing — inferred mode='{mode}' from data shape")
    n_g = pr["g_id"].nunique()
    n_p = pr["probe_id"].nunique()

    if mode == "metric_c1":
        if probe_zquad_path is None:
            raise ValueError("metric_c1 mode requires --probe_zquad path")
        zq = pd.read_parquet(probe_zquad_path)
        zq_cols = sorted([c for c in zq.columns if c.startswith("zq_")],
                         key=lambda s: int(s.split("_")[1]))
        target = (zq.sort_values(["g_id", "probe_id"])[zq_cols].values
                  .reshape(n_g, n_p, len(zq_cols)).astype(np.float32))
    else:
        z_cols = sorted(
            [c for c in pr.columns if c.startswith("z_") and c[2:].isdigit()],
            key=lambda s: int(s.split("_")[1]),
        )
        target = (pr.sort_values(["g_id", "probe_id"])[z_cols].values
                  .reshape(n_g, n_p, len(z_cols)).astype(np.float32))

    # per-g metadata (all rows for given g_id share same g meta, take first)
    g_meta = (pr.sort_values(["g_id", "probe_id"])
                .drop_duplicates(subset="g_id")
                [["g_id", "g_episode", "g_t", "g_body_state",
                  "g_perturbation", "g_valence_zone"]]
                .reset_index(drop=True))

    print(f"[load] mode={mode}  target shape={target.shape}  g_meta rows={len(g_meta)}")
    return target, g_meta, mode


# ---------------------------------------------------------------------------
# PCA via numpy
# ---------------------------------------------------------------------------

def pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Centered PCA via SVD. Returns (Y, explained_var_ratio)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # use SVD on centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = (U[:, :n_components] * S[:n_components])
    var = (S ** 2) / max(len(X) - 1, 1)
    var_ratio = var / var.sum()
    return Y, var_ratio[:n_components]


# ---------------------------------------------------------------------------
# Vis 1: g state PCA scatter
# ---------------------------------------------------------------------------

def vis1_g_pca(
    target: np.ndarray,
    g_meta: pd.DataFrame,
    mode: str,
    outdir: Path,
) -> Dict[str, float]:
    """
    g descriptor = mean target over probes (one vector per g state).
    PCA → 2D. Plot with three encodings:
      - color   = body_state (continuous viridis)
      - marker  = valence_zone (▲ top, ● mid, ▼ bot)
      - edge    = perturb_active (black = on, none = off)

    Saves vis1_g_pca.png and prints summary statistics.
    """
    g_desc = target.mean(axis=1)  # (n_g, dim)
    Y, var_ratio = pca(g_desc, n_components=2)

    body = g_meta["g_body_state"].values
    vz = g_meta["g_valence_zone"].values.astype(int)
    pert = g_meta["g_perturbation"].values.astype(int)

    # Pearson correlation: body_state with PC1, PC2
    corr_pc1 = float(np.corrcoef(Y[:, 0], body)[0, 1]) if len(Y) > 2 else float("nan")
    corr_pc2 = float(np.corrcoef(Y[:, 1], body)[0, 1]) if len(Y) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(8.5, 7))

    marker_map = {0: "^", 1: "o", 2: "v"}     # vz: top=▲ mid=● bot=▼
    label_map = {0: "vz top (positive)", 1: "vz mid", 2: "vz bot (negative)"}

    for vz_id in (0, 1, 2):
        for p_active in (0, 1):
            sel = (vz == vz_id) & (pert == p_active)
            if sel.sum() == 0:
                continue
            edge = "black" if p_active == 1 else "none"
            label = label_map[vz_id] + (" + perturb" if p_active else "")
            sc = ax.scatter(
                Y[sel, 0], Y[sel, 1],
                c=body[sel], cmap="viridis",
                vmin=0.0, vmax=1.0,
                marker=marker_map[vz_id],
                s=180,
                edgecolors=edge, linewidths=1.5,
                label=label, alpha=0.95,
            )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("body_state (0 = depleted ↘ 1 = saturated)", fontsize=11)

    # annotate g_id for traceability
    for i in range(len(Y)):
        ax.annotate(f"  g{i}", (Y[i, 0], Y[i, 1]), fontsize=8, alpha=0.7)

    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% variance)", fontsize=12)
    target_name = "z_quad" if mode == "metric_c1" else "z"
    ax.set_title(
        f"Vis 1: g state PCA in {target_name} space  (mode = {mode})\n"
        f"corr(body, PC1) = {corr_pc1:+.3f}  |  corr(body, PC2) = {corr_pc2:+.3f}",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    out_path = outdir / "vis1_g_pca.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    # Also: best linear axis correlation across first 5 PCs
    Y5, var5 = pca(g_desc, n_components=min(5, g_desc.shape[0] - 1))
    pc_correlations = {
        "body": [float(np.corrcoef(Y5[:, k], body)[0, 1]) for k in range(Y5.shape[1])],
    }
    if len(np.unique(vz)) > 1:
        # for categorical, compute per-PC eta-squared (variance explained by category)
        pc_correlations["valence_zone_eta2"] = []
        for k in range(Y5.shape[1]):
            ss_total = float(np.var(Y5[:, k]) * len(Y5))
            ss_between = 0.0
            grand_mean = Y5[:, k].mean()
            for c in np.unique(vz):
                grp = Y5[vz == c, k]
                ss_between += len(grp) * (grp.mean() - grand_mean) ** 2
            eta2 = ss_between / max(ss_total, 1e-9)
            pc_correlations["valence_zone_eta2"].append(float(eta2))
    if len(np.unique(pert)) > 1:
        pc_correlations["perturb_eta2"] = []
        for k in range(Y5.shape[1]):
            ss_total = float(np.var(Y5[:, k]) * len(Y5))
            ss_between = 0.0
            grand_mean = Y5[:, k].mean()
            for c in np.unique(pert):
                grp = Y5[pert == c, k]
                ss_between += len(grp) * (grp.mean() - grand_mean) ** 2
            eta2 = ss_between / max(ss_total, 1e-9)
            pc_correlations["perturb_eta2"].append(float(eta2))

    print(f"\n  PCA explained variance ratios (first 5 PCs): "
          f"{[round(float(v), 3) for v in var5]}")
    print(f"  cumulative: {[round(float(v), 3) for v in np.cumsum(var5)]}")
    print(f"\n  Body correlation per PC: "
          f"{[round(c, 3) for c in pc_correlations['body']]}")
    if "valence_zone_eta2" in pc_correlations:
        print(f"  Valence zone eta^2 per PC: "
              f"{[round(e, 3) for e in pc_correlations['valence_zone_eta2']]}")
    if "perturb_eta2" in pc_correlations:
        print(f"  Perturb eta^2 per PC: "
              f"{[round(e, 3) for e in pc_correlations['perturb_eta2']]}")

    # Honest interpretation hint
    max_body_corr = max(abs(c) for c in pc_correlations["body"])
    print(f"\n  Strongest body|PC correlation across first 5 PCs: {max_body_corr:.3f}")
    if max_body_corr > 0.5:
        print(f"  → Body organizes a meaningful PC axis (despite silhouette).")
    elif max_body_corr > 0.3:
        print(f"  → Body has moderate PC alignment.")
    else:
        print(f"  → No PC axis strongly aligns with body. silhouette result reflects "
              f"true lack of body-organization.")

    return {
        "pc1_var_ratio": float(var_ratio[0]),
        "pc2_var_ratio": float(var_ratio[1]),
        "corr_body_pc1": corr_pc1,
        "corr_body_pc2": corr_pc2,
        "max_body_pc_corr": max_body_corr,
        "n_g": int(len(Y)),
    }


# ---------------------------------------------------------------------------
# Vis 4: Trajectory through g space + body timeseries
# ---------------------------------------------------------------------------

def _build_pca_basis(
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on g-descriptors and return (mean, components_2d, var_ratio)."""
    g_desc = target.mean(axis=1)                          # (n_g, dim)
    mean = g_desc.mean(axis=0, keepdims=True)             # (1, dim)
    Xc = g_desc - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:2]                                   # (2, dim)
    var = (S ** 2) / max(len(g_desc) - 1, 1)
    var_ratio = var / var.sum()
    return mean, components, var_ratio[:2]


def _project_traj_to_pca(
    g_traj: np.ndarray,                                   # (T, g_dim)
    obs_enc,                                              # encoder for metric_c1 only
    mode: str,
    pca_mean: np.ndarray,                                 # (1, target_dim)
    pca_components: np.ndarray,                           # (2, target_dim)
    z_raw_template: np.ndarray,                           # (n_p, 16) for z_quad recompute
) -> np.ndarray:
    """
    Project a sequence of g vectors into the same PCA basis as Vis 1.

    For metric_c1 mode, the target is z_quad — which is built from a probe
    set z_raw and the M_g(g). Since trajectory g changes step-by-step but the
    "g-descriptor" used in PCA was averaged over the same n_p probes, we
    reproduce that averaging at each timestep:
        descriptor_t = mean over probes of z_quad(g_t, probe_i)

    For film/salience modes, target is z, so descriptor is mean over probes
    of z(g_t, probe_i).
    """
    raise NotImplementedError("Use _project_traj_metric_c1 or _project_traj_simple instead")


def _descriptor_at_g_metric_c1(
    obs_enc, g_t_torch, probe_xs_torch,
) -> np.ndarray:
    """
    Compute descriptor (mean z_quad over probes) for a single g state.
    Returns (target_dim,) numpy array.

    target_dim = z_dim^2 = 256 for z_dim=16.
    """
    import torch
    with torch.no_grad():
        # M_g for this g
        M = obs_enc.build_metric(g_t_torch)               # (1, z_dim, z_dim)
        # z_raw for each probe (g-independent)
        z_raw = torch.tanh(obs_enc.mlp(probe_xs_torch))   # (n_p, z_dim)
        # M_g @ z_raw for each probe
        Mz = torch.matmul(M, z_raw.unsqueeze(-1)).squeeze(-1)  # (n_p, z_dim)
        # quadratic features
        z_outer = z_raw.unsqueeze(-1) * Mz.unsqueeze(-2)  # (n_p, z_dim, z_dim)
        z_quad = z_outer.reshape(z_raw.shape[0], -1)      # (n_p, z_dim^2)
        descriptor = z_quad.mean(dim=0)                    # (z_dim^2,)
    return descriptor.cpu().numpy()


def _descriptor_at_g_simple(
    obs_enc, g_t_torch, probe_xs_torch,
) -> np.ndarray:
    """For film/salience: descriptor = mean z over probes."""
    import torch
    with torch.no_grad():
        # Use encoder forward to apply mode-correct gating
        z = obs_enc(probe_xs_torch, g_t=g_t_torch.expand(probe_xs_torch.shape[0], -1))
        descriptor = z.mean(dim=0)                         # (z_dim,)
    return descriptor.cpu().numpy()


def _select_episodes_stratified(
    traj_df: pd.DataFrame,
    n_low: int = 3,
    n_mid: int = 4,
    n_high: int = 3,
    rng_seed: int = 0,
) -> Dict[str, list]:
    """
    Select episodes from late training (top 1/3 of episodes) stratified by
    terminal body state.

    Returns dict {'low': [ep, ...], 'mid': [ep, ...], 'high': [ep, ...]}.
    """
    rng = np.random.default_rng(rng_seed)
    max_ep = int(traj_df["episode"].max())
    cutoff = max_ep - max_ep // 3
    late_terminal = (traj_df[traj_df["episode"] >= cutoff]
                     .groupby("episode")
                     .agg(terminal_body=("body_state", "last"))
                     .reset_index())

    low_eps = late_terminal[late_terminal["terminal_body"] <= 0.2]["episode"].tolist()
    high_eps = late_terminal[late_terminal["terminal_body"] >= 0.8]["episode"].tolist()
    mid_eps = late_terminal[(late_terminal["terminal_body"] > 0.2) &
                            (late_terminal["terminal_body"] < 0.8)]["episode"].tolist()

    def _pick(pool, n):
        if len(pool) <= n:
            return pool
        return list(rng.choice(pool, size=n, replace=False))

    return {
        "low":  _pick(low_eps, n_low),
        "mid":  _pick(mid_eps, n_mid),
        "high": _pick(high_eps, n_high),
    }


def vis4_trajectory(
    target: np.ndarray,
    g_meta: pd.DataFrame,
    mode: str,
    traj_path: str,
    ckpt_path: str,
    outdir: Path,
    n_low: int = 3,
    n_mid: int = 4,
    n_high: int = 3,
    step_every: int = 10,
) -> None:
    """
    Project training trajectory g(t) into Vis 1's PCA basis. Plot 10 episodes
    selected stratified by terminal body state. Pair with body timeseries.

    Requires importing torch and rebuilding the encoder from the ckpt.
    """
    import torch
    from cear_pilot.models.encoder import EncoderBundle, EncoderConfig
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Config, NZonePhase3Env

    # ---- 1. Fit PCA on probe-data g descriptors (Vis 1 basis) ----
    pca_mean, pca_components, var_ratio = _build_pca_basis(target)
    print(f"  PCA basis fit on {target.shape[0]} g states; "
          f"PC1/PC2 var = {var_ratio[0]:.3f}/{var_ratio[1]:.3f}")

    # ---- 2. Load ckpt and rebuild encoder ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_cfg_dict = ckpt["meta"]["agent_cfg"]["encoder"]
    if "gating_mode" not in enc_cfg_dict:
        enc_cfg_dict["gating_mode"] = mode
    enc_cfg = EncoderConfig(**enc_cfg_dict)
    encoder = EncoderBundle(enc_cfg)
    # filter encoder weights from full agent state
    enc_state = {k.replace("enc.", "", 1): v
                 for k, v in ckpt["agent_state"].items()
                 if k.startswith("enc.")}
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()
    obs_enc = encoder.obs_enc

    # ---- 3. Build the same probe set used by probe_phase3 (env reset) ----
    env_cfg_dict = ckpt["meta"]["env_cfg"]
    cfg = NZonePhase3Config(**{k: v for k, v in env_cfg_dict.items()
                                if k in NZonePhase3Config.__dataclass_fields__})
    env = NZonePhase3Env(config=cfg)

    # Reproduce probe_phase3's stratified probe sampling
    rng_probes = np.random.default_rng(0)
    cz_bds = list(cfg.report_zone_boundaries)
    vz_bds = list(cfg.valence_zone_boundaries)

    def _zone_ranges(bds, total):
        bs = [0] + list(bds) + [total]
        return [(bs[i], bs[i+1]) for i in range(len(bs)-1)]

    cz_ranges = _zone_ranges(cz_bds, cfg.width)
    vz_ranges = _zone_ranges(vz_bds, cfg.height)
    n_probes = 27
    per_stratum = max(1, n_probes // (len(cz_ranges) * len(vz_ranges)))

    def _read_obs(env, x, y, body=0.5):
        env.reset(seed=0)
        env.x = int(x); env.y = int(y)
        env.body_state = np.array([body], dtype=np.float32)
        env._perturbation_active = False
        return env._observe()

    probe_xs_list = []
    for cx_lo, cx_hi in cz_ranges:
        for vy_lo, vy_hi in vz_ranges:
            for _ in range(per_stratum):
                x = int(rng_probes.integers(max(1, cx_lo),
                                             max(cx_lo+1, min(cfg.width-1, cx_hi))))
                y = int(rng_probes.integers(max(1, vy_lo),
                                             max(vy_lo+1, min(cfg.height-1, vy_hi))))
                probe_xs_list.append(_read_obs(env, x, y))
    probe_xs = np.stack(probe_xs_list[:n_probes], axis=0).astype(np.float32)
    probe_xs_torch = torch.tensor(probe_xs, dtype=torch.float32)
    print(f"  rebuilt {len(probe_xs)} probes for descriptor reconstruction")

    # ---- 4. Select episodes ----
    traj_df = pd.read_parquet(traj_path)
    g_cols = sorted([c for c in traj_df.columns if c.startswith("g_") and c[2:].isdigit()],
                    key=lambda s: int(s.split("_")[1]))

    eps_by_band = _select_episodes_stratified(traj_df, n_low, n_mid, n_high)
    print(f"  episodes selected — low: {eps_by_band['low']}  "
          f"mid: {eps_by_band['mid']}  high: {eps_by_band['high']}")

    # ---- 5. Project each episode's g trajectory ----
    # Subsample steps to keep plot manageable
    descriptor_fn = (_descriptor_at_g_metric_c1
                     if mode == "metric_c1" else _descriptor_at_g_simple)

    projected = {}  # ep -> (T_sub, 2) PC coords + body timeseries
    for band, eps in eps_by_band.items():
        for ep in eps:
            sub = traj_df[traj_df["episode"] == ep].sort_values("t")
            g_seq = sub[g_cols].values.astype(np.float32)
            body_seq = sub["body_state"].values
            t_seq = sub["t"].values

            # subsample
            idx = np.arange(0, len(g_seq), step_every)
            g_sub = g_seq[idx]
            body_sub = body_seq[idx]
            t_sub = t_seq[idx]

            # compute descriptor for each subsampled step
            desc_list = []
            for k in range(len(g_sub)):
                g_t_torch = torch.tensor(g_sub[k:k+1], dtype=torch.float32)
                d = descriptor_fn(obs_enc, g_t_torch, probe_xs_torch)
                desc_list.append(d)
            desc_arr = np.stack(desc_list, axis=0)            # (T_sub, target_dim)
            # project to PC1, PC2
            desc_centered = desc_arr - pca_mean              # (T_sub, target_dim)
            pc_coords = desc_centered @ pca_components.T     # (T_sub, 2)

            projected[ep] = {
                "pc": pc_coords,
                "body": body_sub,
                "t": t_sub,
                "band": band,
            }
        print(f"    projected band={band} ({len(eps)} episodes)")

    # ---- 6. Plot: side-by-side. left = PCA + trajectories. right = body timeseries ----
    band_color = {"low": "#762a83", "mid": "#5aae61", "high": "#d6604d"}
    band_label = {"low":  "terminal body ≤ 0.2 (depleted)",
                  "mid":  "terminal body ∈ (0.2, 0.8)",
                  "high": "terminal body ≥ 0.8 (saturated)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: PCA trajectories
    # First: faint Vis 1 g states as backdrop
    g_desc_static = target.mean(axis=1)
    backdrop_centered = g_desc_static - pca_mean
    backdrop_pc = backdrop_centered @ pca_components.T
    ax1.scatter(backdrop_pc[:, 0], backdrop_pc[:, 1],
                c="lightgray", s=60, alpha=0.5, zorder=1,
                edgecolors="gray", linewidths=0.5,
                label="probe-sampled g states")

    # Plot trajectories
    plotted_bands = set()
    for ep, info in projected.items():
        pc = info["pc"]
        band = info["band"]
        # line with time gradient
        for i in range(len(pc) - 1):
            alpha = 0.25 + 0.6 * (i / max(len(pc) - 1, 1))   # early faint → late solid
            ax1.plot(pc[i:i+2, 0], pc[i:i+2, 1],
                     color=band_color[band], alpha=alpha,
                     linewidth=1.4, zorder=2)
        # mark start and end
        ax1.scatter(pc[0, 0], pc[0, 1],
                    s=40, color=band_color[band], marker="o",
                    edgecolors="white", linewidths=1, zorder=3)
        ax1.scatter(pc[-1, 0], pc[-1, 1],
                    s=80, color=band_color[band], marker="*",
                    edgecolors="black", linewidths=1, zorder=4,
                    label=band_label[band] if band not in plotted_bands else None)
        plotted_bands.add(band)

    target_name = "z_quad" if mode == "metric_c1" else "z"
    ax1.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% variance)", fontsize=11)
    ax1.set_title(f"g trajectory in {target_name} PCA  (mode = {mode})\n"
                  f"○ start  ★ end  |  line opacity = time progression",
                  fontsize=11)
    ax1.legend(loc="best", fontsize=8.5, framealpha=0.85)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Right: paired body timeseries
    for ep, info in projected.items():
        band = info["band"]
        ax2.plot(info["t"], info["body"],
                 color=band_color[band], alpha=0.8, linewidth=1.4)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax2.axhline(1, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax2.set_xlabel("step within episode", fontsize=11)
    ax2.set_ylabel("body_state", fontsize=11)
    ax2.set_title("body trajectories  (paired with left)", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = outdir / "vis4_trajectory.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ---------------------------------------------------------------------------
# Vis 5: Trajectory + stance sharpness (entropy) overlay
# ---------------------------------------------------------------------------

def _stance_sharpness_metric_c1(obs_enc, g_t_torch) -> float:
    """
    For metric_c1: stance sharpness = deviation from isotropic M_g.
    Normalized eigenvalue entropy: H = -Σ p_i log p_i,  p_i = λ_i / Σλ.
    Returns 1 - H/log(z_dim), so 0 = fully isotropic (diffuse), 1 = rank-1 (sharp).
    """
    import torch
    if not hasattr(obs_enc, "metric_mlp"):
        return float("nan")
    with torch.no_grad():
        M = obs_enc.build_metric(g_t_torch)               # (1, z_dim, z_dim)
        eigs = torch.linalg.eigvalsh(M[0])                # (z_dim,)
        p = eigs / max(eigs.sum().item(), 1e-9)
        p = p.clamp(min=1e-12)
        H = -(p * torch.log(p)).sum().item()
        H_max = float(np.log(len(eigs)))
        sharpness = 1.0 - H / max(H_max, 1e-9)
    return float(sharpness)


def _stance_sharpness_sigmoid(obs_enc, g_t_torch) -> float:
    """
    For sigmoid salience: sharpness = mean deviation of salience from 0.5.
    Returns value in [0, 0.5]; we rescale to [0, 1] by *2.
    All dims at 0.5 → diffuse (0); all dims at 0 or 1 → sharp (1).
    """
    import torch
    if not hasattr(obs_enc, "salience"):
        return float("nan")
    with torch.no_grad():
        s = torch.sigmoid(obs_enc.salience(g_t_torch))     # (1, z_dim)
        deviation = (s - 0.5).abs().mean().item()
        sharpness = deviation * 2.0
    return float(sharpness)


def _stance_sharpness_film(obs_enc, g_t_torch) -> float:
    """
    For FiLM: sharpness = mean(|gamma| + |beta|) — gating magnitude.
    Larger magnitude = sharper gating. Normalize against rough scale.
    """
    import torch
    if not hasattr(obs_enc, "film"):
        return float("nan")
    with torch.no_grad():
        out = obs_enc.film(g_t_torch)                      # (1, 2*z_dim)
        gamma, beta = out.chunk(2, dim=-1)
        mag = (gamma.abs().mean() + beta.abs().mean()).item() / 2.0
        # rescale: empirically |γ|, |β| are O(0.5) at convergence
        sharpness = float(min(mag, 1.0))
    return sharpness


def _sharpness_fn_for_mode(mode: str, obs_enc=None):
    """Pick sharpness fn for the inferred mode, with fallback to whichever
    branch the encoder actually has if the named one is missing.
    """
    if mode == "metric_c1":
        return _stance_sharpness_metric_c1
    if mode in ("salience", "both"):
        return _stance_sharpness_sigmoid
    if mode == "film":
        if obs_enc is not None and not hasattr(obs_enc, "film"):
            # encoder doesn't have FiLM — fall back to whichever it has
            if hasattr(obs_enc, "salience"):
                return _stance_sharpness_sigmoid
            if hasattr(obs_enc, "metric_mlp"):
                return _stance_sharpness_metric_c1
        return _stance_sharpness_film
    return _stance_sharpness_film


def vis5_trajectory_with_sharpness(
    target: np.ndarray,
    g_meta: pd.DataFrame,
    mode: str,
    traj_path: str,
    ckpt_path: str,
    outdir: Path,
    n_low: int = 3,
    n_mid: int = 4,
    n_high: int = 3,
    step_every: int = 10,
) -> None:
    """
    Vis 5: same as Vis 4 (PCA trajectory + body timeseries) but each
    trajectory point colored/sized by *stance sharpness* — how concentrated
    vs diffuse the gating is at that g state.

    Right panel: body timeseries. Bottom panel: sharpness timeseries paired
    with body.

    Phenomenological reading: does stance become sharper (focus on specific
    aspects) or more diffuse (open/undifferentiated) as body changes?
    """
    import torch
    from cear_pilot.models.encoder import EncoderBundle, EncoderConfig
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Config, NZonePhase3Env

    pca_mean, pca_components, var_ratio = _build_pca_basis(target)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_cfg_dict = ckpt["meta"]["agent_cfg"]["encoder"]
    if "gating_mode" not in enc_cfg_dict:
        enc_cfg_dict["gating_mode"] = mode
    enc_cfg = EncoderConfig(**enc_cfg_dict)
    encoder = EncoderBundle(enc_cfg)
    enc_state = {k.replace("enc.", "", 1): v
                 for k, v in ckpt["agent_state"].items()
                 if k.startswith("enc.")}
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()
    obs_enc = encoder.obs_enc

    # rebuild probes for descriptor reconstruction (same as Vis 4)
    env_cfg_dict = ckpt["meta"]["env_cfg"]
    cfg = NZonePhase3Config(**{k: v for k, v in env_cfg_dict.items()
                                if k in NZonePhase3Config.__dataclass_fields__})
    env = NZonePhase3Env(config=cfg)
    rng_probes = np.random.default_rng(0)

    def _zone_ranges(bds, total):
        bs = [0] + list(bds) + [total]
        return [(bs[i], bs[i+1]) for i in range(len(bs)-1)]

    cz_ranges = _zone_ranges(list(cfg.report_zone_boundaries), cfg.width)
    vz_ranges = _zone_ranges(list(cfg.valence_zone_boundaries), cfg.height)
    n_probes = 27
    per_stratum = max(1, n_probes // (len(cz_ranges) * len(vz_ranges)))

    def _read_obs(env, x, y, body=0.5):
        env.reset(seed=0); env.x = int(x); env.y = int(y)
        env.body_state = np.array([body], dtype=np.float32)
        env._perturbation_active = False
        return env._observe()

    probe_xs_list = []
    for cx_lo, cx_hi in cz_ranges:
        for vy_lo, vy_hi in vz_ranges:
            for _ in range(per_stratum):
                x = int(rng_probes.integers(max(1, cx_lo),
                                             max(cx_lo+1, min(cfg.width-1, cx_hi))))
                y = int(rng_probes.integers(max(1, vy_lo),
                                             max(vy_lo+1, min(cfg.height-1, vy_hi))))
                probe_xs_list.append(_read_obs(env, x, y))
    probe_xs = np.stack(probe_xs_list[:n_probes], axis=0).astype(np.float32)
    probe_xs_torch = torch.tensor(probe_xs, dtype=torch.float32)

    # Episodes
    traj_df = pd.read_parquet(traj_path)
    g_cols = sorted([c for c in traj_df.columns if c.startswith("g_") and c[2:].isdigit()],
                    key=lambda s: int(s.split("_")[1]))
    eps_by_band = _select_episodes_stratified(traj_df, n_low, n_mid, n_high)

    descriptor_fn = (_descriptor_at_g_metric_c1
                     if mode == "metric_c1" else _descriptor_at_g_simple)
    sharpness_fn = _sharpness_fn_for_mode(mode, obs_enc=obs_enc)

    projected = {}
    for band, eps in eps_by_band.items():
        for ep in eps:
            sub = traj_df[traj_df["episode"] == ep].sort_values("t")
            g_seq = sub[g_cols].values.astype(np.float32)
            body_seq = sub["body_state"].values
            t_seq = sub["t"].values

            idx = np.arange(0, len(g_seq), step_every)
            g_sub = g_seq[idx]
            body_sub = body_seq[idx]
            t_sub = t_seq[idx]

            desc_list, sharp_list = [], []
            for k in range(len(g_sub)):
                g_t_torch = torch.tensor(g_sub[k:k+1], dtype=torch.float32)
                d = descriptor_fn(obs_enc, g_t_torch, probe_xs_torch)
                desc_list.append(d)
                sharp_list.append(sharpness_fn(obs_enc, g_t_torch))
            desc_arr = np.stack(desc_list, axis=0)
            sharp_arr = np.array(sharp_list)
            pc_coords = (desc_arr - pca_mean) @ pca_components.T

            projected[ep] = {
                "pc": pc_coords,
                "body": body_sub,
                "sharp": sharp_arr,
                "t": t_sub,
                "band": band,
            }
        print(f"  Vis 5: projected band={band} ({len(eps)} episodes)")

    # ---- Plot: 3-panel layout: left big PCA + sharpness color, right 2 stacked timeseries ----
    fig = plt.figure(figsize=(15.5, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1],
                          hspace=0.32, wspace=0.25)
    ax_pca = fig.add_subplot(gs[:, 0])
    ax_body = fig.add_subplot(gs[0, 1])
    ax_sharp = fig.add_subplot(gs[1, 1], sharex=ax_body)

    # backdrop g states
    g_desc_static = target.mean(axis=1)
    backdrop_pc = (g_desc_static - pca_mean) @ pca_components.T
    ax_pca.scatter(backdrop_pc[:, 0], backdrop_pc[:, 1],
                   c="lightgray", s=60, alpha=0.5, zorder=1,
                   edgecolors="gray", linewidths=0.5)

    # collect global sharpness range for color normalization
    all_sharp = np.concatenate([info["sharp"] for info in projected.values()])
    s_min, s_max = float(all_sharp.min()), float(all_sharp.max())
    norm = mpl.colors.Normalize(vmin=s_min, vmax=s_max)
    cmap = plt.cm.plasma

    # Plot scatter colored by sharpness (single combined scatter for clean colorbar)
    for ep, info in projected.items():
        pc = info["pc"]
        sc = ax_pca.scatter(pc[:, 0], pc[:, 1], c=info["sharp"],
                            cmap=cmap, norm=norm, s=14, alpha=0.85, zorder=2)
        # Connect with thin gray line for trajectory continuity
        ax_pca.plot(pc[:, 0], pc[:, 1], color="gray", alpha=0.3,
                    linewidth=0.7, zorder=1.5)
        # Mark start/end
        ax_pca.scatter(pc[0, 0], pc[0, 1], s=40, marker="o",
                       facecolors="none", edgecolors="black",
                       linewidths=1.2, zorder=3)
        ax_pca.scatter(pc[-1, 0], pc[-1, 1], s=80, marker="*",
                       facecolors="white", edgecolors="black",
                       linewidths=1.2, zorder=4)

    cbar = plt.colorbar(sc, ax=ax_pca, fraction=0.04, pad=0.02)
    cbar.set_label("stance sharpness  (low = diffuse, high = focused)",
                   fontsize=10)

    target_name = "z_quad" if mode == "metric_c1" else "z"
    sharpness_label = {
        "metric_c1": "1 - normalized M_g eigenvalue entropy",
        "salience":  "mean |salience(g) - 0.5| × 2",
        "both":      "mean |salience(g) - 0.5| × 2",
        "film":      "mean(|γ|, |β|)",
    }.get(mode, "stance sharpness")

    ax_pca.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% variance)", fontsize=11)
    ax_pca.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% variance)", fontsize=11)
    ax_pca.set_title(f"Vis 5: g trajectory + stance sharpness  (mode = {mode})\n"
                     f"sharpness = {sharpness_label}",
                     fontsize=11)
    ax_pca.grid(True, alpha=0.3)
    ax_pca.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax_pca.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    band_color = {"low": "#762a83", "mid": "#5aae61", "high": "#d6604d"}
    for ep, info in projected.items():
        ax_body.plot(info["t"], info["body"],
                     color=band_color[info["band"]], alpha=0.8, linewidth=1.4)
        ax_sharp.plot(info["t"], info["sharp"],
                      color=band_color[info["band"]], alpha=0.8, linewidth=1.4)
    ax_body.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    ax_body.axhline(1, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    ax_body.set_ylabel("body_state", fontsize=10)
    ax_body.set_ylim(-0.05, 1.05)
    ax_body.set_title("body and sharpness timeseries", fontsize=10)
    ax_body.grid(True, alpha=0.3)
    ax_body.tick_params(labelbottom=False)

    ax_sharp.set_xlabel("step within episode", fontsize=10)
    ax_sharp.set_ylabel("stance sharpness", fontsize=10)
    ax_sharp.grid(True, alpha=0.3)

    out_path = outdir / "vis5_trajectory_sharpness.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ---------------------------------------------------------------------------
# Vis 6: Probe-discrimination capacity overlaid on g space
# ---------------------------------------------------------------------------

def vis6_probe_discrimination(
    target: np.ndarray,
    g_meta: pd.DataFrame,
    mode: str,
    outdir: Path,
) -> None:
    """
    For each g state, compute *probe-discrimination capacity* = variance of
    target across probes (per dim, then summed). High value = stance reads
    different probes very differently (rich content discrimination). Low value
    = stance reads all probes similarly (uniform/monotonous).

    Plot: same PCA scatter as Vis 1, with marker size and color showing
    discrimination capacity per g state.

    Phenomenological reading: which stance regions afford rich content
    differentiation, vs which give uniform/monotonous reading?
    """
    g_desc = target.mean(axis=1)
    pca_mean = g_desc.mean(axis=0, keepdims=True)
    Xc = g_desc - pca_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = (U[:, :2] * S[:2])
    var_ratio = (S[:2] ** 2) / max(len(g_desc) - 1, 1) / ((S ** 2).sum() / max(len(g_desc) - 1, 1))

    # per-g probe discrimination = sum over dim of variance across probes
    probe_var = target.var(axis=1).sum(axis=-1)            # (n_g,)

    body = g_meta["g_body_state"].values
    vz = g_meta["g_valence_zone"].values.astype(int)
    pert = g_meta["g_perturbation"].values.astype(int)

    # Correlate discrimination with body / valence
    corr_disc_body = float(np.corrcoef(probe_var, body)[0, 1]) if len(probe_var) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(9, 7))

    marker_map = {0: "^", 1: "o", 2: "v"}
    label_map = {0: "vz top", 1: "vz mid", 2: "vz bot"}

    norm = mpl.colors.Normalize(vmin=probe_var.min(), vmax=probe_var.max())
    cmap = plt.cm.viridis

    for vz_id in (0, 1, 2):
        sel = vz == vz_id
        if sel.sum() == 0:
            continue
        # marker size also scales with discrimination
        s_norm = (probe_var[sel] - probe_var.min()) / max(probe_var.max() - probe_var.min(), 1e-9)
        sizes = 80 + 320 * s_norm
        sc = ax.scatter(Y[sel, 0], Y[sel, 1],
                        c=probe_var[sel], cmap=cmap, norm=norm,
                        marker=marker_map[vz_id], s=sizes,
                        edgecolors="black", linewidths=1.0,
                        label=label_map[vz_id], alpha=0.9)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probe-discrimination capacity\n(Σ var of target across probes)",
                   fontsize=10)

    for i in range(len(Y)):
        ax.annotate(f"  g{i}", (Y[i, 0], Y[i, 1]), fontsize=8, alpha=0.7)

    target_name = "z_quad" if mode == "metric_c1" else "z"
    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"Vis 6: probe-discrimination across g states  ({target_name}, mode = {mode})\n"
                 f"corr(discrimination, body_state) = {corr_disc_body:+.3f}",
                 fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    out_path = outdir / "vis6_probe_discrimination.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    print(f"\n  Per-stance probe-discrimination summary:")
    print(f"    range: [{probe_var.min():.4f}, {probe_var.max():.4f}]")
    print(f"    mean: {probe_var.mean():.4f}  std: {probe_var.std():.4f}")
    print(f"    corr with body: {corr_disc_body:+.3f}")
    # Top/bottom 3 g states
    order = np.argsort(probe_var)
    print(f"\n  Lowest 3 discrimination g states (most uniform):")
    for idx in order[:3]:
        m = g_meta.iloc[int(idx)]
        print(f"    g{int(m['g_id'])}: body={m['g_body_state']:.2f}  "
              f"vz={int(m['g_valence_zone'])}  perturb={int(m['g_perturbation'])}  "
              f"disc={probe_var[idx]:.4f}")
    print(f"  Highest 3 discrimination g states (most differentiated):")
    for idx in order[-3:][::-1]:
        m = g_meta.iloc[int(idx)]
        print(f"    g{int(m['g_id'])}: body={m['g_body_state']:.2f}  "
              f"vz={int(m['g_valence_zone'])}  perturb={int(m['g_perturbation'])}  "
              f"disc={probe_var[idx]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_results", type=str, required=True)
    ap.add_argument("--probe_zquad", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs/visualize_phase3")
    # Vis 4 inputs (optional — Vis 4 only runs if both provided)
    ap.add_argument("--traj", type=str, default=None,
                    help="traj.parquet for Vis 4 (training trajectory)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="ckpt_final.pt for Vis 4 (rebuild encoder)")
    ap.add_argument("--n_low", type=int, default=3)
    ap.add_argument("--n_mid", type=int, default=4)
    ap.add_argument("--n_high", type=int, default=3)
    ap.add_argument("--step_every", type=int, default=10,
                    help="subsample step granularity for vis4 (lower = finer)")
    ap.add_argument("--force_mode", type=str, default=None,
                    choices=[None, "film", "salience", "metric_c1", "both"],
                    help="override auto-detected mode (use if inference is wrong)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target, g_meta, mode = load_probe_data(args.probe_results, args.probe_zquad)
    if args.force_mode is not None:
        if args.force_mode != mode:
            print(f"[override] forcing mode '{args.force_mode}' "
                  f"(detected '{mode}')")
            # Reload with forced mode by adjusting target if needed.
            # For metric_c1: requires probe_zquad. Other modes use z from
            # probe_results.
            if args.force_mode == "metric_c1" and not args.probe_zquad:
                raise ValueError("--force_mode metric_c1 requires --probe_zquad")
            mode = args.force_mode
            # If forcing away from metric_c1 to z-based mode, switch target
            # to z (probe_results' z columns).
            if args.force_mode != "metric_c1":
                pr = pd.read_parquet(args.probe_results)
                z_cols = sorted(
                    [c for c in pr.columns if c.startswith("z_") and c[2:].isdigit()],
                    key=lambda s: int(s.split("_")[1]),
                )
                n_g = pr["g_id"].nunique()
                n_p = pr["probe_id"].nunique()
                target = (pr.sort_values(["g_id", "probe_id"])[z_cols].values
                          .reshape(n_g, n_p, len(z_cols)).astype(np.float32))

    print(f"\n=== Vis 1: g state PCA ===")
    summary = vis1_g_pca(target, g_meta, mode, outdir)
    print(f"\nSummary: {summary}")

    if args.traj and args.ckpt:
        print(f"\n=== Vis 4: g trajectory + body timeseries ===")
        vis4_trajectory(target, g_meta, mode, args.traj, args.ckpt, outdir,
                        n_low=args.n_low, n_mid=args.n_mid, n_high=args.n_high,
                        step_every=args.step_every)
        print(f"\n=== Vis 5: trajectory + stance sharpness overlay ===")
        vis5_trajectory_with_sharpness(
            target, g_meta, mode, args.traj, args.ckpt, outdir,
            n_low=args.n_low, n_mid=args.n_mid, n_high=args.n_high,
            step_every=args.step_every,
        )
    else:
        print(f"\n[skip] Vis 4 + Vis 5 (provide --traj and --ckpt to enable)")

    print(f"\n=== Vis 6: probe-discrimination across g states ===")
    vis6_probe_discrimination(target, g_meta, mode, outdir)

    print(f"\n[done] figures in {outdir}/")


if __name__ == "__main__":
    main()