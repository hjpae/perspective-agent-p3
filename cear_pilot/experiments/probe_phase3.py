# cear_pilot/experiments/probe_phase3.py
# -*- coding: utf-8 -*-
"""
Frame 1 probe analysis on a Phase 3 checkpoint.

Q1: Does the same input x produce different FiLM gating (gamma, beta)
    under different g states? Direct test of Frame 1 main claim —
    "same content / different mode of givenness".

Q2: If yes, is that difference systematic or random?
    (a) per-probe consistency of the g-difference vector
    (b) existence of linear transformation between g pairs (R^2)
    (c) cluster structure across many g states (silhouette)
    (d) probe-dependence of g spread (CoV)

Inputs:
  --ckpt        path to ckpt_final.pt
  --traj        path to traj.parquet (used to sample real g states)

Outputs:
  - prints summary
  - probe_results.parquet with per-(g, probe) rows for downstream analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from cear_pilot.envs.nzone_phase3 import NZonePhase3Config, NZonePhase3Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.encoder import EncoderConfig
from cear_pilot.models.world_latent import WorldLatentConfig
from cear_pilot.models.state_head import StateHeadConfig
from cear_pilot.models.policy import PolicyConfig


# ---------------------------------------------------------------------------
# Load ckpt and rebuild agent
# ---------------------------------------------------------------------------

def load_agent(ckpt_path: str, device: str = "cpu") -> Tuple[CEARAgent, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    acfg = meta["agent_cfg"]

    agent_cfg = AgentConfig(
        encoder=EncoderConfig(**acfg["encoder"]),
        world=WorldLatentConfig(**acfg["world"]),
        state=StateHeadConfig(**acfg["state"]),
        policy=PolicyConfig(**acfg["policy"]),
        device=device,
    )
    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"])
    agent.eval()
    print(f"[load] agent restored from {ckpt_path}")
    return agent, meta


# ---------------------------------------------------------------------------
# Sample g states from training trajectory
# ---------------------------------------------------------------------------

def sample_g_states(
    traj_path: str,
    n_states: int = 20,
    strategy: str = "stratified",
    rng_seed: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Sample n_states g vectors from training trajectory.

    Strategies:
      'stratified': sample from late training, stratified by body band x perturb
                    so we see g states from diverse internal contexts.
      'random':     uniform random sample.
      'late':       only late-training (ep >= 2/3 of total).
    """
    df = pd.read_parquet(traj_path)
    g_cols = [c for c in df.columns if c.startswith("g_") and c[2:].isdigit()]
    g_cols = sorted(g_cols, key=lambda s: int(s.split("_")[1]))
    g_dim = len(g_cols)
    print(f"[sample_g] {len(df)} rows in traj; g_dim={g_dim}; strategy={strategy}")

    rng = np.random.default_rng(rng_seed)

    if strategy == "random":
        idx = rng.choice(len(df), size=n_states, replace=False)
        sub = df.iloc[idx]

    elif strategy == "late":
        max_ep = df["episode"].max()
        cutoff = max_ep * 2 // 3
        late_df = df[df["episode"] >= cutoff]
        idx = rng.choice(len(late_df), size=n_states, replace=False)
        sub = late_df.iloc[idx]

    elif strategy == "stratified":
        max_ep = df["episode"].max()
        df = df.copy()
        df["body_band"] = pd.cut(
            df["body_state"], bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
            labels=["very_low", "low", "high", "very_high"],
        )
        df["perturb"] = df["perturbation_active"].map({0: "off", 1: "on"})

        cutoff = max_ep * 2 // 3
        late = df[df["episode"] >= cutoff]
        groups = list(late.groupby(["body_band", "perturb"], observed=True))
        groups = [(k, v) for k, v in groups if len(v) > 0]
        per_group = max(1, n_states // max(1, len(groups)))
        picks = []
        for key, g in groups:
            n = min(per_group, len(g))
            idx = rng.choice(len(g), size=n, replace=False)
            picks.append(g.iloc[idx])
        sub = pd.concat(picks).reset_index(drop=True)
        if len(sub) > n_states:
            keep_idx = rng.choice(len(sub), size=n_states, replace=False)
            sub = sub.iloc[keep_idx]
        elif len(sub) < n_states:
            extra = rng.choice(len(late), size=(n_states - len(sub)), replace=False)
            sub = pd.concat([sub, late.iloc[extra]])
        sub = sub.reset_index(drop=True)

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    g_arr = sub[g_cols].values.astype(np.float32)
    meta_df = sub[["episode", "t", "x", "y", "valence_zone_id", "body_state",
                   "perturbation_active", "alpha", "g_norm"]].reset_index(drop=True)
    print(f"[sample_g] sampled {len(g_arr)} g states")
    return g_arr, meta_df


# ---------------------------------------------------------------------------
# Build probe x set
# ---------------------------------------------------------------------------

def build_probe_set(
    env_cfg_dict: dict,
    n_probes: int = 27,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build a fixed set of probe observations by placing a clean (non-perturbed)
    agent at various (x, y) grid positions, with body_state held at 0.5.

    Probes span (column zone, valence zone) strata for diverse content.
    """
    cfg_dict = dict(env_cfg_dict)
    # filter to known fields
    cfg = NZonePhase3Config(**{k: v for k, v in cfg_dict.items()
                                if k in NZonePhase3Config.__dataclass_fields__})
    env = NZonePhase3Env(config=cfg)

    rng = np.random.default_rng(rng_seed)
    cz_bds = list(cfg.report_zone_boundaries)
    vz_bds = list(cfg.valence_zone_boundaries)
    cz_ranges = _zone_ranges(cz_bds, cfg.width)
    vz_ranges = _zone_ranges(vz_bds, cfg.height)

    per_stratum = max(1, n_probes // (len(cz_ranges) * len(vz_ranges)))
    probes = []
    probe_meta_rows = []
    for ci, (cx_lo, cx_hi) in enumerate(cz_ranges):
        for vi, (vy_lo, vy_hi) in enumerate(vz_ranges):
            for _ in range(per_stratum):
                x = int(rng.integers(max(1, cx_lo), max(cx_lo + 1, min(cfg.width - 1, cx_hi))))
                y = int(rng.integers(max(1, vy_lo), max(vy_lo + 1, min(cfg.height - 1, vy_hi))))
                obs = _read_obs_at(env, x, y, body_state=0.5)
                probes.append(obs)
                probe_meta_rows.append({
                    "probe_id": len(probes) - 1,
                    "x": x, "y": y,
                    "column_zone": ci,
                    "valence_zone": vi,
                })

    probes_arr = np.stack(probes, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(probe_meta_rows)
    if len(probes_arr) > n_probes:
        keep = rng.choice(len(probes_arr), size=n_probes, replace=False)
        probes_arr = probes_arr[keep]
        meta_df = meta_df.iloc[keep].reset_index(drop=True)
        meta_df["probe_id"] = np.arange(len(meta_df))
    n_strata_covered = meta_df.groupby(["column_zone", "valence_zone"]).ngroups
    print(f"[probes] built {len(probes_arr)} probes covering "
          f"{n_strata_covered} strata")
    return probes_arr, meta_df


def _zone_ranges(boundaries: List[int], total: int) -> List[Tuple[int, int]]:
    bs = [0] + list(boundaries) + [total]
    return [(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]


def _read_obs_at(env: NZonePhase3Env, x: int, y: int,
                 body_state: float = 0.5) -> np.ndarray:
    """Read an observation at given (x, y) without perturbation."""
    env.reset(seed=0)
    env.x = int(x)
    env.y = int(y)
    env.body_state = np.array([body_state], dtype=np.float32)
    env._perturbation_active = False
    return env._observe()


# ---------------------------------------------------------------------------
# Forward probes through the agent (with controlled g)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_probes(
    agent: CEARAgent,
    probe_xs: np.ndarray,
    g_states: np.ndarray,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    For every (g, x) pair, single forward through encoder + state head's
    metric machinery (when applicable), record:

      - z_raw   pre-gating (encoder mlp output, tanh applied). g-independent.
      - z       post-encoder representation. In metric_c1 mode, equals z_raw.
                In film/salience modes, gating applied here.
      - mode-dependent gating params:
          film     → gamma, beta
          salience → salience
          metric_c1 → M_g (z_dim, z_dim), and z_metric = M_g @ z_raw
          both     → gamma, beta, salience

    For metric_c1 mode, the *probe-dependent reorganization* signature lives
    in z_quad = z_raw_outer (M_g @ z_raw). We compute it for analysis.

    Critical for analysis:
      - In film/salience modes, "z" is the meaningful post-gating output.
      - In metric_c1 mode, the agent's gating effect appears in z_metric
        (linear in z_raw) and z_quad (quadratic in z_raw — probe-dependent).
        We expose all three (z=z_raw, z_metric, z_quad) so analysis code
        can pick the right one.
    """
    agent.eval()
    n_g = g_states.shape[0]
    n_p = probe_xs.shape[0]
    z_dim = agent.cfg.encoder.z_dim

    Z = np.zeros((n_g, n_p, z_dim), dtype=np.float32)
    Z_RAW = np.zeros((n_g, n_p, z_dim), dtype=np.float32)
    Z_METRIC = np.zeros((n_g, n_p, z_dim), dtype=np.float32)
    Z_QUAD = np.zeros((n_g, n_p, z_dim * z_dim), dtype=np.float32)
    GAMMA = np.zeros((n_g, n_p, z_dim), dtype=np.float32)
    BETA = np.zeros((n_g, n_p, z_dim), dtype=np.float32)
    SALIENCE = np.zeros((n_g, n_p, z_dim), dtype=np.float32)
    M_G = np.zeros((n_g, n_p, z_dim, z_dim), dtype=np.float32)

    obs_enc = agent.enc.obs_enc
    mode = obs_enc.gating_mode
    g_tensor_full = torch.tensor(g_states, dtype=torch.float32, device=device)
    x_tensor_full = torch.tensor(probe_xs, dtype=torch.float32, device=device)

    for gi in range(n_g):
        g = g_tensor_full[gi:gi+1]
        gp = obs_enc.gating_params(g)

        for pi in range(n_p):
            x = x_tensor_full[pi:pi+1]
            z = obs_enc(x, g_t=g)               # post-gating in film/salience
            z_raw = torch.tanh(obs_enc.mlp(x))   # encoder mlp output

            Z[gi, pi]     = z.cpu().numpy()[0]
            Z_RAW[gi, pi] = z_raw.cpu().numpy()[0]

            if "gamma" in gp:
                GAMMA[gi, pi] = gp["gamma"].cpu().numpy()[0]
            if "beta" in gp:
                BETA[gi, pi]  = gp["beta"].cpu().numpy()[0]
            if "salience" in gp:
                SALIENCE[gi, pi] = gp["salience"].cpu().numpy()[0]
            if "M_g" in gp:
                M = gp["M_g"]                    # (1, z_dim, z_dim)
                M_G[gi, pi] = M.cpu().numpy()[0]
                # z_metric = M_g @ z_raw         (linear in z_raw)
                Mz = torch.bmm(M, z_raw.unsqueeze(-1)).squeeze(-1)  # (1, z_dim)
                Z_METRIC[gi, pi] = Mz.cpu().numpy()[0]
                # z_quad[i, j] = z_raw[i] * (M_g z_raw)[j]   (quadratic)
                z_outer = z_raw.unsqueeze(-1) * Mz.unsqueeze(-2)
                Z_QUAD[gi, pi] = z_outer.reshape(1, -1).cpu().numpy()[0]

    return {
        "z": Z,
        "z_raw": Z_RAW,
        "z_metric": Z_METRIC,
        "z_quad": Z_QUAD,
        "gamma": GAMMA,
        "beta": BETA,
        "salience": SALIENCE,
        "M_g": M_G,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Q1: Same x / different g → different gating?
# ---------------------------------------------------------------------------

def _select_target(out: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Select the meaningful measurement target based on mode:
      - film/salience/both: 'z' (post-gating output)
      - metric_c1:          'z_quad' (quadratic features — where probe-dependent
                            reorganization actually lives in this architecture)

    For metric_c1, returning z (= z_raw) would show no g effect at all, since
    z is unmodulated in this mode. The phenomenological gating effect is
    realized when state head consumes z_quad downstream — that's the actual
    representation conditioning agent behavior.
    """
    mode = out.get("mode", "film")
    if mode == "metric_c1":
        return out["z_quad"]
    return out["z"]


def q1_summary(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Q1: does same x produce different processed-z under different g?

    Primary metric (target = z_quad in metric_c1, z elsewhere):
      var across g states, per probe, then averaged.

    Sanity: variance of z_raw across g should be ~0 (encoder mlp doesn't see g).
    Reference: variance of target across probes (purely content-driven variance).

    g_to_probe_ratio compares the magnitude of g-induced variation to
    probe-content variation.
    """
    target = _select_target(out)
    Z_RAW = out["z_raw"]
    mode = out.get("mode", "film")

    var_target_across_g     = float(target.var(axis=0).mean())
    var_zraw_across_g       = float(Z_RAW.var(axis=0).mean())  # sanity: ~0
    var_target_across_probes = float(target.var(axis=1).mean())
    var_zraw_across_probes  = float(Z_RAW.var(axis=1).mean())
    g_to_probe_ratio = var_target_across_g / max(var_target_across_probes, 1e-9)

    result = {
        "mode": mode,
        "target": "z_quad" if mode == "metric_c1" else "z",
        "var_target_across_g": var_target_across_g,
        "var_zraw_across_g_sanity": var_zraw_across_g,
        "var_target_across_probes": var_target_across_probes,
        "var_zraw_across_probes": var_zraw_across_probes,
        "g_to_probe_ratio": g_to_probe_ratio,
    }

    # Mode-specific gating param diagnostics
    if mode in ("film", "both"):
        result["var_gamma_across_g"] = float(out["gamma"].var(axis=0).mean())
        result["var_beta_across_g"]  = float(out["beta"].var(axis=0).mean())
    if mode in ("salience", "both"):
        result["var_salience_across_g"] = float(out["salience"].var(axis=0).mean())
        result["mean_salience"] = float(out["salience"].mean())
        result["std_salience"]  = float(out["salience"].std())
    if mode == "metric_c1":
        # M_g eigenstructure summary: anisotropy across g states
        M = out["M_g"]   # (n_g, n_p, z_dim, z_dim) — but constant across probes
        # use g-axis only (collapse probe axis since M_g is g-only function)
        M_per_g = M[:, 0, :, :]   # (n_g, z_dim, z_dim)
        eigs = np.linalg.eigvalsh(M_per_g)   # (n_g, z_dim) ascending
        result["mean_max_eigenvalue"] = float(eigs[:, -1].mean())
        result["mean_min_eigenvalue"] = float(eigs[:, 0].mean())
        result["mean_condition_number"] = float(
            (eigs[:, -1] / np.maximum(eigs[:, 0], 1e-9)).mean()
        )
        # variance of eigenvalues across g states — how much M_g changes with stance
        result["var_max_eigenvalue_across_g"] = float(eigs[:, -1].var())

    return result


# ---------------------------------------------------------------------------
# Q2: Systematic vs random?
# ---------------------------------------------------------------------------

def q2_consistency(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    For each (g_A, g_B) pair: per-probe delta = target(g_A) - target(g_B).
    Mean pairwise cosine similarity of delta vectors across probes.

    Target: z (film/salience/both) or z_quad (metric_c1) — the representation
    that actually conditions agent's downstream behavior.

    cos ≈ 1 → probe-independent shift in latent space
    cos < 1 → probe-dependent component exists (Frame 1's stronger signature)
    cos ≈ 0 → strongly probe-dependent

    metric_c1's expectation: z_quad is *quadratic* in z_raw, so per-probe
    delta vectors should be more probe-dependent (lower consistency) than
    in film/salience modes where target is linear in z_raw.
    """
    target = _select_target(out)
    n_g, n_p, _ = target.shape

    pairwise = []
    for ga in range(n_g):
        for gb in range(ga + 1, n_g):
            deltas = target[ga] - target[gb]
            norms = np.linalg.norm(deltas, axis=-1, keepdims=True) + 1e-9
            unit = deltas / norms
            cos = unit @ unit.T
            mask = ~np.eye(n_p, dtype=bool)
            mean_cos = float(cos[mask].mean())
            pairwise.append(mean_cos)

    return {
        "mean_pairwise_consistency": float(np.mean(pairwise)),
        "std_pairwise_consistency": float(np.std(pairwise)),
        "min_pairwise_consistency": float(np.min(pairwise)),
        "max_pairwise_consistency": float(np.max(pairwise)),
    }


def q2_linear_transform(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    For each pair (g_A, g_B): fit linear map target(g_B, x) ≈ W target(g_A, x) + b.
    R^2 across probes per pair.

    Architecture-dependent interpretation:

      mode="film"/"salience"/"both": target is linear-in-z_raw, so by
        construction R^2 = 1 trivially. Reported only for completeness.

      mode="metric_c1": target = z_quad is *quadratic* in z_raw. A linear
        map between z_quad(g_A, x) and z_quad(g_B, x) is NO LONGER guaranteed
        to fit perfectly. R^2 < 1 here is meaningful: it quantifies how much
        the g-induced transformation differs from a single global linear map
        across probes — i.e., how much probe-dependent reorganization occurs.

    Caveat for film/salience: even though R^2 ≈ 1, the consistency metric
    (Q2(a)) still captures probe-dependence via cosine alignment of deltas.
    Both metrics should be read together.
    """
    target = _select_target(out)
    n_g, n_p, d = target.shape

    r2 = []
    for ga in range(n_g):
        for gb in range(ga + 1, n_g):
            X = target[ga]
            Y = target[gb]
            X_aug = np.concatenate([X, np.ones((n_p, 1))], axis=1)
            sol, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
            Y_hat = X_aug @ sol
            ss_res = ((Y - Y_hat) ** 2).sum()
            ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
            r2.append(1.0 - ss_res / max(ss_tot, 1e-9))

    return {
        "mean_linear_R2": float(np.mean(r2)),
        "std_linear_R2": float(np.std(r2)),
        "min_linear_R2": float(np.min(r2)),
        "max_linear_R2": float(np.max(r2)),
    }


def q2_cluster_structure(out: Dict[str, np.ndarray],
                         meta_g: pd.DataFrame) -> Dict[str, float]:
    """
    Average target across probes → one descriptor per g.
    Compute silhouette using candidate labels (body, perturb, valence_zone).

    silhouette > 0.3 → meaningful clustering by that label.
    """
    target = _select_target(out)
    g_descriptors = target.mean(axis=1)

    results = {}
    if "body_state" in meta_g.columns:
        body = meta_g["body_state"].values
        labels = pd.cut(body, bins=[-0.01, 0.33, 0.67, 1.01],
                        labels=[0, 1, 2])
        labels = np.asarray(labels.astype(int))
        results["silhouette_body"] = float(_silhouette(g_descriptors, labels))

    if "perturbation_active" in meta_g.columns:
        labels = meta_g["perturbation_active"].values.astype(int)
        results["silhouette_perturb"] = float(_silhouette(g_descriptors, labels))

    if "valence_zone_id" in meta_g.columns:
        labels = meta_g["valence_zone_id"].values.astype(int)
        results["silhouette_valence_zone"] = float(_silhouette(g_descriptors, labels))

    return results


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    n = len(X)
    if n < 4:
        return float("nan")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    sil = []
    for i in range(n):
        same = (labels == labels[i]) & (np.arange(n) != i)
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b_per = []
        for L in uniq:
            if L == labels[i]:
                continue
            mask = labels == L
            if mask.sum() > 0:
                b_per.append(D[i, mask].mean())
        if not b_per:
            continue
        b = min(b_per)
        sil.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(sil)) if sil else float("nan")


def q2_probe_dependence(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Spread of target across g states, per probe.
    CoV high → some probes much more stance-sensitive than others.
    """
    target = _select_target(out)
    spread_per_probe = target.var(axis=0).sum(axis=-1)
    cov = float(spread_per_probe.std() / max(spread_per_probe.mean(), 1e-9))
    return {
        "spread_mean": float(spread_per_probe.mean()),
        "spread_std": float(spread_per_probe.std()),
        "spread_cov": cov,
        "spread_min": float(spread_per_probe.min()),
        "spread_max": float(spread_per_probe.max()),
    }


# ---------------------------------------------------------------------------
# Save raw
# ---------------------------------------------------------------------------

def save_raw(out: Dict[str, np.ndarray],
             meta_g: pd.DataFrame, meta_probes: pd.DataFrame,
             outdir: Path) -> None:
    n_g, n_p, z_dim = out["z"].shape
    mode = out.get("mode", "film")
    rows = []
    for gi in range(n_g):
        for pi in range(n_p):
            row = {
                "g_id": gi, "probe_id": pi,
                "g_episode": int(meta_g.iloc[gi]["episode"]),
                "g_t": int(meta_g.iloc[gi]["t"]),
                "g_body_state": float(meta_g.iloc[gi]["body_state"]),
                "g_perturbation": int(meta_g.iloc[gi]["perturbation_active"]),
                "g_valence_zone": int(meta_g.iloc[gi]["valence_zone_id"]),
                "probe_x": int(meta_probes.iloc[pi]["x"]),
                "probe_y": int(meta_probes.iloc[pi]["y"]),
                "probe_column_zone": int(meta_probes.iloc[pi]["column_zone"]),
                "probe_valence_zone": int(meta_probes.iloc[pi]["valence_zone"]),
                "mode": mode,
            }
            for d in range(z_dim):
                row[f"z_{d}"]     = float(out["z"][gi, pi, d])
                row[f"z_raw_{d}"] = float(out["z_raw"][gi, pi, d])

            if mode in ("film", "both"):
                for d in range(z_dim):
                    row[f"gamma_{d}"] = float(out["gamma"][gi, pi, d])
                    row[f"beta_{d}"]  = float(out["beta"][gi, pi, d])
            if mode in ("salience", "both"):
                for d in range(z_dim):
                    row[f"salience_{d}"] = float(out["salience"][gi, pi, d])
            if mode == "metric_c1":
                for d in range(z_dim):
                    row[f"z_metric_{d}"] = float(out["z_metric"][gi, pi, d])
                # store M_g eigenvalues only (M_g itself = 256 floats too large)
                M = out["M_g"][gi, pi]   # (z_dim, z_dim)
                eigs = np.linalg.eigvalsh(M)  # ascending
                for d in range(z_dim):
                    row[f"M_eig_{d}"] = float(eigs[d])
                # also store the trace and log-determinant as compact summaries
                row["M_trace"] = float(np.trace(M))
                # log-det via Cholesky for stability
                try:
                    L = np.linalg.cholesky(M)
                    row["M_logdet"] = float(2.0 * np.sum(np.log(np.diag(L))))
                except np.linalg.LinAlgError:
                    row["M_logdet"] = float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    out_path = outdir / "probe_results.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[save] {out_path}  ({len(df)} rows)")

    # In metric_c1 mode, also save z_quad as separate parquet (z_dim^2 = 256
    # extra columns would bloat probe_results; downstream analysis can load on
    # demand)
    if mode == "metric_c1":
        n_quad = out["z_quad"].shape[-1]
        rows_q = []
        for gi in range(n_g):
            for pi in range(n_p):
                row = {"g_id": gi, "probe_id": pi}
                for d in range(n_quad):
                    row[f"zq_{d}"] = float(out["z_quad"][gi, pi, d])
                rows_q.append(row)
        df_q = pd.DataFrame(rows_q)
        out_q = outdir / "probe_zquad.parquet"
        df_q.to_parquet(out_q, index=False)
        print(f"[save] {out_q}  ({len(df_q)} rows × {n_quad} z_quad dims)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--traj", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/probe_phase3")
    ap.add_argument("--n_g_states", type=int, default=20)
    ap.add_argument("--n_probes", type=int, default=27)
    ap.add_argument("--g_strategy", type=str, default="stratified",
                    choices=["stratified", "random", "late"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    agent, meta = load_agent(args.ckpt, device=args.device)
    g_states, meta_g = sample_g_states(
        args.traj, n_states=args.n_g_states,
        strategy=args.g_strategy, rng_seed=args.rng_seed,
    )
    probes, meta_probes = build_probe_set(
        meta["env_cfg"], n_probes=args.n_probes, rng_seed=args.rng_seed,
    )
    out = forward_probes(agent, probes, g_states, device=args.device)
    print(f"[forward] z {out['z'].shape}  gamma {out['gamma'].shape}  "
          f"z_raw {out['z_raw'].shape}")

    print("\n=== Q1: Same x / different g → different gating? ===")
    q1 = q1_summary(out)
    for k, v in q1.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.6f}")

    print("\n=== Q2 (a): Per-pair delta consistency across probes ===")
    q2a = q2_consistency(out)
    for k, v in q2a.items():
        print(f"  {k}: {v:+.4f}")

    print("\n=== Q2 (b): Linear transform R^2 between g pairs ===")
    q2b = q2_linear_transform(out)
    for k, v in q2b.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Q2 (c): Cluster structure of g-descriptors ===")
    q2c = q2_cluster_structure(out, meta_g)
    for k, v in q2c.items():
        print(f"  {k}: {v:+.4f}")

    print("\n=== Q2 (d): Probe-dependence of g spread ===")
    q2d = q2_probe_dependence(out)
    for k, v in q2d.items():
        print(f"  {k}: {v:.4f}")

    save_raw(out, meta_g, meta_probes, outdir)
    print("\n[done]")


if __name__ == "__main__":
    main()
