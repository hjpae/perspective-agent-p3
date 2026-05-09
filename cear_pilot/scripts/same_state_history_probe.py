#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Same-state history-g probe for Phase 3 perturbation assays.

Question
--------
Do different perturbation-history g states reorganize the *same* input world?

This script compares condition-specific history g vectors from assay trajectories:

  g_control_recovery  vs  g_body_shock_recovery

Then it injects those g vectors into the same fixed probe states and measures:

  - BodyDecoder trend field: pred_trend_UP - pred_trend_DOWN
  - conative target q_UP - q_DOWN
  - metric spectrum summaries from encoder.gating_params(g)
  - state vector s distance, if available

This is the stronger CEAR-style test:

  same input + different perturbation-history g -> different organization?

Example
-------
PYTHONPATH=. python cear_pilot/scripts/same_state_history_probe.py \
  --base outputs/p3_v3_assay \
  --train_base outputs/p3_v3_cohorts \
  --seed_start 0 --seed_end 29 \
  --outdir outputs/p3_v3_same_state_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_COHORTS = ["full", "no_body_in_g", "no_conative"]
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return columns like g_0, g_1, ... or M_g_eig_0, ...
    Excludes non-index columns such as g_norm, g_delta_norm.
    """
    cols = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        suffix = c.split("_")[-1]
        if suffix.isdigit():
            cols.append(c)
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def build_models_from_checkpoint(ckpt_path: Path, device: str):
    from cear_pilot.training import train_phase3_v3_conative as tr
    ckpt = torch.load(ckpt_path, map_location=device)
    args_dict = dict(ckpt.get("meta", {}).get("args", {}))
    if not args_dict:
        raise RuntimeError(f"Checkpoint missing meta.args: {ckpt_path}")
    args_dict["device"] = device
    args = SimpleNamespace(**args_dict)
    agent, decoder, body_decoder = tr.build_agent_and_decoder(args)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    if body_decoder is not None:
        body_decoder.load_state_dict(ckpt["body_decoder_state"])
    agent.to(device).eval()
    decoder.to(device).eval()
    if body_decoder is not None:
        body_decoder.to(device).eval()
    return ckpt, args, agent, decoder, body_decoder


def build_env_from_checkpoint(ckpt: Dict):
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Env, NZonePhase3Config
    env_cfg_dict = dict(ckpt.get("meta", {}).get("env_cfg", {}))
    if not env_cfg_dict:
        raise RuntimeError("Checkpoint missing meta.env_cfg")
    env = NZonePhase3Env(config=NZonePhase3Config(**env_cfg_dict))
    return env


def load_history_g(assay_base: Path, cohort: str, seed: int, condition: str, history_phase: str, strategy: str) -> Optional[np.ndarray]:
    traj_path = assay_base / cohort / f"s{seed}" / condition / "assay_traj.parquet"
    if not traj_path.exists():
        print(f"[missing] {traj_path}")
        return None
    df = pd.read_parquet(traj_path)
    g_cols = [c for c in sorted_prefixed_cols(df, "g_") if c.split("_")[-1].isdigit()]
    if not g_cols:
        raise RuntimeError(f"No g columns in {traj_path}")
    if "phase" not in df.columns:
        # Assumes default assay timing.
        def phase(t):
            if t < 60: return "pre"
            if t < 80: return "shock"
            return "recovery"
        df["phase"] = [phase(int(t)) for t in df["t"]]
    d = df[df["phase"] == history_phase]
    if d.empty:
        return None
    if strategy == "mean":
        return d[g_cols].mean().to_numpy(dtype=np.float32)
    if strategy == "last":
        return d[g_cols].iloc[-1].to_numpy(dtype=np.float32)
    if strategy == "early_mean":
        return d.head(max(1, len(d)//3))[g_cols].mean().to_numpy(dtype=np.float32)
    if strategy == "late_mean":
        return d.tail(max(1, len(d)//3))[g_cols].mean().to_numpy(dtype=np.float32)
    raise ValueError(f"unknown history strategy: {strategy}")


def make_probe_specs() -> List[Dict]:
    """Fixed grid/body probe states.

    x positions: left/mid/right; y positions: top/mid/bottom.
    body values: low/mid/high. We keep this modest to avoid huge runtime.
    """
    xs = [(2, "L"), (7, "M"), (12, "R")]
    ys = [(2, "T"), (7, "M"), (12, "B")]
    bodies = [(0.25, "low"), (0.50, "mid"), (0.80, "high")]
    specs = []
    for x, xl in xs:
        for y, yl in ys:
            for b, bl in bodies:
                specs.append({"x": x, "y": y, "body_state": b, "x_zone": xl, "y_zone": yl, "body_level": bl})
    return specs


def impose_env_state(env, spec: Dict, seed: int = 123):
    env.reset(seed=seed)
    env.x = int(spec["x"])
    env.y = int(spec["y"])
    env.t = 0
    # env uses body_u -> body_state. Set both consistently.
    if hasattr(env, "body_u"):
        env.body_u = logit(float(spec["body_state"]))
    env.body_state = np.array([float(spec["body_state"])], dtype=np.float32)
    if hasattr(env, "_last_body_u"):
        env._last_body_u = float(getattr(env, "body_u", logit(float(spec["body_state"]))))
    return env._observe(), env._info_dict()


def metric_stats_from_g(agent, g_t: torch.Tensor) -> Dict[str, float]:
    out = {}
    with torch.no_grad():
        gp = agent.enc.obs_enc.gating_params(g_t.detach())
    if "M_g" in gp:
        M = gp["M_g"][0].detach().cpu().numpy()
        eigs = np.linalg.eigvalsh(M)
        eigs = np.clip(eigs, 1e-12, None)
        p = eigs / eigs.sum()
        out["metric_eig_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        out["metric_cond"] = float(eigs.max() / eigs.min())
        out["metric_top1_mass"] = float(eigs.max() / eigs.sum())
        out["metric_trace"] = float(eigs.sum())
        for i, v in enumerate(eigs):
            out[f"metric_eig_{i}"] = float(v)
    else:
        out["metric_eig_entropy"] = float("nan")
        out["metric_cond"] = float("nan")
        out["metric_top1_mass"] = float("nan")
        out["metric_trace"] = float("nan")
    return out


def conative_q_from_body_outputs(body_pred_all, tendency_all, args):
    # Mirror training conative target. Defaults if args missing.
    temp = float(getattr(args, "conative_temperature", 0.10))
    trend_w = float(getattr(args, "conative_trend_weight", 1.0))
    body_w = float(getattr(args, "conative_body_weight", 0.25))
    field_value = trend_w * tendency_all.squeeze(-1) + body_w * body_pred_all.squeeze(-1)
    q = torch.softmax(field_value / max(temp, 1e-6), dim=-1)
    return q


@torch.no_grad()
def probe_one_seed(
    train_base: Path,
    assay_base: Path,
    cohort: str,
    seed: int,
    device: str,
    history_phase: str,
    history_strategy: str,
    include_pre_history: bool,
) -> Optional[pd.DataFrame]:
    ckpt_path = train_base / cohort / f"s{seed}" / "ckpt_final.pt"
    if not ckpt_path.exists():
        print(f"[missing] {ckpt_path}")
        return None

    g_control = load_history_g(assay_base, cohort, seed, "control", history_phase, history_strategy)
    g_shock = load_history_g(assay_base, cohort, seed, "body_shock", history_phase, history_strategy)
    if g_control is None or g_shock is None:
        return None
    histories = {
        "control": g_control,
        "body_shock": g_shock,
    }
    if include_pre_history:
        g_pre_control = load_history_g(assay_base, cohort, seed, "control", "pre", history_strategy)
        g_pre_shock = load_history_g(assay_base, cohort, seed, "body_shock", "pre", history_strategy)
        if g_pre_control is not None:
            histories["control_pre"] = g_pre_control
        if g_pre_shock is not None:
            histories["body_shock_pre"] = g_pre_shock

    ckpt, args, agent, decoder, body_decoder = build_models_from_checkpoint(ckpt_path, device)
    if body_decoder is None:
        raise RuntimeError("BodyDecoder is required for same-state probe.")
    env = build_env_from_checkpoint(ckpt)

    probe_specs = make_probe_specs()
    n_actions = 5
    rows = []
    err_dim = int(getattr(agent.cfg.world, "err_dim", 6))
    zero_err = torch.zeros((1, err_dim), dtype=torch.float32, device=device)
    p_t = F.one_hot(torch.tensor([4], device=device), num_classes=n_actions).float()  # previous action STAY

    for spec_id, spec in enumerate(probe_specs):
        obs, info = impose_env_state(env, spec, seed=1000 + seed + spec_id)
        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        body_t = torch.tensor([[float(info["body_state"])]], dtype=torch.float32, device=device)
        body_silhouette_t = None
        if "body_silhouette" in info:
            body_silhouette_t = torch.tensor(info["body_silhouette"][None, :], dtype=torch.float32, device=device)

        # For each history g, inject g and run one forward pass with same inputs.
        outputs = {}
        for hist_name, g_np in histories.items():
            agent.reset(batch_size=1)
            g_t = torch.tensor(g_np[None, :], dtype=torch.float32, device=device)
            agent.set_g(g_t)
            if getattr(agent, "_body_pred_prev", None) is not None:
                agent._body_pred_prev = body_t.detach().clone()
            out = agent.forward_step(
                x_t, p_t, err_t=zero_err,
                body_actual_t=body_t,
                body_silhouette_t=body_silhouette_t,
            )
            # Use the injected/evolved out['g']; this reflects one-step organization under the same input.
            body_pred_all, tendency_all = body_decoder.predict_all_actions(out["z"], out["g"], body_t)
            q = conative_q_from_body_outputs(body_pred_all, tendency_all, args)
            trend = tendency_all[0, :, 0].detach().cpu().numpy()
            bpred = body_pred_all[0, :, 0].detach().cpu().numpy()
            q_np = q[0].detach().cpu().numpy()
            met = metric_stats_from_g(agent, out["g"].detach())
            s_np = out["s"].detach().cpu().numpy()[0]
            g_out_np = out["g"].detach().cpu().numpy()[0]
            row = {
                "cohort": cohort,
                "seed": seed,
                "probe_id": spec_id,
                "history": hist_name,
                **spec,
                "body_u_probe": float(info.get("body_u", logit(float(info["body_state"])))),
                "pred_trend_UP": float(trend[0]),
                "pred_trend_DOWN": float(trend[1]),
                "pred_trend_UP_minus_DOWN": float(trend[0] - trend[1]),
                "body_pred_UP": float(bpred[0]),
                "body_pred_DOWN": float(bpred[1]),
                "body_pred_UP_minus_DOWN": float(bpred[0] - bpred[1]),
                "q_UP": float(q_np[0]),
                "q_DOWN": float(q_np[1]),
                "q_LEFT": float(q_np[2]),
                "q_RIGHT": float(q_np[3]),
                "q_STAY": float(q_np[4]),
                "q_UP_minus_DOWN": float(q_np[0] - q_np[1]),
                "q_entropy": float(-(q_np * np.log(q_np + 1e-12)).sum()),
                "g_norm": float(np.linalg.norm(g_out_np)),
                **met,
            }
            for i, v in enumerate(g_out_np):
                row[f"g_out_{i}"] = float(v)
            for i, v in enumerate(s_np):
                row[f"s_{i}"] = float(v)
            rows.append(row)
            outputs[hist_name] = row

    return pd.DataFrame(rows)


def summarize_probe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Long summary by cohort/history/probe y-zone/body-level.
    agg_cols = [
        "pred_trend_UP_minus_DOWN", "body_pred_UP_minus_DOWN", "q_UP_minus_DOWN", "q_entropy",
        "metric_eig_entropy", "metric_cond", "metric_top1_mass", "metric_trace", "g_norm",
    ]
    agg_cols = [c for c in agg_cols if c in df.columns]
    by_hist = (
        df.groupby(["cohort", "seed", "history", "y_zone", "body_level"], as_index=False)
        .agg({c: "mean" for c in agg_cols})
    )

    # Paired shock-control differences for same cohort/seed/probe_id.
    rows = []
    value_cols = agg_cols + [c for c in df.columns if c.startswith("metric_eig_") and c.split("_")[-1].isdigit()]
    s_cols = [c for c in df.columns if c.startswith("s_") and c[2:].isdigit()]
    g_cols = [c for c in df.columns if c.startswith("g_out_")]
    for (cohort, seed, probe_id), grp in df.groupby(["cohort", "seed", "probe_id"]):
        if set(grp["history"]) < {"control", "body_shock"}:
            continue
        ctrl = grp[grp["history"] == "control"].iloc[0]
        shock = grp[grp["history"] == "body_shock"].iloc[0]
        row = {
            "cohort": cohort,
            "seed": seed,
            "probe_id": probe_id,
            "x": ctrl["x"], "y": ctrl["y"],
            "x_zone": ctrl["x_zone"], "y_zone": ctrl["y_zone"], "body_level": ctrl["body_level"],
            "body_state": ctrl["body_state"],
        }
        for c in value_cols:
            if c in ctrl.index and c in shock.index:
                row[f"shock_minus_control.{c}"] = float(shock[c] - ctrl[c])
        if s_cols:
            row["s_distance_shock_control"] = float(np.linalg.norm(shock[s_cols].to_numpy(float) - ctrl[s_cols].to_numpy(float)))
        if g_cols:
            row["g_distance_shock_control"] = float(np.linalg.norm(shock[g_cols].to_numpy(float) - ctrl[g_cols].to_numpy(float)))
        rows.append(row)
    paired = pd.DataFrame(rows)
    return by_hist, paired


def print_mean_se(df: pd.DataFrame, cols: List[str], title: str) -> None:
    print(f"\n=== {title} ===")
    for cohort, g in df.groupby("cohort"):
        print(f"\n[{cohort}] n={len(g)}")
        for c in cols:
            if c not in g.columns:
                continue
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(x) == 0:
                continue
            se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            print(f"  {c:48s} {x.mean(): .5f} ± {se:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="outputs/p3_v3_assay", help="Assay output base containing cohort/sSeed/condition.")
    ap.add_argument("--train_base", type=str, default="outputs/p3_v3_cohorts", help="Training checkpoint base.")
    ap.add_argument("--cohorts", nargs="+", default=DEFAULT_COHORTS)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=29)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--history_phase", type=str, default="recovery", choices=["pre", "shock", "recovery"])
    ap.add_argument("--history_strategy", type=str, default="mean", choices=["mean", "last", "early_mean", "late_mean"])
    ap.add_argument("--include_pre_history", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    assay_base = Path(args.base)
    train_base = Path(args.train_base)
    outdir = Path(args.outdir) if args.outdir else assay_base / "same_state_history_probe"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for cohort in args.cohorts:
        for seed in range(args.seed_start, args.seed_end + 1):
            print(f"[probe] {cohort} s{seed}")
            d = probe_one_seed(
                train_base=train_base,
                assay_base=assay_base,
                cohort=cohort,
                seed=seed,
                device=args.device,
                history_phase=args.history_phase,
                history_strategy=args.history_strategy,
                include_pre_history=args.include_pre_history,
            )
            if d is not None and not d.empty:
                all_rows.append(d)
    if not all_rows:
        raise SystemExit("No probe rows created.")
    df = pd.concat(all_rows, ignore_index=True)
    samples_path = outdir / f"same_state_history_probe_samples_{args.history_phase}_{args.history_strategy}.csv"
    df.to_csv(samples_path, index=False)

    by_hist, paired = summarize_probe(df)
    by_hist_path = outdir / f"same_state_history_probe_by_history_{args.history_phase}_{args.history_strategy}.csv"
    paired_path = outdir / f"same_state_history_probe_shock_control_diff_{args.history_phase}_{args.history_strategy}.csv"
    by_hist.to_csv(by_hist_path, index=False)
    paired.to_csv(paired_path, index=False)

    print(f"[save] {samples_path}")
    print(f"[save] {by_hist_path}")
    print(f"[save] {paired_path}")

    print_mean_se(
        paired,
        cols=[
            "shock_minus_control.pred_trend_UP_minus_DOWN",
            "shock_minus_control.q_UP_minus_DOWN",
            "shock_minus_control.metric_eig_entropy",
            "shock_minus_control.metric_cond",
            "shock_minus_control.metric_top1_mass",
            "shock_minus_control.metric_trace",
            "s_distance_shock_control",
            "g_distance_shock_control",
        ],
        title="SAME-STATE EFFECT OF BODY-SHOCK HISTORY G: shock - control",
    )

    print("\nInterpretation hints:")
    print("  This is the core CEAR probe: same probe input, different perturbation-history g.")
    print("  Larger full than no_body_in_g shifts suggest body shock is incorporated into perspective geometry.")
    print("  If shifts are small/equal, body_in_g may matter only under stronger shocks or longer recovery/probe protocols.")


if __name__ == "__main__":
    main()
