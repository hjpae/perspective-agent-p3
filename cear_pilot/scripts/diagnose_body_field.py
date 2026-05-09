#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diagnose Phase 3 body / affordance field.

Run:
  python scripts/diagnose_body_field.py \
    --outdir outputs/p3_v3_bdec_metric_s0_sil010_lb010 \
    --device cuda \
    --late_episodes 80 \
    --n_per_zone 200

Outputs:
  <outdir>/diagnostics/body_zone_summary.csv
  <outdir>/diagnostics/realized_action_delta_by_zone.csv
  <outdir>/diagnostics/bodydecoder_counterfactuals.csv
  <outdir>/diagnostics/bodydecoder_counterfactual_summary.csv

Main things to inspect:
  - body_state_mean by zone
  - body_delta_mean by zone
  - pred_UP, pred_DOWN, pred_STAY
  - pred_UP_minus_DOWN
  - true_UP_minus_DOWN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "STAY",
}


def zone3_from_xy(x: int, y: int) -> str:
    xz = "L" if x <= 4 else ("M" if x <= 9 else "R")
    yz = "T" if y <= 4 else ("M" if y <= 9 else "B")
    return yz + xz


def valence_name(vz: int) -> str:
    return {0: "top", 1: "mid", 2: "bottom"}.get(int(vz), f"vz{vz}")


def action_to_next_xy(x: int, y: int, action: int, W: int = 15, H: int = 15):
    dx, dy = 0, 0
    if action == 0:      # UP
        dy = -1
    elif action == 1:    # DOWN
        dy = 1
    elif action == 2:    # LEFT
        dx = -1
    elif action == 3:    # RIGHT
        dx = 1
    elif action == 4:    # STAY
        dx, dy = 0, 0
    else:
        raise ValueError(f"unknown action {action}")

    nx = int(np.clip(x + dx, 0, W - 1))
    ny = int(np.clip(y + dy, 0, H - 1))
    return nx, ny


def true_body_next_from_env(env, x: int, y: int, body: float, action: int) -> float:
    """Deterministic body transition according to env's body dynamics."""
    cfg = env.cfg3
    nx, ny = action_to_next_xy(x, y, action, W=env.W, H=env.H)
    moved = action != 4

    metabolic_delta = -float(cfg.metabolic_cost)
    if moved:
        metabolic_delta -= float(cfg.movement_cost)

    affordance = float(env._affordance_map[ny, nx])
    affordance_gain = affordance * float(cfg.affordance_to_body_gain)

    b_next = float(body) + metabolic_delta + affordance_gain
    b_next = float(np.clip(b_next, float(cfg.body_min), float(cfg.body_max)))
    return b_next


def summarize_trajectory(df: pd.DataFrame, outdir: Path, late_episodes: int) -> pd.DataFrame:
    df = df.copy()
    max_ep = int(df["episode"].max())
    df = df[df["episode"] >= max_ep - late_episodes + 1].copy()

    df["zone3"] = [zone3_from_xy(x, y) for x, y in zip(df["x"], df["y"])]
    df["valence_name"] = df["valence_zone_id"].map(valence_name)

    # next body within same episode
    df["body_next_logged"] = df.groupby("episode")["body_state"].shift(-1)
    df["body_delta_logged"] = df["body_next_logged"] - df["body_state"]
    df["body_drive"] = 1.0 - df["body_state"]
    df["action_name"] = df["action"].map(ACTION_NAMES)

    # Zone-level actual body statistics
    zone_summary = (
        df.dropna(subset=["body_delta_logged"])
        .groupby(["valence_name", "zone3"], as_index=False)
        .agg(
            n=("body_state", "size"),
            body_state_mean=("body_state", "mean"),
            body_state_std=("body_state", "std"),
            body_delta_mean=("body_delta_logged", "mean"),
            body_delta_std=("body_delta_logged", "std"),
            body_drive_mean=("body_drive", "mean"),
            body_pe_mean=("body_pe", "mean"),
            obs_pe_mean=("obs_pe", "mean"),
            alpha_mean=("alpha", "mean"),
            g_norm_mean=("g_norm", "mean"),
        )
        .sort_values(["valence_name", "zone3"])
    )

    # Realized action → realized body_delta
    realized_action = (
        df.dropna(subset=["body_delta_logged"])
        .groupby(["valence_name", "zone3", "action_name"], as_index=False)
        .agg(
            n=("body_delta_logged", "size"),
            body_delta_mean=("body_delta_logged", "mean"),
            body_delta_std=("body_delta_logged", "std"),
            body_state_mean=("body_state", "mean"),
            body_next_mean=("body_next_logged", "mean"),
        )
        .sort_values(["valence_name", "zone3", "action_name"])
    )

    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    zone_summary.to_csv(diag_dir / "body_zone_summary.csv", index=False)
    realized_action.to_csv(diag_dir / "realized_action_delta_by_zone.csv", index=False)

    print("\n=== TRAJECTORY: body by valence zone ===")
    print(
        df.dropna(subset=["body_delta_logged"])
        .groupby("valence_name")
        .agg(
            n=("body_state", "size"),
            body_state=("body_state", "mean"),
            body_delta=("body_delta_logged", "mean"),
            body_drive=("body_drive", "mean"),
            body_pe=("body_pe", "mean"),
        )
        .round(5)
    )

    print("\n=== TRAJECTORY: realized body_delta by valence x action ===")
    pivot = (
        realized_action
        .pivot_table(
            index="valence_name",
            columns="action_name",
            values="body_delta_mean",
            aggfunc="mean",
        )
        .reindex(["top", "mid", "bottom"])
    )
    print(pivot.round(5))

    return df


def build_models_from_checkpoint(ckpt_path: Path, device: str):
    """
    Uses the user's current train_phase3_v3.build_agent_and_decoder(args)
    so this stays compatible with local implementation.
    """
    from cear_pilot.training import train_phase3_v3 as tr

    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    args_dict = dict(meta.get("args", {}))
    if not args_dict:
        raise RuntimeError("Checkpoint meta['args'] missing. Cannot rebuild models robustly.")

    args_dict["device"] = device
    args = SimpleNamespace(**args_dict)

    built = tr.build_agent_and_decoder(args)
    if not isinstance(built, tuple):
        raise RuntimeError("build_agent_and_decoder(args) did not return tuple.")

    if len(built) == 3:
        agent, obs_decoder, body_decoder = built
    elif len(built) == 2:
        raise RuntimeError(
            "build_agent_and_decoder returned only (agent, decoder). "
            "Your current code needs to return BodyDecoder too for this diagnostic."
        )
    else:
        raise RuntimeError(f"Unexpected build_agent_and_decoder return length: {len(built)}")

    agent.load_state_dict(ckpt["agent_state"])
    obs_decoder.load_state_dict(ckpt["decoder_state"])

    # Try common keys for body decoder state.
    body_state_key = None
    for k in ["body_decoder_state", "bodydec_state", "body_decoder"]:
        if k in ckpt:
            body_state_key = k
            break

    if body_state_key is None:
        raise RuntimeError(
            "Could not find BodyDecoder state in checkpoint. "
            "Expected key like 'body_decoder_state'."
        )

    body_decoder.load_state_dict(ckpt[body_state_key])

    agent.to(device).eval()
    obs_decoder.to(device).eval()
    body_decoder.to(device).eval()

    return ckpt, args, agent, obs_decoder, body_decoder


def build_env_from_checkpoint(ckpt: Dict[str, Any]):
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Env, NZonePhase3Config

    meta = ckpt.get("meta", {})
    env_cfg_dict = dict(meta.get("env_cfg", {}))
    if not env_cfg_dict:
        raise RuntimeError("Checkpoint meta['env_cfg'] missing.")

    env_cfg = NZonePhase3Config(**env_cfg_dict)
    env = NZonePhase3Env(config=env_cfg)
    return env


@torch.no_grad()
def counterfactual_bodydecoder_diagnostic(
    df: pd.DataFrame,
    outdir: Path,
    ckpt_path: Path,
    device: str,
    n_per_zone: int,
    late_episodes: int,
    seed: int = 0,
):
    ckpt, args, agent, obs_decoder, body_decoder = build_models_from_checkpoint(ckpt_path, device)
    env = build_env_from_checkpoint(ckpt)

    rng = np.random.default_rng(seed)

    max_ep = int(df["episode"].max())
    late = df[df["episode"] >= max_ep - late_episodes + 1].copy()
    late["zone3"] = [zone3_from_xy(x, y) for x, y in zip(late["x"], late["y"])]
    late["valence_name"] = late["valence_zone_id"].map(valence_name)
    late["prev_action"] = late.groupby("episode")["action"].shift(1).fillna(4).astype(int)

    # Stratified sample: top/mid/bottom × zone3 if possible
    samples = []
    group_cols = ["valence_name", "zone3"]
    for _, gdf in late.groupby(group_cols):
        if len(gdf) == 0:
            continue
        take = min(n_per_zone, len(gdf))
        samples.append(gdf.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    sample_df = pd.concat(samples, ignore_index=True) if samples else late.sample(
        n=min(n_per_zone, len(late)), random_state=seed
    )

    rows: List[Dict[str, Any]] = []
    n_actions = 5
    g_cols = sorted(
        [c for c in sample_df.columns if c.startswith("g_") and c[2:].isdigit()],
        key=lambda s: int(s.split("_")[1])
    )

    if not g_cols:
        raise RuntimeError("No g_i columns found in traj parquet. Need save_traj with g columns.")

    for idx, row in sample_df.iterrows():
        # Reset env and impose logged state.
        env.reset(seed=int(row["episode"]))
        env.x = int(row["x"])
        env.y = int(row["y"])
        env.t = int(row["t"])
        env.body_state = np.array([float(row["body_state"])], dtype=np.float32)

        # Best-effort perturbation state.
        if hasattr(env, "_perturbation_active"):
            env._perturbation_active = bool(int(row.get("perturbation_active", 0)))

        obs = env._observe()
        info = env._info_dict()

        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        body_t = torch.tensor([[float(row["body_state"])]], dtype=torch.float32, device=device)

        prev_action = int(row.get("prev_action", 4))
        p_t = F.one_hot(
            torch.tensor([prev_action], device=device),
            num_classes=n_actions
        ).float()

        body_silhouette_t = None
        if "body_silhouette" in info:
            body_silhouette_t = torch.tensor(
                info["body_silhouette"][None, :],
                dtype=torch.float32,
                device=device,
            )

        # Put agent into logged g state.
        g_vec = torch.tensor(
            row[g_cols].to_numpy(dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=device,
        )
        agent.reset(batch_size=1)
        agent.set_g(g_vec)

        # Avoid artificial body PE during this diagnostic.
        if getattr(agent, "_body_pred_prev", None) is not None:
            agent._body_pred_prev = body_t.detach().clone()

        # Zero env PE for diagnostic; we are not probing AlphaNet here.
        err_dim = int(getattr(agent.cfg.world, "err_dim", 6))
        err_t = torch.zeros((1, err_dim), dtype=torch.float32, device=device)

        out = agent.forward_step(
            x_t,
            p_t,
            err_t=err_t,
            body_actual_t=body_t,
            body_silhouette_t=body_silhouette_t,
        )

        # BodyDecoder counterfactual prediction.
        # Expected API: predict_all_actions(z_t, g_t, body_t) -> (B, A, 1)
        try:
            bpred_all = body_decoder.predict_all_actions(out["z"], out["g"], body_t)
        except TypeError:
            # Some implementations may use (z, g, body_t=...)
            bpred_all = body_decoder.predict_all_actions(out["z"], out["g"], body_t=body_t)

        bpred = bpred_all.detach().cpu().numpy()[0, :, 0]

        # True deterministic body consequences under each action.
        true_next = np.array([
            true_body_next_from_env(
                env,
                x=int(row["x"]),
                y=int(row["y"]),
                body=float(row["body_state"]),
                action=a,
            )
            for a in range(n_actions)
        ], dtype=np.float32)

        rows.append({
            "episode": int(row["episode"]),
            "t": int(row["t"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "zone3": zone3_from_xy(int(row["x"]), int(row["y"])),
            "valence_name": valence_name(int(row["valence_zone_id"])),
            "body_state": float(row["body_state"]),
            "affordance_here": float(row.get("affordance_here", np.nan)),
            "logged_action": ACTION_NAMES.get(int(row["action"]), str(row["action"])),

            "pred_UP": float(bpred[0]),
            "pred_DOWN": float(bpred[1]),
            "pred_LEFT": float(bpred[2]),
            "pred_RIGHT": float(bpred[3]),
            "pred_STAY": float(bpred[4]),
            "pred_UP_minus_DOWN": float(bpred[0] - bpred[1]),
            "pred_UP_minus_STAY": float(bpred[0] - bpred[4]),

            "true_UP": float(true_next[0]),
            "true_DOWN": float(true_next[1]),
            "true_LEFT": float(true_next[2]),
            "true_RIGHT": float(true_next[3]),
            "true_STAY": float(true_next[4]),
            "true_UP_minus_DOWN": float(true_next[0] - true_next[1]),
            "true_UP_minus_STAY": float(true_next[0] - true_next[4]),

            "abs_err_UP": float(abs(bpred[0] - true_next[0])),
            "abs_err_DOWN": float(abs(bpred[1] - true_next[1])),
            "abs_err_STAY": float(abs(bpred[4] - true_next[4])),
        })

    cf = pd.DataFrame(rows)

    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    cf.to_csv(diag_dir / "bodydecoder_counterfactuals.csv", index=False)

    summary = (
        cf.groupby(["valence_name", "zone3"], as_index=False)
        .agg(
            n=("body_state", "size"),
            body_state=("body_state", "mean"),
            pred_UP=("pred_UP", "mean"),
            pred_DOWN=("pred_DOWN", "mean"),
            pred_STAY=("pred_STAY", "mean"),
            pred_UP_minus_DOWN=("pred_UP_minus_DOWN", "mean"),
            pred_UP_minus_STAY=("pred_UP_minus_STAY", "mean"),
            true_UP=("true_UP", "mean"),
            true_DOWN=("true_DOWN", "mean"),
            true_STAY=("true_STAY", "mean"),
            true_UP_minus_DOWN=("true_UP_minus_DOWN", "mean"),
            true_UP_minus_STAY=("true_UP_minus_STAY", "mean"),
            abs_err_UP=("abs_err_UP", "mean"),
            abs_err_DOWN=("abs_err_DOWN", "mean"),
            abs_err_STAY=("abs_err_STAY", "mean"),
        )
        .sort_values(["valence_name", "zone3"])
    )
    summary.to_csv(diag_dir / "bodydecoder_counterfactual_summary.csv", index=False)

    print("\n=== BODYDECODER: counterfactual summary by valence ===")
    val_summary = (
        cf.groupby("valence_name")
        .agg(
            n=("body_state", "size"),
            body_state=("body_state", "mean"),
            pred_UP=("pred_UP", "mean"),
            pred_DOWN=("pred_DOWN", "mean"),
            pred_STAY=("pred_STAY", "mean"),
            pred_UP_minus_DOWN=("pred_UP_minus_DOWN", "mean"),
            true_UP_minus_DOWN=("true_UP_minus_DOWN", "mean"),
            abs_err_UP=("abs_err_UP", "mean"),
            abs_err_DOWN=("abs_err_DOWN", "mean"),
        )
        .reindex(["top", "mid", "bottom"])
    )
    print(val_summary.round(5))

    print("\n=== BODYDECODER: desired directional check ===")
    print("Want pred_UP_minus_DOWN > 0, especially in mid/bottom.")
    print(
        cf.groupby("valence_name")["pred_UP_minus_DOWN"]
        .agg(["mean", "std", "min", "max"])
        .reindex(["top", "mid", "bottom"])
        .round(5)
    )

    print(f"\nSaved diagnostics to: {diag_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--traj", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--late_episodes", type=int, default=80)
    ap.add_argument("--n_per_zone", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_counterfactual", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    traj_path = Path(args.traj) if args.traj else outdir / "traj.parquet"
    ckpt_path = Path(args.ckpt) if args.ckpt else outdir / "ckpt_final.pt"

    if not traj_path.exists():
        raise FileNotFoundError(f"traj not found: {traj_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    df = pd.read_parquet(traj_path)

    # 1) trajectory-only diagnostic
    df_late = summarize_trajectory(df, outdir, late_episodes=args.late_episodes)

    # 2) model counterfactual diagnostic
    if not args.skip_counterfactual:
        counterfactual_bodydecoder_diagnostic(
            df=df,
            outdir=outdir,
            ckpt_path=ckpt_path,
            device=args.device,
            n_per_zone=args.n_per_zone,
            late_episodes=args.late_episodes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()