#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose Phase 3 interoceptive field and BodyDecoder tendency head.

Run from repo root:
  PYTHONPATH=. python cear_pilot/scripts/diagnose_body_field.py \
    --outdir outputs/p3_v3_softu_trend_s0 \
    --device cuda \
    --late_episodes 80 \
    --n_per_zone 200

Main checks:
  1) realized body_state / body_delta / body_u by top-mid-bottom
  2) BodyDecoder next-body counterfactual ordering
  3) BodyDecoder latent tendency counterfactual ordering:
       pred_tendency_UP - pred_tendency_DOWN
       true_tendency_UP - true_tendency_DOWN
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


def zone3_from_xy(x: int, y: int) -> str:
    xz = "L" if x <= 4 else ("M" if x <= 9 else "R")
    yz = "T" if y <= 4 else ("M" if y <= 9 else "B")
    return yz + xz


def valence_name(vz: int) -> str:
    return {0: "top", 1: "mid", 2: "bottom"}.get(int(vz), f"vz{vz}")


def summarize_trajectory(df: pd.DataFrame, outdir: Path, late_episodes: int) -> pd.DataFrame:
    df = df.copy()
    max_ep = int(df["episode"].max())
    df = df[df["episode"] >= max_ep - late_episodes + 1].copy()

    df["zone3"] = [zone3_from_xy(x, y) for x, y in zip(df["x"], df["y"])]
    df["valence_name"] = df["valence_zone_id"].map(valence_name)
    df["action_name"] = df["action"].map(ACTION_NAMES)

    df["body_next_logged"] = df.groupby("episode")["body_state"].shift(-1)
    df["body_delta_logged"] = df["body_next_logged"] - df["body_state"]
    df["body_drive"] = 1.0 - df["body_state"]

    if "body_u" in df.columns:
        df["body_u_next_logged"] = df.groupby("episode")["body_u"].shift(-1)
        df["body_u_delta_logged"] = df["body_u_next_logged"] - df["body_u"]
    else:
        df["body_u"] = np.nan
        df["body_u_delta_logged"] = np.nan

    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    zone_summary = (
        df.dropna(subset=["body_delta_logged"])
        .groupby(["valence_name", "zone3"], as_index=False)
        .agg(
            n=("body_state", "size"),
            body_state_mean=("body_state", "mean"),
            body_u_mean=("body_u", "mean"),
            body_delta_mean=("body_delta_logged", "mean"),
            body_u_delta_mean=("body_u_delta_logged", "mean"),
            body_drive_mean=("body_drive", "mean"),
            body_pe_mean=("body_pe", "mean"),
            body_loss_mean=("body_loss", "mean") if "body_loss" in df.columns else ("body_pe", "mean"),
            body_loss_tendency_mean=("body_loss_tendency", "mean") if "body_loss_tendency" in df.columns else ("body_pe", "mean"),
            tendency_target_mean=("tendency_target", "mean") if "tendency_target" in df.columns else ("body_pe", "mean"),
            tendency_pred_mean=("tendency_pred", "mean") if "tendency_pred" in df.columns else ("body_pe", "mean"),
            obs_pe_mean=("obs_pe", "mean"),
            alpha_mean=("alpha", "mean"),
            g_norm_mean=("g_norm", "mean"),
        )
        .sort_values(["valence_name", "zone3"])
    )
    zone_summary.to_csv(diag_dir / "body_zone_summary.csv", index=False)

    realized_action = (
        df.dropna(subset=["body_delta_logged"])
        .groupby(["valence_name", "zone3", "action_name"], as_index=False)
        .agg(
            n=("body_delta_logged", "size"),
            body_delta_mean=("body_delta_logged", "mean"),
            body_u_delta_mean=("body_u_delta_logged", "mean"),
            body_state_mean=("body_state", "mean"),
            body_next_mean=("body_next_logged", "mean"),
            tendency_target_mean=("tendency_target", "mean") if "tendency_target" in df.columns else ("body_delta_logged", "mean"),
            tendency_pred_mean=("tendency_pred", "mean") if "tendency_pred" in df.columns else ("body_delta_logged", "mean"),
        )
        .sort_values(["valence_name", "zone3", "action_name"])
    )
    realized_action.to_csv(diag_dir / "realized_action_delta_by_zone.csv", index=False)

    print("\n=== TRAJECTORY: body by valence zone ===")
    print(
        df.dropna(subset=["body_delta_logged"])
        .groupby("valence_name")
        .agg(
            n=("body_state", "size"),
            body_state=("body_state", "mean"),
            body_u=("body_u", "mean"),
            body_delta=("body_delta_logged", "mean"),
            body_u_delta=("body_u_delta_logged", "mean"),
            body_drive=("body_drive", "mean"),
            body_pe=("body_pe", "mean"),
            trend_target=("tendency_target", "mean") if "tendency_target" in df.columns else ("body_pe", "mean"),
        )
        .reindex(["top", "mid", "bottom"])
        .round(5)
    )

    print("\n=== TRAJECTORY: realized body_delta by valence x action ===")
    pivot = realized_action.pivot_table(
        index="valence_name", columns="action_name", values="body_delta_mean", aggfunc="mean"
    ).reindex(["top", "mid", "bottom"])
    print(pivot.round(5))

    print("\n=== TRAJECTORY: realized body_u_delta by valence x action ===")
    pivot_u = realized_action.pivot_table(
        index="valence_name", columns="action_name", values="body_u_delta_mean", aggfunc="mean"
    ).reindex(["top", "mid", "bottom"])
    print(pivot_u.round(5))

    # Optional conative diagnostics from the training trajectory. These are
    # available only when train_phase3_v3 was run with --w_conative > 0.
    con_cols = [
        "conative_loss", "conative_target_entropy", "conative_pref_action",
        "conative_pref_UP", "conative_pref_DOWN", "conative_pref_LEFT",
        "conative_pref_RIGHT", "conative_pref_STAY",
    ]
    if all(c in df.columns for c in con_cols):
        con_summary = (
            df.groupby("valence_name")
            .agg(
                n=("body_state", "size"),
                conative_loss=("conative_loss", "mean"),
                q_entropy=("conative_target_entropy", "mean"),
                q_UP=("conative_pref_UP", "mean"),
                q_DOWN=("conative_pref_DOWN", "mean"),
                q_LEFT=("conative_pref_LEFT", "mean"),
                q_RIGHT=("conative_pref_RIGHT", "mean"),
                q_STAY=("conative_pref_STAY", "mean"),
            )
            .reindex(["top", "mid", "bottom"])
        )
        print("\n=== CONATIVE TARGET q(a) by valence zone ===")
        print(con_summary.round(5))
        con_summary.to_csv(diag_dir / "conative_target_by_valence.csv")

    return df


def build_models_from_checkpoint(ckpt_path: Path, device: str):
    from cear_pilot.training import train_phase3_v3 as tr

    ckpt = torch.load(ckpt_path, map_location=device)
    args_dict = dict(ckpt.get("meta", {}).get("args", {}))
    if not args_dict:
        raise RuntimeError("Checkpoint meta['args'] missing.")
    args_dict["device"] = device
    args = SimpleNamespace(**args_dict)

    built = tr.build_agent_and_decoder(args)
    if len(built) != 3:
        raise RuntimeError("Expected build_agent_and_decoder(args) -> (agent, decoder, body_decoder).")
    agent, obs_decoder, body_decoder = built

    agent.load_state_dict(ckpt["agent_state"])
    obs_decoder.load_state_dict(ckpt["decoder_state"])
    if body_decoder is None:
        raise RuntimeError("BodyDecoder is None; cannot run counterfactual diagnostic.")
    if "body_decoder_state" not in ckpt:
        raise RuntimeError("Checkpoint missing 'body_decoder_state'.")
    body_decoder.load_state_dict(ckpt["body_decoder_state"])

    agent.to(device).eval()
    obs_decoder.to(device).eval()
    body_decoder.to(device).eval()
    return ckpt, args, agent, obs_decoder, body_decoder


def build_env_from_checkpoint(ckpt: Dict[str, Any]):
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Env, NZonePhase3Config
    env_cfg = NZonePhase3Config(**dict(ckpt["meta"]["env_cfg"]))
    return NZonePhase3Env(config=env_cfg)


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

    samples = []
    for _, gdf in late.groupby(["valence_name", "zone3"]):
        take = min(n_per_zone, len(gdf))
        if take:
            samples.append(gdf.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    sample_df = pd.concat(samples, ignore_index=True) if samples else late.sample(n=min(n_per_zone, len(late)), random_state=seed)

    g_cols = sorted([c for c in sample_df.columns if c.startswith("g_") and c[2:].isdigit()], key=lambda s: int(s.split("_")[1]))
    if not g_cols:
        raise RuntimeError("No g_i columns found in traj parquet.")

    rows: List[Dict[str, Any]] = []
    n_actions = 5
    err_dim = int(getattr(agent.cfg.world, "err_dim", 6))

    for _, row in sample_df.iterrows():
        env.reset(seed=int(row["episode"]))
        env.x = int(row["x"])
        env.y = int(row["y"])
        env.t = int(row["t"])
        if "body_u" in row and not pd.isna(row["body_u"]):
            env.body_u = float(row["body_u"])
            env.body_state = np.array([1.0 / (1.0 + np.exp(-env.body_u))], dtype=np.float32)
        else:
            env.body_state = np.array([float(row["body_state"])], dtype=np.float32)
            # best-effort logit if the old traj has no body_u
            b = float(np.clip(row["body_state"], 1e-6, 1 - 1e-6))
            env.body_u = float(np.log(b / (1 - b)))
        if hasattr(env, "_last_body_u"):
            env._last_body_u = float(env.body_u)

        if hasattr(env, "_perturbation_active"):
            env._perturbation_active = bool(int(row.get("perturbation_active", 0)))

        obs = env._observe()
        info = env._info_dict()
        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        body_t = torch.tensor([[float(env.body_state[0])]], dtype=torch.float32, device=device)
        p_t = F.one_hot(torch.tensor([int(row.get("prev_action", 4))], device=device), num_classes=n_actions).float()

        sil_t = None
        if "body_silhouette" in info:
            sil_t = torch.tensor(info["body_silhouette"][None, :], dtype=torch.float32, device=device)

        g_vec = torch.tensor(row[g_cols].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32, device=device)
        agent.reset(batch_size=1)
        agent.set_g(g_vec)
        if getattr(agent, "_body_pred_prev", None) is not None:
            agent._body_pred_prev = body_t.detach().clone()

        err_t = torch.zeros((1, err_dim), dtype=torch.float32, device=device)
        out = agent.forward_step(x_t, p_t, err_t=err_t, body_actual_t=body_t, body_silhouette_t=sil_t)

        body_pred_all, tendency_all = body_decoder.predict_all_actions(out["z"], out["g"], body_t)
        bpred = body_pred_all.detach().cpu().numpy()[0, :, 0]
        tpred = tendency_all.detach().cpu().numpy()[0, :, 0]

        true_next = []
        true_trend = []
        # For true_next, simulate one env step in copied local variables using env's helper logic indirectly.
        for a in range(n_actions):
            # true tendency over horizon
            true_trend.append(float(env.counterfactual_body_tendency(a, horizon=int(getattr(args, "body_tendency_horizon", 5)))))
            # true one-step body: reproduce one update from current state without mutation
            x0, y0 = int(env.x), int(env.y)
            u = float(env.body_u)
            dx, dy = 0, 0
            if a == env.ACTION_UP: dy = -1
            elif a == env.ACTION_DOWN: dy = 1
            elif a == env.ACTION_LEFT: dx = -1
            elif a == env.ACTION_RIGHT: dx = 1
            nx, ny = env._clip_xy(x0 + dx, y0 + dy)
            moved = a != env.ACTION_STAY
            metabolic_delta = -env.cfg3.metabolic_cost - (env.cfg3.movement_cost if moved else 0.0)
            affordance_gain = float(env._affordance_map[ny, nx]) * float(env.cfg3.affordance_to_body_gain)
            u1 = float(env.cfg3.body_u_decay) * u + metabolic_delta + affordance_gain
            true_next.append(float(1.0 / (1.0 + np.exp(-u1))))

        rows.append({
            "episode": int(row["episode"]),
            "t": int(row["t"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "zone3": zone3_from_xy(int(row["x"]), int(row["y"])),
            "valence_name": valence_name(int(row["valence_zone_id"])),
            "body_state": float(body_t.item()),
            "body_u": float(env.body_u),
            "logged_action": ACTION_NAMES.get(int(row["action"]), str(row["action"])),
            "pred_UP": float(bpred[0]), "pred_DOWN": float(bpred[1]), "pred_STAY": float(bpred[4]),
            "pred_UP_minus_DOWN": float(bpred[0] - bpred[1]),
            "true_UP": float(true_next[0]), "true_DOWN": float(true_next[1]), "true_STAY": float(true_next[4]),
            "true_UP_minus_DOWN": float(true_next[0] - true_next[1]),
            "pred_trend_UP": float(tpred[0]), "pred_trend_DOWN": float(tpred[1]), "pred_trend_STAY": float(tpred[4]),
            "pred_trend_UP_minus_DOWN": float(tpred[0] - tpred[1]),
            "true_trend_UP": float(true_trend[0]), "true_trend_DOWN": float(true_trend[1]), "true_trend_STAY": float(true_trend[4]),
            "true_trend_UP_minus_DOWN": float(true_trend[0] - true_trend[1]),
            "abs_err_trend_UP": float(abs(tpred[0] - true_trend[0])),
            "abs_err_trend_DOWN": float(abs(tpred[1] - true_trend[1])),
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
            body_u=("body_u", "mean"),
            pred_UP_minus_DOWN=("pred_UP_minus_DOWN", "mean"),
            true_UP_minus_DOWN=("true_UP_minus_DOWN", "mean"),
            pred_trend_UP_minus_DOWN=("pred_trend_UP_minus_DOWN", "mean"),
            true_trend_UP_minus_DOWN=("true_trend_UP_minus_DOWN", "mean"),
            abs_err_trend_UP=("abs_err_trend_UP", "mean"),
            abs_err_trend_DOWN=("abs_err_trend_DOWN", "mean"),
        )
        .sort_values(["valence_name", "zone3"])
    )
    summary.to_csv(diag_dir / "bodydecoder_counterfactual_summary.csv", index=False)

    print("\n=== BODYDECODER: next-body counterfactual summary by valence ===")
    print(
        cf.groupby("valence_name")
        .agg(
            n=("body_state", "size"),
            body_state=("body_state", "mean"),
            pred_UP_minus_DOWN=("pred_UP_minus_DOWN", "mean"),
            true_UP_minus_DOWN=("true_UP_minus_DOWN", "mean"),
        )
        .reindex(["top", "mid", "bottom"])
        .round(5)
    )

    print("\n=== BODYDECODER: TENDENCY counterfactual summary by valence ===")
    print(
        cf.groupby("valence_name")
        .agg(
            n=("body_state", "size"),
            body_u=("body_u", "mean"),
            pred_trend_UP_minus_DOWN=("pred_trend_UP_minus_DOWN", "mean"),
            true_trend_UP_minus_DOWN=("true_trend_UP_minus_DOWN", "mean"),
            abs_err_trend_UP=("abs_err_trend_UP", "mean"),
            abs_err_trend_DOWN=("abs_err_trend_DOWN", "mean"),
        )
        .reindex(["top", "mid", "bottom"])
        .round(5)
    )

    print("\n=== TENDENCY directional check ===")
    print("Want pred_trend_UP_minus_DOWN > 0 and aligned with true_trend_UP_minus_DOWN.")
    print(
        cf.groupby("valence_name")["pred_trend_UP_minus_DOWN"]
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
    summarize_trajectory(df, outdir, late_episodes=args.late_episodes)
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
