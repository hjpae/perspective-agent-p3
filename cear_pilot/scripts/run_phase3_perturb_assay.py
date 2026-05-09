#cear_pilot/scripts/run_phase3_perturb_assay.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frozen perturbation assay for Phase 3 CEAR agents.

Purpose:
  Test perturbation response, recovery, and residue after training.
  The agent parameters are frozen. The recurrent perspective g is allowed to
  evolve online during the assay via agent.forward_step, but no gradients or
  optimizer updates occur.

Single run example:
  PYTHONPATH=. python cear_pilot/scripts/run_phase3_perturb_assay.py \
    --ckpt outputs/p3_v3_cohorts/full/s0/ckpt_final.pt \
    --condition body_shock \
    --steps 160 \
    --shock_start 60 \
    --shock_duration 20 \
    --body_u_shock_delta -0.08 \
    --device cuda \
    --outdir outputs/p3_v3_assay/full/s0/body_shock

Batch example:
  PYTHONPATH=. python cear_pilot/scripts/run_phase3_perturb_assay.py \
    --base outputs/p3_v3_cohorts \
    --cohorts full no_body_in_g no_conative \
    --seed_start 0 --seed_end 2 \
    --conditions control body_shock env_shock \
    --device cuda \
    --assay_base outputs/p3_v3_assay_smoke

Conditions:
  control     : no externally imposed perturbation; random env perturbations disabled.
  body_shock  : during shock window, env.body_u += body_u_shock_delta each step.
                Agent receives only body_state = sigmoid(body_u), not body_u.
  env_shock   : during shock window, exteroceptive perturbation distortion is forced on.

Notes:
  - body_u_shock_delta is an assay perturbation magnitude, not a loss weight.
  - BodyDecoder/conative q are computed only for diagnostics; q is not directly
    injected into policy at assay time. The trained policy already contains the
    conative attunement learned during training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
ERR_DIM = 6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def zone3(x: int, y: int) -> str:
    xz = "L" if x <= 4 else ("M" if x <= 9 else "R")
    yz = "T" if y <= 4 else ("M" if y <= 9 else "B")
    return yz + xz


def valence_name(v: int) -> str:
    return {0: "top", 1: "mid", 2: "bottom"}.get(int(v), f"vz{v}")


def build_err_t(pred_err: float,
                pred_err_ema_short: float,
                pred_err_ema_long: float,
                pred_err_prev: float,
                perturbation_active: int,
                perturbation_trace: float,
                device: str) -> torch.Tensor:
    ema_safe = max(float(pred_err_ema_long), 1e-6)
    feats = [
        min(float(pred_err) / ema_safe, 5.0),
        min(float(pred_err_ema_short) / ema_safe, 5.0),
        float(np.log1p(pred_err_ema_long)),
        float(np.tanh((float(pred_err) - float(pred_err_prev)) * 10.0)),
        float(perturbation_active),
        float(perturbation_trace),
    ]
    return torch.tensor([feats], dtype=torch.float32, device=device)


def load_training_module():
    try:
        from cear_pilot.training import train_phase3_v3_conative as tr
        return tr
    except Exception:
        from cear_pilot.training import train_phase3_v3 as tr
        return tr


def build_models_from_checkpoint(ckpt_path: Path, device: str):
    tr = load_training_module()
    ckpt = torch.load(ckpt_path, map_location=device)
    args_dict = dict(ckpt.get("meta", {}).get("args", {}))
    if not args_dict:
        raise RuntimeError(f"Checkpoint missing meta['args']: {ckpt_path}")
    args_dict["device"] = device
    args = SimpleNamespace(**args_dict)

    built = tr.build_agent_and_decoder(args)
    if len(built) != 3:
        raise RuntimeError("Expected build_agent_and_decoder(args) -> (agent, decoder, body_decoder).")
    agent, obs_decoder, body_decoder = built

    agent.load_state_dict(ckpt["agent_state"])
    obs_decoder.load_state_dict(ckpt["decoder_state"])
    if body_decoder is not None and "body_decoder_state" in ckpt:
        body_decoder.load_state_dict(ckpt["body_decoder_state"])

    agent.to(device).eval()
    obs_decoder.to(device).eval()
    if body_decoder is not None:
        body_decoder.to(device).eval()
    return ckpt, args, agent, obs_decoder, body_decoder


def build_env_from_checkpoint(ckpt: Dict[str, Any], *, max_steps: int, disable_random_perturbations: bool = True):
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Env, NZonePhase3Config

    env_cfg_dict = dict(ckpt.get("meta", {}).get("env_cfg", {}))
    if not env_cfg_dict:
        raise RuntimeError("Checkpoint missing meta['env_cfg'].")
    env_cfg_dict["max_steps"] = int(max_steps)
    if disable_random_perturbations:
        env_cfg_dict["n_perturbations"] = 0
        env_cfg_dict["perturbation_duration"] = 0
    env = NZonePhase3Env(config=NZonePhase3Config(**env_cfg_dict))
    return env


def force_body_u(env, new_u: float) -> None:
    env.body_u = float(new_u)
    env.body_state = np.array([sigmoid(env.body_u)], dtype=np.float32)
    if hasattr(env, "_last_body_u"):
        env._last_body_u = float(env.body_u)


def apply_pre_observation_perturbation(env, condition: str, t: int, args: argparse.Namespace) -> bool:
    """Apply external perturbation before obs/info are read at step t.
    Returns whether an external perturbation is active on this step.
    """
    in_window = int(args.shock_start) <= int(t) < int(args.shock_start + args.shock_duration)
    active = False

    if condition == "body_shock" and in_window:
        # Directly perturb latent viability potential u.
        force_body_u(env, float(env.body_u) + float(args.body_u_shock_delta))
        active = True

    if condition == "env_shock" and in_window:
        # Force exteroceptive perturbation for this observation.
        if hasattr(env, "_perturbation_active"):
            env._perturbation_active = True
        if hasattr(env, "_perturbation_trace"):
            env._perturbation_trace = 1.0
        try:
            env.cfg.perturbation_scale = float(args.env_perturb_scale)
        except Exception:
            pass
        active = True
    elif condition != "env_shock":
        # Ensure manual env perturb is off outside env_shock condition.
        if hasattr(env, "_perturbation_active"):
            env._perturbation_active = False
        if hasattr(env, "_perturbation_trace"):
            env._perturbation_trace = 0.0

    return active


@torch.no_grad()
def compute_conative_q(body_pred_all: torch.Tensor,
                       tendency_all: torch.Tensor,
                       train_args: SimpleNamespace) -> Tuple[np.ndarray, float]:
    trend_w = float(getattr(train_args, "conative_trend_weight", 1.0))
    body_w = float(getattr(train_args, "conative_body_weight", 0.25))
    temp = max(float(getattr(train_args, "conative_temperature", 0.10)), 1e-6)

    field_value = (
        trend_w * tendency_all.squeeze(-1).detach()
        + body_w * body_pred_all.squeeze(-1).detach()
    )
    q = torch.softmax(field_value / temp, dim=-1)[0]
    entropy = -(q * torch.log(q + 1e-9)).sum().item()
    return q.detach().cpu().numpy(), float(entropy)


@torch.no_grad()
def run_assay_single(ckpt_path: Path,
                     outdir: Path,
                     condition: str,
                     device: str,
                     args: argparse.Namespace) -> None:
    seed_everything(int(args.assay_seed))
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt, train_args, agent, obs_decoder, body_decoder = build_models_from_checkpoint(ckpt_path, device)
    env = build_env_from_checkpoint(ckpt, max_steps=int(args.steps), disable_random_perturbations=True)

    n_actions = int(env.action_space.n)
    obs, info = env.reset(seed=int(args.assay_seed))

    # Reset assay latents. We let g evolve during the frozen assay.
    agent.reset(batch_size=1)
    agent.reset_body_pred()
    g_prev = agent.get_latents()["g"].detach().clone()

    pe_ema_s = float(args.init_pe)
    pe_ema_l = float(args.init_pe)
    pe_prev = float(args.init_pe)
    last_action = STAY

    rows: List[Dict[str, Any]] = []

    for t in range(int(args.steps)):
        external_perturb_active = apply_pre_observation_perturbation(env, condition, t, args)
        obs = env._observe()
        info = env._info_dict()

        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()
        body_actual_t = torch.tensor([[float(info["body_state"])]], dtype=torch.float32, device=device)

        body_silhouette_t = None
        if "body_silhouette" in info:
            body_silhouette_t = torch.tensor(info["body_silhouette"][None, :], dtype=torch.float32, device=device)

        err_t = build_err_t(
            pe_prev,
            pe_ema_s,
            pe_ema_l,
            pe_prev,
            int(external_perturb_active or info.get("perturbation_active", 0)),
            float(info.get("perturbation_trace", 0.0)),
            device,
        )

        out = agent.forward_step(
            x_t,
            p_t,
            err_t=err_t,
            body_actual_t=body_actual_t,
            body_silhouette_t=body_silhouette_t,
        )

        # Policy action. Conative attunement has already shaped policy weights during training;
        # q below is diagnostic only and is not injected into logits.
        if agent.cfg.policy.body_dim > 0:
            logits_act = agent.policy(out["s"].detach(), body_t=body_actual_t.detach())
        else:
            logits_act = agent.policy(out["s"].detach())
        pi = torch.softmax(logits_act, dim=-1)[0]
        if args.greedy:
            a_int = int(torch.argmax(pi).item())
        else:
            a_int = int(torch.distributions.Categorical(probs=pi).sample().item())

        # Diagnostics before env.step.
        if body_decoder is not None:
            body_pred_all, tendency_all = body_decoder.predict_all_actions(out["z"], out["g"], body_actual_t)
            body_pred_action = body_pred_all[0, a_int].unsqueeze(0)
            tendency_action = tendency_all[0, a_int].unsqueeze(0)
            q, q_entropy = compute_conative_q(body_pred_all, tendency_all, train_args)
            pred_trend_ud = float((tendency_all[0, UP, 0] - tendency_all[0, DOWN, 0]).item())
            pred_body_ud = float((body_pred_all[0, UP, 0] - body_pred_all[0, DOWN, 0]).item())
            tendency_target = float(env.counterfactual_body_tendency(a_int, horizon=int(getattr(train_args, "body_tendency_horizon", args.body_tendency_horizon))))
            true_trend_ud = float(
                env.counterfactual_body_tendency(UP, horizon=int(getattr(train_args, "body_tendency_horizon", args.body_tendency_horizon)))
                - env.counterfactual_body_tendency(DOWN, horizon=int(getattr(train_args, "body_tendency_horizon", args.body_tendency_horizon)))
            )
        else:
            body_pred_action = None
            tendency_action = None
            q = np.ones(n_actions, dtype=np.float32) / n_actions
            q_entropy = float(np.log(n_actions))
            pred_trend_ud = np.nan
            pred_body_ud = np.nan
            tendency_target = np.nan
            true_trend_ud = np.nan

        # Step environment.
        obs_next, _, terminated, truncated, info_next = env.step(a_int)
        x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
        body_actual_next = torch.tensor([[float(info_next["body_state"])]], dtype=torch.float32, device=device)

        # Prediction errors / internal costs.
        xhat_all = obs_decoder.predict_all_actions(out["z"], out["g"])
        xhat_executed = xhat_all[0, a_int].unsqueeze(0)
        obs_cost = float(F.mse_loss(xhat_executed, x_next).item())
        body_drive = float(1.0 - body_actual_next.item())
        lambda_body = float(getattr(train_args, "lambda_body", 0.10))
        cost_t = float(obs_cost + lambda_body * body_drive)

        if body_pred_action is not None:
            body_next_pe = float(F.mse_loss(body_pred_action, body_actual_next).item())
            tendency_pe = float((float(tendency_action.item()) - tendency_target) ** 2)
        else:
            body_next_pe = np.nan
            tendency_pe = np.nan

        g_np = out["g"].detach().cpu().numpy()[0]
        z_dim = int(agent.cfg.encoder.z_dim)
        try:
            gp = agent.enc.obs_enc.gating_params(out["g"].detach())
            if "M_g" in gp:
                M = gp["M_g"][0].detach().cpu().numpy()
                eigs = np.linalg.eigvalsh(M)
            else:
                eigs = np.zeros(z_dim, dtype=np.float32)
        except Exception:
            eigs = np.zeros(z_dim, dtype=np.float32)

        row: Dict[str, Any] = {
            "t": int(t),
            "condition": condition,
            "phase": "pre" if t < args.shock_start else ("shock" if t < args.shock_start + args.shock_duration else "recovery"),
            "external_perturb_active": int(external_perturb_active),
            "x": int(info["x"]),
            "y": int(info["y"]),
            "zone3": zone3(int(info["x"]), int(info["y"])),
            "valence_zone_id": int(info.get("valence_zone_id", 0)),
            "valence_name": valence_name(int(info.get("valence_zone_id", 0))),
            "action": a_int,
            "action_name": ACTION_NAMES.get(a_int, str(a_int)),
            "pi_UP": float(pi[UP].item()),
            "pi_DOWN": float(pi[DOWN].item()),
            "pi_LEFT": float(pi[LEFT].item()),
            "pi_RIGHT": float(pi[RIGHT].item()),
            "pi_STAY": float(pi[STAY].item()),
            "q_UP": float(q[UP]),
            "q_DOWN": float(q[DOWN]),
            "q_LEFT": float(q[LEFT]),
            "q_RIGHT": float(q[RIGHT]),
            "q_STAY": float(q[STAY]),
            "q_UP_minus_DOWN": float(q[UP] - q[DOWN]),
            "q_entropy": float(q_entropy),
            "body_state": float(info["body_state"]),
            "body_next": float(info_next["body_state"]),
            "body_u": float(info.get("body_u", np.nan)),
            "body_raw_delta": float(info.get("body_raw_delta", np.nan)),
            "affordance_here": float(info.get("affordance_here", np.nan)),
            "obs_pe": obs_cost,
            "body_next_pe": body_next_pe,
            "tendency_pe": tendency_pe,
            "pred_trend_UP_minus_DOWN": pred_trend_ud,
            "pred_body_UP_minus_DOWN": pred_body_ud,
            "true_trend_UP_minus_DOWN": true_trend_ud,
            "tendency_pred_action": float(tendency_action.item()) if tendency_action is not None else np.nan,
            "tendency_target_action": tendency_target,
            "body_drive": body_drive,
            "cost_t": cost_t,
            "alpha": float(out["alpha"].item()),
            "g_norm": float(np.linalg.norm(g_np)),
            "g_delta_norm": float(torch.norm(out["g"].detach() - g_prev.detach()).item()),
        }
        for i, gv in enumerate(g_np):
            row[f"g_{i}"] = float(gv)
        for i, ev in enumerate(eigs):
            row[f"M_g_eig_{i}"] = float(ev)
        rows.append(row)

        # Update PE traces for next step.
        pe_ema_s = 0.1 * obs_cost + 0.9 * pe_ema_s
        pe_ema_l = 0.01 * obs_cost + 0.99 * pe_ema_l
        pe_prev = obs_cost
        g_prev = out["g"].detach().clone()
        last_action = a_int
        obs = obs_next
        info = info_next

        if bool(terminated or truncated):
            break

    traj = pd.DataFrame(rows)
    traj.to_parquet(outdir / "assay_traj.parquet", index=False)

    summary = summarize_assay(traj, args)
    with open(outdir / "assay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save({
        "final_g": agent.get_latents()["g"].detach().cpu(),
        "condition": condition,
        "ckpt": str(ckpt_path),
        "assay_args": vars(args),
    }, outdir / "final_g.pt")

    print(f"[save] {outdir / 'assay_traj.parquet'} ({len(traj)} rows)")
    print(f"[save] {outdir / 'assay_summary.json'}")
    print(json.dumps(summary, indent=2))


def summarize_assay(traj: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for phase in ["pre", "shock", "recovery"]:
        d = traj[traj["phase"] == phase]
        if d.empty:
            continue
        out[phase] = {
            "n": int(len(d)),
            "top_occ": float((d["valence_name"] == "top").mean()),
            "TR_occ": float((d["zone3"] == "TR").mean()),
            "bottom_occ": float((d["valence_name"] == "bottom").mean()),
            "body_state_mean": float(d["body_state"].mean()),
            "body_u_mean": float(d["body_u"].mean()) if "body_u" in d else float("nan"),
            "obs_pe_mean": float(d["obs_pe"].mean()),
            "q_UP_minus_DOWN_mean": float(d["q_UP_minus_DOWN"].mean()),
            "pred_trend_UP_minus_DOWN_mean": float(d["pred_trend_UP_minus_DOWN"].mean()),
            "alpha_mean": float(d["alpha"].mean()),
            "g_norm_mean": float(d["g_norm"].mean()),
            "g_delta_norm_mean": float(d["g_delta_norm"].mean()),
        }
    if "pre" in out and "recovery" in out:
        out["residue_recovery_minus_pre"] = {
            k: float(out["recovery"][k] - out["pre"][k])
            for k in out["pre"].keys()
            if k != "n" and k in out["recovery"]
        }
    return out


def run_batch(args: argparse.Namespace) -> None:
    base = Path(args.base)
    assay_base = Path(args.assay_base)
    for cohort in args.cohorts:
        for seed in range(int(args.seed_start), int(args.seed_end) + 1):
            ckpt = base / cohort / f"s{seed}" / "ckpt_final.pt"
            if not ckpt.exists():
                print(f"[missing] {ckpt}")
                continue
            for condition in args.conditions:
                outdir = assay_base / cohort / f"s{seed}" / condition
                done_file = outdir / "assay_traj.parquet"
                if done_file.exists() and not args.overwrite:
                    print(f"[skip] {done_file}")
                    continue
                print(f"\n=== assay cohort={cohort} seed={seed} condition={condition} ===")
                run_assay_single(ckpt, outdir, condition, args.device, args)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Single-run mode
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--condition", type=str, default="body_shock", choices=["control", "body_shock", "env_shock"])

    # Batch mode
    ap.add_argument("--base", type=str, default="")
    ap.add_argument("--assay_base", type=str, default="outputs/p3_v3_assay")
    ap.add_argument("--cohorts", nargs="+", default=["full", "no_body_in_g", "no_conative"])
    ap.add_argument("--conditions", nargs="+", default=["control", "body_shock", "env_shock"], choices=["control", "body_shock", "env_shock"])
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")

    # Assay parameters
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--assay_seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--shock_start", type=int, default=60)
    ap.add_argument("--shock_duration", type=int, default=20)
    ap.add_argument("--body_u_shock_delta", type=float, default=-0.08,
                    help="Per-step additive perturbation to env.body_u during body_shock window.")
    ap.add_argument("--env_perturb_scale", type=float, default=0.20)
    ap.add_argument("--body_tendency_horizon", type=int, default=5)
    ap.add_argument("--init_pe", type=float, default=0.05)
    ap.add_argument("--greedy", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.base:
        run_batch(args)
    else:
        if not args.ckpt or not args.outdir:
            raise SystemExit("Single-run mode requires --ckpt and --outdir, or use --base for batch mode.")
        run_assay_single(Path(args.ckpt), Path(args.outdir), args.condition, args.device, args)


if __name__ == "__main__":
    main()
