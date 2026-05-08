# cear_pilot/training/train_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 training: minimal embodiment.

Differences from Phase 2:
  - Env: NZonePhase3Env (15x15, body state, vertical valence affordance,
         self-cell as Position 2 — self-cell ch1 = current cell affordance).
  - obs_dim = 18 (vs phase 2's 8).
  - State head has separate body-prediction head (state.body_dim = 1).
  - World latent receives body PE as separate AlphaNet channel
    (world.body_err_dim = 1).
  - Loss = obs prediction error + body prediction error + actor objective
           (AAAI-style REINFORCE with PE as internal cost).
  - Trained from scratch (phase 1/2 obs-incompatible).

Body PE flow:
  At step t, agent has body_pred(t-1) stored from previous forward.
  body_actual_t is read from env info (after env.step).
  body_pe_t = body_actual_t - body_pred(t-1)
  This body_pe is fed to AlphaNet so plasticity reacts to body surprisal.
  body_pred(t) is then computed by state head and stored for next step.

Loss:
  L_obs    = MSE(decoder(g_t, a_t), obs_{t+1})        — phase 2 style
  L_body   = MSE(body_pred_t, body_actual_{t+1})       — phase 3 specific
  L_actor  = -(c_t - b_t) * log π(a_t | s_t)           — AAAI/phase 1 style
             where c_t = MSE(decoder(g_t, a_t), obs_{t+1}) is action-
             conditioned prediction error. Stop-gradient at policy state
             ensures actor objective doesn't flow into perspective layer.
  L_smooth = MSE(g_t, stopgrad(g_{t-1}))               — slow latent regularizer
  L_ent    = -H(π(·|s_t))                              — entropy bonus
  L_total  = L_obs + body_loss_weight * L_body
            + w_actor * L_actor + w_smooth * L_smooth + w_ent * L_ent

The actor objective is disabled during a warm-up phase (warmup_episodes
episodes), during which only L_obs + L_body train the world model and
body head. This follows AAAI/phase 1's commitment that the perspective
formation cycle and policy optimization cycle remain separable: backbone
must reach predictive stability before policy learning is introduced.

The body_loss_weight is a *training-time* hyperparameter (gradient scaling),
not a value-function weight. It only controls how strongly the body head is
trained, not the agent's runtime behavior. Body PE in AlphaNet has no such
weight — AlphaNet learns the weighting itself.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cear_pilot.envs.nzone_phase3 import NZonePhase3Config, NZonePhase3Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.encoder import EncoderConfig
from cear_pilot.models.world_latent import WorldLatentConfig
from cear_pilot.models.state_head import StateHeadConfig
from cear_pilot.models.policy import PolicyConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def count_params(module: nn.Module, only_trainable=False):
    return sum(p.numel() for p in module.parameters() if (not only_trainable or p.requires_grad))


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set all RNG seeds for full reproducibility (matches phase 1 spec).
    Network init, action sampling, and dropout all use PyTorch's global
    random state, so without this the same `--seed` argument can yield
    different results across runs."""
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


class EMAMeanVar:
    """Exponential moving average tracker for advantage normalization
    (AAAI/phase 1 style). Tracks mean and variance of a scalar stream."""
    def __init__(self, beta: float = 0.99):
        self.beta = beta
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, x: float) -> None:
        if not self.initialized:
            self.mean = float(x)
            self.var = 1.0
            self.initialized = True
            return
        m = self.beta * self.mean + (1.0 - self.beta) * float(x)
        d = float(x) - self.mean
        v = self.beta * self.var + (1.0 - self.beta) * d * d
        self.mean = m
        self.var = v

    @property
    def std(self) -> float:
        return float(max(self.var, 1e-6) ** 0.5)


ERR_DIM = 6
BODY_DIM = 1


def build_err_t(pred_err, pred_err_ema_short, pred_err_ema_long,
                pred_err_prev, perturbation_active, perturbation_trace, device):
    """Same env-PE feature vector as phase 2."""
    ema_safe = max(pred_err_ema_long, 1e-6)
    feats = [
        min(pred_err / ema_safe, 5.0),
        min(pred_err_ema_short / ema_safe, 5.0),
        float(np.log1p(pred_err_ema_long)),
        float(np.tanh((pred_err - pred_err_prev) * 10.0)),
        float(perturbation_active),
        float(perturbation_trace),
    ]
    return torch.tensor([feats], dtype=torch.float32, device=device)


def parse_mixed_schedule(s: str) -> List[Tuple[int, int]]:
    blocks = []
    for part in s.split(","):
        n, eps = part.strip().split(":")
        blocks.append((int(n), int(eps)))
    return blocks


# ── Build agent + decoder from scratch (body-coupled architecture) ──

def build_agent_and_decoder(args):
    device = args.device

    # Body-coupled architecture (Layer 2 — Option C):
    # - z_obs (16-dim) + z_body (4-dim) → z_fused (20-dim)
    # - z_fused is what gating (FiLM/salience/metric_c1) sees, so the
    #   gating layer modulates body-coupled perception.
    # - Policy receives body_state directly (Layer "willing": Safron's
    #   body→effector mapping).
    # - State head uses z_fused (20-dim) for both s prediction and
    #   body prediction; metric_c1 builds M_g over the full 20-dim space.
    enc_cfg = EncoderConfig(
        obs_dim=8,
        proprio_dim=5,
        z_dim=args.z_dim,                # fused dim (default 20)
        z_obs_dim=args.z_obs_dim,        # default 16
        body_z_dim=args.body_z_dim,      # default 4
        p_dim=8,
        g_dim=12,
        hidden=64,
        use_salience_gate=True,
        gating_mode=args.gating_mode,
        body_dim=BODY_DIM,                # 1
        body_hidden=32,
        silhouette_dim=int(args.silhouette_dim),
    )

    world_cfg = WorldLatentConfig(
        z_dim=enc_cfg.z_dim,
        p_dim=enc_cfg.p_dim,
        g_dim=enc_cfg.g_dim,
        update_mode=args.update_mode,
        alpha_fixed=args.alpha_fixed,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        use_error_feedback=True,
        err_dim=ERR_DIM,
        body_err_dim=BODY_DIM,
        body_in_g=bool(args.body_in_g),
        body_g_scale=float(args.body_g_scale),
    )

    # Auto-set use_metric based on encoder mode
    use_metric = (args.gating_mode == "metric_c1")
    state_cfg = StateHeadConfig(
        z_dim=enc_cfg.z_dim,
        p_dim=enc_cfg.p_dim,
        g_dim=enc_cfg.g_dim,
        s_dim=16,
        hidden=64,
        body_dim=BODY_DIM,
        body_hidden=32,
        use_metric=use_metric,
    )

    # Policy receives body_state directly (Layer 1 willing).
    policy_cfg = PolicyConfig(
        s_dim=state_cfg.s_dim,
        n_actions=5,
        hidden=64,
        body_dim=BODY_DIM,
    )

    agent_cfg = AgentConfig(
        encoder=enc_cfg, world=world_cfg, state=state_cfg, policy=policy_cfg,
        device=device,
    )
    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(
        g_dim=enc_cfg.g_dim,
        n_actions=5,
        obs_dim=8,
        hidden=64,
    )
    decoder = ObsDecoder(dec_cfg)

    print(f"[build] Agent params: {count_params(agent)}")
    print(f"[build] Decoder params: {count_params(decoder)}")
    print(f"[build] z_fused={enc_cfg.z_dim} (z_obs={enc_cfg.z_obs_dim} + "
          f"z_body={enc_cfg.body_z_dim}), body_dim={enc_cfg.body_dim}")
    print(f"[build] body-coupled: encoder={enc_cfg.body_dim>0} "
          f"policy={policy_cfg.body_dim>0} state={state_cfg.body_dim>0}")
    print(f"[build] body_in_g={world_cfg.body_in_g} "
          f"(scale={world_cfg.body_g_scale})  obs_dim={enc_cfg.obs_dim}")
    sil_str = (f"silhouette_dim={enc_cfg.silhouette_dim} "
               f"(σ={args.silhouette_sigma})"
               if enc_cfg.silhouette_dim > 0 else "silhouette=disabled")
    print(f"[build] {sil_str}")

    return agent, decoder


# ── Training ──

def train(args):
    device = args.device

    # Seed all RNGs BEFORE module construction for reproducibility.
    seed_everything(int(args.seed), deterministic=True)

    agent, decoder = build_agent_and_decoder(args)
    agent.to(device)
    decoder.to(device)

    env_cfg = NZonePhase3Config(
        max_steps=args.max_steps,
        sigma_left=args.sigma_left,
        sigma_right=args.sigma_right,
        n_perturbations=args.n_perturbations,
        perturbation_duration=args.perturbation_duration,
        perturbation_scale=args.perturbation_scale,
        affordance_top=args.affordance_top,
        affordance_bottom=args.affordance_bottom,
        affordance_sigmoid_slope=args.affordance_slope,
        metabolic_cost=args.metabolic_cost,
        movement_cost=args.movement_cost,
        affordance_to_body_gain=args.affordance_gain,
        silhouette_dim=int(args.silhouette_dim),
        silhouette_noise_sigma=float(args.silhouette_sigma),
    )
    env = NZonePhase3Env(config=env_cfg)

    all_params = list(agent.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    outdir = Path(args.outdir) if args.outdir else Path(f"outputs/phase3_s{args.seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Schedule
    mixed = None
    if args.mixed_schedule:
        mixed = parse_mixed_schedule(args.mixed_schedule)
        total_eps = sum(eps for _, eps in mixed)
        print(f"[mixed] Schedule: {mixed} → {total_eps} episodes")
    else:
        total_eps = args.episodes

    ep_schedule = []
    if mixed:
        for block_id, (n_p, n_eps) in enumerate(mixed):
            for _ in range(n_eps):
                ep_schedule.append((n_p, block_id))
    else:
        for _ in range(total_eps):
            ep_schedule.append((args.n_perturbations, 0))

    traj_rows = []
    n_actions = int(env.action_space.n)
    pe_ema_s, pe_ema_l, pe_prev = 0.05, 0.05, 0.05

    # Actor advantage normalization (AAAI/phase 1 style)
    baseline_stats = EMAMeanVar(beta=args.actor_baseline_beta)
    adv_stats = EMAMeanVar(beta=args.actor_std_beta)

    # Adaptive entropy weight (AAAI/phase 1 spec).
    entropy_weight = float(args.w_entropy_init)
    entropy_target = float(np.log(n_actions) * args.entropy_target_ratio)
    print(f"[train] adaptive entropy: init={entropy_weight:.4f} "
          f"target={entropy_target:.4f} (={args.entropy_target_ratio:.0%} of max) "
          f"bounds=[{args.w_entropy_min}, {args.w_entropy_max}]")

    # Warm-up calculation: the actor objective is disabled until
    # warmup_episodes have been completed.
    warmup_episodes = int(max(0, args.warmup_episodes))
    warmup_steps = warmup_episodes * args.max_steps  # for logging only
    print(f"[train] warmup={warmup_episodes} episodes "
          f"({warmup_steps} steps), total={len(ep_schedule)} episodes")

    agent.reset(batch_size=1)
    g_prev = agent.get_latents()["g"].detach().clone()
    t0 = time.time()
    global_step = 0

    # Per-print-block accumulators for action distribution + spatial diagnostics.
    # Reset every args.print_every episodes so we see fresh dynamics.
    block_actions: List[int] = []
    block_xs: List[int] = []
    block_ys: List[int] = []
    block_advs: List[float] = []
    block_costs: List[float] = []

    for global_ep, (n_perturb_now, block_id) in enumerate(ep_schedule):
        env.cfg.n_perturbations = n_perturb_now

        obs, info = env.reset(seed=args.seed + global_ep)
        last_action = 4
        ep_obs_pe: List[float] = []
        ep_body_pe: List[float] = []
        done = False

        # Reset agent latents and body_pred between episodes
        agent.reset(batch_size=1)
        g_prev = agent.get_latents()["g"].detach().clone()

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device),
                            num_classes=n_actions).float()

            err_t = build_err_t(pe_prev, pe_ema_s, pe_ema_l, pe_prev,
                                int(info.get("perturbation_active", 0)),
                                float(info.get("perturbation_trace", 0.0)),
                                device)

            # Phase 3: pass body_actual_t to agent so it can compute body PE
            body_actual_t = torch.tensor(
                [[float(info["body_state"])]],
                dtype=torch.float32, device=device,
            )
            # Phase 3 (optional): body silhouette — directional NSEW
            # affordance silhouette via body, with Gaussian blur.
            # None when silhouette_dim == 0 (env didn't produce it).
            body_silhouette_t: Optional[torch.Tensor] = None
            if "body_silhouette" in info:
                body_silhouette_t = torch.tensor(
                    info["body_silhouette"][None, :],   # (1, 4)
                    dtype=torch.float32, device=device,
                )

            out = agent.forward_step(
                x_t, p_t,
                err_t=err_t,
                body_actual_t=body_actual_t,
                body_silhouette_t=body_silhouette_t,
            )
            # logits returned by forward_step come from policy(s_t) WITHOUT
            # stop-gradient. We compute a separate set of "actor logits" from
            # detached s_t to enforce the AAAI commitment: actor objective
            # must not flow into the perspective formation cycle.
            logits_pred = out["logits"]                  # gradient-attached, not used for sampling here
            s_t = out["s"]
            # Phase 3 body-coupled: policy receives body input directly.
            # body_actual_t is the live interoceptive signal at this step
            # (already detached from any backprop graph).
            if agent.cfg.policy.body_dim > 0:
                logits_act = agent.policy(
                    s_t.detach(), body_t=body_actual_t.detach()
                )
            else:
                logits_act = agent.policy(s_t.detach())
            action = agent.policy.sample_action(
                logits_act, greedy=args.greedy,
            )
            a_int = int(action.item())

            obs_next, _, terminated, truncated, info_next = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
            a_oh = F.one_hot(torch.tensor([a_int], device=device),
                             num_classes=n_actions).float()
            body_actual_next = torch.tensor(
                [[float(info_next["body_state"])]],
                dtype=torch.float32, device=device,
            )

            # ─── World-model loss ───
            # AAAI uses an action-conditioned mixture target; for phase 3 we
            # follow the same logic. Predict obs under each candidate action,
            # weight by the (detached) policy distribution. Then the executed
            # action's prediction error serves as an internal cost for the
            # actor objective.
            xhat_all = decoder.predict_all_actions(out["g"])
            # xhat_all: (B=1, A=n_actions, obs_dim)
            with torch.no_grad():
                pi_pred = torch.softmax(logits_pred, dim=-1)  # (1, A)
            xhat_mix = (pi_pred.unsqueeze(-1) * xhat_all).sum(dim=1)  # (1, obs_dim)
            obs_loss = F.mse_loss(xhat_mix, x_next)

            # ─── Body loss ───
            body_pred_t = out["body_pred"]               # (1, 1)
            body_loss = F.mse_loss(body_pred_t, body_actual_next)

            # ─── Smoothness loss on g (slow latent regularizer) ───
            smooth_loss = F.mse_loss(out["g"], g_prev.detach())

            # ─── Actor loss (REINFORCE with action-conditioned PE as cost) ───
            # Cost combines obs PE (visual predictability) and body PE
            # (homeostatic stability) — both grounded prediction errors.
            #   c_t = MSE(decoder(g_t, a_t), obs_{t+1}) + λ · body_PE
            # This instantiates Layer 1 (instinct): the agent's
            # predictability-seeking is grounded both in visual stability
            # ─── Actor cost = obs prediction error + body prediction error ───
            # Layer 1 cost combines two grounded prediction errors:
            #   obs_cost  = MSE(decoder(g, a), x_next)  — visual predictability
            #   body_pe²  = (body_pred - body_actual)²  — interoceptive surprisal
            # cost_t = obs_cost + λ_body · body_pe²
            #
            # Both terms are squared prediction errors. The agent's actor
            # learns to minimize cost — i.e., select actions that produce
            # both predictable visual outcomes and predictable body
            # outcomes. Body PE squared (not body deviation from a target)
            # keeps the cost interpretation as predictive: body is a
            # learnable signal whose stability the agent seeks via action.
            #
            # Sign convention (matches phase 1 train_phase1.py:344):
            #   advantage = baseline - cost
            # → low cost (good action) yields *positive* advantage.
            xhat_executed = xhat_all[0, a_int].unsqueeze(0)       # (1, obs_dim)
            obs_cost = F.mse_loss(xhat_executed, x_next).detach().item()
            body_pe_squared = float(
                ((body_pred_t - body_actual_next) ** 2).detach().item()
            )
            cost_t = float(obs_cost + args.lambda_body * body_pe_squared)
            baseline_stats.update(cost_t)
            advantage = baseline_stats.mean - cost_t
            adv_stats.update(advantage)
            adv_norm = float(np.clip(
                advantage / max(adv_stats.std, 1e-6),
                -args.adv_clip, args.adv_clip,
            ))
            log_pi_act = F.log_softmax(logits_act, dim=-1)[0, a_int]
            actor_loss = -(adv_norm * log_pi_act)

            # ─── Entropy regularizer ───
            ent = -(F.softmax(logits_act, dim=-1)
                    * F.log_softmax(logits_act, dim=-1)).sum(dim=-1)
            ent_loss = -ent.mean()  # negate so reducing this *increases* entropy

            # ─── Total loss with warm-up gating + adaptive entropy ───
            # Actor objective is disabled during warmup_episodes.
            in_warmup = global_ep < warmup_episodes
            actor_active = not in_warmup
            if actor_active:
                w_actor_now = float(args.w_actor)
                w_ent_now = float(entropy_weight)   # adaptive
            else:
                w_actor_now = 0.0
                w_ent_now = 0.0

            total_loss = (
                obs_loss
                + args.body_loss_weight * body_loss
                + args.w_smooth * smooth_loss
                + w_actor_now * actor_loss
                + w_ent_now * ent_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(all_params, args.clip_grad)
            optimizer.step()

            # ─── Adaptive entropy weight update (AAAI/phase 1 line 371-377) ───
            if actor_active:
                ent_val = float(ent.detach().item())
                if ent_val < entropy_target:
                    entropy_weight *= (1.0 + float(args.entropy_adapt_rate))
                else:
                    entropy_weight *= (1.0 - 0.5 * float(args.entropy_adapt_rate))
                entropy_weight = float(np.clip(
                    entropy_weight, args.w_entropy_min, args.w_entropy_max
                ))

            obs_pe_val = float(obs_loss.item())
            body_pe_val = float(body_loss.item())
            pe_ema_s = 0.1 * obs_pe_val + 0.9 * pe_ema_s
            pe_ema_l = 0.01 * obs_pe_val + 0.99 * pe_ema_l
            pe_prev = obs_pe_val
            ep_obs_pe.append(obs_pe_val)
            ep_body_pe.append(body_pe_val)

            # Per-print-block diagnostics
            block_actions.append(a_int)
            block_xs.append(int(info["x"]))
            block_ys.append(int(info["y"]))
            block_advs.append(adv_norm)
            block_costs.append(cost_t)

            # advance g_prev for next-step smoothness
            g_prev = out["g"].detach().clone()
            global_step += 1

            if args.save_traj:
                g_np = out["g"].detach().cpu().numpy()[0]
                z_dim = agent.cfg.encoder.z_dim

                # mode-aware gating params via introspection helper
                with torch.no_grad():
                    gp = agent.enc.obs_enc.gating_params(out["g"].detach())
                gamma_np = (gp["gamma"].cpu().numpy()[0]
                            if "gamma" in gp else np.zeros(z_dim))
                beta_np = (gp["beta"].cpu().numpy()[0]
                           if "beta" in gp else np.zeros(z_dim))
                salience_np = (gp["salience"].cpu().numpy()[0]
                               if "salience" in gp else np.zeros(z_dim))
                # Metric_c1 mode: capture M_g eigenvalues (compact summary).
                # Full M_g (z_dim^2 = 256 floats) is too large for traj rows;
                # eigenvalues capture metric anisotropy structure.
                if "M_g" in gp:
                    M = gp["M_g"][0].cpu().numpy()  # (z_dim, z_dim)
                    eigs = np.linalg.eigvalsh(M)    # ascending
                else:
                    eigs = np.zeros(z_dim)

                row = {
                    "episode": global_ep, "t": int(info["t"]),
                    "x": int(info["x"]), "y": int(info["y"]),
                    "zone_id": int(info["zone_id"]),
                    "valence_zone_id": int(info.get("valence_zone_id", 0)),
                    "action": a_int,
                    "obs_pe": obs_pe_val,
                    "body_pe": body_pe_val,
                    "alpha": float(out["alpha"].item()),
                    "g_norm": float(np.linalg.norm(g_np)),
                    "body_state": float(info["body_state"]),
                    "body_pred": float(body_pred_t.detach().item()),
                    "affordance_here": float(info.get("affordance_here", 0.0)),
                    "perturbation_active": int(info.get("perturbation_active", 0)),
                    "perturbation_trace": float(info.get("perturbation_trace", 0.0)),
                    "n_perturb_setting": n_perturb_now,
                    "block_id": block_id,
                    "gating_mode": agent.cfg.encoder.gating_mode,
                }
                for gi in range(len(g_np)):
                    row[f"g_{gi}"] = float(g_np[gi])
                for zi in range(z_dim):
                    row[f"gamma_{zi}"] = float(gamma_np[zi])
                    row[f"beta_{zi}"] = float(beta_np[zi])
                    row[f"salience_{zi}"] = float(salience_np[zi])
                    row[f"M_g_eig_{zi}"] = float(eigs[zi])
                traj_rows.append(row)

            obs = obs_next
            info = info_next
            last_action = a_int
            done = bool(terminated or truncated)

        if (global_ep + 1) % args.print_every == 0:
            elapsed = time.time() - t0
            phase_str = "warmup" if (global_ep < warmup_episodes) else "actor "
            # Action distribution as compact "U/D/L/R/S" fractions
            acts = np.array(block_actions)
            n_a = max(len(acts), 1)
            fU = float((acts == 0).sum() / n_a)
            fD = float((acts == 1).sum() / n_a)
            fL = float((acts == 2).sum() / n_a)
            fR = float((acts == 3).sum() / n_a)
            fS = float((acts == 4).sum() / n_a)
            mean_x = float(np.mean(block_xs)) if block_xs else float("nan")
            mean_y = float(np.mean(block_ys)) if block_ys else float("nan")
            mean_adv = float(np.mean(block_advs)) if block_advs else 0.0
            mean_cost = float(np.mean(block_costs)) if block_costs else 0.0

            # 3×3 zone occupancy: x=[0,4]L/[5,9]M/[10,14]R, y=[0,4]T/[5,9]M/[10,14]B
            zone_counts = {
                "TL": 0, "TM": 0, "TR": 0,
                "ML": 0, "MM": 0, "MR": 0,
                "BL": 0, "BM": 0, "BR": 0,
            }
            for x, y in zip(block_xs, block_ys):
                xz = "L" if x <= 4 else ("M" if x <= 9 else "R")
                yz = "T" if y <= 4 else ("M" if y <= 9 else "B")
                zone_counts[yz + xz] += 1
            n_zones = max(sum(zone_counts.values()), 1)
            sorted_zones = sorted(zone_counts.items(),
                                  key=lambda kv: kv[1], reverse=True)
            zone_str = " ".join(
                f"{name}={cnt / n_zones:.2f}"
                for name, cnt in sorted_zones[:2]
                if cnt > 0
            )
            ent_val_now = float(ent.detach().item())
            body_now = float(body_actual_next.item())
            print(
                f"[ep {global_ep+1:4d}/{len(ep_schedule)}] [{phase_str}] "
                f"obs_PE={np.mean(ep_obs_pe):.4f} "
                f"body_PE={np.mean(ep_body_pe):.4f} "
                f"||g||={out['g'].detach().cpu().norm().item():.3f} "
                f"α={out['alpha'].item():.3f} body={body_now:.3f} | "
                f"xy=({mean_x:.1f},{mean_y:.1f}) zones[{zone_str}] "
                f"UDLRS={fU:.2f}/{fD:.2f}/{fL:.2f}/{fR:.2f}/{fS:.2f} "
                f"H={ent_val_now:.3f} wH={entropy_weight:.4f} "
                f"adv={mean_adv:+.3f} c̄={mean_cost:.4f} "
                f"({elapsed:.0f}s)"
            )
            # Reset block accumulators for next print window
            block_actions.clear()
            block_xs.clear()
            block_ys.clear()
            block_advs.clear()
            block_costs.clear()

    # Save
    if args.save_traj and traj_rows:
        traj_df = pd.DataFrame(traj_rows)
        traj_df.to_parquet(outdir / "traj.parquet", index=False)
        print(f"[save] {outdir / 'traj.parquet'} ({len(traj_df)} rows)")

    torch.save({
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": {
            "agent_cfg": {
                "encoder": agent.cfg.encoder.__dict__,
                "world": agent.cfg.world.__dict__,
                "state": agent.cfg.state.__dict__,
                "policy": agent.cfg.policy.__dict__,
            },
            "decoder_cfg": decoder.cfg.__dict__,
            "env_cfg": asdict(env_cfg),
            "args": vars(args),
        },
    }, outdir / "ckpt_final.pt")
    print(f"[save] {outdir / 'ckpt_final.pt'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--save_traj", action="store_true")

    # ── Body-coupled architecture (Layer 2 — Option C) ─────────────
    ap.add_argument("--z_dim", type=int, default=20,
                    help="Fused z dimension. Must equal z_obs_dim + body_z_dim.")
    ap.add_argument("--z_obs_dim", type=int, default=16,
                    help="Obs-only z dim (output of ObservationEncoder).")
    ap.add_argument("--body_z_dim", type=int, default=4,
                    help="Body-only z dim (output of BodyEncoder).")

    # ── Layer 1 cost design ────────────────────────────────────────
    # Actor cost = obs_PE + λ_body · body_PE².
    # Both components are squared prediction errors. lambda_body=0
    # disables body PE in actor cost. lambda_body~3 makes body PE
    # substantial alongside obs PE.
    ap.add_argument("--lambda_body", type=float, default=3.0,
                    help="Weight of body PE² in actor cost. "
                         "cost_t = obs_PE + λ · (body_pred - body_actual)².")

    # ── Phase 3 commitment (III): body PE → g GRU update content ──
    # When True, body_err_t (signed scalar) is concatenated into the GRU
    # input (scaled by body_g_scale) in addition to its existing AlphaNet
    # role. Asymmetric: env PE → AlphaNet only; body PE → both AlphaNet
    # and g GRU update content. Rationale: interoceptive grounding of
    # subjectivity (body PE is constitutive of g, not just a plasticity
    # rate driver).
    ap.add_argument("--body_in_g", action="store_true", default=True,
                    help="Route body PE into the g GRU update content.")
    ap.add_argument("--no_body_in_g", action="store_false", dest="body_in_g",
                    help="Disable body PE → g GRU (ablation).")
    ap.add_argument("--body_g_scale", type=float, default=20.0,
                    help="Scale factor for body_err_t in GRU input "
                         "(body PE magnitude ~0.05; scale to ~1.0).")

    # ── Body silhouette (interoceptive directional affordance) ──
    # silhouette_dim=0 disables (BodyEncoder receives only body_state).
    # silhouette_dim=4 enables NSEW directional silhouette with Gaussian
    # noise σ. This is "어느 정도 직감" — the agent feels (through body)
    # the affordance of its 4 cardinal neighbors, blurred by σ.
    ap.add_argument("--silhouette_dim", type=int, default=0,
                    help="Body silhouette dim. 0 disables, 4 enables NSEW.")
    ap.add_argument("--silhouette_sigma", type=float, default=0.2,
                    help="Gaussian noise σ for silhouette (0 = clean).")

    # AAAI/phase 1-style learning objective + warm-up
    # Warm-up: actor objective disabled for the first warmup_episodes,
    # during which only obs prediction + body prediction train the backbone.
    # Default 50 episodes matches AAAI's ratio (~25% of training).
    ap.add_argument("--warmup_episodes", type=int, default=30)
    ap.add_argument("--w_actor", type=float, default=0.5)
    ap.add_argument("--w_smooth", type=float, default=0.25)
    # Adaptive entropy weight (AAAI/phase 1 spec)
    ap.add_argument("--w_entropy_init", type=float, default=0.01)
    ap.add_argument("--w_entropy_min", type=float, default=0.002)
    ap.add_argument("--w_entropy_max", type=float, default=0.05)
    ap.add_argument("--entropy_target_ratio", type=float, default=0.60)
    ap.add_argument("--entropy_adapt_rate", type=float, default=0.02)
    ap.add_argument("--actor_baseline_beta", type=float, default=0.98)
    ap.add_argument("--actor_std_beta", type=float, default=0.99)
    ap.add_argument("--adv_clip", type=float, default=3.0)

    ap.add_argument("--update_mode", type=str, default="adaptive",
                    choices=["adaptive", "fixed"])
    ap.add_argument("--alpha_fixed", type=float, default=0.10)
    ap.add_argument("--alpha_min", type=float, default=0.03)
    ap.add_argument("--alpha_max", type=float, default=0.30)

    # Phase 3 env: perception
    ap.add_argument("--sigma_left", type=float, default=0.40)
    ap.add_argument("--sigma_right", type=float, default=0.05)
    ap.add_argument("--n_perturbations", type=int, default=4)
    ap.add_argument("--perturbation_duration", type=int, default=15)
    ap.add_argument("--perturbation_scale", type=float, default=0.12)

    # Phase 3 env: valence (vertical sigmoid affordance)
    ap.add_argument("--affordance_top", type=float, default=0.5)
    ap.add_argument("--affordance_bottom", type=float, default=-0.5)
    ap.add_argument("--affordance_slope", type=float, default=1.6)

    # Phase 3 env: body dynamics (defaults from random-walk diagnostic)
    ap.add_argument("--metabolic_cost", type=float, default=0.01)
    ap.add_argument("--movement_cost", type=float, default=0.01)
    ap.add_argument("--affordance_gain", type=float, default=0.10)

    # Body loss weight (training scaling, not runtime preference)
    ap.add_argument("--body_loss_weight", type=float, default=1.0)

    # Encoder gating mode. metric_c1 is the Phase 3 main commitment
    # (default). film, salience, and "both" are sanity-check baselines /
    # ablations: pass --gating_mode <name> to use them.
    ap.add_argument("--gating_mode", type=str, default="metric_c1",
                    choices=["film", "salience", "metric_c1", "both"])

    ap.add_argument("--mixed_schedule", type=str, default="",
                    help="Block schedule: 'n_perturb:episodes,...'")

    return ap.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()