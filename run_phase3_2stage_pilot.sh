# cear_pilot/training/train_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 two-stage training (Phase 2 commitment, instantiated for Phase 3 substrate).

Stage 1 — Backbone training (default 80 episodes):
  - All modules trainable: encoder, decoder, state head, policy, AlphaNet, gating layer
  - Forward pass uses ablate_g=True so g is forced to zero throughout the
    episode. This makes the gating layer's effect a no-op (g=0 → identity
    modulation under all gating modes), and AlphaNet receives no learning
    signal because the world latent update path is skipped.
  - Net effect: a *plain backbone* learns predictive structure of the
    embodied substrate (encoder, decoder, state head, policy) without
    perspective dynamics.
  - Actor objective is active after the warmup phase, providing
    predictability seeking. Body PE flows into the state head's body
    prediction loss as usual.
  - Perturbation is OFF in stage 1 for both P3 and P4 protocols.

Stage 2 — Perspective formation (default 170 episodes):
  - Policy frozen (requires_grad=False). Everything else (encoder + gating,
    decoder, state head main + body head, AlphaNet) trainable.
  - The frozen policy carries the predictability-seeking behavioral tendency
    learned in stage 1. The rest of the world model adapts to the newly
    re-enabled affordance + body homeostasis.
  - Forward pass with ablate_g=False — g flows freely, gating layer
    modulates encoder output, AlphaNet computes adaptive plasticity.
  - Actor objective is *disabled* (w_actor=0). Loss = obs PE + body PE
    + smoothness on g.
  - Perturbation handling depends on protocol:
      * P3: perturbation OFF throughout stage 2 (assay-only protocol)
      * P4: perturbation ON in stage 2 (perturbation as training signal)

Total: 80 + 170 = 250 episodes.

Distinctions from train_phase3.py:
  - --stage1_episodes, --stage2_episodes (replace --episodes)
  - --perturb_protocol {P3, P4}
  - During stage 1: ablate_g=True, world.update_mode=fixed (alpha=0.1),
                    actor active after warmup
  - At stage 1 → stage 2 transition: ckpt save, freeze backbone params,
                                     reset agent latents, set update_mode=adaptive
  - During stage 2: ablate_g=False, AlphaNet trainable, actor disabled
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


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set all RNG seeds for full reproducibility (matches phase 1 spec).

    Without this, network init / action sampling / dropout etc. all use
    PyTorch's global random state, so the same `--seed` argument can yield
    different stage 1 attractors across runs. This is essential for any
    seed-paired analysis (P3 vs P4 with same nominal seed must share stage 1).
    """
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


# ── Build agent + decoder from scratch (no phase 1/2 ckpt loading) ──

def build_agent_and_decoder(args):
    device = args.device

    enc_cfg = EncoderConfig(
        obs_dim=18,
        proprio_dim=5,
        z_dim=16,
        p_dim=8,
        g_dim=12,
        hidden=64,
        use_salience_gate=True,
        gating_mode=args.gating_mode,
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

    policy_cfg = PolicyConfig(
        s_dim=state_cfg.s_dim,
        n_actions=5,
        hidden=64,
    )

    agent_cfg = AgentConfig(
        encoder=enc_cfg, world=world_cfg, state=state_cfg, policy=policy_cfg,
        device=device,
    )
    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(
        g_dim=enc_cfg.g_dim,
        n_actions=5,
        obs_dim=18,
        hidden=64,
    )
    decoder = ObsDecoder(dec_cfg)

    print(f"[build] Agent params: {count_params(agent)}")
    print(f"[build] Decoder params: {count_params(decoder)}")

    return agent, decoder


# ── Training ──

def train(args):
    device = args.device

    # Seed all RNGs BEFORE any module is constructed. Network initialization
    # is sensitive to global random state, so this must come first.
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
    )
    env = NZonePhase3Env(config=env_cfg)

    # Save original env config values for stage 2 restoration. We need these
    # because env.cfg may share the same instance as env_cfg, so mutating
    # env.cfg in stage 1 also mutates env_cfg — losing the originals.
    orig_env_cfg = {
        "affordance_top": float(env_cfg.affordance_top),
        "affordance_bottom": float(env_cfg.affordance_bottom),
        "metabolic_cost": float(env_cfg.metabolic_cost),
        "movement_cost": float(env_cfg.movement_cost),
        "affordance_to_body_gain": float(env_cfg.affordance_to_body_gain),
    }

    all_params = list(agent.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    outdir = Path(args.outdir) if args.outdir else Path(f"outputs/phase3_s{args.seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Two-stage schedule. Stage 1 = backbone training (ablate_g=True,
    # alpha fixed, perturbation OFF, actor active after warmup).
    # Stage 2 = perspective formation (backbone frozen, gating + AlphaNet
    # trainable, actor OFF, perturbation per protocol).
    stage1_eps = int(max(0, args.stage1_episodes))
    stage2_eps = int(max(0, args.stage2_episodes))
    total_eps = stage1_eps + stage2_eps

    if args.mixed_schedule:
        raise ValueError(
            "--mixed_schedule is not supported in two-stage training. "
            "Use --perturb_protocol {P3,P4} instead."
        )

    # Stage 2 perturbation count by protocol
    if args.perturb_protocol == "P3":
        stage2_n_perturb = 0
    elif args.perturb_protocol == "P4":
        stage2_n_perturb = int(args.n_perturbations)
    else:
        raise ValueError(f"unknown perturb_protocol: {args.perturb_protocol}")

    # Build per-episode schedule: list of dicts with stage info
    # stage1: ablate_g=True, fixed alpha, n_perturb=0, actor on after warmup
    # stage2: ablate_g=False, adaptive alpha, n_perturb per protocol, actor off
    ep_schedule = []
    for ep in range(stage1_eps):
        ep_schedule.append({
            "stage": 1,
            "n_perturb": 0,
            "block_id": 0,
        })
    for ep in range(stage2_eps):
        ep_schedule.append({
            "stage": 2,
            "n_perturb": stage2_n_perturb,
            "block_id": 1,
        })

    traj_rows = []
    n_actions = int(env.action_space.n)
    pe_ema_s, pe_ema_l, pe_prev = 0.05, 0.05, 0.05

    # Actor advantage normalization (AAAI/phase 1 style). Used in stage 1 only.
    baseline_stats = EMAMeanVar(beta=args.actor_baseline_beta)
    adv_stats = EMAMeanVar(beta=args.actor_std_beta)

    warmup_episodes = int(max(0, args.warmup_episodes))
    if warmup_episodes >= stage1_eps:
        print(f"[warn] warmup_episodes ({warmup_episodes}) >= stage1_episodes "
              f"({stage1_eps}) — actor will never train. Continuing.")
    print(f"[train] protocol={args.perturb_protocol}  "
          f"stage1={stage1_eps} eps (warmup {warmup_episodes}, then actor on)  "
          f"stage2={stage2_eps} eps (n_perturb={stage2_n_perturb})  "
          f"total={total_eps} eps")

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

    # Optimizer setup. We start with all params trainable (stage 1).
    # At stage 1 → 2 transition we'll rebuild the optimizer with only
    # the gating + AlphaNet params trainable.
    prev_stage = 0  # neither stage; will fire transition logic on first ep

    for global_ep, ep_info in enumerate(ep_schedule):
        cur_stage = ep_info["stage"]
        n_perturb_now = ep_info["n_perturb"]
        block_id = ep_info["block_id"]

        # ─── Stage transitions ──────────────────────────────────────────
        if cur_stage != prev_stage:
            if cur_stage == 1:
                # Entering stage 1: backbone training, ablate_g=True,
                # alpha fixed, actor active after warmup.
                # World update_mode is irrelevant during ablate_g but set
                # to "fixed" for clarity / trajectory logging.
                agent.world.cfg.update_mode = "fixed"
                agent.world.cfg.alpha_fixed = float(args.stage1_alpha_fixed)
                # Disable valenced affordance + body homeostasis. The
                # commitment is that stage 1 actor learns from the sigma
                # gradient ONLY: predictability seeking, no body / valence
                # confound. Body remains static at body_init (0.5).
                env.cfg.affordance_top = 0.0
                env.cfg.affordance_bottom = 0.0
                env.cfg.metabolic_cost = 0.0
                env.cfg.movement_cost = 0.0
                env.cfg.affordance_to_body_gain = 0.0
                # Rebuild the affordance map with the disabled values so
                # the env's per-cell affordance channel reads as zero.
                env._affordance_map = env._build_affordance_map()
                # All params trainable; build optimizer fresh
                for p in all_params:
                    p.requires_grad_(True)
                optimizer = torch.optim.Adam(all_params, lr=args.lr)
                print(f"[stage 1] backbone training: ablate_g=True, "
                      f"alpha_fixed={args.stage1_alpha_fixed}, "
                      f"affordance OFF, body homeostasis OFF, "
                      f"all params trainable")
            elif cur_stage == 2:
                # Entering stage 2: freeze the POLICY only. Everything else
                # (encoder + gating, decoder, state head incl. body head,
                # AlphaNet) remains trainable.
                #
                # Commitment: the agent retains the predictability-seeking
                # *behavioral tendency* learned in stage 1 (frozen policy),
                # while the rest of the world model adapts to the newly
                # re-enabled affordance + body homeostasis.
                agent.world.cfg.update_mode = "adaptive"
                env.cfg.affordance_top = orig_env_cfg["affordance_top"]
                env.cfg.affordance_bottom = orig_env_cfg["affordance_bottom"]
                env.cfg.metabolic_cost = orig_env_cfg["metabolic_cost"]
                env.cfg.movement_cost = orig_env_cfg["movement_cost"]
                env.cfg.affordance_to_body_gain = \
                    orig_env_cfg["affordance_to_body_gain"]
                env._affordance_map = env._build_affordance_map()
                # Start with everything trainable
                for p in all_params:
                    p.requires_grad_(True)
                # Then freeze policy only
                for p in agent.policy.parameters():
                    p.requires_grad_(False)
                trainable_params = [p for p in all_params if p.requires_grad]
                if not trainable_params:
                    raise RuntimeError(
                        "No trainable params in stage 2 — check policy/agent setup."
                    )
                optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
                # Save intermediate stage-1 ckpt
                stage1_ckpt_path = (Path(args.outdir) / "ckpt_stage1.pt") \
                    if args.outdir else None
                if stage1_ckpt_path:
                    stage1_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "agent_state_dict": agent.state_dict(),
                        "decoder_state_dict": decoder.state_dict(),
                        "meta": {"stage": 1, "args": vars(args),
                                 "stage1_episodes_done": global_ep},
                    }, stage1_ckpt_path)
                    print(f"[stage 1→2] saved {stage1_ckpt_path}")
                n_trainable = sum(p.numel() for p in trainable_params)
                n_total = sum(p.numel() for p in all_params)
                gating_mode = agent.cfg.encoder.gating_mode.lower()
                print(f"[stage 2] perspective formation: policy frozen, "
                      f"affordance ON, body homeostasis ON, "
                      f"trainable={n_trainable}/{n_total} params, "
                      f"gating_mode={gating_mode}, n_perturb={n_perturb_now}")
            prev_stage = cur_stage

        # Stage-specific flags for forward + loss
        ablate_g_now = (cur_stage == 1)
        # Actor active in stage 1 only, after warmup
        actor_active = (cur_stage == 1 and global_ep >= warmup_episodes)

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

            out = agent.forward_step(
                x_t, p_t,
                err_t=err_t,
                body_actual_t=body_actual_t,
                ablate_g=ablate_g_now,
            )
            # logits returned by forward_step come from policy(s_t) WITHOUT
            # stop-gradient. We compute a separate set of "actor logits" from
            # detached s_t to enforce the AAAI commitment: actor objective
            # must not flow into the perspective formation cycle.
            logits_pred = out["logits"]                  # gradient-attached, not used for sampling here
            s_t = out["s"]
            logits_act = agent.policy(s_t.detach())      # detached → actor grad blocked at s
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
            #   c_t = MSE(decoder(g_t, a_t), obs_{t+1})  — executed-action PE
            #   advantage = (c_t - baseline) / std   (clipped)
            #   L_actor = advantage * log π(a_t | s_t)
            # Note: cost c_t is detached (treated as scalar advantage signal),
            # so backprop from L_actor flows only through log π(a_t | s_t),
            # i.e., into policy parameters. With s_t detached, no gradient
            # reaches the perspective layer.
            xhat_executed = xhat_all[0, a_int].unsqueeze(0)       # (1, obs_dim)
            cost_t = F.mse_loss(xhat_executed, x_next).detach().item()
            baseline_stats.update(cost_t)
            # Sign convention (matches phase 1 train_phase1.py:344):
            #   advantage = baseline - cost
            # → low cost (good action) yields *positive* advantage
            # → loss = -(advantage · log π); minimizing loss = increasing log π
            #   for low-cost actions = strengthening preference for them.
            # The opposite sign (cost - baseline) trains the agent to AVOID
            # low-cost actions, which is the wrong direction for predictability
            # seeking.
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

            # ─── Total loss with stage-aware weighting ───
            # Stage 1 warmup:    obs + body + smooth                (actor off)
            # Stage 1 post-warm: obs + body + smooth + actor + ent  (actor on)
            # Stage 2:           obs + body + smooth                (actor off)
            #   In stage 2, actor + entropy are off because policy is frozen
            #   anyway; including them would add noise to the gradient stream
            #   without changing trainable params.
            if actor_active:
                w_actor_now = float(args.w_actor)
                w_ent_now = float(args.w_entropy)
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
            if cur_stage == 1:
                phase_str = "S1-wrm" if global_ep < warmup_episodes else "S1-act"
            else:
                phase_str = "S2-prs"
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

            # 3×3 zone occupancy in this print block:
            #   x ∈ [0,4]=L, [5,9]=M, [10,14]=R   (predictability axis)
            #   y ∈ [0,4]=T, [5,9]=M, [10,14]=B   (valence axis: T=positive)
            # Cell labels: TL TM TR / ML MM MR / BL BM BR
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
            # Top-2 dominant zones
            sorted_zones = sorted(
                zone_counts.items(), key=lambda kv: kv[1], reverse=True
            )
            zone_str = " ".join(
                f"{name}={cnt / n_zones:.2f}"
                for name, cnt in sorted_zones[:2]
                if cnt > 0
            )
            print(
                f"[ep {global_ep+1:4d}/{len(ep_schedule)}] [{phase_str}] "
                f"obs_PE={np.mean(ep_obs_pe):.4f} "
                f"body_PE={np.mean(ep_body_pe):.4f} "
                f"||g||={out['g'].detach().cpu().norm().item():.3f} "
                f"α={out['alpha'].item():.3f} | "
                f"xy=({mean_x:.1f},{mean_y:.1f}) zones[{zone_str}] "
                f"UDLRS={fU:.2f}/{fD:.2f}/{fL:.2f}/{fR:.2f}/{fS:.2f} "
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
    ap.add_argument("--stage1_episodes", type=int, default=80,
                    help="Number of episodes for stage 1 (backbone training, "
                         "ablate_g=True, fixed alpha)")
    ap.add_argument("--stage2_episodes", type=int, default=170,
                    help="Number of episodes for stage 2 (perspective formation, "
                         "backbone frozen, gating + AlphaNet trainable)")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--save_traj", action="store_true")

    # Two-stage protocol
    ap.add_argument("--perturb_protocol", type=str, default="P3",
                    choices=["P3", "P4"],
                    help="P3: perturbation off in both stages (assay-only); "
                         "P4: perturbation off in stage 1, on in stage 2")
    ap.add_argument("--stage1_alpha_fixed", type=float, default=0.10,
                    help="Fixed alpha value during stage 1 backbone training. "
                         "Used because stage 1 forward bypasses world latent.")

    # AAAI/phase 1-style learning objective (used in stage 1 only)
    # Warm-up: actor objective disabled for the first warmup_episodes,
    # during which only obs prediction + body prediction train the backbone.
    ap.add_argument("--warmup_episodes", type=int, default=20,
                    help="Within stage 1: episodes during which actor is OFF. "
                         "After warmup, actor is ON until end of stage 1. "
                         "Stage 2 has actor permanently OFF.")
    ap.add_argument("--w_actor", type=float, default=0.5)
    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_entropy", type=float, default=0.01)
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
    ap.add_argument("--metabolic_cost", type=float, default=0.002)
    ap.add_argument("--movement_cost", type=float, default=0.002)
    ap.add_argument("--affordance_gain", type=float, default=0.02)

    # Body loss weight (training scaling, not runtime preference)
    ap.add_argument("--body_loss_weight", type=float, default=1.0)

    # Encoder gating mode. metric_c1 is the Phase 3 main commitment;
    # film and salience are sanity-check baselines.
    ap.add_argument("--gating_mode", type=str, default="film",
                    choices=["film", "salience", "metric_c1", "both"])

    ap.add_argument("--mixed_schedule", type=str, default="",
                    help="Block schedule: 'n_perturb:episodes,...'")

    return ap.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()