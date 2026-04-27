# cear_pilot/models/agent.py
# -*- coding: utf-8 -*-
"""
CEAR Agent.

Phase 2: g_prev gates perception via FiLM (encoder).
Phase 3 addition: body prediction error flows into AlphaNet as a separate
channel (Option I). Mechanism:
  - state head outputs (s, body_pred) when body_dim > 0
  - body_pred(t-1) is stored on the agent
  - at step t, body_actual_t (read from observation/info) is compared
    against body_pred(t-1) to compute body_pe
  - body_pe is passed to world_latent forward as body_err_t
  - the world_latent's AlphaNet learns to weight env PE vs body PE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cear_pilot.models.encoder import EncoderBundle, EncoderConfig
from cear_pilot.models.world_latent import WorldLatent, WorldLatentConfig
from cear_pilot.models.state_head import StateHead, StateHeadConfig
from cear_pilot.models.policy import PolicyNet, PolicyConfig


@dataclass
class AgentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    world: WorldLatentConfig = field(default_factory=WorldLatentConfig)
    state: StateHeadConfig = field(default_factory=StateHeadConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    device: str = "cpu"


class CEARAgent(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.encoder.z_dim == cfg.world.z_dim
        assert cfg.encoder.p_dim == cfg.world.p_dim
        assert cfg.world.g_dim == cfg.state.g_dim
        assert cfg.encoder.z_dim == cfg.state.z_dim
        assert cfg.encoder.p_dim == cfg.state.p_dim
        assert cfg.state.s_dim == cfg.policy.s_dim
        # Phase 3: state.body_dim must match world.body_err_dim if body
        # PE is wired through. body_err_dim has the same dim as body_dim
        # because PE = (pred - actual) is element-wise.
        if cfg.state.body_dim > 0:
            assert cfg.state.body_dim == cfg.world.body_err_dim, (
                f"state.body_dim ({cfg.state.body_dim}) must equal "
                f"world.body_err_dim ({cfg.world.body_err_dim})"
            )

        self.enc = EncoderBundle(cfg.encoder)
        self.world = WorldLatent(cfg.world)
        self.state = StateHead(cfg.state)
        self.policy = PolicyNet(cfg.policy)

        self.device_ = torch.device(cfg.device)
        self.to(self.device_)

        self._g: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        # Phase 3: body prediction from previous step (used to compute body PE
        # at the *next* step when the actual next-step body is known).
        self._body_pred_prev: Optional[torch.Tensor] = None

    @property
    def has_body(self) -> bool:
        return self.cfg.state.body_dim > 0

    def reset(self, batch_size: int = 1) -> None:
        gd = self.cfg.world.g_dim
        self._g = torch.zeros((batch_size, gd), device=self.device_, dtype=torch.float32)
        self._alpha = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)
        if self.has_body:
            # init body_pred at 0.5 (matches env body_init under default cfg);
            # this gives a defined-but-uninformative first body PE.
            self._body_pred_prev = torch.full(
                (batch_size, self.cfg.state.body_dim),
                0.5,
                device=self.device_,
                dtype=torch.float32,
            )
        else:
            self._body_pred_prev = None

    def get_latents(self) -> Dict[str, torch.Tensor]:
        if self._g is None:
            raise RuntimeError("Call reset() first.")
        out = {"g": self._g, "alpha": self._alpha}
        if self.has_body and self._body_pred_prev is not None:
            out["body_pred_prev"] = self._body_pred_prev
        return out

    @torch.no_grad()
    def set_g(self, g_new: torch.Tensor) -> None:
        if g_new.ndim != 2 or g_new.shape[-1] != self.cfg.world.g_dim:
            raise ValueError(f"Expected (B, {self.cfg.world.g_dim}), got {tuple(g_new.shape)}")
        self._g = g_new.to(self.device_).detach().clone()

    def forward_step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        ablate_g: bool = False,
        err_t: Optional[torch.Tensor] = None,
        body_actual_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 3 forward step.

        Args:
          x_t: observation (B, obs_dim)
          p_t: proprioception (optional)
          ablate_g: zero out g (ablation)
          err_t: env prediction error feature (B, err_dim) for AlphaNet
          body_actual_t: actual body state at time t (B, body_dim).
                         Required if agent has body. Used to compute body PE
                         against self._body_pred_prev (which is body_pred(t-1)
                         pointing at the body at time t).
        """
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        # Phase 3: body PE for AlphaNet
        body_pe: Optional[torch.Tensor] = None
        if self.has_body and body_actual_t is not None and self._body_pred_prev is not None:
            body_actual_t = body_actual_t.to(self.device_).float()
            # PE as signed error; AlphaNet receives this directly so it can
            # learn signed vs squared weighting.
            body_pe = body_actual_t - self._body_pred_prev
            body_pe = body_pe.detach()  # PE is a feedback signal, not a path
                                        # for backprop into the previous step's
                                        # body head (we train body head from
                                        # next-step body label directly).

        # encoder (Phase 2 FiLM gating preserved)
        z_t, p_emb = self.enc(x_t, p_t, g_t=self._g)

        if ablate_g:
            g_t = torch.zeros_like(self._g)
            alpha_t = torch.zeros((x_t.shape[0], 1), device=x_t.device)
        else:
            out_world = self.world(
                self._g, z_t, p_emb,
                err_t=err_t,
                body_err_t=body_pe,
            )
            g_t = out_world["g"]
            alpha_t = out_world["alpha"]

        # state head + (phase 3) body head
        if self.has_body:
            s_t, body_pred_t = self.state(z_t, p_emb, g_t)
        else:
            s_t = self.state(z_t, p_emb, g_t)
            body_pred_t = None

        logits = self.policy(s_t)

        # update internal state
        self._g = g_t.detach()
        self._alpha = alpha_t.detach()
        if self.has_body and body_pred_t is not None:
            # store for next step's PE computation
            self._body_pred_prev = body_pred_t.detach()

        out: Dict[str, torch.Tensor] = {
            "z": z_t,
            "p_emb": p_emb,
            "g": g_t,
            "alpha": alpha_t,
            "s": s_t,
            "logits": logits,
        }
        if body_pred_t is not None:
            out["body_pred"] = body_pred_t
        if body_pe is not None:
            out["body_pe"] = body_pe
        return out

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        greedy: bool = False,
        ablate_g: bool = False,
        err_t: Optional[torch.Tensor] = None,
        body_actual_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward_step(
            x_t, p_t,
            ablate_g=ablate_g,
            err_t=err_t,
            body_actual_t=body_actual_t,
        )
        action = self.policy.sample_action(out["logits"], greedy=greedy)
        return action, out
