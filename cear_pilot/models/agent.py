# cear_pilot/models/agent.py
# -*- coding: utf-8 -*-
"""
CEAR Agent.

Phase 2: g_prev gates perception via FiLM (encoder).
Phase 3 additions:
  - body PE flows into AlphaNet as a separate channel (Option I)
  - encoder.gating_mode in {"film", "salience", "metric_c1", "both"}
  - if encoder is metric_c1: M_g (Riemannian metric on z) is computed by the
    encoder and passed through to the state head, which consumes it via
    quadratic features (Form C-B). z_t is the *unmodulated z_raw* in this mode;
    the modulation lives entirely in M_g, applied downstream.
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
        if cfg.state.body_dim > 0:
            assert cfg.state.body_dim == cfg.world.body_err_dim, (
                f"state.body_dim ({cfg.state.body_dim}) must equal "
                f"world.body_err_dim ({cfg.world.body_err_dim})"
            )
        # metric_c1 + use_metric must agree
        if cfg.encoder.gating_mode == "metric_c1" and not cfg.state.use_metric:
            raise ValueError("encoder gating_mode='metric_c1' requires "
                             "state.use_metric=True")

        self.enc = EncoderBundle(cfg.encoder)
        self.world = WorldLatent(cfg.world)
        self.state = StateHead(cfg.state)
        self.policy = PolicyNet(cfg.policy)

        self.device_ = torch.device(cfg.device)
        self.to(self.device_)

        self._g: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._body_pred_prev: Optional[torch.Tensor] = None

    @property
    def has_body(self) -> bool:
        return self.cfg.state.body_dim > 0

    @property
    def has_metric(self) -> bool:
        return self.cfg.encoder.gating_mode == "metric_c1"

    def reset(self, batch_size: int = 1) -> None:
        gd = self.cfg.world.g_dim
        self._g = torch.zeros((batch_size, gd), device=self.device_, dtype=torch.float32)
        self._alpha = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)
        if self.has_body:
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
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        # Body PE (signed) — feedback signal, not a backprop path
        body_pe: Optional[torch.Tensor] = None
        if self.has_body and body_actual_t is not None and self._body_pred_prev is not None:
            body_actual_t = body_actual_t.to(self.device_).float()
            body_pe = body_actual_t - self._body_pred_prev
            body_pe = body_pe.detach()

        # encoder
        z_t, p_emb = self.enc(x_t, p_t, g_t=self._g)

        # world latent — alpha modulated by env PE + body PE
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

        # state head — receives M_g if metric_c1
        M_g: Optional[torch.Tensor] = None
        if self.has_metric:
            M_g = self.enc.obs_enc.build_metric(g_t)

        if self.has_body:
            s_t, body_pred_t = self.state(z_t, p_emb, g_t, M_g=M_g)
        else:
            s_t = self.state(z_t, p_emb, g_t, M_g=M_g)
            body_pred_t = None

        logits = self.policy(s_t)

        self._g = g_t.detach()
        self._alpha = alpha_t.detach()
        if self.has_body and body_pred_t is not None:
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
        if M_g is not None:
            out["M_g"] = M_g
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
