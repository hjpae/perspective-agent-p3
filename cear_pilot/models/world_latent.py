# cear_pilot/models/world_latent.py
# -*- coding: utf-8 -*-
"""
Perspective latent with self-modulating plasticity.

Phase 1-2: alpha is modulated by env prediction error (err_t).
Phase 3 addition: body prediction error (body_err_t) as a separate channel.
This is the architectural extension of self-modulating plasticity to embodied
surprisal — body surprise drives perspective reconfiguration alongside env
surprise. AlphaNet learns the weighting between the two channels (no
hyperparameter for body-vs-env weight).

g_t = (1 - alpha_t) * g_prev + alpha_t * GRU(z_t, p_emb)
alpha_t = sigmoid(alpha_net(z_t, p_emb, g_prev, err_t [, body_err_t]))
                                                           ^^^^^^^^
                                                phase 3: separate channel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class WorldLatentConfig:
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12

    layernorm: bool = True
    update_mode: str = "adaptive"

    alpha_fixed: float = 0.10
    alpha_min: float = 0.03
    alpha_max: float = 0.30
    alpha_hidden: int = 32

    use_error_feedback: bool = True
    err_dim: int = 6  # env PE features + perturbation signal

    # Phase 3: body PE channel(s).
    # body_err_dim = 0 disables body PE entirely (phase 1-2 compat).
    # When body_err_dim > 0, body_err_t (1-d signed scalar) is available
    # for routing to AlphaNet and/or g GRU separately:
    #   body_err_in_alpha → AlphaNet input (plasticity rate)
    #   body_in_g         → g GRU input (perspective content; commitment III)
    # The two paths are independently toggleable. Default Phase 3:
    #   body_err_in_alpha = False (env PE drives plasticity rate only)
    #   body_in_g         = True  (body PE constitutive of perspective content)
    # Rationale: epistemic surprise (env PE) governs *how fast* perspective
    # revises (plasticity rate). Interoceptive surprise (body PE) governs
    # *what* the perspective is grounded on (content). Decoupling avoids
    # AlphaNet saturation cycles driven by spontaneous body PE spikes.
    body_err_dim: int = 0
    body_err_in_alpha: bool = False
    body_in_g: bool = False
    body_g_scale: float = 20.0  # scale up body_err_t (~0.05 magnitude) to be
                                # comparable to z_t/p_emb (~1.0 magnitude).

    g_damping: float = 0.10  # legacy fallback


class WorldLatent(nn.Module):
    def __init__(self, cfg: WorldLatentConfig):
        super().__init__()
        self.cfg = cfg

        # GRU input: z_t + p_emb [+ body_err_t * scale] when body_in_g=True.
        # body_err_t (1-d, signed) directly enters the perspective update
        # content — interoceptive grounding of subjectivity.
        gru_in_dim = cfg.z_dim + cfg.p_dim
        if cfg.body_in_g and int(cfg.body_err_dim) > 0:
            gru_in_dim += int(cfg.body_err_dim)
        self.gru = nn.GRUCell(
            input_size=gru_in_dim,
            hidden_size=cfg.g_dim,
        )
        self.ln = nn.LayerNorm(cfg.g_dim) if cfg.layernorm else nn.Identity()

        # AlphaNet input: z_t + p_emb + g_prev [+ err_t] [+ body_err_t]
        # body_err_t included only when body_err_in_alpha flag is True.
        alpha_in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim
        if cfg.use_error_feedback:
            alpha_in_dim += int(cfg.err_dim)
        if int(cfg.body_err_dim) > 0 and cfg.body_err_in_alpha:
            alpha_in_dim += int(cfg.body_err_dim)

        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_in_dim, cfg.alpha_hidden),
            nn.Tanh(),
            nn.Linear(cfg.alpha_hidden, 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        last_linear = self.alpha_net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.constant_(last_linear.bias, -0.75)

    def _compute_alpha(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: Optional[torch.Tensor] = None,
        body_err_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = str(self.cfg.update_mode).lower().strip()

        if mode == "fixed":
            a = float(self.cfg.alpha_fixed)
            return torch.full(
                (g_prev.shape[0], 1), a,
                dtype=g_prev.dtype, device=g_prev.device,
            )

        parts = [z_t, p_emb, g_prev]

        if self.cfg.use_error_feedback and err_t is not None:
            err_t = err_t.to(device=g_prev.device, dtype=g_prev.dtype)
            if err_t.ndim == 1:
                err_t = err_t.unsqueeze(0)
            if err_t.shape[0] == 1 and g_prev.shape[0] > 1:
                err_t = err_t.expand(g_prev.shape[0], -1)
            parts.append(err_t)

        # Phase 3: body PE channel — included in AlphaNet only when
        # body_err_in_alpha=True. When False, body PE bypasses AlphaNet
        # and only enters g GRU update content (commitment III), so
        # plasticity rate is governed solely by env PE (epistemic
        # surprise).
        if (int(self.cfg.body_err_dim) > 0
                and self.cfg.body_err_in_alpha
                and body_err_t is not None):
            body_err_t = body_err_t.to(device=g_prev.device, dtype=g_prev.dtype)
            if body_err_t.ndim == 1:
                body_err_t = body_err_t.unsqueeze(0)
            if body_err_t.shape[0] == 1 and g_prev.shape[0] > 1:
                body_err_t = body_err_t.expand(g_prev.shape[0], -1)
            parts.append(body_err_t)

        raw = self.alpha_net(torch.cat(parts, dim=-1))

        center = 0.5 * (float(self.cfg.alpha_min) + float(self.cfg.alpha_max))
        half = 0.5 * (float(self.cfg.alpha_max) - float(self.cfg.alpha_min))
        alpha = center + half * torch.tanh(raw)
        return alpha.clamp(min=float(self.cfg.alpha_min), max=float(self.cfg.alpha_max))

    def forward(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: Optional[torch.Tensor] = None,
        body_err_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # GRU input: [z_t, p_emb, body_err_t * scale] when body_in_g=True.
        # body_err_t enters the perspective update content (Phase 3
        # commitment III: interoceptive PE is constitutive of subjectivity).
        gru_parts = [z_t, p_emb]
        if (self.cfg.body_in_g and int(self.cfg.body_err_dim) > 0
                and body_err_t is not None):
            be = body_err_t.to(device=g_prev.device, dtype=g_prev.dtype)
            if be.ndim == 1:
                be = be.unsqueeze(0)
            if be.shape[0] == 1 and g_prev.shape[0] > 1:
                be = be.expand(g_prev.shape[0], -1)
            gru_parts.append(be * float(self.cfg.body_g_scale))
        x = torch.cat(gru_parts, dim=-1)
        h_t = self.ln(self.gru(x, g_prev))

        alpha_t = self._compute_alpha(g_prev, z_t, p_emb, err_t, body_err_t)
        g_t = (1.0 - alpha_t) * g_prev + alpha_t * h_t

        return {
            "g": g_t,
            "alpha": alpha_t,
        }