# cear_pilot/models/decoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    z_dim: int = 20      # gated percept dim (z_fused: z_obs + z_body)
    g_dim: int = 12
    n_actions: int = 5
    obs_dim: int = 8
    hidden: int = 64
    dropout: float = 0.0


class ObsDecoder(nn.Module):
    """One-step observation prediction: x_hat_{t+1} = D(z_t, g_t, a_t).

    Phase 3 commitment: z_t carries the fast predictive state (current
    local percept), g_t carries the slow perspective / context bias.
    Earlier phases used D(g_t, a_t) only — that forced g to absorb fast
    predictive state info, breaking g's role as a slow perspective
    latent. With z_t routed in directly, local prediction is grounded
    in z, and g modulates only stance/context.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.z_dim + cfg.g_dim + cfg.n_actions
        self.g_ln = nn.LayerNorm(cfg.g_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.obs_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        g_t: torch.Tensor,
        a_onehot: torch.Tensor,
    ) -> torch.Tensor:
        g = self.g_ln(g_t)
        x = torch.cat([z_t, g, a_onehot], dim=-1)
        return self.net(x)

    def predict_all_actions(
        self,
        z_t: torch.Tensor,
        g_t: torch.Tensor,
    ) -> torch.Tensor:
        B = g_t.shape[0]
        A = self.cfg.n_actions
        device = g_t.device
        dtype = g_t.dtype

        g = self.g_ln(g_t)

        eye = torch.eye(A, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        g_rep = g.unsqueeze(1).repeat(1, A, 1)
        z_rep = z_t.unsqueeze(1).repeat(1, A, 1)

        x = torch.cat([z_rep, g_rep, eye], dim=-1)
        out = self.net(x.view(B * A, -1)).view(B, A, -1)
        return out