# cear_pilot/models/state_head.py
# -*- coding: utf-8 -*-
"""
State head: predicts next-step state s_t and (Phase 3) body state.

Phase 1-2 backward compat (body_dim=0, use_metric=False):
  s_t = tanh(LayerNorm(net(z, p, g)))

Phase 3 default (body_dim=1):
  + separate body_pred head from same input.

Phase 3 metric_c1 (body_dim=1, use_metric=True): Form C-B quadratic features.
  M_g (z_dim, z_dim) Riemannian metric on z space (from encoder, stance-dep).
  Quadratic features: z_quad_ij = z_raw_i * (M_g z_raw)_j  → (z_dim, z_dim) → flat.
  State head input: [z_raw, z_quad_flat, p_emb, g].

  Inter-dim coupling enters through M_g's off-diagonals.
  Probe-dependent reorganization arises because z_quad is quadratic in z_raw —
  same g induces different z_quad patterns for different probe x.

  Body head shares the same metric-aware input.

The body head uses sigmoid output to match env's body range [0, 1].
Loss in train_phase3.py compares body_pred(t) vs body_actual(t+1).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StateHeadConfig:
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12
    s_dim: int = 16
    hidden: int = 64
    dropout: float = 0.0
    g_influence: float = 1.0
    body_dim: int = 0
    body_hidden: int = 32
    # Phase 3 metric_c1: enable quadratic features from M_g.
    # When True, state head expects M_g passed to forward().
    use_metric: bool = False


class StateHead(nn.Module):
    def __init__(self, cfg: StateHeadConfig):
        super().__init__()
        self.cfg = cfg

        # Compute total input dim for state head.
        # Standard: z + p + g
        # With metric: z + z_quad (z_dim^2) + p + g
        in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim
        if cfg.use_metric:
            in_dim += cfg.z_dim * cfg.z_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.s_dim),
        )
        self.ln = nn.LayerNorm(cfg.s_dim)

        # Phase 3: body head shares the same input.
        if cfg.body_dim > 0:
            self.body_head = nn.Sequential(
                nn.Linear(in_dim, cfg.body_hidden),
                nn.Tanh(),
                nn.Linear(cfg.body_hidden, cfg.body_dim),
            )
            last = self.body_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    @staticmethod
    def _quadratic_features(
        z: torch.Tensor, M: torch.Tensor,
    ) -> torch.Tensor:
        """
        z:  (B, z_dim)
        M:  (B, z_dim, z_dim)
        returns: z_quad_flat (B, z_dim * z_dim)

        z_quad[b, i, j] = z[b, i] * (M[b] @ z[b])[j]
        """
        Mz = torch.bmm(M, z.unsqueeze(-1)).squeeze(-1)         # (B, z_dim)
        z_outer = z.unsqueeze(-1) * Mz.unsqueeze(-2)            # (B, z_dim, z_dim)
        return z_outer.reshape(z.shape[0], -1)

    def _build_input(
        self,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        g_t: torch.Tensor,
        M_g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g_scaled = g_t * float(self.cfg.g_influence)
        if self.cfg.use_metric:
            if M_g is None:
                # fallback: identity metric → quadratic = z_i * z_j
                B = z_t.shape[0]
                M_g = torch.eye(self.cfg.z_dim,
                                device=z_t.device, dtype=z_t.dtype)
                M_g = M_g.unsqueeze(0).expand(B, -1, -1)
            z_quad = self._quadratic_features(z_t, M_g)
            return torch.cat([z_t, z_quad, p_emb, g_scaled], dim=-1)
        return torch.cat([z_t, p_emb, g_scaled], dim=-1)

    def forward(
        self,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        g_t: torch.Tensor,
        M_g: Optional[torch.Tensor] = None,
    ):
        """
        If body_dim == 0: returns s_t.
        If body_dim  > 0: returns (s_t, body_pred_t).
        M_g (B, z_dim, z_dim) required if cfg.use_metric is True.
        """
        x = self._build_input(z_t, p_emb, g_t, M_g=M_g)
        s = torch.tanh(self.net(x))
        s = self.ln(s)

        if self.cfg.body_dim > 0:
            body_pred = torch.sigmoid(self.body_head(x))
            return s, body_pred
        return s
