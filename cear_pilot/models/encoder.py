# cear_pilot/models/encoder.py
# -*- coding: utf-8 -*-
"""
Observation encoder with g-conditioned gating.

Three gating modes (cfg.gating_mode):

  "film"     — Phase 1-2 baseline. Affine modulation:
                   z_t = (1 + gamma(g)) * z_raw + beta(g)
               Init: zeros so gating starts as identity.

  "salience" — Phase 3 Frame 1 commitment. Sigmoid salience field:
                   s_t = sigmoid(Linear(g_t))    # in (0, 1), per dim
                   z_t = z_raw * s_t
               z_raw is *not reshaped*; only weighted dim-by-dim.
               Init: zeros so sigmoid(0) = 0.5 (neutral starting stance).

  "both"     — ablation: FiLM applied first, salience applied after.
                   z_t = ((1 + gamma) * z_raw + beta) * s_t

The "salience" mode is the architectural realization of "same content,
different mode of givenness": z_raw carries the representational content,
salience modulates which dimensions are foregrounded under stance g.
Unlike FiLM (affine in z_raw), salience is nonlinear in z_raw via sigmoid,
so probe-dependent transformations become possible.

Backward compat: cfg.gating_mode defaults to "film". Existing checkpoints
load cleanly with mode="film".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    obs_dim: int = 8
    proprio_dim: int = 5
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12
    hidden: int = 64
    dropout: float = 0.0
    use_salience_gate: bool = True
    # Gating mode for g-conditioned modulation. Default "film" preserves
    # phase 1-2 behavior. "salience" is the phase 3 Frame 1 commitment.
    gating_mode: str = "film"


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(cfg.obs_dim, cfg.z_dim, cfg.hidden, cfg.dropout)

        mode = (cfg.gating_mode or "film").lower()
        if mode not in ("film", "salience", "both"):
            raise ValueError(f"unknown gating_mode: {cfg.gating_mode}")

        # FiLM branch (used by mode in {"film", "both"})
        if cfg.use_salience_gate and mode in ("film", "both"):
            self.film = nn.Linear(cfg.g_dim, cfg.z_dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        # Salience branch (used by mode in {"salience", "both"})
        if cfg.use_salience_gate and mode in ("salience", "both"):
            self.salience = nn.Linear(cfg.g_dim, cfg.z_dim)
            # zero init → sigmoid(0) = 0.5, neutral starting stance.
            # Learning is symmetrically free to amplify (>0.5) or
            # suppress (<0.5) any z dimension.
            nn.init.zeros_(self.salience.weight)
            nn.init.zeros_(self.salience.bias)

    @property
    def gating_mode(self) -> str:
        return (self.cfg.gating_mode or "film").lower()

    def forward(
        self,
        x_t: torch.Tensor,
        g_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_raw = torch.tanh(self.mlp(x_t))

        if g_t is None or not self.cfg.use_salience_gate:
            return z_raw

        mode = self.gating_mode
        z_t = z_raw

        if mode in ("film", "both") and hasattr(self, "film"):
            gamma, beta = self.film(g_t).chunk(2, dim=-1)
            z_t = (1.0 + gamma) * z_t + beta

        if mode in ("salience", "both") and hasattr(self, "salience"):
            s = torch.sigmoid(self.salience(g_t))
            z_t = z_t * s

        return z_t

    # ----- introspection helpers (used by probe analysis) -----

    @torch.no_grad()
    def gating_params(self, g_t: torch.Tensor) -> dict:
        """Return raw gating parameters for diagnostic / probe use.
        Keys depend on mode:
          - "film":     {"gamma", "beta"}
          - "salience": {"salience"}
          - "both":     {"gamma", "beta", "salience"}
        """
        out: dict = {}
        mode = self.gating_mode
        if mode in ("film", "both") and hasattr(self, "film"):
            gamma, beta = self.film(g_t).chunk(2, dim=-1)
            out["gamma"] = gamma
            out["beta"] = beta
        if mode in ("salience", "both") and hasattr(self, "salience"):
            out["salience"] = torch.sigmoid(self.salience(g_t))
        return out


class ProprioEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.mlp = MLP(cfg.proprio_dim, cfg.p_dim, cfg.hidden, cfg.dropout)

    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(p_t))


class EncoderBundle(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_enc = ObservationEncoder(cfg)
        self.prop_enc = ProprioEncoder(cfg)

    def forward(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        g_t: Optional[torch.Tensor] = None,
    ):
        z_t = self.obs_enc(x_t, g_t=g_t)
        if p_t is None:
            B = x_t.shape[0]
            p_emb = torch.zeros((B, self.cfg.p_dim), device=x_t.device, dtype=x_t.dtype)
        else:
            p_emb = self.prop_enc(p_t)
        return z_t, p_emb
