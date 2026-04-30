# cear_pilot/models/encoder.py
# -*- coding: utf-8 -*-
"""
Observation encoder with g-conditioned modulation.

Modes (cfg.gating_mode):

  "film"      — Phase 1-2 baseline (sanity check). Affine modulation:
                    z_t = (1 + gamma(g)) * z_raw + beta(g)

  "salience"  — Phase 3 first attempt (sanity check). Sigmoid salience field:
                    s_t = sigmoid(Linear(g))
                    z_t = z_raw * s_t
                Dim-wise gating, no inter-dimensional structure.

  "metric_c1" — Phase 3 main commitment. Stance-dependent Riemannian metric
                on z space (Cholesky-parameterized, positive definite):
                    L_g = lower_triangular(MLP(g))
                    M_g = L_g L_g^T + epsilon * I
                z_raw is *preserved unchanged*. Downstream computation
                receives both z_raw and M_g, and uses M_g via Form C-B
                (quadratic features) inside the state head:
                    z_quad_ij = z_raw_i * (M_g z_raw)_j
                Inter-dimensional coupling is stance-dependent through
                the off-diagonals of M_g; probe-dependent reorganization
                arises through the quadratic interaction with z_raw.

  "both"      — sanity check ablation: FiLM + salience composed.

Init: zeros so all modes start as identity-like.
- film: gamma=0, beta=0 → z_raw passes through.
- salience: sigmoid(0)=0.5 → z_raw scaled by 0.5.
- metric_c1: L_g = 0 → M_g = epsilon*I (isotropic), no preferred direction.

Backward compat: gating_mode defaults to "film". Old ckpts load cleanly.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

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
    gating_mode: str = "film"
    # metric_c1 settings
    metric_epsilon: float = 0.1     # added to L L^T for positive definiteness
    metric_hidden: int = 64         # hidden dim of MLP that produces L_g entries


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
        if mode not in ("film", "salience", "metric_c1", "both"):
            raise ValueError(f"unknown gating_mode: {cfg.gating_mode}")

        # FiLM branch (mode in {"film", "both"})
        if cfg.use_salience_gate and mode in ("film", "both"):
            self.film = nn.Linear(cfg.g_dim, cfg.z_dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        # Salience branch (mode in {"salience", "both"})
        if cfg.use_salience_gate and mode in ("salience", "both"):
            self.salience = nn.Linear(cfg.g_dim, cfg.z_dim)
            nn.init.zeros_(self.salience.weight)
            nn.init.zeros_(self.salience.bias)

        # Metric C-1 branch (mode == "metric_c1")
        if mode == "metric_c1":
            n_lower = cfg.z_dim * (cfg.z_dim + 1) // 2  # entries of lower-triangular
            self.metric_mlp = MLP(cfg.g_dim, n_lower,
                                  hidden=cfg.metric_hidden, dropout=cfg.dropout)
            # zero-init last layer so L_g starts at 0 → M_g = epsilon * I
            last = self.metric_mlp.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            # Precompute lower-triangular indices (z_dim, z_dim) → (n_lower,)
            tril_idx = torch.tril_indices(cfg.z_dim, cfg.z_dim, offset=0)
            self.register_buffer("tril_row", tril_idx[0])
            self.register_buffer("tril_col", tril_idx[1])
            # epsilon scaled identity for positive definiteness
            self.register_buffer("metric_eye",
                                 torch.eye(cfg.z_dim) * float(cfg.metric_epsilon))

    @property
    def gating_mode(self) -> str:
        return (self.cfg.gating_mode or "film").lower()

    def _build_metric(self, g_t: torch.Tensor) -> torch.Tensor:
        """Build M_g = L L^T + epsilon * I (positive definite) per batch."""
        B = g_t.shape[0]
        z_dim = self.cfg.z_dim
        l_entries = self.metric_mlp(g_t)              # (B, n_lower)

        # Construct lower-triangular L_g
        L = torch.zeros(B, z_dim, z_dim,
                        device=g_t.device, dtype=g_t.dtype)
        # Apply softplus to diagonal entries to ensure positive diagonals
        # (this makes M_g strictly positive definite, not just semi-definite)
        diag_mask = (self.tril_row == self.tril_col)
        l_diag_part = torch.nn.functional.softplus(l_entries[:, diag_mask])
        l_offdiag_part = l_entries[:, ~diag_mask]
        # scatter
        L[:, self.tril_row[diag_mask], self.tril_col[diag_mask]] = l_diag_part
        L[:, self.tril_row[~diag_mask], self.tril_col[~diag_mask]] = l_offdiag_part

        # M_g = L L^T + epsilon * I
        M = torch.bmm(L, L.transpose(-2, -1)) + self.metric_eye.unsqueeze(0)
        return M

    def forward(
        self,
        x_t: torch.Tensor,
        g_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns z_t. For metric_c1 mode, z_t IS z_raw (no in-encoder gating);
        the metric M_g is consumed downstream via gating_params() / state head.
        """
        z_raw = torch.tanh(self.mlp(x_t))

        if g_t is None or not self.cfg.use_salience_gate:
            return z_raw

        mode = self.gating_mode
        if mode == "metric_c1":
            # z_raw is preserved; metric is delivered through gating_params()
            return z_raw

        z_t = z_raw
        if mode in ("film", "both") and hasattr(self, "film"):
            gamma, beta = self.film(g_t).chunk(2, dim=-1)
            z_t = (1.0 + gamma) * z_t + beta
        if mode in ("salience", "both") and hasattr(self, "salience"):
            s = torch.sigmoid(self.salience(g_t))
            z_t = z_t * s
        return z_t

    @torch.no_grad()
    def gating_params(self, g_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return raw gating parameters for diagnostic / probe use.

        Keys depend on mode:
          "film":      {"gamma", "beta"}
          "salience":  {"salience"}
          "metric_c1": {"M_g"}        (B, z_dim, z_dim) symmetric PD
          "both":      {"gamma", "beta", "salience"}
        """
        out: Dict[str, torch.Tensor] = {}
        mode = self.gating_mode
        if mode in ("film", "both") and hasattr(self, "film"):
            gamma, beta = self.film(g_t).chunk(2, dim=-1)
            out["gamma"] = gamma
            out["beta"] = beta
        if mode in ("salience", "both") and hasattr(self, "salience"):
            out["salience"] = torch.sigmoid(self.salience(g_t))
        if mode == "metric_c1" and hasattr(self, "metric_mlp"):
            out["M_g"] = self._build_metric(g_t)
        return out

    def build_metric(self, g_t: torch.Tensor) -> torch.Tensor:
        """Public differentiable accessor for M_g (used by state head in
        metric_c1 mode). For other modes, returns identity."""
        if self.gating_mode == "metric_c1" and hasattr(self, "metric_mlp"):
            return self._build_metric(g_t)
        # fallback: identity (no metric structure)
        B = g_t.shape[0]
        return self.metric_eye_fallback(B, g_t.device, g_t.dtype)

    def metric_eye_fallback(self, B: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(self.cfg.z_dim, device=device, dtype=dtype)
        return eye.unsqueeze(0).expand(B, -1, -1)


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
