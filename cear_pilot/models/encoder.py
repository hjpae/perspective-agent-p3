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
    # z_dim is the *fused* z dimension (what downstream modules see).
    # When body_dim > 0, z_dim = z_obs_dim + body_z_dim.
    # When body_dim == 0, z_dim = z_obs_dim and there is no body encoder.
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
    # ── Phase 3 body-coupled architecture (Layer 2 — Option C) ────────
    # When body_dim > 0, a separate BodyEncoder produces z_body of dim
    # body_z_dim, and z_obs of dim z_obs_dim is produced from obs only.
    # z_fused = concat(z_obs, z_body) has dim z_dim, which is what
    # downstream (state_head, decoder, world_latent, metric_c1) sees
    # as the *effective z dimension*.
    # Constraint when body_dim > 0: z_obs_dim + body_z_dim must equal z_dim.
    # Default body_dim=0 keeps phase 1/2 backward compatibility (no body
    # encoder, all encoder capacity goes to obs, z_dim = z_obs_dim).
    body_dim: int = 0                # raw body state dim (e.g. 1 scalar)
    z_obs_dim: int = 16              # obs-only z dim (must add up with body_z_dim to z_dim if body_dim > 0)
    body_z_dim: int = 4              # body-only z dim
    body_hidden: int = 32            # hidden dim of body encoder MLP
    # Optional: directional affordance silhouette (interoceptive feature).
    # When silhouette_dim > 0, BodyEncoder receives [body_state, silhouette]
    # concatenated as input (dim = body_dim + silhouette_dim).
    # Default 0 disables — BodyEncoder receives body_state only.
    silhouette_dim: int = 0

    def __post_init__(self):
        if self.body_dim > 0:
            if self.z_obs_dim + self.body_z_dim != self.z_dim:
                raise ValueError(
                    f"body_dim>0 requires z_obs_dim ({self.z_obs_dim}) + "
                    f"body_z_dim ({self.body_z_dim}) = z_dim ({self.z_dim}); "
                    f"got {self.z_obs_dim + self.body_z_dim}."
                )
        else:
            # body_dim == 0: z_obs_dim must equal z_dim
            if self.z_obs_dim != self.z_dim:
                # Auto-fix for back-compat: if user did not set z_obs_dim,
                # silently align it.
                self.z_obs_dim = self.z_dim
                self.body_z_dim = 0


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
        # ObservationEncoder produces z_obs of dim z_obs_dim.
        # When body_dim == 0, z_obs_dim == z_dim (no body encoder, no fusion).
        self.mlp = MLP(cfg.obs_dim, cfg.z_obs_dim, cfg.hidden, cfg.dropout)

        mode = (cfg.gating_mode or "film").lower()
        if mode not in ("film", "salience", "metric_c1", "both"):
            raise ValueError(f"unknown gating_mode: {cfg.gating_mode}")

        # FiLM / salience operate on the *fused* z (z_dim), so they take g_t
        # and produce a (z_dim,) modulation. The body encoder produces z_body
        # ungated (body state is an internal signal, not stance-modulated).
        # Metric_c1 builds M_g as (z_dim, z_dim) over the fused z.
        z_full = cfg.z_dim                    # fused dim seen by gating

        # FiLM branch (mode in {"film", "both"})
        if cfg.use_salience_gate and mode in ("film", "both"):
            self.film = nn.Linear(cfg.g_dim, z_full * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        # Salience branch (mode in {"salience", "both"})
        if cfg.use_salience_gate and mode in ("salience", "both"):
            self.salience = nn.Linear(cfg.g_dim, z_full)
            nn.init.zeros_(self.salience.weight)
            nn.init.zeros_(self.salience.bias)

        # Metric C-1 branch (mode == "metric_c1")
        if mode == "metric_c1":
            n_lower = z_full * (z_full + 1) // 2  # entries of lower-triangular over fused z
            self.metric_mlp = MLP(cfg.g_dim, n_lower,
                                  hidden=cfg.metric_hidden, dropout=cfg.dropout)
            # zero-init last layer so L_g starts at 0 → M_g = epsilon * I
            last = self.metric_mlp.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            # Precompute lower-triangular indices (z_full, z_full) → (n_lower,)
            tril_idx = torch.tril_indices(z_full, z_full, offset=0)
            self.register_buffer("tril_row", tril_idx[0])
            self.register_buffer("tril_col", tril_idx[1])
            # epsilon scaled identity for positive definiteness (over fused z)
            self.register_buffer("metric_eye",
                                 torch.eye(z_full) * float(cfg.metric_epsilon))

    @property
    def gating_mode(self) -> str:
        return (self.cfg.gating_mode or "film").lower()

    def _build_metric(self, g_t: torch.Tensor) -> torch.Tensor:
        """Build M_g = L L^T + epsilon * I (positive definite) per batch.
        M_g is over the *fused* z dimension (z_dim), so it modulates both
        z_obs and z_body when the body encoder is active."""
        B = g_t.shape[0]
        z_full = self.cfg.z_dim
        l_entries = self.metric_mlp(g_t)              # (B, n_lower)

        # Construct lower-triangular L_g
        L = torch.zeros(B, z_full, z_full,
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

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Returns z_obs (raw, ungated). Gating is applied at the
        EncoderBundle level after fusion with z_body, so that body-coupled
        perception is what the gating modulates."""
        return torch.tanh(self.mlp(x_t))

    def apply_gating(
        self,
        z_fused: torch.Tensor,
        g_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply g-conditioned gating to the fused z (z_obs ⊕ z_body).
        For metric_c1, z_fused is preserved (M_g delivered separately
        via build_metric). For film/salience, modulation is applied
        to z_fused as a whole — body-coupled perception."""
        if g_t is None or not self.cfg.use_salience_gate:
            return z_fused

        mode = self.gating_mode
        if mode == "metric_c1":
            # z_fused preserved; metric M_g consumed downstream
            return z_fused

        z_t = z_fused
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


class BodyEncoder(nn.Module):
    """Phase 3 (Layer 2 — Option C): separate encoder for body state.

    Maps interoceptive input → z_body (B, body_z_dim).
    Input dimensionality:
      - body_dim only:                     body_state (e.g. 1-d energy)
      - body_dim + silhouette_dim:         [body_state, body_silhouette]
                                            where silhouette is the
                                            directional (NSEW) affordance
                                            felt sense (Gaussian-blurred).

    The output is concatenated with z_obs to form z_fused, which is what
    gating (FiLM / salience / metric_c1) and downstream modules see.

    This instantiates body as part of perception itself (Safron's
    synesthetic affects: interoceptive states infused into other
    percepts). When silhouette is included, the agent's interoceptive
    channel carries both the scalar energy state AND a directional
    affordance silhouette — "어느 정도 직감" of surrounding affordance
    via body, not via vision.

    No g-conditioning here — body's contribution to perception is
    pre-stance, while gating (above) re-weights the *body-coupled*
    perception based on stance g.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.body_dim <= 0:
            raise ValueError("BodyEncoder requires cfg.body_dim > 0")
        in_dim = cfg.body_dim + max(0, int(cfg.silhouette_dim))
        self.mlp = MLP(in_dim, cfg.body_z_dim, cfg.body_hidden, cfg.dropout)

    @property
    def has_silhouette(self) -> bool:
        return int(self.cfg.silhouette_dim) > 0

    def forward(
        self,
        body_t: torch.Tensor,
        silhouette_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.has_silhouette:
            if silhouette_t is None:
                raise ValueError(
                    "BodyEncoder.silhouette_dim > 0 but silhouette_t was not "
                    "provided."
                )
            x = torch.cat([body_t, silhouette_t], dim=-1)
        else:
            x = body_t
        return torch.tanh(self.mlp(x))


class EncoderBundle(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_enc = ObservationEncoder(cfg)
        self.prop_enc = ProprioEncoder(cfg)
        # Optional body encoder: only constructed when body_dim > 0.
        # Phase 1/2 ckpts (body_dim=0) load cleanly because this attribute
        # does not exist for them.
        self.has_body_encoder = (cfg.body_dim > 0)
        if self.has_body_encoder:
            self.body_enc = BodyEncoder(cfg)

    def forward(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        g_t: Optional[torch.Tensor] = None,
        body_t: Optional[torch.Tensor] = None,
        silhouette_t: Optional[torch.Tensor] = None,
    ):
        """Produce (z_t, p_emb, z_body).

        Pipeline (body_dim > 0):
            z_obs  = obs_enc(x_t)                      # (B, z_obs_dim)  raw
            z_body = body_enc(body_t [, silhouette_t]) # (B, body_z_dim)
            z_fused = concat(z_obs, z_body)             # (B, z_dim)
            z_t = obs_enc.apply_gating(z_fused, g_t)

        Pipeline (body_dim == 0):
            z_obs = obs_enc(x_t)
            z_t = obs_enc.apply_gating(z_obs, g_t)
            z_body = None

        silhouette_t (B, silhouette_dim) is consumed by BodyEncoder when
        cfg.silhouette_dim > 0; otherwise ignored.

        z_t has dim cfg.z_dim regardless.

        z_body (when not None) is exposed as a separate output so that
        downstream modules — e.g., the policy in body-to-effector
        coupling — can use the learned interoceptive representation
        directly without re-encoding it. Always return z_body raw
        (pre-gating); gating is for percept synthesis, while z_body is
        the raw interoceptive channel.
        """
        z_obs = self.obs_enc(x_t)
        if self.has_body_encoder:
            if body_t is None:
                raise ValueError(
                    "EncoderBundle has body_enc but body_t was not provided. "
                    "Pass body_t to forward()."
                )
            z_body = self.body_enc(body_t, silhouette_t=silhouette_t)
            z_fused = torch.cat([z_obs, z_body], dim=-1)
        else:
            z_fused = z_obs
            z_body = None

        # Apply g-conditioned gating to the (possibly body-coupled) fused z
        z_t = self.obs_enc.apply_gating(z_fused, g_t=g_t)

        if p_t is None:
            B = x_t.shape[0]
            p_emb = torch.zeros((B, self.cfg.p_dim), device=x_t.device, dtype=x_t.dtype)
        else:
            p_emb = self.prop_enc(p_t)
        return z_t, p_emb, z_body