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
        # Phase 3 body-coupled architecture: encoder.body_dim and
        # policy.body_dim must match state.body_dim (single raw body dim
        # propagated through encoder, policy, state head).
        if cfg.encoder.body_dim > 0:
            assert cfg.encoder.body_dim == cfg.state.body_dim, (
                f"encoder.body_dim ({cfg.encoder.body_dim}) must equal "
                f"state.body_dim ({cfg.state.body_dim})"
            )
        if cfg.policy.body_dim > 0:
            assert cfg.policy.body_dim == cfg.state.body_dim, (
                f"policy.body_dim ({cfg.policy.body_dim}) must equal "
                f"state.body_dim ({cfg.state.body_dim})"
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

    def reset_body_pred(self) -> None:
        """Reset only body_pred to env body_init (0.5). Leave g and alpha
        unchanged — used at episode boundaries when g must carry across
        episodes (long-horizon perspective formation) but body_state is
        reset by env at episode start.

        This separates two reset semantics:
          - reset(): full latent reset (training start, assay start)
          - reset_body_pred(): episode-boundary reset of body_pred only
                               so that body_PE on step 1 of the new episode
                               is computed against env's reset body_init,
                               not against the carried-over final body_pred
                               of the previous episode.
        """
        if self._g is None:
            raise RuntimeError("reset_body_pred() called before reset(). "
                               "Call reset() first to initialize batch_size.")
        if self.has_body:
            bs = self._g.shape[0]
            self._body_pred_prev = torch.full(
                (bs, self.cfg.state.body_dim),
                0.5,
                device=self.device_,
                dtype=torch.float32,
            )

    def decay_g(self, factor: float) -> None:
        """Multiplicatively decay the carried perspective g by `factor`.
        Used at episode boundaries when g is carried across episodes:
        full carry leads to monotonic ||g|| growth and eventual
        saturation-driven instability. Partial decay (e.g. 0.9) preserves
        most of the perspective representation while preventing
        unbounded magnitude growth — phenomenologically, "previous
        perspective is mostly preserved but with slight refresh upon
        entering a new context".

        factor in [0, 1]: 0 = full reset (equivalent to reset()'s g part),
        1 = no decay (full carry).
        """
        if self._g is None:
            raise RuntimeError("decay_g() called before reset(). "
                               "Call reset() first to initialize.")
        f = float(factor)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"decay factor must be in [0, 1], got {f}")
        self._g = (self._g * f).detach()

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
        body_silhouette_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        # Body PE (signed) — feedback signal, not a backprop path
        body_pe: Optional[torch.Tensor] = None
        body_t_for_modules: Optional[torch.Tensor] = None
        if self.has_body and body_actual_t is not None and self._body_pred_prev is not None:
            body_actual_t = body_actual_t.to(self.device_).float()
            body_pe = body_actual_t - self._body_pred_prev
            body_pe = body_pe.detach()
            # body_t_for_modules is the *current actual body state*, used as
            # input to the body encoder and the policy. This is a live
            # interoceptive signal (not a backprop quantity from the agent's
            # own prediction).
            body_t_for_modules = body_actual_t.detach()

        # Silhouette: directional affordance felt sense (interoceptive,
        # blurred). Detach (no backprop into env-noise sample).
        sil_for_modules: Optional[torch.Tensor] = None
        if body_silhouette_t is not None:
            sil_for_modules = body_silhouette_t.to(self.device_).float().detach()

        # encoder — body_coupled when encoder.body_dim > 0
        if self.cfg.encoder.body_dim > 0:
            if body_t_for_modules is None:
                raise RuntimeError(
                    "encoder.body_dim > 0 but body_actual_t was not provided. "
                    "Phase 3 body-coupled architecture requires body input each step."
                )
            # Validate silhouette presence/absence matches encoder config
            if int(self.cfg.encoder.silhouette_dim) > 0 and sil_for_modules is None:
                raise RuntimeError(
                    "encoder.silhouette_dim > 0 but body_silhouette_t was not "
                    "provided. Pass body_silhouette_t each step."
                )
            z_t, p_emb = self.enc(
                x_t, p_t, g_t=self._g,
                body_t=body_t_for_modules,
                silhouette_t=sil_for_modules,
            )
        else:
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
            # body_t_for_modules is the live interoceptive body_state (detached);
            # the body_head needs it directly to ground body prediction in
            # current body context (avoid lock-in to past g context across
            # carried episodes).
            s_t, body_pred_t = self.state(
                z_t, p_emb, g_t, M_g=M_g, body_t=body_t_for_modules,
            )
        else:
            s_t = self.state(z_t, p_emb, g_t, M_g=M_g)
            body_pred_t = None

        # policy — body_coupled when policy.body_dim > 0
        if self.cfg.policy.body_dim > 0:
            if body_t_for_modules is None:
                raise RuntimeError(
                    "policy.body_dim > 0 but body_actual_t was not provided."
                )
            logits = self.policy(s_t, body_t=body_t_for_modules)
        else:
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
        body_silhouette_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward_step(
            x_t, p_t,
            ablate_g=ablate_g,
            err_t=err_t,
            body_actual_t=body_actual_t,
            body_silhouette_t=body_silhouette_t,
        )
        action = self.policy.sample_action(out["logits"], greedy=greedy)
        return action, out