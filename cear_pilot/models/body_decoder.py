# cear_pilot/models/body_decoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class BodyDecoderConfig:
    z_dim: int = 20         # gated percept dim (z_fused: z_obs + z_body)
    g_dim: int = 12
    body_dim: int = 1
    n_actions: int = 5
    hidden: int = 64
    dropout: float = 0.0


class BodyDecoder(nn.Module):
    """Action-conditioned body prediction — counterfactual interoceptive
    anticipation, with two heads:

      b̂_{t+1}^{(a)}  : next-step body prediction (bounded, sigmoid)
      τ̂_t^{(a)}     : short-horizon viability tendency (real-valued)

    Phase 3 phenomenological commitment:
      - NOT a policy feature. Both outputs are never routed into policy
        logits. Affordance does not become an action knob.
      - Trains interoceptive *anticipation* and *tendency* models. The
        agent learns "what bodily consequence does this action open" and
        "what direction does my viability move under this action".
      - Loss flows back through z (encoder including BodyEncoder) and g
        (perspective). This is how g organizes a qualitative field of
        bodily viability around action consequences.

    Why two heads:
      Next-step body alone is flat at body extremes (clipped or sigmoid-
      saturated): UP and DOWN look near-identical at body≈0. Tendency
      target — Δu over k steps in latent viability space — preserves
      directional gradient even at extremes. Tendency captures "this
      direction recovers / further depletes" where one-step body cannot.

    Conceptual separation from state head's body_head:
      - state body_head: interoceptive *registration* / continuity
                         anchor. "How does my current body state
                         continue?" Stabilizes body_pred_prev across
                         carried g.
      - BodyDecoder:     counterfactual interoceptive *anticipation* +
                         viability *tendency*. "What body opens / what
                         direction does viability move under this
                         action?"

    Both serve the generative/perspectival pathway; neither feeds policy.
    """

    def __init__(self, cfg: BodyDecoderConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.z_dim + cfg.g_dim + cfg.body_dim + cfg.n_actions
        self.g_ln = nn.LayerNorm(cfg.g_dim)
        # Output dim = body_dim (next body) + body_dim (tendency).
        # body_dim=1 → 2-d output: [body_next_logit, tendency_real].
        self.out_dim = 2 * cfg.body_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, self.out_dim),
        )

    def _split_outputs(
        self, raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """raw: (..., 2*body_dim) -> (body_next, tendency).
        body_next is sigmoid-squashed to [0, 1].
        tendency is left real-valued (target is Δu, unbounded).
        """
        D = self.cfg.body_dim
        body_next = torch.sigmoid(raw[..., :D])
        tendency = raw[..., D:]
        return body_next, tendency

    def forward(
        self,
        z_t: torch.Tensor,
        g_t: torch.Tensor,
        body_t: torch.Tensor,
        a_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict (b̂_{t+1}, τ̂_t) under the given action onehot."""
        g = self.g_ln(g_t)
        x = torch.cat([z_t, g, body_t, a_onehot], dim=-1)
        raw = self.net(x)
        return self._split_outputs(raw)

    def predict_all_actions(
        self,
        z_t: torch.Tensor,
        g_t: torch.Tensor,
        body_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict (b̂_{t+1}^{(a)}, τ̂_t^{(a)}) for every candidate action.
        Returns (body_next_all (B, A, D), tendency_all (B, A, D)).
        """
        B = g_t.shape[0]
        A = self.cfg.n_actions
        device = g_t.device
        dtype = g_t.dtype

        g = self.g_ln(g_t)
        eye = torch.eye(A, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        z_rep = z_t.unsqueeze(1).repeat(1, A, 1)
        g_rep = g.unsqueeze(1).repeat(1, A, 1)
        b_rep = body_t.unsqueeze(1).repeat(1, A, 1)

        x = torch.cat([z_rep, g_rep, b_rep, eye], dim=-1)
        raw = self.net(x.view(B * A, -1)).view(B, A, self.out_dim)
        return self._split_outputs(raw)