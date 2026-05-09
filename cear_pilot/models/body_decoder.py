# cear_pilot/models/body_decoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

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
    anticipation.

    For each candidate action a, predict the body state at t+1 under
    that action:

        b̂_{t+1}^{(a)} = D_body(z_t, g_t, b_t, a)

    Phase 3 phenomenological commitment:
      - This is NOT a policy feature. b̂^{(a)} is never routed into
        policy logits.
      - This trains an interoceptive *anticipation* model: the agent
        learns "what bodily consequence does this action open" without
        affordance becoming a direct action knob.
      - Its loss flows back through z (encoder including BodyEncoder)
        and g (perspective). This is how g learns to organize a
        qualitative field of bodily viability — affordance enters as
        anticipated bodily consequence, not as policy preference.

    Conceptual separation from state head's body_head:
      - state body_head: interoceptive *registration* / continuity
                         anchor. "How does my current body state
                         continue?" Stabilizes body_pred_prev across
                         carried g.
      - BodyDecoder:     counterfactual interoceptive *anticipation*.
                         "What body opens with this action?" Trains
                         the perspectival field around bodily viability.

    Both serve the generative/perspectival pathway; neither feeds policy.
    """

    def __init__(self, cfg: BodyDecoderConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.z_dim + cfg.g_dim + cfg.body_dim + cfg.n_actions
        self.g_ln = nn.LayerNorm(cfg.g_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.body_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        g_t: torch.Tensor,
        body_t: torch.Tensor,
        a_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Predict b̂_{t+1} under the given action onehot.
        Output passed through sigmoid → in [0, 1] consistent with body
        state range.
        """
        g = self.g_ln(g_t)
        x = torch.cat([z_t, g, body_t, a_onehot], dim=-1)
        return torch.sigmoid(self.net(x))

    def predict_all_actions(
        self,
        z_t: torch.Tensor,
        g_t: torch.Tensor,
        body_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict b̂_{t+1}^{(a)} for every candidate action.
        Returns (B, n_actions, body_dim).
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
        out = self.net(x.view(B * A, -1)).view(B, A, -1)
        return torch.sigmoid(out)
