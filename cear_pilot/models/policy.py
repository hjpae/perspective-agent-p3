# cear_pilot/models/policy.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PolicyConfig:
    s_dim: int = 16
    body_dim: int = 0          # raw body_t scalar (1-d when present)
    hidden: int = 64
    n_actions: int = 5
    dropout: float = 0.0


class PolicyNet(nn.Module):
    """Action selection over [s_t, body_t].

    Phase 3 commitment — minimal body-effector coupling. Policy receives
    s_t (g-grounded perspective state) and body_t (raw 1-d interoceptive
    scalar). Both are detached upstream so actor gradient is confined to
    PolicyNet.

    z_body is NOT routed into policy: that would turn the learned
    interoceptive representation into a direct policy knob (functionalist
    body-informed action control). Instead, body / affordance enters
    action selection only indirectly — through the perspective-organized
    state representation s_t. Counterfactual interoceptive anticipation
    is handled by the separate BodyDecoder, whose outputs are also kept
    out of policy logits.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.s_dim + cfg.body_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.n_actions),
        )

    def forward(
        self,
        s_t: torch.Tensor,
        body_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.body_dim > 0:
            if body_t is None:
                raise ValueError(
                    "PolicyConfig.body_dim > 0 requires body_t to be provided."
                )
            x = torch.cat([s_t, body_t], dim=-1)
        else:
            x = s_t
        return self.net(x)

    @torch.no_grad()
    def sample_action(self, logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        if greedy:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)