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
    hidden: int = 64
    n_actions: int = 5
    dropout: float = 0.0
    # Phase 3: optional direct body input. When body_dim > 0, the policy
    # receives [s_t, body_state] instead of just s_t.
    # This instantiates Safron's "willing": interoceptive states map onto
    # the effector systems by which intentions are realized — body
    # directly informs action selection, not just through downstream
    # feedback via state head.
    # Default body_dim=0 keeps phase 1/2 backward compatibility.
    body_dim: int = 0


class PolicyNet(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.s_dim + max(0, cfg.body_dim)
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
                    "PolicyNet has body_dim > 0 but body_t was not provided."
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