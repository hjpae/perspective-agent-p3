# cear_pilot/models/state_head.py
# -*- coding: utf-8 -*-
"""
State head: predicts next-step state s_t and (Phase 3) body state.

Phase 1-2 behavior preserved when body_dim=0 (default for backward compat):
  s_t = tanh(LayerNorm(net(z, p, g)))

Phase 3 addition (body_dim > 0):
  s_t (same as before) + body_pred_t (separate small head)

The body head is intentionally a separate small MLP — it shares no parameters
with the s-head trunk so the s representation isn't contaminated by body
prediction. The body head is what enables body prediction error to flow into
the AlphaNet as a separate channel (Option I in the design discussion).

The body head outputs body in [0, 1] via sigmoid, matching the env's body
range. Loss term in train_phase3.py compares body_pred(t) vs body_actual(t+1).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

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
    # Phase 3: body prediction head. body_dim=0 disables it (phase 1-2 compat).
    body_dim: int = 0
    body_hidden: int = 32


class StateHead(nn.Module):
    def __init__(self, cfg: StateHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim
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

        # Phase 3: separate small body-prediction head from same input.
        # Outputs body in [0, 1] via sigmoid.
        if cfg.body_dim > 0:
            self.body_head = nn.Sequential(
                nn.Linear(in_dim, cfg.body_hidden),
                nn.Tanh(),
                nn.Linear(cfg.body_hidden, cfg.body_dim),
            )
            # init last layer near zero so initial prediction ~ 0.5 (sigmoid(0))
            last = self.body_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        g_t: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        If body_dim == 0: returns s_t (phase 1-2 backward compat).
        If body_dim  > 0: returns (s_t, body_pred_t) tuple.
        """
        g_scaled = g_t * float(self.cfg.g_influence)
        x = torch.cat([z_t, p_emb, g_scaled], dim=-1)
        s = torch.tanh(self.net(x))
        s = self.ln(s)

        if self.cfg.body_dim > 0:
            body_pred = torch.sigmoid(self.body_head(x))
            return s, body_pred
        return s
