# cear_pilot/envs/nzone_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 environment: minimal embodiment extension of Phase 2.

Continuity with Phase 2:
- Same world geometry style (sigma gradient, perturbation mechanism)
- Inherits NZonePhase2Env; only the additions and grid-shape changes below are new

What is new (the "filling of the intentional gap"):
1. Self-cell observation
   Phase 1-2 agent perceives only the 8 surrounding cells; the central cell
   (the agent itself) was an intentional blank. Phase 3 fills it: the patch
   has 9 cells, with the self-cell at index 8.

2. Per-cell affordance channel
   Each cell now carries TWO channels:
     channel 0: env_sample  (or body_state, for the self-cell)
     channel 1: affordance value at that cell (Position 2: self-cell also
                carries the affordance of its current cell)
   So obs_dim = 9 cells x 2 channels = 18.

3. Body state
   A minimal scalar (energy, in [body_min, body_max]) carried by the agent.
   It evolves each step via:
       body <- body  - metabolic_cost
                     - movement_cost (if moved)
                     + affordance_at_current_cell * affordance_to_body_gain
   This is *environment dynamics* (a physical fact, like sigma gradient),
   not an imposed value function. The agent has no preference about body
   state; it only learns to predict body dynamics via prediction error.

4. Grid redesign (15 x 15 with orthogonal gradients)
   - Horizontal axis: perception challenge (sigma gradient, linear).
     Left noisy, right clean. Same role as phase 2.
   - Vertical axis: valence gradient (affordance, sigmoid).
     Top positive (appetitive), bottom negative (aversive).
   - Two gradients are *orthogonal* — perception and valence are
     independent dimensions for stance differentiation.
   - 3 horizontal zones (5 cols each) x 3 vertical zones (5 rows each)
     for analytical convenience; gradient itself is smooth.

5. Action consequences flow movement -> body_state -> next observation.
   "Action returns to body" architecturally instantiated.

Notes:
- Environment field still NEVER changes between episodes; sigma gradient
  and affordance map are static. Differentiation between agents arises
  through their *history* (body trajectory, perturbation exposure).
- Phase 1/2 trained checkpoints are NOT obs-compatible with Phase 3
  (obs_dim differs). New training is required for Phase 3 agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("gymnasium required") from e

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# 9-cell patch order: 8 surrounding + 1 self at index 8.
# Self-cell is intentionally placed last so existing 8-cell indexing logic
# maps cleanly to the surrounding cells [0..7].
PHASE3_PATCH_ORDER: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
    ( 0,  0),  # self-cell
)


@dataclass
class NZonePhase3Config(NZonePhase2Config):
    """Phase 3 config extends Phase 2 with body & affordance settings,
    plus grid redesign for orthogonal perception/valence gradients."""

    # ----- grid redesign: 15x15 with column/row partitioned into 3 zones each -----
    width: int = 15
    height: int = 15
    start_xy: Tuple[int, int] = (7, 7)             # geometric center
    max_steps: int = 300                           # bump to 400 if needed
    # Column zones (perception): 3 zones of 5 cols each. Boundaries at 5 and 10.
    report_zone_boundaries: Tuple[int, ...] = (5, 10)
    # Row zones (valence): 3 zones of 5 rows each. Boundaries at 5 and 10.
    valence_zone_boundaries: Tuple[int, ...] = (5, 10)

    # ----- horizontal: perception (sigma) gradient (linear) -----
    sigma_left: float = 0.20
    sigma_right: float = 0.10
    # zone_mu_scale / row_mu_scale inherited; row_mu_scale=0.10 keeps mild
    # vertical signal in mu so vertical movement also has perceptual content
    # (otherwise vertical would carry only valence, making the two axes
    # confounded with "what the agent perceives" vs "what it doesn't").

    # ----- vertical: affordance (valence) gradient (sigmoid) -----
    affordance_top: float = 0.5                    # row 0
    affordance_bottom: float = -0.5                # row H-1
    affordance_sigmoid_slope: float = 1.6          # tanh slope, mild

    # ----- observation layout -----
    patch_order: Tuple[Tuple[int, int], ...] = PHASE3_PATCH_ORDER
    n_channels_per_cell: int = 2
    # 9 cells * 2 channels = 18; include_xy adds 2 more downstream.
    obs_dim: int = 18

    # ----- body state -----
    body_dim: int = 1
    body_init: float = 0.5
    body_min: float = 0.0
    body_max: float = 1.0
    # Body dynamics magnitudes — amplified for the body-coupled architecture.
    # Original (Apr 2026 minimal embodiment): metabolic_cost=0.002,
    # movement_cost=0.002, affordance_to_body_gain=0.02. Step-wise body
    # change ~0.01 max → body PE ~1e-4. Too small to be a meaningful
    # learning signal once body PE is integrated into actor cost and used
    # as a body encoder input (Layer 1 + Layer 2 commitments).
    # Amplified (May 2026): step-wise body change up to ~0.05, body PE
    # comfortably in 1e-3 to 1e-2 range, body state ranging meaningfully
    # in [0, 1] within a 300-step episode under non-trivial behavior.
    metabolic_cost: float = 0.01
    movement_cost: float = 0.01
    affordance_to_body_gain: float = 0.10

    # ----- termination -----
    terminate_on_body_min: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NZonePhase3Env(NZonePhase2Env):
    """Minimal embodied extension of Phase 2."""

    SELF_CELL_INDEX: int = 8  # index of (0, 0) in PHASE3_PATCH_ORDER

    def __init__(
        self,
        config: Optional[NZonePhase3Config] = None,
        render_mode: Optional[str] = None,
    ):
        cfg = config or NZonePhase3Config()
        super().__init__(config=cfg, render_mode=render_mode)

        # Phase 3 specific state
        self.body_state: np.ndarray = np.array([cfg.body_init], dtype=np.float32)

        # Static affordance map indexed by (y, x): vertical sigmoid gradient
        self._affordance_map: np.ndarray = self._build_affordance_map()

        # Step-level diagnostics
        self._last_metabolic_delta: float = 0.0
        self._last_affordance_gain: float = 0.0

    # ----- typed accessor -----

    @property
    def cfg3(self) -> NZonePhase3Config:
        return self.cfg  # type: ignore[return-value]

    # ----- affordance map (vertical sigmoid) -----

    def _build_affordance_map(self) -> np.ndarray:
        """Per-cell affordance value (H, W). Vertical sigmoid gradient:
        top rows positive (appetitive), bottom rows negative (aversive).
        Constant in time."""
        H, W = self.H, self.W
        cfg = self.cfg3

        # Map row index to [+1, -1]: top = +1, bottom = -1
        row_center = (H - 1) / 2.0
        # y_norm in [+1, -1] from top to bottom
        y_norm = -(np.arange(H, dtype=np.float32) - row_center) / max(1.0, row_center)

        # Sigmoid (tanh) gradient with mild slope
        gradient = np.tanh(cfg.affordance_sigmoid_slope * y_norm).astype(np.float32)

        # Scale to [affordance_bottom, affordance_top]
        a_top = float(cfg.affordance_top)
        a_bot = float(cfg.affordance_bottom)
        mid = 0.5 * (a_top + a_bot)
        half = 0.5 * (a_top - a_bot)
        col_affordance = mid + half * gradient  # shape (H,)

        afford = np.tile(col_affordance[:, None], (1, W)).astype(np.float32)
        return afford

    def valence_zone_id(self, y: int) -> int:
        """Row-based valence zone id (top = 0). For analytical convenience.
        Vertical analogue of report_zone_id."""
        y = int(np.clip(y, 0, self.H - 1))
        for zi, b in enumerate(self.cfg3.valence_zone_boundaries):
            if y < int(b):
                return zi
        return len(self.cfg3.valence_zone_boundaries)

    # ----- observation -----

    def _observe(self) -> np.ndarray:
        """
        9-cell x 2-channel observation, flattened.
          For surrounding cells (indices 0..7):
            ch 0: env sample (mu + noise)  -- with perturbation distortion
            ch 1: affordance value at that cell
          For self-cell (index 8):  [Position 2]
            ch 0: body state
            ch 1: affordance at current cell (the cell the agent is in)

        Optionally appends normalized (x, y) if cfg.include_xy.
        """
        n_cells = len(self.cfg3.patch_order)
        n_chan = int(self.cfg3.n_channels_per_cell)
        obs = np.zeros((n_cells, n_chan), dtype=np.float32)

        for i, (dx, dy) in enumerate(self.cfg3.patch_order):
            is_self = (dx == 0 and dy == 0)
            if is_self:
                obs[i, 0] = float(self.body_state[0])
                # Position 2: self-cell ch1 = current cell's affordance
                obs[i, 1] = float(self._affordance_map[self.y, self.x])
            else:
                obs[i, 0] = self._sample_cell(self.x + dx, self.y + dy)
                px, py = self._patch_coord(self.x + dx, self.y + dy)
                obs[i, 1] = float(self._affordance_map[py, px])

        # Apply perturbation distortion only to surrounding cells' env-sample channel.
        if self._perturbation_active:
            distortion = self._perturbation_distortion_per_cell()
            obs[:, 0] += distortion

        flat = obs.reshape(-1)

        if self.cfg.include_xy:
            xy = np.array(
                [
                    self.x / max(1, self.W - 1),
                    self.y / max(1, self.H - 1),
                ],
                dtype=np.float32,
            )
            flat = np.concatenate([flat, xy])

        return flat.astype(np.float32)

    def _perturbation_distortion_per_cell(self) -> np.ndarray:
        """Phase 2 distortion adapted to 9-cell layout. Self-cell unaffected
        (body state is internal; environmental distortion does not apply)."""
        scale = float(self.cfg.perturbation_scale)
        n_cells = len(self.cfg3.patch_order)
        distortion = np.zeros(n_cells, dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg3.patch_order):
            if dx == 0 and dy == 0:
                continue
            px, _ = self._patch_coord(self.x + dx, self.y + dy)
            distortion[i] = scale * self._inversion_pattern[px]
        return distortion

    # ----- step -----

    def step(self, action: int):
        moved = (action != self.ACTION_STAY)

        dx, dy = 0, 0
        if action == self.ACTION_UP:
            dy = -1
        elif action == self.ACTION_DOWN:
            dy = 1
        elif action == self.ACTION_LEFT:
            dx = -1
        elif action == self.ACTION_RIGHT:
            dx = 1
        self.x, self.y = self._clip_xy(self.x + dx, self.y + dy)
        self.t += 1
        self.visited.add((self.x, self.y))
        self._update_perturbation()

        self._update_body_state(moved=moved)

        terminated = False
        if (
            self.cfg3.terminate_on_body_min
            and self.body_state[0] <= self.cfg3.body_min + 1e-6
        ):
            terminated = True
        truncated = self.t >= self.max_steps

        obs = self._observe()
        return obs, float(self.cfg.reward_scale), terminated, truncated, self._info_dict()

    def _update_body_state(self, moved: bool) -> None:
        cfg = self.cfg3
        metabolic_delta = -cfg.metabolic_cost
        if moved:
            metabolic_delta -= cfg.movement_cost

        affordance = float(self._affordance_map[self.y, self.x])
        affordance_gain = affordance * cfg.affordance_to_body_gain

        delta = metabolic_delta + affordance_gain
        new_body = float(self.body_state[0]) + delta
        new_body = float(np.clip(new_body, cfg.body_min, cfg.body_max))

        self.body_state = np.array([new_body], dtype=np.float32)
        self._last_metabolic_delta = float(metabolic_delta)
        self._last_affordance_gain = float(affordance_gain)

    # ----- reset -----

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        _, _ = super().reset(seed=seed, options=options)
        self.body_state = np.array([self.cfg3.body_init], dtype=np.float32)
        self._last_metabolic_delta = 0.0
        self._last_affordance_gain = 0.0
        obs = self._observe()
        info = self._info_dict()
        return obs, info

    # ----- info -----

    def _info_dict(self) -> Dict[str, Any]:
        info = super()._info_dict()
        info["body_state"] = float(self.body_state[0])
        info["affordance_here"] = float(self._affordance_map[self.y, self.x])
        info["valence_zone_id"] = int(self.valence_zone_id(self.y))
        info["metabolic_delta"] = float(self._last_metabolic_delta)
        info["affordance_gain"] = float(self._last_affordance_gain)
        return info


def make_env(**kwargs) -> NZonePhase3Env:
    return NZonePhase3Env(config=NZonePhase3Config(**kwargs))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}  (expected (18,))")
    print(f"observation_space: {env.observation_space}")
    print(f"grid: {env.W} x {env.H}, start: ({env.x}, {env.y})")
    print(f"initial body_state: {info['body_state']:.4f}")
    print(f"initial affordance_here: {info['affordance_here']:+.4f}  "
          f"(valence zone {info['valence_zone_id']})")
    print(f"perturbation_steps: {env.perturbation_steps}")

    print(f"\naffordance map (vertical gradient, sample column 7):")
    for y in range(env.H):
        a = env._affordance_map[y, 7]
        vz = env.valence_zone_id(y)
        marker = "  <- agent start" if y == env.y else ""
        print(f"  row {y:2d} (vz={vz}):  affordance={a:+.4f}{marker}")

    print("\nfirst 10 steps (UP, UP, UP, UP, UP, STAY, STAY, DOWN, DOWN, DOWN):")
    actions = [env.ACTION_UP] * 5 + [env.ACTION_STAY] * 2 + [env.ACTION_DOWN] * 3
    for action in actions:
        obs, r, term, trunc, info = env.step(action)
        self_ch0 = obs[2 * env.SELF_CELL_INDEX]
        self_ch1 = obs[2 * env.SELF_CELL_INDEX + 1]
        action_name = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][action]
        print(
            f"  t={info['t']:3d}  {action_name:5s}  pos=({info['x']:2d},{info['y']:2d})  "
            f"vz={info['valence_zone_id']}  "
            f"body={info['body_state']:.4f}  "
            f"afford_here={info['affordance_here']:+.3f}  "
            f"self=(b={self_ch0:.3f}, a={self_ch1:+.3f})"
        )