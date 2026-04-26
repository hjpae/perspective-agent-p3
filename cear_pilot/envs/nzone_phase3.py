# cear_pilot/envs/nzone_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 environment: minimal embodiment extension of Phase 2.

Continuity with Phase 2:
- Same world geometry (W x H grid, sigma gradient, zone boundaries)
- Same perturbation mechanism (transient inversion of sigma gradient)
- Inherits NZonePhase2Env; only the additions below are new

What is new (the "filling of the intentional gap"):
1. Self-cell observation
   Phase 1-2 agent perceives only the 8 surrounding cells; the central cell
   (the agent itself) was an intentional blank. Phase 3 fills it: the patch
   has 9 cells, with the self-cell at index 8.

2. Per-cell affordance channel
   Each cell now carries TWO channels:
     channel 0: env_sample  (or body_state, for the self-cell)
     channel 1: affordance  (zone-dependent valenced property)
   So obs_dim = 9 cells x 2 channels = 18 (vs. Phase 2's 8 x 1 = 8).
   Self-cell's channel 0 reads body_state; its channel 1 is 0 by convention.

3. Body state
   A minimal scalar (energy, in [body_min, body_max]) carried by the agent.
   It evolves each step via:
       body <- body  - metabolic_cost
                     - movement_cost (if moved)
                     + affordance_at_current_cell * affordance_to_body_gain
   The agent reads its own body_state through the self-cell's channel 0.

4. Affordance structure
   Each zone has a fixed affordance value (default: graded with sigma —
   cleaner zones positive, noisier zones negative). The agent observes
   the affordance of every cell in its 9-cell patch (including its own
   cell, via the affordance map; channel 1 of self-cell is held at 0
   to keep the self-cell semantically distinct).
   For probing different stances, override `affordance_per_zone` in config.

Action consequences flow from movement -> body_state -> next observation,
which is the architectural minimum for "action returns to body."

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
# (e.g. for visualization) maps cleanly to the surrounding cells [0..7].
PHASE3_PATCH_ORDER: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
    ( 0,  0),  # self-cell
)


@dataclass
class NZonePhase3Config(NZonePhase2Config):
    """Phase 3 config extends Phase 2 with body & affordance settings.

    Field overrides from Phase 2:
      - patch_order: 9 cells (was 8)
      - obs_dim:     18 (= 9 cells x 2 channels)  (was 8)
    """

    # ----- observation layout (overrides phase 2 defaults) -----
    patch_order: Tuple[Tuple[int, int], ...] = PHASE3_PATCH_ORDER
    n_channels_per_cell: int = 2
    # 9 cells * 2 channels = 18; include_xy adds 2 more downstream.
    obs_dim: int = 18

    # ----- body state -----
    body_dim: int = 1                 # minimal scalar energy
    body_init: float = 0.5            # initial value
    body_min: float = 0.0
    body_max: float = 1.0
    metabolic_cost: float = 0.005     # per-step base cost
    movement_cost: float = 0.005      # extra cost when action != STAY

    # ----- affordance structure -----
    # One value per report-zone (left -> right). Default: cleaner = positive,
    # noisier = negative (mirrors sigma gradient). Override to design
    # differentiated formation conditions.
    # Zone count = len(report_zone_boundaries) + 1.
    affordance_per_zone: Tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
    affordance_to_body_gain: float = 0.02

    # ----- termination -----
    # If True, episode ends when body_state hits body_min.
    # Default False so episode length is decoupled from body dynamics
    # (matches phase 2 max_steps semantics).
    terminate_on_body_min: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NZonePhase3Env(NZonePhase2Env):
    """Minimal embodied extension of Phase 2.

    Inherits world geometry, sigma gradient, and perturbation logic from
    NZonePhase2Env; overrides observation, step, reset, and info dict to
    add the embodied layer.
    """

    SELF_CELL_INDEX: int = 8  # index of (0, 0) in PHASE3_PATCH_ORDER

    def __init__(
        self,
        config: Optional[NZonePhase3Config] = None,
        render_mode: Optional[str] = None,
    ):
        cfg = config or NZonePhase3Config()
        # Phase 2 __init__ reads cfg.obs_dim and cfg.patch_order, so the
        # phase 3 overrides are honored.
        super().__init__(config=cfg, render_mode=render_mode)

        # Phase 3 specific state
        self.body_state: np.ndarray = np.array([cfg.body_init], dtype=np.float32)

        # Static affordance map indexed by (y, x)
        self._affordance_map: np.ndarray = self._build_affordance_map()

        # Step-level diagnostics (reset each step)
        self._last_metabolic_delta: float = 0.0
        self._last_affordance_gain: float = 0.0

    # ----- typed accessor -----

    @property
    def cfg3(self) -> NZonePhase3Config:
        return self.cfg  # type: ignore[return-value]

    # ----- affordance map -----

    def _build_affordance_map(self) -> np.ndarray:
        """Per-cell affordance value (H, W). Constant in time."""
        afford = np.zeros((self.H, self.W), dtype=np.float32)
        affs = self.cfg3.affordance_per_zone
        n_zones_expected = len(self.cfg.report_zone_boundaries) + 1
        if len(affs) < n_zones_expected:
            # Pad with zeros if user provided fewer values.
            affs = tuple(list(affs) + [0.0] * (n_zones_expected - len(affs)))
        for x in range(self.W):
            zone = self.report_zone_id(x)
            zone = min(zone, len(affs) - 1)
            afford[:, x] = affs[zone]
        return afford

    # ----- observation -----

    def _observe(self) -> np.ndarray:
        """
        9-cell x 2-channel observation, flattened.
          For surrounding cells (indices 0..7):
            ch 0: env sample (mu + noise)  -- with perturbation distortion
            ch 1: affordance value at that cell
          For self-cell (index 8):
            ch 0: body state
            ch 1: 0  (held distinct from environmental affordance)

        Optionally appends normalized (x, y) if cfg.include_xy.
        """
        n_cells = len(self.cfg3.patch_order)
        n_chan = int(self.cfg3.n_channels_per_cell)
        obs = np.zeros((n_cells, n_chan), dtype=np.float32)

        for i, (dx, dy) in enumerate(self.cfg3.patch_order):
            is_self = (dx == 0 and dy == 0)
            if is_self:
                obs[i, 0] = float(self.body_state[0])
                obs[i, 1] = 0.0
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
        """Phase 2's distortion adapted to 9-cell layout. Self-cell gets 0
        (body state is internal; no environmental distortion applies)."""
        scale = float(self.cfg.perturbation_scale)
        n_cells = len(self.cfg3.patch_order)
        distortion = np.zeros(n_cells, dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg3.patch_order):
            if dx == 0 and dy == 0:
                continue  # self-cell unaffected
            px, _ = self._patch_coord(self.x + dx, self.y + dy)
            distortion[i] = scale * self._inversion_pattern[px]
        return distortion

    # ----- step -----

    def step(self, action: int):
        moved = (action != self.ACTION_STAY)

        # Phase 2 movement
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

        # Phase 3 body dynamics
        self._update_body_state(moved=moved)

        # Termination
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
        # Phase 2 reset handles position, perturbation schedule, etc.
        _, _ = super().reset(seed=seed, options=options)

        # Reset body state
        self.body_state = np.array([self.cfg3.body_init], dtype=np.float32)
        self._last_metabolic_delta = 0.0
        self._last_affordance_gain = 0.0

        # Re-observe with body state initialized (super().reset returned an
        # observation built before we reset body_state, so rebuild it)
        obs = self._observe()
        info = self._info_dict()
        return obs, info

    # ----- info -----

    def _info_dict(self) -> Dict[str, Any]:
        info = super()._info_dict()
        info["body_state"] = float(self.body_state[0])
        info["affordance_here"] = float(self._affordance_map[self.y, self.x])
        info["metabolic_delta"] = float(self._last_metabolic_delta)
        info["affordance_gain"] = float(self._last_affordance_gain)
        return info


def make_env(**kwargs) -> NZonePhase3Env:
    return NZonePhase3Env(config=NZonePhase3Config(**kwargs))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test: reset, step a few times, verify shapes & body dynamics.
    env = make_env()
    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}  (expected (18,) without include_xy)")
    print(f"observation_space: {env.observation_space}")
    print(f"initial body_state: {info['body_state']}")
    print(f"initial affordance_here: {info['affordance_here']}")
    print(f"perturbation_steps: {env.perturbation_steps}")

    print("\nfirst 5 steps (alternating LEFT, RIGHT):")
    for t in range(5):
        action = env.ACTION_LEFT if t % 2 == 0 else env.ACTION_RIGHT
        obs, r, term, trunc, info = env.step(action)
        # Read self-cell (channels 16, 17 in flat layout = index 8 * 2)
        self_ch0 = obs[2 * env.SELF_CELL_INDEX]
        self_ch1 = obs[2 * env.SELF_CELL_INDEX + 1]
        print(
            f"  t={info['t']:3d}  pos=({info['x']:2d},{info['y']:2d})  "
            f"zone={info['zone_id']}  "
            f"body={info['body_state']:.4f}  "
            f"afford_here={info['affordance_here']:+.3f}  "
            f"self_cell=(ch0={self_ch0:.3f}, ch1={self_ch1:.3f})  "
            f"perturb={info['perturbation_active']}"
        )
