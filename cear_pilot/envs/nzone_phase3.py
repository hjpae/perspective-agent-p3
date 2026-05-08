# cear_pilot/envs/nzone_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 environment: minimal embodiment extension of Phase 2.

Continuity with Phase 2:
- Same world geometry style (sigma gradient, perturbation mechanism)
- Inherits NZonePhase2Env; only the additions and grid-shape changes below are new

What is new:

1. Pure exteroceptive obs (visual surround only)
   The agent perceives 8 surrounding cells' env_sample (sigma noise) only.
   - obs_dim = 8.
   - Affordance is NOT a visual feature: agent cannot "see" which cells
     are appetitive vs aversive. Affordance is felt only through body
     state changes.
   - Body state is NOT in obs: it enters the agent through a separate
     interoceptive channel (body input to BodyEncoder + policy).

2. Body state (interoceptive, separate from obs)
   A minimal scalar (energy, in [body_min, body_max]) carried by the agent.
   It evolves each step via:
       body <- body  - metabolic_cost
                     - movement_cost (if moved)
                     + affordance_at_current_cell * affordance_to_body_gain
   This makes affordance an interoceptive feature: it is felt only
   through its effect on body. The agent must explore (act) and feel
   the body change to learn the affordance structure of the environment.

3. Grid layout (15 x 15 with orthogonal gradients)
   - Horizontal axis: perception challenge (sigma gradient, linear).
     Left noisy, right clean. Same role as phase 2.
   - Vertical axis: valence gradient (affordance, sigmoid).
     Top positive (appetitive), bottom negative (aversive).
   - 3 horizontal zones (5 cols each) x 3 vertical zones (5 rows each)
     for analytical convenience; gradient itself is smooth.

4. Action consequences flow movement → body_state → next interoception.
   "Action returns to body" architecturally instantiated.

Notes:
- Environment field still NEVER changes between episodes (during training);
  sigma gradient and affordance map are static.
- Differentiation between agents arises through their *history* (body
  trajectory, perturbation exposure during assay).
- Phase 1/2 trained checkpoints are NOT obs-compatible with Phase 3
  (obs_dim differs). New training is required for Phase 3 agents.

Architectural commitment summary:
- exteroceptive (visual) → obs vector (8-dim, env_sample only)
- interoceptive (body)   → separate body_state input (1-dim)
- affordance is interoceptive (sensed via body, not visually)
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

# 8-cell surrounding patch (no self-cell). Body is interoceptive — entered
# into the agent's representation through a separate body channel
# (BodyEncoder), not through the obs vector. Removing the self-cell from
# obs ensures obs is exclusively *exteroceptive* (visual surround).
#
# Earlier versions included a self-cell at index 8 carrying body_state and
# affordance_at_current_cell. That conflated interoceptive (body) with
# exteroceptive (visual) channels. Phase 3 final commitment: clean
# separation.
PHASE3_PATCH_ORDER: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
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
    # Single channel: env_sample only. Affordance is *not* visually
    # perceptible — it is felt only through the body via the affordance →
    # body-state pathway. Agent must explore to learn the affordance
    # structure, since visual perception does not directly reveal it.
    # This is the architectural instantiation of "interoceptive affect":
    # affordance is a quasi-synesthetic interoceptive feature, not an
    # exteroceptive feature.
    n_channels_per_cell: int = 1
    # 8 surrounding cells × 1 channel = 8. Body is interoceptive (separate
    # path). Affordance is interoceptive (felt through body changes only).
    obs_dim: int = 8

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

    # ----- body silhouette (optional, toggleable) -----
    # When silhouette_dim > 0, the env produces a body_silhouette feature
    # in info dict: a directional affordance "felt sense" of the agent's
    # 4 cardinal neighbors (N, S, W, E), with Gaussian noise σ.
    # This instantiates "어느 정도 직감" — partial/blurred interoceptive
    # sense of surrounding affordance via body. Visual perception remains
    # affordance-blind; silhouette is interoceptive.
    # silhouette_dim = 0 disables (agent receives only body_state, 1-d).
    # silhouette_dim = 4 enables N/S/W/E directional silhouette.
    silhouette_dim: int = 0
    silhouette_noise_sigma: float = 0.2

    # ----- termination -----
    terminate_on_body_min: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NZonePhase3Env(NZonePhase2Env):
    """Minimal embodied extension of Phase 2."""

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
        # Single-channel obs: env_sample only. Affordance is interoceptive
        # (entered into the agent only through body state dynamics, not as
        # a visual feature). Body is also interoceptive (separate body
        # encoder path).
        obs = np.zeros((n_cells,), dtype=np.float32)

        for i, (dx, dy) in enumerate(self.cfg3.patch_order):
            obs[i] = self._sample_cell(self.x + dx, self.y + dy)

        # Apply perturbation distortion to env-sample channel.
        if self._perturbation_active:
            distortion = self._perturbation_distortion_per_cell()
            obs += distortion

        flat = obs  # already 1-D shape (n_cells,)

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
        """Phase 2 obs distortion, applied to all 8 surrounding cells'
        env_sample channel. Note: this is *exteroceptive* perturbation
        (sigma-channel noise). The Phase 3 commitment is to add a separate
        *affordance perturbation* mechanism later, which is not yet
        instantiated here."""
        scale = float(self.cfg.perturbation_scale)
        n_cells = len(self.cfg3.patch_order)
        distortion = np.zeros(n_cells, dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg3.patch_order):
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

    # ----- body silhouette (interoceptive) -----

    def _compute_body_silhouette(self) -> Optional[np.ndarray]:
        """Return 4-d directional affordance silhouette for the agent's
        N/S/W/E cardinal neighbors (in that order), with Gaussian blur.

        This is *interoceptive*: the agent feels (through body) the
        affordance of the cells it is adjacent to, but with noise — only
        a "silhouette" of the directional affordance structure.

        Returns None when silhouette_dim == 0 (feature disabled).
        """
        if int(self.cfg3.silhouette_dim) <= 0:
            return None
        # Cardinal neighbors: N=(0,-1), S=(0,+1), W=(-1,0), E=(+1,0).
        cardinals = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        sigma = float(self.cfg3.silhouette_noise_sigma)
        out = np.zeros(4, dtype=np.float32)
        for i, (dx, dy) in enumerate(cardinals):
            px, py = self._patch_coord(self.x + dx, self.y + dy)
            clean = float(self._affordance_map[py, px])
            noise = float(self._rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0
            out[i] = clean + noise
        return out

    def _info_dict(self) -> Dict[str, Any]:
        info = super()._info_dict()
        info["body_state"] = float(self.body_state[0])
        info["affordance_here"] = float(self._affordance_map[self.y, self.x])
        info["valence_zone_id"] = int(self.valence_zone_id(self.y))
        info["metabolic_delta"] = float(self._last_metabolic_delta)
        info["affordance_gain"] = float(self._last_affordance_gain)
        sil = self._compute_body_silhouette()
        if sil is not None:
            info["body_silhouette"] = sil
        return info


def make_env(**kwargs) -> NZonePhase3Env:
    return NZonePhase3Env(config=NZonePhase3Config(**kwargs))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}  (expected (16,) — 8 surrounding cells × 2 channels)")
    print(f"observation_space: {env.observation_space}")
    print(f"grid: {env.W} x {env.H}, start: ({env.x}, {env.y})")
    print(f"initial body_state: {info['body_state']:.4f}  (interoceptive, separate from obs)")
    print(f"initial affordance_here: {info['affordance_here']:+.4f}  "
          f"(valence zone {info['valence_zone_id']}, env-internal — not in obs)")
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
        action_name = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][action]
        print(
            f"  t={info['t']:3d}  {action_name:5s}  pos=({info['x']:2d},{info['y']:2d})  "
            f"vz={info['valence_zone_id']}  "
            f"body={info['body_state']:.4f}  "
            f"afford_here={info['affordance_here']:+.3f}"
        )