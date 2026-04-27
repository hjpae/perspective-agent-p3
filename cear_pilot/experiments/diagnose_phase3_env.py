# cear_pilot/experiments/diagnose_phase3_env.py
# -*- coding: utf-8 -*-
"""
Phase 3 environment dynamics diagnostic.

Run random-policy episodes and measure:
  1. Body trajectory distribution (saturation frequency, terminal body dist)
  2. Visit pattern (does random walk show vertical bias from affordance alone?
     it shouldn't — only body trajectory should reflect affordance)
  3. Affordance-body systematic coupling (body change correlates with visit
     row's affordance)
  4. Comparison across metabolic/affordance_gain settings to find a regime
     where body is informative (not trivially saturated, coupled to position)

Usage:
  python -m cear_pilot.experiments.diagnose_phase3_env
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from cear_pilot.envs.nzone_phase3 import NZonePhase3Config, make_env


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def random_rollout(
    cfg: NZonePhase3Config,
    seed: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """One random-policy episode. Returns trajectories of body, position, etc."""
    env = make_env(**asdict(cfg))
    obs, info = env.reset(seed=seed)

    T = cfg.max_steps
    body = np.zeros(T + 1, dtype=np.float32)
    xs = np.zeros(T + 1, dtype=np.int32)
    ys = np.zeros(T + 1, dtype=np.int32)
    affordance_here = np.zeros(T + 1, dtype=np.float32)
    valence_zone = np.zeros(T + 1, dtype=np.int32)
    actions = np.zeros(T, dtype=np.int32)
    body[0] = info["body_state"]
    xs[0] = info["x"]
    ys[0] = info["y"]
    affordance_here[0] = info["affordance_here"]
    valence_zone[0] = info["valence_zone_id"]

    t = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        a = int(rng.integers(0, 5))  # uniform random over 5 actions
        actions[t] = a
        obs, r, terminated, truncated, info = env.step(a)
        t += 1
        body[t] = info["body_state"]
        xs[t] = info["x"]
        ys[t] = info["y"]
        affordance_here[t] = info["affordance_here"]
        valence_zone[t] = info["valence_zone_id"]

    return {
        "body": body[: t + 1],
        "x": xs[: t + 1],
        "y": ys[: t + 1],
        "affordance_here": affordance_here[: t + 1],
        "valence_zone": valence_zone[: t + 1],
        "actions": actions[:t],
        "T": t,
    }


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    cfg_label: str
    n_episodes: int

    # Body trajectory stats
    body_terminal_mean: float
    body_terminal_std: float
    body_min_seen: float
    body_max_seen: float
    saturation_low_rate: float    # fraction of steps where body <= body_min + eps
    saturation_high_rate: float   # fraction of steps where body >= body_max - eps
    body_episode_range_mean: float  # mean over episodes of (max-min) within episode

    # Visit pattern (should NOT be biased under random policy: random walk 결과로
    # spatial distribution이 affordance와 무관해야 — intentional bias가 없으니까.
    # bias 나오면 grid boundary effect)
    vz_visit_fractions: Tuple[float, float, float]  # (top, mid, bot)
    column_zone_visit_fractions: Tuple[float, float, float]  # (left, mid, right)
    mean_y: float
    mean_x: float

    # Affordance-body coupling
    body_change_vs_affordance_corr: float  # per-step: corr(delta_body, affordance_here)


def run_diagnostic(
    cfg: NZonePhase3Config,
    cfg_label: str,
    n_episodes: int = 30,
    base_seed: int = 0,
) -> DiagnosticResult:
    rng = np.random.default_rng(base_seed)

    all_body: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_x: List[np.ndarray] = []
    all_vz: List[np.ndarray] = []
    all_cz: List[np.ndarray] = []
    all_terminal_body: List[float] = []
    all_episode_ranges: List[float] = []
    all_delta_body: List[np.ndarray] = []
    all_affordance: List[np.ndarray] = []

    for ep in range(n_episodes):
        traj = random_rollout(cfg, seed=base_seed + ep, rng=rng)
        body = traj["body"]
        all_body.append(body)
        all_y.append(traj["y"])
        all_x.append(traj["x"])
        all_vz.append(traj["valence_zone"])
        # column zone for each step
        cz = np.array(
            [
                _column_zone_id(int(xx), cfg.report_zone_boundaries)
                for xx in traj["x"]
            ],
            dtype=np.int32,
        )
        all_cz.append(cz)
        all_terminal_body.append(float(body[-1]))
        all_episode_ranges.append(float(body.max() - body.min()))
        # delta body and affordance pairs (step transitions)
        delta = body[1:] - body[:-1]
        # affordance_here at the *destination* cell (after step)
        afford = traj["affordance_here"][1:]
        all_delta_body.append(delta)
        all_affordance.append(afford)

    body_concat = np.concatenate(all_body)
    body_min = float(cfg.body_min)
    body_max = float(cfg.body_max)
    eps = 1e-3

    sat_low = float(np.mean(body_concat <= body_min + eps))
    sat_high = float(np.mean(body_concat >= body_max - eps))

    vz_concat = np.concatenate(all_vz)
    cz_concat = np.concatenate(all_cz)
    vz_fracs = tuple(float(np.mean(vz_concat == z)) for z in (0, 1, 2))
    cz_fracs = tuple(float(np.mean(cz_concat == z)) for z in (0, 1, 2))

    delta_concat = np.concatenate(all_delta_body)
    afford_concat = np.concatenate(all_affordance)

    if delta_concat.std() > 1e-9 and afford_concat.std() > 1e-9:
        corr = float(np.corrcoef(delta_concat, afford_concat)[0, 1])
    else:
        corr = float("nan")

    return DiagnosticResult(
        cfg_label=cfg_label,
        n_episodes=n_episodes,
        body_terminal_mean=float(np.mean(all_terminal_body)),
        body_terminal_std=float(np.std(all_terminal_body)),
        body_min_seen=float(body_concat.min()),
        body_max_seen=float(body_concat.max()),
        saturation_low_rate=sat_low,
        saturation_high_rate=sat_high,
        body_episode_range_mean=float(np.mean(all_episode_ranges)),
        vz_visit_fractions=vz_fracs,  # type: ignore[arg-type]
        column_zone_visit_fractions=cz_fracs,  # type: ignore[arg-type]
        mean_y=float(np.mean(np.concatenate(all_y))),
        mean_x=float(np.mean(np.concatenate(all_x))),
        body_change_vs_affordance_corr=corr,
    )


def _column_zone_id(x: int, boundaries: Tuple[int, ...]) -> int:
    for zi, b in enumerate(boundaries):
        if x < int(b):
            return zi
    return len(boundaries)


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_result(r: DiagnosticResult) -> None:
    print(f"\n=== {r.cfg_label}  (n_episodes={r.n_episodes}) ===")
    print(f"  body terminal: {r.body_terminal_mean:.3f} ± {r.body_terminal_std:.3f}")
    print(f"  body range seen: [{r.body_min_seen:.3f}, {r.body_max_seen:.3f}]")
    print(f"  body saturation: low={r.saturation_low_rate:.3f}  high={r.saturation_high_rate:.3f}")
    print(f"  body episode range (mean): {r.body_episode_range_mean:.3f}")
    print(f"  visits valence zones (top/mid/bot): "
          f"{r.vz_visit_fractions[0]:.3f} / {r.vz_visit_fractions[1]:.3f} / "
          f"{r.vz_visit_fractions[2]:.3f}")
    print(f"  visits column zones (L/M/R):       "
          f"{r.column_zone_visit_fractions[0]:.3f} / "
          f"{r.column_zone_visit_fractions[1]:.3f} / "
          f"{r.column_zone_visit_fractions[2]:.3f}")
    print(f"  mean position: x={r.mean_x:.2f} (center=7), y={r.mean_y:.2f} (center=7)")
    print(f"  corr(delta_body, affordance_here): {r.body_change_vs_affordance_corr:+.3f}")


# ---------------------------------------------------------------------------
# Settings comparison
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--base_seed", type=int, default=0)
    args = p.parse_args()

    base = NZonePhase3Config()

    # Compare multiple settings to find informative regime.
    settings: List[Tuple[str, NZonePhase3Config]] = [
        ("default (metabolic=0.005, move=0.005, gain=0.02)", base),
        ("low gain (gain=0.01)",
         replace(base, affordance_to_body_gain=0.01)),
        ("high gain (gain=0.04)",
         replace(base, affordance_to_body_gain=0.04)),
        ("low metabolic (metabolic=0.002, move=0.002)",
         replace(base, metabolic_cost=0.002, movement_cost=0.002)),
        ("zero metabolic, only movement (metabolic=0, move=0.005)",
         replace(base, metabolic_cost=0.0, movement_cost=0.005)),
        ("balanced (metabolic=0, move=0.005, gain=0.01)",
         replace(base, metabolic_cost=0.0, movement_cost=0.005,
                 affordance_to_body_gain=0.01)),
    ]

    results = []
    for label, cfg in settings:
        r = run_diagnostic(cfg, label, n_episodes=args.n_episodes,
                           base_seed=args.base_seed)
        results.append(r)
        print_result(r)

    print("\n=== summary heuristic ===")
    print("Want: body episode range > 0.1 (informative dynamics),")
    print("      saturation_low + saturation_high < 0.2 (not trivially stuck),")
    print("      |corr(delta_body, affordance)| > 0.3 (clear coupling).")
    print()
    for r in results:
        ok_range = r.body_episode_range_mean > 0.1
        ok_sat = (r.saturation_low_rate + r.saturation_high_rate) < 0.2
        ok_corr = abs(r.body_change_vs_affordance_corr) > 0.3 \
            if not np.isnan(r.body_change_vs_affordance_corr) else False
        flags = "".join([
            "R" if ok_range else "·",
            "S" if ok_sat else "·",
            "C" if ok_corr else "·",
        ])
        print(f"  [{flags}]  {r.cfg_label}")


if __name__ == "__main__":
    main()
