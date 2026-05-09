#cear_pilot/scripts/probe_g_dependency.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Probe whether the action-conditioned interoceptive tendency field depends on g.

Single run:
  PYTHONPATH=. python cear_pilot/scripts/probe_g_dependency.py \
    --outdir outputs/p3_v3_cohorts/full/s0 \
    --device cuda \
    --late_episodes 50 \
    --n_samples 600

Cohort batch:
  PYTHONPATH=. python cear_pilot/scripts/probe_g_dependency.py \
    --base outputs/p3_v3_cohorts \
    --cohorts full no_body_in_g no_conative \
    --seed_start 0 --seed_end 29 \
    --device cuda \
    --late_episodes 50 \
    --n_samples 600

Main interpretation:
  If replacing g with zero/mean/shuffled/random changes pred_trend_UP-DOWN
  or conative q_UP-DOWN much more in full than in no_body_in_g, then the
  interoceptive tendency field is more g-dependent in the full model.

  If interventions barely change the field in both full and no_body_in_g,
  then ordinary tendency prediction is being carried mostly by z/body/silhouette
  rather than by g; body_in_g may still matter for perturbation residue / same-input
  perspective probes, but not for baseline field learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def zone3(x: int, y: int) -> str:
    xz = "L" if x <= 4 else ("M" if x <= 9 else "R")
    yz = "T" if y <= 4 else ("M" if y <= 9 else "B")
    return yz + xz


def valence_name(v: int) -> str:
    return {0: "top", 1: "mid", 2: "bottom"}.get(int(v), f"vz{v}")


def softmax_np(x: np.ndarray, temp: float) -> np.ndarray:
    temp = max(float(temp), 1e-6)
    z = x / temp
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def kl_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def load_builders():
    """Prefer conative trainer, fallback to train_phase3_v3."""
    try:
        from cear_pilot.training import train_phase3_v3_conative as tr
        return tr
    except Exception:
        from cear_pilot.training import train_phase3_v3 as tr
        return tr


def build_models_from_checkpoint(ckpt_path: Path, device: str):
    tr = load_builders()
    ckpt = torch.load(ckpt_path, map_location=device)
    args_dict = dict(ckpt.get("meta", {}).get("args", {}))
    if not args_dict:
        raise RuntimeError(f"Checkpoint missing meta['args']: {ckpt_path}")
    args_dict["device"] = device
    args = SimpleNamespace(**args_dict)

    built = tr.build_agent_and_decoder(args)
    if len(built) != 3:
        raise RuntimeError("Expected build_agent_and_decoder(args) -> (agent, decoder, body_decoder).")
    agent, obs_decoder, body_decoder = built

    agent.load_state_dict(ckpt["agent_state"])
    obs_decoder.load_state_dict(ckpt["decoder_state"])
    if body_decoder is None:
        raise RuntimeError("body_decoder is None; g-dependency probe requires BodyDecoder.")
    body_decoder.load_state_dict(ckpt["body_decoder_state"])

    agent.to(device).eval()
    obs_decoder.to(device).eval()
    body_decoder.to(device).eval()
    return ckpt, args, agent, obs_decoder, body_decoder


def build_env_from_checkpoint(ckpt: Dict[str, Any]):
    from cear_pilot.envs.nzone_phase3 import NZonePhase3Env, NZonePhase3Config
    env_cfg_dict = dict(ckpt.get("meta", {}).get("env_cfg", {}))
    if not env_cfg_dict:
        raise RuntimeError("Checkpoint missing meta['env_cfg'].")
    env = NZonePhase3Env(config=NZonePhase3Config(**env_cfg_dict))
    return env


def set_env_state_from_row(env, row: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Reset to initialize RNG/maps/perturbation internals, then impose logged state.
    env.reset(seed=int(row["episode"]))
    env.x = int(row["x"])
    env.y = int(row["y"])
    env.t = int(row["t"])

    if "body_u" in row.index:
        env.body_u = float(row["body_u"])
        # Use env helper if present; otherwise compute directly.
        body = 1.0 / (1.0 + float(np.exp(-env.body_u)))
        env.body_state = np.array([body], dtype=np.float32)
    else:
        env.body_state = np.array([float(row["body_state"])], dtype=np.float32)

    if hasattr(env, "_perturbation_active") and "perturbation_active" in row.index:
        env._perturbation_active = bool(int(row["perturbation_active"]))
    if hasattr(env, "_perturbation_trace") and "perturbation_trace" in row.index:
        env._perturbation_trace = float(row["perturbation_trace"])

    obs = env._observe()
    info = env._info_dict()
    return obs, info


def sample_late_rows(df: pd.DataFrame, late_episodes: int, n_samples: int, seed: int) -> pd.DataFrame:
    max_ep = int(df["episode"].max())
    d = df[df["episode"] >= max_ep - late_episodes + 1].copy()
    d["valence_name"] = d["valence_zone_id"].map(valence_name)
    d["zone3"] = [zone3(x, y) for x, y in zip(d["x"], d["y"])]
    d["prev_action"] = d.groupby("episode")["action"].shift(1).fillna(STAY).astype(int)

    if len(d) <= n_samples:
        return d.reset_index(drop=True)

    # Stratify by top/mid/bottom so rare bottom states are represented.
    rng = np.random.default_rng(seed)
    parts = []
    per = max(1, n_samples // 3)
    for vn in ["top", "mid", "bottom"]:
        sub = d[d["valence_name"] == vn]
        if len(sub) == 0:
            continue
        take = min(per, len(sub))
        parts.append(sub.sample(n=take, random_state=int(rng.integers(1_000_000))))
    out = pd.concat(parts, ignore_index=True) if parts else d.sample(n=n_samples, random_state=seed)
    if len(out) < n_samples and len(d) > len(out):
        rest = d.drop(index=out.index, errors="ignore")
        take = min(n_samples - len(out), len(rest))
        if take > 0:
            out = pd.concat([out, rest.sample(n=take, random_state=seed + 1)], ignore_index=True)
    return out.reset_index(drop=True)


def g_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("g_") and c[2:].isdigit()]
    return sorted(cols, key=lambda s: int(s.split("_")[1]))


@torch.no_grad()
def forward_state(agent, env, row, device: str, n_actions: int):
    obs, info = set_env_state_from_row(env, row)
    x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    body_t = torch.tensor([[float(info["body_state"])]], dtype=torch.float32, device=device)

    prev_action = int(row.get("prev_action", STAY))
    p_t = F.one_hot(torch.tensor([prev_action], device=device), num_classes=n_actions).float()

    err_dim = int(getattr(agent.cfg.world, "err_dim", 6))
    err_t = torch.zeros((1, err_dim), dtype=torch.float32, device=device)

    body_silhouette_t = None
    if "body_silhouette" in info:
        body_silhouette_t = torch.tensor(info["body_silhouette"][None, :], dtype=torch.float32, device=device)

    # Initialize agent g from logged pre-forward g if available.
    gcols = [c for c in row.index if c.startswith("g_") and c[2:].isdigit()]
    if gcols:
        gcols = sorted(gcols, key=lambda s: int(s.split("_")[1]))
        logged_g = torch.tensor(row[gcols].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32, device=device)
        agent.reset(batch_size=1)
        agent.set_g(logged_g)
    else:
        agent.reset(batch_size=1)

    if getattr(agent, "_body_pred_prev", None) is not None:
        agent._body_pred_prev = body_t.detach().clone()

    out = agent.forward_step(
        x_t, p_t,
        err_t=err_t,
        body_actual_t=body_t,
        body_silhouette_t=body_silhouette_t,
    )

    # True tendency for each action from current imposed state.
    true_trends = []
    if hasattr(env, "counterfactual_body_tendency"):
        for a in range(n_actions):
            true_trends.append(float(env.counterfactual_body_tendency(a, horizon=5)))
    else:
        true_trends = [np.nan] * n_actions

    return out, body_t, np.array(true_trends, dtype=np.float32), info


def predict_field(body_decoder, z, g, body_t, temp: float, trend_w: float, body_w: float):
    body_all, trend_all = body_decoder.predict_all_actions(z, g, body_t)
    body_np = body_all.detach().cpu().numpy()[0, :, 0]
    trend_np = trend_all.detach().cpu().numpy()[0, :, 0]
    field_value = trend_w * trend_np + body_w * body_np
    q = softmax_np(field_value, temp=temp)
    return body_np, trend_np, q, field_value


def probe_one_run(
    outdir: Path,
    device: str,
    late_episodes: int,
    n_samples: int,
    seed: int,
    conative_temperature: float | None = None,
    conative_trend_weight: float | None = None,
    conative_body_weight: float | None = None,
    cohort: str = "",
    seed_id: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    traj_path = outdir / "traj.parquet"
    ckpt_path = outdir / "ckpt_final.pt"
    if not traj_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing traj/checkpoint in {outdir}")

    ckpt, args, agent, obs_decoder, body_decoder = build_models_from_checkpoint(ckpt_path, device)
    env = build_env_from_checkpoint(ckpt)
    n_actions = int(env.action_space.n)

    temp = float(conative_temperature if conative_temperature is not None else getattr(args, "conative_temperature", 0.10))
    trend_w = float(conative_trend_weight if conative_trend_weight is not None else getattr(args, "conative_trend_weight", 1.0))
    body_w = float(conative_body_weight if conative_body_weight is not None else getattr(args, "conative_body_weight", 0.25))

    df = pd.read_parquet(traj_path)
    sample = sample_late_rows(df, late_episodes=late_episodes, n_samples=n_samples, seed=seed)
    gcols = g_columns(df)
    if not gcols:
        raise RuntimeError(f"No g columns in {traj_path}")

    late = df[df["episode"] >= int(df["episode"].max()) - late_episodes + 1]
    mean_g_np = late[gcols].mean().to_numpy(dtype=np.float32)
    std_g_np = late[gcols].std().fillna(0.0).to_numpy(dtype=np.float32)
    std_g_np = np.where(std_g_np < 1e-6, 1.0, std_g_np)

    rng = np.random.default_rng(seed)
    shuffled_g_rows = sample[gcols].sample(frac=1.0, random_state=seed + 17).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for i, row in sample.iterrows():
        out, body_t, true_trends, info = forward_state(agent, env, row, device, n_actions)
        z = out["z"].detach()
        g_actual = out["g"].detach()
        g_zero = torch.zeros_like(g_actual)
        g_mean = torch.tensor(mean_g_np[None, :], dtype=torch.float32, device=device)
        g_shuf = torch.tensor(shuffled_g_rows.iloc[i].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32, device=device)
        rand_np = rng.normal(loc=mean_g_np, scale=std_g_np).astype(np.float32)
        g_random = torch.tensor(rand_np[None, :], dtype=torch.float32, device=device)

        variants = {
            "actual": g_actual,
            "zero": g_zero,
            "mean": g_mean,
            "shuffled": g_shuf,
            "random": g_random,
        }

        base_body, base_trend, base_q, base_field = predict_field(body_decoder, z, g_actual, body_t, temp, trend_w, body_w)
        base_trend_ud = float(base_trend[UP] - base_trend[DOWN])
        base_q_ud = float(base_q[UP] - base_q[DOWN])

        for mode, gv in variants.items():
            body_np, trend_np, q_np, field_np = predict_field(body_decoder, z, gv, body_t, temp, trend_w, body_w)
            row_out = {
                "cohort": cohort,
                "seed": seed_id if seed_id is not None else seed,
                "sample_id": i,
                "g_mode": mode,
                "episode": int(row["episode"]),
                "t": int(row["t"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "zone3": zone3(int(row["x"]), int(row["y"])),
                "valence_name": valence_name(int(row["valence_zone_id"])),
                "body_state": float(info["body_state"]),
                "body_u": float(getattr(env, "body_u", np.nan)),
                "true_trend_UP_minus_DOWN": float(true_trends[UP] - true_trends[DOWN]),
                "pred_trend_UP": float(trend_np[UP]),
                "pred_trend_DOWN": float(trend_np[DOWN]),
                "pred_trend_STAY": float(trend_np[STAY]),
                "pred_trend_UP_minus_DOWN": float(trend_np[UP] - trend_np[DOWN]),
                "pred_body_UP_minus_DOWN": float(body_np[UP] - body_np[DOWN]),
                "q_UP": float(q_np[UP]),
                "q_DOWN": float(q_np[DOWN]),
                "q_LEFT": float(q_np[LEFT]),
                "q_RIGHT": float(q_np[RIGHT]),
                "q_STAY": float(q_np[STAY]),
                "q_UP_minus_DOWN": float(q_np[UP] - q_np[DOWN]),
                "q_entropy": float(-np.sum(q_np * np.log(q_np + 1e-9))),
                "delta_trend_ud_from_actual": float(abs((trend_np[UP] - trend_np[DOWN]) - base_trend_ud)),
                "delta_q_ud_from_actual": float(abs((q_np[UP] - q_np[DOWN]) - base_q_ud)),
                "kl_q_actual_to_mode": float(kl_np(base_q, q_np)),
                "g_norm_mode": float(gv.detach().cpu().norm().item()),
                "g_norm_actual": float(g_actual.detach().cpu().norm().item()),
            }
            rows.append(row_out)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["cohort", "seed", "g_mode", "valence_name"], as_index=False)
        .agg(
            n=("sample_id", "size"),
            pred_trend_UP_minus_DOWN=("pred_trend_UP_minus_DOWN", "mean"),
            true_trend_UP_minus_DOWN=("true_trend_UP_minus_DOWN", "mean"),
            q_UP_minus_DOWN=("q_UP_minus_DOWN", "mean"),
            q_entropy=("q_entropy", "mean"),
            delta_trend_ud_from_actual=("delta_trend_ud_from_actual", "mean"),
            delta_q_ud_from_actual=("delta_q_ud_from_actual", "mean"),
            kl_q_actual_to_mode=("kl_q_actual_to_mode", "mean"),
            body_state=("body_state", "mean"),
            body_u=("body_u", "mean"),
        )
    )

    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(diag_dir / "g_dependency_probe_samples.csv", index=False)
    summary.to_csv(diag_dir / "g_dependency_probe_summary.csv", index=False)
    return raw, summary


def mean_se_str(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "NA"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{x.mean():.4f} ± {se:.4f}"


def print_batch_summary(all_summary: pd.DataFrame):
    print("\n=== G-DEPENDENCY PROBE SUMMARY ===")
    metrics = [
        "pred_trend_UP_minus_DOWN",
        "true_trend_UP_minus_DOWN",
        "q_UP_minus_DOWN",
        "delta_trend_ud_from_actual",
        "delta_q_ud_from_actual",
        "kl_q_actual_to_mode",
    ]
    # Average over valence zones first within seed/mode, then report over seeds.
    seed_level = (
        all_summary.groupby(["cohort", "seed", "g_mode"], as_index=False)[metrics]
        .mean()
    )
    for cohort, cg in seed_level.groupby("cohort"):
        print(f"\n[{cohort}] seeds={cg['seed'].nunique()}")
        for mode in ["actual", "zero", "mean", "shuffled", "random"]:
            mg = cg[cg["g_mode"] == mode]
            if mg.empty:
                continue
            print(f"  g_mode={mode}")
            for m in metrics:
                print(f"    {m:30s} {mean_se_str(mg[m])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="", help="Single run directory containing traj.parquet and ckpt_final.pt")
    ap.add_argument("--base", type=str, default="", help="Base directory containing cohort/sSEED subdirs")
    ap.add_argument("--cohorts", nargs="*", default=["full", "no_body_in_g", "no_conative"])
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=29)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--late_episodes", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=600)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--conative_temperature", type=float, default=None)
    ap.add_argument("--conative_trend_weight", type=float, default=None)
    ap.add_argument("--conative_body_weight", type=float, default=None)
    args = ap.parse_args()

    if args.outdir:
        outdir = Path(args.outdir)
        raw, summary = probe_one_run(
            outdir=outdir,
            device=args.device,
            late_episodes=args.late_episodes,
            n_samples=args.n_samples,
            seed=args.sample_seed,
            conative_temperature=args.conative_temperature,
            conative_trend_weight=args.conative_trend_weight,
            conative_body_weight=args.conative_body_weight,
            cohort=outdir.parent.name,
            seed_id=int(outdir.name[1:]) if outdir.name.startswith("s") and outdir.name[1:].isdigit() else None,
        )
        print(summary)
        print(f"\nSaved to {outdir / 'diagnostics'}")
        return

    if not args.base:
        raise SystemExit("Provide either --outdir or --base.")

    base = Path(args.base)
    summaries = []
    for cohort in args.cohorts:
        for sid in range(args.seed_start, args.seed_end + 1):
            outdir = base / cohort / f"s{sid}"
            if not (outdir / "traj.parquet").exists():
                print(f"[skip missing] {outdir}")
                continue
            print(f"[probe] {cohort} seed={sid}")
            _, summ = probe_one_run(
                outdir=outdir,
                device=args.device,
                late_episodes=args.late_episodes,
                n_samples=args.n_samples,
                seed=args.sample_seed + sid,
                conative_temperature=args.conative_temperature,
                conative_trend_weight=args.conative_trend_weight,
                conative_body_weight=args.conative_body_weight,
                cohort=cohort,
                seed_id=sid,
            )
            summaries.append(summ)

    if not summaries:
        raise SystemExit("No summaries produced.")
    all_summary = pd.concat(summaries, ignore_index=True)
    out_csv = base / f"g_dependency_probe_summary_s{args.seed_start}_s{args.seed_end}.csv"
    all_summary.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")
    print_batch_summary(all_summary)


if __name__ == "__main__":
    main()
