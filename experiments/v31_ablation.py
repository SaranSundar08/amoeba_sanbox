#!/usr/bin/env python3
"""Reproducible V3.1 stability ablation for the Nav2-oriented hybrid controller.

Examples:
  python3 -m experiments.v31_ablation --worlds 4,48,49 --seeds 2 --jobs 3
  python3 -m experiments.v31_ablation --worlds 48 --seeds 1 --configs baseline,default
"""
import argparse
import concurrent.futures
import csv
import os

import numpy as np

from astar import astar_path
from controllers import AmoebaHybrid
from env import BarnEnv
from run import episode


CONFIGS = {
    "baseline": {"grouped_sampling": False},
    "path25": {
        "grouped_sampling": True, "fallback_share": 0.25,
        "mode_min_dwell": 10, "mode_confirm_cycles": 3,
        "mode_switch_margin": 0.3,
    },
    "default": {
        "grouped_sampling": True, "fallback_share": 0.50,
        "mode_min_dwell": 10, "mode_confirm_cycles": 3,
        "mode_switch_margin": 0.3,
    },
    "path70": {
        "grouped_sampling": True, "fallback_share": 0.70,
        "mode_min_dwell": 10, "mode_confirm_cycles": 3,
        "mode_switch_margin": 0.3,
    },
    "conservative": {
        "grouped_sampling": True, "fallback_share": 0.50,
        "mode_min_dwell": 20, "mode_confirm_cycles": 5,
        "mode_switch_margin": 0.6,
    },
}


def _path_error(trace, path):
    d = np.linalg.norm(trace[:, None, :2] - path[None, :, :], axis=-1)
    nearest = d.min(axis=1)
    return float(nearest.mean()), float(nearest.max())


def _summarize(rows):
    summaries = []
    for name in CONFIGS:
        selected = [row for row in rows if row["config"] == name]
        if not selected:
            continue
        successes = [row for row in selected if row["success"]]
        summaries.append({
            "config": name,
            "runs": len(selected),
            "successes": len(successes),
            "collisions": sum(row["result"] == "collision"
                              for row in selected),
            "mean_time_s": float(np.mean(
                [row["time_s"] for row in successes]))
                if successes else np.nan,
            "mean_path_m": float(np.mean(
                [row["path_len_m"] for row in successes]))
                if successes else np.nan,
            "worst_clear_m": float(np.min(
                [row["min_clear_m"] for row in selected])),
            "mean_clear_m": float(np.mean(
                [row["min_clear_m"] for row in selected])),
            "mean_sign_flips": float(np.mean(
                [row["omega_sign_flips"] for row in selected])),
            "mean_mode_switches": float(np.mean(
                [row["mode_switches"] for row in selected])),
            "mean_fallback_fraction": float(np.mean(
                [row["fallback_fraction"] for row in selected])),
            "mean_ess": float(np.mean(
                [row["mean_ess"] for row in selected])),
            "mean_path_error_m": float(np.mean(
                [row["mean_path_error_m"] for row in selected])),
        })
    return summaries


def _run_case(case):
    world, seed, radius, max_steps, name = case
    env = BarnEnv(world, robot_r=radius)
    path = astar_path(env, env.start[:2], env.goal)
    kwargs = dict(CONFIGS[name])
    if kwargs.get("grouped_sampling"):
        kwargs.update(extract_branches=True, max_branches=3)
    ctrl = AmoebaHybrid(env, seed=seed, path=path, **kwargs)

    previous_sign = 0
    sign_flips = 0
    selected_keys = []
    ess_values = []
    branch_ids = set()

    def observe(_k, _state, info, _trace):
        nonlocal previous_sign, sign_flips
        omega = float(ctrl.last_applied[1])
        sign = int(np.sign(omega)) if abs(omega) >= 0.05 else 0
        if sign and previous_sign and sign != previous_sign:
            sign_flips += 1
        if sign:
            previous_sign = sign
        if "selected_mode" in info:
            selected_keys.append(info["selected_mode"]["key"])
            ess_values.extend(item["ess"] for item in info["mode_stats"])
        branch_ids.update(branch.branch_id
                          for branch in getattr(ctrl, "branches", []))

    metrics = episode(
        env, ctrl, max_steps=max_steps, on_step=observe)
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    grouped_steps = len(selected_keys)
    return {
        "config": name,
        "world": world,
        "seed": seed,
        **metrics,
        "omega_sign_flips": sign_flips,
        "mode_switches": int(getattr(ctrl, "mode_switch_count", 0)),
        "fallback_fraction": (
            selected_keys.count(-1) / grouped_steps
            if grouped_steps else 1.0),
        "unique_branch_ids": len(branch_ids),
        "mean_ess": float(np.mean(ess_values)) if ess_values else np.nan,
        "min_ess": float(np.min(ess_values)) if ess_values else np.nan,
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
        "mean_field_ms": round(float(np.mean(ctrl.build_ms)), 3),
        "mean_proposal_ms": round(
            float(np.mean(ctrl.proposal_ms)) if ctrl.proposal_ms else 0.0, 3),
        "mean_path_proposal_ms": round(
            float(np.mean(ctrl.path_proposal_ms))
            if ctrl.path_proposal_ms else 0.0, 3),
    }


def _print_summary(summaries):
    print("\nconfig          success  coll  time   path  worst_clr  flips  switches "
          "fallback  ESS   path_err")
    for row in summaries:
        print(
            f"{row['config']:13s} "
            f"{row['successes']:2d}/{row['runs']:<2d} "
            f"{row['collisions']:5d} "
            f"{row['mean_time_s']:5.1f} "
            f"{row['mean_path_m']:6.2f} "
            f"{row['worst_clear_m']:9.3f} "
            f"{row['mean_sign_flips']:6.1f} "
            f"{row['mean_mode_switches']:8.1f} "
            f"{row['mean_fallback_fraction']:8.2f} "
            f"{row['mean_ess']:5.1f} "
            f"{row['mean_path_error_m']:8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="4,48,49")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--out", default="artifacts/results/milestones/v31_ablation.csv")
    args = parser.parse_args()

    worlds = [int(value) for value in args.worlds.split(",")]
    names = args.configs.split(",")
    unknown = sorted(set(names) - set(CONFIGS))
    if unknown:
        parser.error(f"unknown configs: {','.join(unknown)}")
    cases = [(world, seed, args.radius, args.max_steps, name)
             for world in worlds for seed in range(args.seeds)
             for name in names]

    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['config']:13s} world={row['world']:2d} "
                  f"seed={row['seed']} {row['result']:9s} "
                  f"time={row['time_s']:5.1f}s clear={row['min_clear_m']:.3f} "
                  f"flips={row['omega_sign_flips']:3d}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_case, case) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{row['config']:13s} world={row['world']:2d} "
                      f"seed={row['seed']} {row['result']:9s} "
                      f"time={row['time_s']:5.1f}s "
                      f"clear={row['min_clear_m']:.3f} "
                      f"flips={row['omega_sign_flips']:3d}", flush=True)

    rows.sort(key=lambda row: (
        row["world"], row["seed"], names.index(row["config"])))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    _print_summary(_summarize(rows))


if __name__ == "__main__":
    main()
