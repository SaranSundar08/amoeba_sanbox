#!/usr/bin/env python3
"""Reproducible fixed-versus-adaptive covariance ablation for V5.

Examples:
  python3 -m experiments.v5_ablation --worlds 4,48,49 --seeds 2 --jobs 4
  python3 -m experiments.v5_ablation --worlds 48 --seeds 1 --scenarios dynamic_blocked
"""
import argparse
import concurrent.futures
import csv
import os

import numpy as np

from astar import astar_path
from controllers import AmoebaHybrid
from dynabarn import DynaBarnEnv
from env import BarnEnv
from path_test import lane_path
from simulation import episode
from experiments.v31_ablation import _path_error


CONFIGS = {
    "fixed": {"adaptive_covariance": False},
    "adaptive": {"adaptive_covariance": True},
}
SCENARIOS = ("normal", "blocked", "dynamic", "dynamic_blocked")


def _make_env(world, seed, radius, scenario, dynamic_seed, n_moving,
              moving_speed, moving_amplitude):
    if scenario.startswith("dynamic"):
        dyn_seed = (world * 1000 + seed if dynamic_seed is None
                    else dynamic_seed + world * 1000 + seed)
        return DynaBarnEnv(
            world, robot_r=radius, n_moving=n_moving,
            speed=moving_speed, amplitude=moving_amplitude, seed=dyn_seed)
    return BarnEnv(world, robot_r=radius)


def _make_path(env, scenario, lane):
    if scenario in ("normal", "dynamic"):
        return astar_path(env, env.start[:2], env.goal)
    lane_x = {"left": -3.5, "center": -2.25, "right": -1.0}[lane]
    return lane_path(env, lane_x)


def _run_case(case):
    (world, seed, radius, max_steps, scenario, lane, name, dynamic_seed,
     n_moving, moving_speed, moving_amplitude) = case
    env = _make_env(
        world, seed, radius, scenario, dynamic_seed, n_moving,
        moving_speed, moving_amplitude)
    path = _make_path(env, scenario, lane)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, fallback_share=0.25,
        mode_min_dwell=10, mode_confirm_cycles=3,
        mode_switch_margin=0.3, gate="auto", path_validity_gate=True,
        **CONFIGS[name])

    previous_sign = 0
    sign_flips = 0
    selected_keys = []
    ess_values = []
    branch_ids = set()
    sigma_v, sigma_w = [], []

    def observe(_k, _state, info, _trace):
        nonlocal previous_sign, sign_flips
        omega = float(ctrl.last_applied[1])
        sign = int(np.sign(omega)) if abs(omega) >= 0.05 else 0
        if sign and previous_sign and sign != previous_sign:
            sign_flips += 1
        if sign:
            previous_sign = sign
        selected_keys.append(info["selected_mode"]["key"])
        ess_values.extend(item["ess"] for item in info["mode_stats"])
        branch_ids.update(branch.branch_id
                          for branch in getattr(ctrl, "branches", []))
        mode_stds = info["mode_stds"]
        sigma_v.extend(mode_stds[..., 0].ravel())
        sigma_w.extend(mode_stds[..., 1].ravel())

    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe)
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    return {
        "scenario": scenario,
        "config": name,
        "world": world,
        "seed": seed,
        **metrics,
        "omega_sign_flips": sign_flips,
        "mode_switches": int(ctrl.mode_switch_count),
        "branch_selection_fraction": float(np.mean(
            [key != -1 for key in selected_keys])),
        "unique_branch_ids": len(branch_ids),
        "mean_ess": float(np.mean(ess_values)),
        "min_ess": float(np.min(ess_values)),
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
        "mean_sigma_v": float(np.mean(sigma_v)),
        "mean_sigma_w": float(np.mean(sigma_w)),
    }


def _summarize(rows):
    output = []
    for scenario in SCENARIOS:
        for name in CONFIGS:
            group = [row for row in rows
                     if row["scenario"] == scenario
                     and row["config"] == name]
            if not group:
                continue
            successes = [row for row in group if row["success"]]
            output.append({
                "scenario": scenario,
                "config": name,
                "runs": len(group),
                "successes": len(successes),
                "collisions": sum(row["result"] == "collision"
                                  for row in group),
                "timeouts": sum(row["result"] == "timeout"
                                for row in group),
                "mean_time_s": float(np.mean(
                    [row["time_s"] for row in successes]))
                    if successes else np.nan,
                "mean_path_m": float(np.mean(
                    [row["path_len_m"] for row in successes]))
                    if successes else np.nan,
                "worst_clear_m": float(np.min(
                    [row["min_clear_m"] for row in group])),
                "mean_stall_s": float(np.mean(
                    [row["stall_s"] for row in group])),
                "mean_flips": float(np.mean(
                    [row["omega_sign_flips"] for row in group])),
                "mean_switches": float(np.mean(
                    [row["mode_switches"] for row in group])),
                "mean_ess": float(np.mean(
                    [row["mean_ess"] for row in group])),
                "mean_path_error_m": float(np.mean(
                    [row["mean_path_error_m"] for row in group])),
            })
    return output


def _print_summary(summaries):
    print("\nscenario        covariance success coll tout time  path  worst_clr "
          "stall flips switches ESS path_err")
    for row in summaries:
        print(
            f"{row['scenario']:15s} {row['config']:8s} "
            f"{row['successes']:2d}/{row['runs']:<2d} "
            f"{row['collisions']:4d} {row['timeouts']:4d} "
            f"{row['mean_time_s']:5.1f} {row['mean_path_m']:5.2f} "
            f"{row['worst_clear_m']:9.3f} {row['mean_stall_s']:5.1f} "
            f"{row['mean_flips']:5.1f} {row['mean_switches']:8.1f} "
            f"{row['mean_ess']:5.1f} {row['mean_path_error_m']:8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="4,48,49")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--lane", choices=("left", "center", "right"),
                        default="center")
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dynamic-seed", type=int)
    parser.add_argument("--n-moving", type=int, default=4)
    parser.add_argument("--moving-speed", type=float, default=0.4)
    parser.add_argument("--moving-amplitude", type=float, default=0.6)
    parser.add_argument(
        "--out", default="artifacts/results/milestones/v5_ablation.csv")
    args = parser.parse_args()

    worlds = [int(value) for value in args.worlds.split(",")]
    scenarios = args.scenarios.split(",")
    names = args.configs.split(",")
    if set(scenarios) - set(SCENARIOS):
        parser.error("unknown scenario")
    if set(names) - set(CONFIGS):
        parser.error("unknown configuration")
    cases = [
        (world, seed, args.radius, args.max_steps, scenario, args.lane,
         name, args.dynamic_seed, args.n_moving, args.moving_speed,
         args.moving_amplitude)
        for scenario in scenarios for world in worlds
        for seed in range(args.seeds) for name in names]

    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['scenario']:15s} {row['config']:8s} "
                  f"world={row['world']:2d} seed={row['seed']} "
                  f"{row['result']:9s} clear={row['min_clear_m']:.3f}",
                  flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_case, case) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{row['scenario']:15s} {row['config']:8s} "
                      f"world={row['world']:2d} seed={row['seed']} "
                      f"{row['result']:9s} clear={row['min_clear_m']:.3f}",
                      flush=True)

    scenario_order = {name: index for index, name in enumerate(scenarios)}
    config_order = {name: index for index, name in enumerate(names)}
    rows.sort(key=lambda row: (
        scenario_order[row["scenario"]], row["world"], row["seed"],
        config_order[row["config"]]))
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
