#!/usr/bin/env python3
"""Attribute gains to path controls, topology proposals, and commitment.

The controller critics and path input remain fixed within each case. Only the
proposal mechanism changes.
"""
import argparse
import concurrent.futures
import csv
import os

import numpy as np

from astar import astar_path
from controllers import AmoebaHybrid
from env import BarnEnv
from path_test import lane_path
from run import episode
from experiments.v31_ablation import _path_error


CONFIGS = {
    "v0": {
        "grouped_sampling": False,
        "use_branches": False,
    },
    "path_only": {
        "grouped_sampling": True,
        "use_branches": False,
        "fallback_share": 1.0,
    },
    "path_branches": {
        "grouped_sampling": True,
        "use_branches": True,
        "fallback_share": 0.25,
        "mode_min_dwell": 10,
        "mode_confirm_cycles": 3,
        "mode_switch_margin": 0.3,
    },
    "no_commit": {
        "grouped_sampling": True,
        "use_branches": True,
        "fallback_share": 0.25,
        "mode_min_dwell": 0,
        "mode_confirm_cycles": 1,
        "mode_switch_margin": 0.0,
    },
}


def _make_path(env, scenario, lane):
    if scenario == "normal":
        return astar_path(env, env.start[:2], env.goal)
    lane_x = {"left": -3.5, "center": -2.25, "right": -1.0}[lane]
    return lane_path(env, lane_x)


def _run_case(case):
    world, seed, radius, max_steps, scenario, lane, name = case
    env = BarnEnv(world, robot_r=radius)
    path = _make_path(env, scenario, lane)
    config = dict(CONFIGS[name])
    use_branches = config.pop("use_branches")
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=use_branches,
        max_branches=3, path_validity_gate=True, **config)

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
            ess_values.extend(stat["ess"] for stat in info["mode_stats"])
        branch_ids.update(branch.branch_id
                          for branch in getattr(ctrl, "branches", []))

    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe)
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    plan_clearance = float(np.min(env.clearance(path) - env.robot_r))
    grouped_steps = len(selected_keys)
    return {
        "scenario": scenario,
        "lane": lane if scenario == "blocked" else "astar",
        "config": name,
        "world": world,
        "seed": seed,
        "plan_min_clear_m": round(plan_clearance, 4),
        **metrics,
        "omega_sign_flips": sign_flips,
        "mode_switches": int(getattr(ctrl, "mode_switch_count", 0)),
        "branch_selection_fraction": (
            sum(key != -1 for key in selected_keys) / grouped_steps
            if grouped_steps else 0.0),
        "unique_branch_ids": len(branch_ids),
        "mean_ess": float(np.mean(ess_values)) if ess_values else np.nan,
        "min_ess": float(np.min(ess_values)) if ess_values else np.nan,
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
    }


def _summarize(rows):
    output = []
    for scenario in ("normal", "blocked"):
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
                "timeouts": sum(row["result"] == "timeout" for row in group),
                "mean_time_s": float(np.mean(
                    [row["time_s"] for row in successes]))
                    if successes else np.nan,
                "mean_path_m": float(np.mean(
                    [row["path_len_m"] for row in successes]))
                    if successes else np.nan,
                "worst_clear_m": float(np.min(
                    [row["min_clear_m"] for row in group])),
                "mean_flips": float(np.mean(
                    [row["omega_sign_flips"] for row in group])),
                "mean_switches": float(np.mean(
                    [row["mode_switches"] for row in group])),
                "branch_fraction": float(np.mean(
                    [row["branch_selection_fraction"] for row in group])),
                "mean_path_error_m": float(np.mean(
                    [row["mean_path_error_m"] for row in group])),
            })
    return output


def _print_summary(rows):
    print("\nscenario config         success coll tout time  path  worst_clr "
          "flips switches branch_use path_err")
    for row in _summarize(rows):
        print(
            f"{row['scenario']:8s} {row['config']:13s} "
            f"{row['successes']:2d}/{row['runs']:<2d} "
            f"{row['collisions']:4d} {row['timeouts']:4d} "
            f"{row['mean_time_s']:5.1f} {row['mean_path_m']:5.2f} "
            f"{row['worst_clear_m']:9.3f} {row['mean_flips']:5.1f} "
            f"{row['mean_switches']:8.1f} {row['branch_fraction']:10.2f} "
            f"{row['mean_path_error_m']:8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="4,48,49")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--scenarios", default="normal,blocked")
    parser.add_argument("--lane", choices=("left", "center", "right"),
                        default="center")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--out", default="artifacts/results/milestones/attribution.csv")
    args = parser.parse_args()

    worlds = [int(value) for value in args.worlds.split(",")]
    scenarios = args.scenarios.split(",")
    names = args.configs.split(",")
    if set(scenarios) - {"normal", "blocked"}:
        parser.error("scenarios must contain only normal and blocked")
    if set(names) - set(CONFIGS):
        parser.error("unknown configuration")
    cases = [(world, seed, args.radius, args.max_steps, scenario, args.lane,
              name)
             for scenario in scenarios for world in worlds
             for seed in range(args.seeds) for name in names]

    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['scenario']:7s} {row['config']:13s} "
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
                print(f"{row['scenario']:7s} {row['config']:13s} "
                      f"world={row['world']:2d} seed={row['seed']} "
                      f"{row['result']:9s} clear={row['min_clear_m']:.3f}",
                      flush=True)

    order = {name: index for index, name in enumerate(names)}
    rows.sort(key=lambda row: (
        scenarios.index(row["scenario"]), row["world"], row["seed"],
        order[row["config"]]))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
