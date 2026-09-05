#!/usr/bin/env python3
"""Paired current-only versus horizon-predicted dynamic-obstacle ablation.

Every configuration pair receives the same BARN world, MPPI seed, moving
obstacle placement, and motion phases.  Rendering is intentionally absent so
the reported wall time measures controller/simulation work only.
"""
import argparse
import concurrent.futures
import csv
import os
import time

import numpy as np

from astar import astar_path
from controllers import AmoebaHybrid
from dynabarn import DynaBarnEnv
from simulation import episode
from experiments.v31_ablation import _path_error


CONFIGS = {
    "current": {"dynamic_prediction": False},
    "predicted": {"dynamic_prediction": True},
}


def _run_case(case):
    (world, seed, radius, max_steps, name, dynamic_seed, n_moving,
     moving_speed, moving_amplitude, samples, horizon,
     prediction_horizon, uncertainty_rate) = case
    env = DynaBarnEnv(
        world, robot_r=radius, n_moving=n_moving, speed=moving_speed,
        amplitude=moving_amplitude, seed=dynamic_seed)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, K=samples, T=horizon,
        extract_branches=True, max_branches=3, grouped_sampling=True,
        importance_weighting=True, importance_ess_guard=True,
        adaptive_covariance=True, fallback_share=0.25,
        mode_min_dwell=10, mode_confirm_cycles=3,
        mode_switch_margin=0.3, gate="auto", path_validity_gate=True,
        prediction_horizon=prediction_horizon,
        prediction_uncertainty_rate=uncertainty_rate,
        **CONFIGS[name])

    selected_keys = []
    ess_values = []

    def observe(_step, _state, info, _trace):
        selected_keys.append(info["selected_mode"]["key"])
        ess_values.extend(item["ess"] for item in info["mode_stats"])

    started = time.perf_counter()
    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe)
    wall_time = time.perf_counter() - started
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    return {
        "config": name,
        "world": world,
        "seed": seed,
        "dynamic_seed": dynamic_seed,
        "samples": samples,
        "horizon": horizon,
        "prediction_horizon_s": prediction_horizon,
        "uncertainty_rate_mps": uncertainty_rate,
        **metrics,
        "wall_time_s": round(wall_time, 3),
        "controller_hz": round((len(trace) - 1) / max(wall_time, 1e-9), 3),
        "mode_switches": int(ctrl.mode_switch_count),
        "omega_sign_flips": int(ctrl.omega_sign_flips),
        "branch_selection_fraction": float(np.mean(
            [key != -1 for key in selected_keys])) if selected_keys else 0.0,
        "mean_ess": float(np.mean(ess_values)) if ess_values else np.nan,
        "min_ess": float(np.min(ess_values)) if ess_values else np.nan,
        "ess_guard_cycles": int(ctrl.importance_guard_cycles),
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
    }


def summarize(rows):
    summaries = []
    for name in CONFIGS:
        group = [row for row in rows if row["config"] == name]
        if not group:
            continue
        successes = [row for row in group if row["success"]]
        summaries.append({
            "config": name,
            "runs": len(group),
            "successes": len(successes),
            "success_rate": len(successes) / len(group),
            "collisions": sum(row["result"] == "collision" for row in group),
            "timeouts": sum(row["result"] == "timeout" for row in group),
            "mean_time_success_s": (float(np.mean(
                [row["time_s"] for row in successes]))
                if successes else np.nan),
            "mean_path_success_m": (float(np.mean(
                [row["path_len_m"] for row in successes]))
                if successes else np.nan),
            "mean_clearance_m": float(np.mean(
                [row["min_clear_m"] for row in group])),
            "worst_clearance_m": float(np.min(
                [row["min_clear_m"] for row in group])),
            "mean_stall_s": float(np.mean([row["stall_s"] for row in group])),
            "mean_controller_hz": float(np.mean(
                [row["controller_hz"] for row in group])),
            "mean_ess": float(np.mean([row["mean_ess"] for row in group])),
            "mean_mode_switches": float(np.mean(
                [row["mode_switches"] for row in group])),
            "mean_path_error_m": float(np.mean(
                [row["mean_path_error_m"] for row in group])),
        })
    return summaries


def paired_deltas(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row["world"], row["seed"]), {})[
            row["config"]] = row
    complete = [pair for pair in pairs.values()
                if set(pair) == set(CONFIGS)]
    if not complete:
        return {}
    return {
        "pairs": len(complete),
        "success_delta": sum(
            pair["predicted"]["success"] - pair["current"]["success"]
            for pair in complete),
        "collision_delta": sum(
            (pair["predicted"]["result"] == "collision")
            - (pair["current"]["result"] == "collision")
            for pair in complete),
        "mean_clearance_delta_m": float(np.mean([
            pair["predicted"]["min_clear_m"]
            - pair["current"]["min_clear_m"] for pair in complete])),
        "mean_path_delta_m": float(np.mean([
            pair["predicted"]["path_len_m"]
            - pair["current"]["path_len_m"] for pair in complete])),
    }


def _write_csv(path, rows):
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return absolute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="4,42,48,49,100,120,138,160,180,200")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dynamic-seed", type=int, default=61000)
    parser.add_argument("--n-moving", type=int, default=4)
    parser.add_argument("--moving-speed", type=float, default=0.4)
    parser.add_argument("--moving-amplitude", type=float, default=0.6)
    parser.add_argument("--prediction-horizon", type=float, default=2.0)
    parser.add_argument("--uncertainty-rate", type=float, default=0.03)
    parser.add_argument(
        "--out", default="artifacts/results/milestones/v61_ablation.csv")
    args = parser.parse_args()

    worlds = [int(value) for value in args.worlds.split(",") if value]
    names = [value for value in args.configs.split(",") if value]
    if not worlds or args.seeds < 1:
        parser.error("at least one world and seed are required")
    if set(names) - set(CONFIGS):
        parser.error("unknown configuration")
    cases = []
    for world in worlds:
        for seed in range(args.seeds):
            # Pair identity is explicit and independent of execution order.
            dyn_seed = args.dynamic_seed + world * 1000 + seed
            for name in names:
                cases.append((
                    world, seed, args.radius, args.max_steps, name, dyn_seed,
                    args.n_moving, args.moving_speed, args.moving_amplitude,
                    args.samples, args.horizon, args.prediction_horizon,
                    args.uncertainty_rate))

    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['config']:9s} world={row['world']:3d} "
                  f"seed={row['seed']} {row['result']:9s} "
                  f"clear={row['min_clear_m']:.3f} "
                  f"hz={row['controller_hz']:.1f}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_case, case) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{row['config']:9s} world={row['world']:3d} "
                      f"seed={row['seed']} {row['result']:9s} "
                      f"clear={row['min_clear_m']:.3f} "
                      f"hz={row['controller_hz']:.1f}", flush=True)

    order = {name: index for index, name in enumerate(names)}
    rows.sort(key=lambda row: (row["world"], row["seed"],
                               order[row["config"]]))
    output = _write_csv(args.out, rows)
    summaries = summarize(rows)
    summary_output = _write_csv(
        os.path.splitext(args.out)[0] + "_summary.csv", summaries)
    print(f"\nwrote {output}\nwrote {summary_output}")
    print("\nconfig     success collision timeout mean_clr worst_clr ctrl_hz ESS")
    for row in summaries:
        print(f"{row['config']:9s} {row['successes']:2d}/{row['runs']:<2d} "
              f"{row['collisions']:9d} {row['timeouts']:7d} "
              f"{row['mean_clearance_m']:8.3f} "
              f"{row['worst_clearance_m']:9.3f} "
              f"{row['mean_controller_hz']:7.1f} {row['mean_ess']:5.1f}")
    delta = paired_deltas(rows)
    if delta:
        print("\npaired predicted-current: "
              f"success={delta['success_delta']:+d}, "
              f"collision={delta['collision_delta']:+d}, "
              f"clearance={delta['mean_clearance_delta_m']:+.3f} m, "
              f"path={delta['mean_path_delta_m']:+.3f} m")


if __name__ == "__main__":
    main()
