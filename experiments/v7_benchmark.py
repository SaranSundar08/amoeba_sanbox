#!/usr/bin/env python3
"""Matched V7 benchmark: vanilla, path-only, and final Amoeba MPPI."""
import argparse
import concurrent.futures
import csv
import os
import time

import numpy as np

from astar import astar_path
from controllers import AmoebaHybrid, MPPI
from dynabarn import DynaBarnEnv
from env import BarnEnv
from path_test import PathTrackMPPI
from simulation import episode
from experiments.v31_ablation import _path_error


CONFIGS = ("vanilla", "path_mppi", "path_biased", "amoeba", "amoeba_nominal_fb")
SCENARIOS = ("static", "dynamic")


def _environment(world, seed, scenario, radius, dynamic_seed, n_moving,
                 moving_speed, moving_amplitude):
    if scenario == "static":
        return BarnEnv(world, robot_r=radius)
    return DynaBarnEnv(
        world, robot_r=radius, n_moving=n_moving, speed=moving_speed,
        amplitude=moving_amplitude,
        seed=dynamic_seed + world * 1000 + seed)


def _controller(name, env, path, seed, samples, horizon, robot_model,
                dynamic_prediction, prediction_horizon, uncertainty_rate):
    common = dict(
        seed=seed, K=samples, T=horizon, robot_model=robot_model,
        dynamic_prediction=dynamic_prediction,
        prediction_horizon=prediction_horizon,
        prediction_uncertainty_rate=uncertainty_rate)
    if name == "vanilla":
        return MPPI(env, **common)
    if name == "path_mppi":
        return PathTrackMPPI(env, path=path, **common)
    if name == "path_biased":
        # Single-ancillary Biased MPPI reference: convert the A* path into
        # one feasible control mean and sample the full batch around it.
        # Flow guidance and pseudopods are disabled for clean attribution.
        return AmoebaHybrid(
            env, path=path, extract_branches=False, grouped_sampling=True,
            fallback_share=1.0, importance_weighting=False,
            adaptive_covariance=False, flow_w=0.0, bias_strength=0.0,
            gate="auto", path_validity_gate=True, **common)
    reference_kw = {}
    if name == "amoeba_nominal_fb":
        # Accepted candidate from docs/experiments/REFERENCE_DISTINCTNESS.md:
        # nominal-speed branch reference, falling back to the shaped one per
        # branch when infeasible. Opt-in; 'amoeba' keeps the shaped default.
        reference_kw = dict(
            reference_min_speed_ratio=1.0, reference_curvature_slowdown=0.0,
            reference_infeasible_fallback=True)
    return AmoebaHybrid(
        env, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, importance_weighting=False,
        importance_ess_guard=False, adaptive_covariance=False,
        fallback_share=0.25, mode_min_dwell=10, mode_confirm_cycles=3,
        mode_switch_margin=0.3, gate="auto", path_validity_gate=True,
        **reference_kw, **common)


def _run_case(case):
    (scenario, world, seed, name, radius, max_steps, samples, horizon,
     robot_model, dynamic_seed, n_moving, moving_speed, moving_amplitude,
     prediction_horizon, uncertainty_rate) = case
    env = _environment(
        world, seed, scenario, radius, dynamic_seed, n_moving,
        moving_speed, moving_amplitude)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = _controller(
        name, env, path, seed, samples, horizon, robot_model,
        scenario == "dynamic", prediction_horizon, uncertainty_rate)
    selected, ess, active_branches, assist = [], [], [], []

    def observe(_step, _state, info, _trace):
        if "selected_mode" in info:
            selected.append(info["selected_mode"]["key"])
            ess.extend(item["ess"] for item in info["mode_stats"])
        proposals = info.get("branch_proposals", ())
        active_branches.append(sum(proposal.feasible for proposal in proposals))
        assist.append(info.get("assist_state") == "ASSIST")

    started = time.perf_counter()
    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe)
    wall_time = time.perf_counter() - started
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    initial_distance = float(np.linalg.norm(env.start[:2] - env.goal))
    final_distance = float(np.linalg.norm(trace[-1, :2] - env.goal))
    steps = len(trace) - 1
    return {
        "scenario": scenario,
        "config": name,
        "world": world,
        "seed": seed,
        "samples": samples,
        "horizon": horizon,
        "robot_model": robot_model,
        **metrics,
        "goal_progress_m": round(initial_distance - final_distance, 4),
        "final_goal_distance_m": round(final_distance, 4),
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
        "wall_time_s": round(wall_time, 3),
        "controller_hz": round(steps / max(wall_time, 1e-9), 3),
        "mode_switches": int(getattr(ctrl, "mode_switch_count", 0)),
        "omega_sign_flips": int(getattr(ctrl, "omega_sign_flips", 0)),
        "branch_selection_fraction": (float(np.mean(
            [key != -1 for key in selected])) if selected else 0.0),
        "fallback_selection_fraction": (float(np.mean(
            [key == -1 for key in selected])) if selected else 1.0),
        "ancillary_available_fraction": (float(np.mean(
            [count > 0 for count in active_branches]))
            if active_branches else 0.0),
        "mean_active_ancillary_modes": (float(np.mean(active_branches))
                                        if active_branches else 0.0),
        "assist_fraction": float(np.mean(assist)) if assist else 0.0,
        "mean_ess": float(np.mean(ess)) if ess else np.nan,
        "min_ess": float(np.min(ess)) if ess else np.nan,
    }


def summarize(rows):
    output = []
    for scenario in SCENARIOS:
        for name in CONFIGS:
            group = [row for row in rows if row["scenario"] == scenario
                     and row["config"] == name]
            if not group:
                continue
            successes = [row for row in group if row["success"]]
            output.append({
                "scenario": scenario,
                "config": name,
                "runs": len(group),
                "successes": len(successes),
                "success_rate": len(successes) / len(group),
                "collisions": sum(row["result"] == "collision"
                                  for row in group),
                "timeouts": sum(row["result"] == "timeout" for row in group),
                "mean_success_time_s": (float(np.mean(
                    [row["time_s"] for row in successes]))
                    if successes else np.nan),
                "mean_success_path_m": (float(np.mean(
                    [row["path_len_m"] for row in successes]))
                    if successes else np.nan),
                "mean_clearance_m": float(np.mean(
                    [row["min_clear_m"] for row in group])),
                "worst_clearance_m": float(np.min(
                    [row["min_clear_m"] for row in group])),
                "mean_goal_progress_m": float(np.mean(
                    [row["goal_progress_m"] for row in group])),
                "mean_stall_s": float(np.mean(
                    [row["stall_s"] for row in group])),
                "mean_controller_hz": float(np.mean(
                    [row["controller_hz"] for row in group])),
                "mean_controller_ms": float(np.mean(
                    [row["controller_mean_ms"] for row in group])),
                "mean_controller_p95_ms": float(np.mean(
                    [row["controller_p95_ms"] for row in group])),
                "worst_controller_max_ms": float(np.max(
                    [row["controller_max_ms"] for row in group])),
                "mean_path_error_m": float(np.mean(
                    [row["mean_path_error_m"] for row in group])),
            })
    return output


def _write(path, rows):
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return absolute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="0,7,48,1,5,20,40,42,93,94")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--robot-model", choices=("ideal", "slip"), default="slip")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dynamic-seed", type=int, default=71000)
    parser.add_argument("--n-moving", type=int, default=4)
    parser.add_argument("--moving-speed", type=float, default=0.4)
    parser.add_argument("--moving-amplitude", type=float, default=0.6)
    parser.add_argument("--prediction-horizon", type=float, default=2.0)
    parser.add_argument("--uncertainty-rate", type=float, default=0.03)
    parser.add_argument(
        "--out", default="artifacts/results/milestones/v7_benchmark.csv")
    args = parser.parse_args()
    worlds = [int(value) for value in args.worlds.split(",") if value]
    scenarios = [value for value in args.scenarios.split(",") if value]
    configs = [value for value in args.configs.split(",") if value]
    if set(scenarios) - set(SCENARIOS):
        parser.error("unknown scenario")
    if set(configs) - set(CONFIGS):
        parser.error("unknown configuration")
    cases = [
        (scenario, world, seed, name, args.radius, args.max_steps,
         args.samples, args.horizon, args.robot_model, args.dynamic_seed,
         args.n_moving, args.moving_speed, args.moving_amplitude,
         args.prediction_horizon, args.uncertainty_rate)
        for scenario in scenarios for world in worlds
        for seed in range(args.seeds) for name in configs]
    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['scenario']:7s} {row['config']:9s} "
                  f"world={row['world']:3d} seed={row['seed']} "
                  f"{row['result']:9s} clear={row['min_clear_m']:.3f} "
                  f"hz={row['controller_hz']:.1f}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_case, case) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{row['scenario']:7s} {row['config']:9s} "
                      f"world={row['world']:3d} seed={row['seed']} "
                      f"{row['result']:9s} clear={row['min_clear_m']:.3f} "
                      f"hz={row['controller_hz']:.1f}", flush=True)
    scenario_order = {name: i for i, name in enumerate(scenarios)}
    config_order = {name: i for i, name in enumerate(configs)}
    rows.sort(key=lambda row: (scenario_order[row["scenario"]], row["world"],
                               row["seed"], config_order[row["config"]]))
    output = _write(args.out, rows)
    summaries = summarize(rows)
    summary_output = _write(os.path.splitext(args.out)[0] + "_summary.csv",
                            summaries)
    print(f"\nwrote {output}\nwrote {summary_output}")
    print("\nscenario config     success coll tout time  clearance progress hz")
    for row in summaries:
        print(f"{row['scenario']:8s} {row['config']:9s} "
              f"{row['successes']:2d}/{row['runs']:<2d} "
              f"{row['collisions']:4d} {row['timeouts']:4d} "
              f"{row['mean_success_time_s']:5.1f} "
              f"{row['mean_clearance_m']:9.3f} "
              f"{row['mean_goal_progress_m']:8.3f} "
              f"{row['mean_controller_hz']:5.1f}")


if __name__ == "__main__":
    main()
