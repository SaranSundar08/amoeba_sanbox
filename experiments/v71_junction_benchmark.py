#!/usr/bin/env python3
"""V7.1 Stage-1 benchmark against a matched scripted peer robot."""
import argparse
import concurrent.futures
import csv
import os
import time

import numpy as np

from amoeba_controllers import AmoebaHybrid
from astar import astar_path
from mppi import MPPI
from path_test import PathTrackMPPI
from junction_env import SCENARIOS, ScriptedJunctionEnv


CONFIGS = ("vanilla", "path", "flow_only", "amoeba")


def _make_controller(name, env, path, seed, samples, horizon,
                     robot_model="slip"):
    common = dict(
        seed=seed, K=samples, T=horizon, dynamic_prediction=True,
        prediction_horizon=2.0, prediction_uncertainty_rate=0.03,
        robot_model=robot_model)
    if name == "vanilla":
        return MPPI(env, **common)
    if name == "path":
        return PathTrackMPPI(env, path=path, **common)
    extract_branches = name == "amoeba"
    return AmoebaHybrid(
        env, path=path, extract_branches=extract_branches, max_branches=3,
        grouped_sampling=True, importance_weighting=False,
        importance_ess_guard=False, adaptive_covariance=False,
        fallback_share=0.25, mode_min_dwell=10, mode_confirm_cycles=3,
        mode_switch_margin=0.3, gate="auto", path_validity_gate=True,
        **common)


def _run_case(case):
    scenario, seed, name, samples, horizon, max_steps, robot_model = case
    env = ScriptedJunctionEnv(scenario, robot_r=0.405)
    # The plan is intentionally computed before the peer enters the costmap.
    env.dynamic_enabled = False
    path = astar_path(env, env.start[:2], env.goal)
    env.dynamic_enabled = True
    ctrl = _make_controller(
        name, env, path, seed, samples, horizon, robot_model)
    state = env.start.copy()
    trace = [state.copy()]
    minimum_peer_clearance = np.inf
    minimum_static_clearance = np.inf
    stall_time = 0.0
    peer_collision = False
    static_collision = False
    assist_cycles = 0
    branch_cycles = 0
    mode_switches = 0
    result = "timeout"
    controller_times_ms = []
    started = time.perf_counter()
    for _step in range(max_steps):
        env.step_time(ctrl.dt)
        controller_start = time.perf_counter()
        control, info = ctrl.step(state)
        controller_times_ms.append(
            1.0e3 * (time.perf_counter() - controller_start))
        state = ctrl.robot_model.integrate(state, control, ctrl.dt)
        trace.append(state.copy())
        points = ctrl.robot_model.footprint_points(state)
        if ctrl.robot_model.is_slip:
            # Static walls are rectangular, not circular: still the sampled
            # 17-point check (see docs/experiments/PLANT_MISMATCH_PILOT.md's
            # sibling caveat -- an exact oriented-rectangle-vs-rectangle check
            # is deferred, not yet implemented).
            static_clearance = float(np.min(env.static_clearance(points)))
            # The peer is circular, so this gets the same exact analytic
            # check as BarnEnv's obstacles (RobotModel.circle_set_clearance,
            # added 2026-09-04): nothing can hide between footprint samples.
            peer_clearance = float(ctrl.robot_model.circle_set_clearance(
                state, env.moving_obs(), env.peer_r))
        else:
            static_clearance = float(
                env.static_clearance(state[None, :2])[0]
                - ctrl.robot_model.radius)
            peer_clearance = env.inter_robot_clearance(state[:2])
        minimum_peer_clearance = min(minimum_peer_clearance, peer_clearance)
        minimum_static_clearance = min(minimum_static_clearance,
                                       static_clearance)
        if abs(control[0]) < 0.05:
            stall_time += ctrl.dt
        if info.get("assist_state") == "ASSIST":
            assist_cycles += 1
        selected = info.get("selected_mode")
        if selected is not None and selected["key"] != -1:
            branch_cycles += 1
        mode_switches = int(getattr(ctrl, "mode_switch_count", 0))
        if peer_clearance < 0.0:
            peer_collision, result = True, "peer_collision"
            break
        if static_clearance < 0.0:
            static_collision, result = True, "static_collision"
            break
        if np.linalg.norm(state[:2] - env.goal) < env.goal_tol:
            result = "success"
            break
    wall_time = time.perf_counter() - started
    trace = np.asarray(trace)
    steps = len(trace) - 1
    path_length = float(np.linalg.norm(np.diff(trace[:, :2], axis=0), axis=1).sum())
    return {
        "scenario": scenario,
        "config": name,
        "seed": seed,
        "result": result,
        "success": int(result == "success"),
        "peer_collision": int(peer_collision),
        "static_collision": int(static_collision),
        "time_s": round(steps * ctrl.dt, 3),
        "path_length_m": round(path_length, 4),
        "minimum_peer_clearance_m": round(minimum_peer_clearance, 4),
        "minimum_static_clearance_m": round(minimum_static_clearance, 4),
        "stall_time_s": round(stall_time, 3),
        "final_goal_distance_m": round(float(np.linalg.norm(
            state[:2] - env.goal)), 4),
        "assist_fraction": assist_cycles / max(steps, 1),
        "branch_fraction": branch_cycles / max(steps, 1),
        "mode_switches": mode_switches,
        "repair_count": int(getattr(ctrl, "repair_count", 0)),
        "repair_failures": int(getattr(ctrl, "repair_failures", 0)),
        "wall_time_s": round(wall_time, 3),
        "controller_hz": round(steps / max(wall_time, 1e-9), 3),
        "controller_mean_ms": round(float(np.mean(controller_times_ms)), 3),
        "controller_p95_ms": round(float(np.percentile(
            controller_times_ms, 95)), 3),
        "controller_max_ms": round(float(np.max(controller_times_ms)), 3),
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
                "peer_collisions": sum(row["peer_collision"] for row in group),
                "static_collisions": sum(row["static_collision"] for row in group),
                "timeouts": sum(row["result"] == "timeout" for row in group),
                "mean_success_time_s": (float(np.mean(
                    [row["time_s"] for row in successes]))
                    if successes else np.nan),
                "mean_peer_clearance_m": float(np.mean(
                    [row["minimum_peer_clearance_m"] for row in group])),
                "worst_peer_clearance_m": float(np.min(
                    [row["minimum_peer_clearance_m"] for row in group])),
                "mean_stall_time_s": float(np.mean(
                    [row["stall_time_s"] for row in group])),
                "mean_branch_fraction": float(np.mean(
                    [row["branch_fraction"] for row in group])),
                "mean_controller_hz": float(np.mean(
                    [row["controller_hz"] for row in group])),
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
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first MPPI seed; enables non-overlapping extensions")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--robot-model", choices=("ideal", "slip"),
                        default="slip")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--out", default="artifacts/results/milestones/v71_junction.csv")
    args = parser.parse_args()
    scenarios = [x for x in args.scenarios.split(",") if x]
    configs = [x for x in args.configs.split(",") if x]
    if set(scenarios) - set(SCENARIOS):
        parser.error("unknown scenario")
    if set(configs) - set(CONFIGS):
        parser.error("unknown configuration")
    cases = [(scenario, seed, name, args.samples, args.horizon, args.max_steps,
              args.robot_model)
             for scenario in scenarios
             for seed in range(args.seed_start, args.seed_start + args.seeds)
             for name in configs]
    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['scenario']:8s} {row['config']:7s} seed={row['seed']} "
                  f"{row['result']:16s} peer={row['minimum_peer_clearance_m']:+.3f} "
                  f"stall={row['stall_time_s']:.1f}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(_run_case, case) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{row['scenario']:8s} {row['config']:7s} seed={row['seed']} "
                      f"{row['result']:16s} peer={row['minimum_peer_clearance_m']:+.3f} "
                      f"stall={row['stall_time_s']:.1f}", flush=True)
    so, co = ({x: i for i, x in enumerate(scenarios)},
              {x: i for i, x in enumerate(configs)})
    rows.sort(key=lambda r: (so[r["scenario"]], r["seed"], co[r["config"]]))
    output = _write(args.out, rows)
    summary = summarize(rows)
    summary_output = _write(os.path.splitext(args.out)[0] + "_summary.csv", summary)
    print(f"\nwrote {output}\nwrote {summary_output}")
    print("\nscenario config  success peer_coll static_coll timeout peer_clr stall")
    for row in summary:
        print(f"{row['scenario']:8s} {row['config']:7s} "
              f"{row['successes']:2d}/{row['runs']:<2d} "
              f"{row['peer_collisions']:9d} {row['static_collisions']:11d} "
              f"{row['timeouts']:7d} {row['mean_peer_clearance_m']:+8.3f} "
              f"{row['mean_stall_time_s']:5.1f}")


if __name__ == "__main__":
    main()
