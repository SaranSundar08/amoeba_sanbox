#!/usr/bin/env python3
"""Matched ideal-versus-analytical-SLIP V6 development ablation."""
import argparse
import concurrent.futures
import csv
import os

import numpy as np

from controllers import AmoebaHybrid
from robot_model import sample_plant_mismatch
from simulation import episode
from experiments.v31_ablation import _path_error
from experiments.v5_ablation import SCENARIOS, _make_env, _make_path


CONFIGS = {
    "ideal": {"robot_model": "ideal"},
    "slip": {
        "robot_model": "slip",
        # 0.84 x 0.68 m = the Nav2 costmap footprint (2026-09-09). The banked
        # v6_ablation*.csv / v6_ablation_plant_mismatch*.csv were produced at
        # the earlier provisional 0.90 x 0.65 m; set these back to reproduce
        # them exactly.
        "footprint_length": 0.84,
        "footprint_width": 0.68,
        "skid_yaw_gain": 0.82,
        "skid_speed_yaw_loss": 0.55,
        "turn_safety_margin": 0.08,
        "speed_safety_margin": 0.03,
    },
}


def _run_case(case):
    (world, seed, radius, max_steps, scenario, lane, name, dynamic_seed,
     n_moving, moving_speed, moving_amplitude, plant_mismatch, draw) = case
    env = _make_env(
        world, seed, radius, scenario, dynamic_seed, n_moving,
        moving_speed, moving_amplitude)
    if name == "slip":
        env.robot_r = max(
            env.robot_r,
            0.5 * CONFIGS[name]["footprint_width"]
            + CONFIGS[name]["turn_safety_margin"])
    path = _make_path(env, scenario, lane)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, adaptive_covariance=False,
        fallback_share=0.25, mode_min_dwell=10, mode_confirm_cycles=3,
        mode_switch_margin=0.3, gate="auto", path_validity_gate=True,
        **CONFIGS[name])

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
        selected_keys.append(info["selected_mode"]["key"])
        ess_values.extend(item["ess"] for item in info["mode_stats"])
        branch_ids.update(branch.branch_id
                          for branch in getattr(ctrl, "branches", []))

    # Plan/plant separation (see simulation.episode): the controller keeps
    # its nominal model; the plant executes with independently jittered
    # yaw-gain and speed-yaw-loss. Only 'slip' has slip parameters to jitter,
    # so 'ideal' is unchanged, as is everything when the factor is 0.
    # `draw` selects one of N independent mismatch draws per (world, seed):
    # a 4-part seed sequence keeps every draw statistically independent (not
    # just a different arithmetic offset into the same stream). This reseeds
    # relative to the original n=1 pilot, so draw=0 here is a different
    # (still valid, still reproducible) sample than that pilot's CSV, not a
    # bit-identical replay of it.
    plant_model = None
    plant_yaw_gain = plant_speed_yaw_loss = np.nan
    if plant_mismatch > 0.0 and ctrl.robot_model.is_slip:
        plant_model = sample_plant_mismatch(
            ctrl.robot_model,
            np.random.default_rng([world, seed, draw, 7]),
            yaw_gain_range=(1.0 - plant_mismatch, 1.0 + plant_mismatch),
            speed_yaw_loss_range=(1.0 - plant_mismatch, 1.0 + plant_mismatch))
        plant_yaw_gain = plant_model.yaw_gain
        plant_speed_yaw_loss = plant_model.speed_yaw_loss

    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe,
                      plant_model=plant_model)
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    return {
        "scenario": scenario,
        "config": name,
        "world": world,
        "seed": seed,
        "plant_mismatch": plant_mismatch,
        "draw": draw,
        "plant_yaw_gain": plant_yaw_gain,
        "plant_speed_yaw_loss": plant_speed_yaw_loss,
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
    print("\nscenario        model success coll tout time  path  worst_clr "
          "stall flips switches ESS path_err")
    for row in summaries:
        print(
            f"{row['scenario']:15s} {row['config']:5s} "
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
        "--plant-mismatch", type=float, default=0.0,
        help="execute with a plant whose slip yaw-gain and speed-yaw-loss "
             "are scaled by uniform factors in [1-x, 1+x], unknown to the "
             "controller (0 = self-consistent plant, the original ablation)")
    parser.add_argument(
        "--plant-mismatch-draws", type=int, default=1,
        help="independent mismatch draws per (world, seed), reused across "
             "all scenarios for that pair, same as the single-draw case "
             "(default 1). Only takes effect when --plant-mismatch > 0; "
             "ignored (fixed at one no-op draw) otherwise, since there is "
             "nothing random to redraw with no mismatch.")
    parser.add_argument(
        "--out", default="artifacts/results/milestones/v6_ablation.csv")
    args = parser.parse_args()

    worlds = [int(value) for value in args.worlds.split(",")]
    scenarios = args.scenarios.split(",")
    names = args.configs.split(",")
    if set(scenarios) - set(SCENARIOS):
        parser.error("unknown scenario")
    if set(names) - set(CONFIGS):
        parser.error("unknown configuration")
    draws = (range(args.plant_mismatch_draws) if args.plant_mismatch > 0.0
             else range(1))
    cases = [
        (world, seed, args.radius, args.max_steps, scenario, args.lane,
         name, args.dynamic_seed, args.n_moving, args.moving_speed,
         args.moving_amplitude, args.plant_mismatch, draw)
        for scenario in scenarios for world in worlds
        for seed in range(args.seeds) for name in names for draw in draws]

    rows = []
    if args.jobs == 1:
        iterator = map(_run_case, cases)
        for row in iterator:
            rows.append(row)
            print(f"{row['scenario']:15s} {row['config']:5s} "
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
                print(f"{row['scenario']:15s} {row['config']:5s} "
                      f"world={row['world']:2d} seed={row['seed']} "
                      f"{row['result']:9s} clear={row['min_clear_m']:.3f}",
                      flush=True)

    scenario_order = {name: index for index, name in enumerate(scenarios)}
    config_order = {name: index for index, name in enumerate(names)}
    rows.sort(key=lambda row: (
        scenario_order[row["scenario"]], row["world"], row["seed"],
        config_order[row["config"]], row["draw"]))
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
