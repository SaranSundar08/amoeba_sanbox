#!/usr/bin/env python3
"""Homotopy-consistency (mode-purity) cost ablation.

Matched with/without comparison of the T-MPC-style side-of-obstacle penalty
(`homotopy.py`) on the blocked-path scenario, where the hybrid controller is
in ASSIST with genuinely distinct pseudopod modes. The `off` configuration
runs the unchanged controller with `homotopy_monitor=True`, so mode purity is
measured identically in both arms without altering the baseline.

Examples:
  python3 -m experiments.homotopy_ablation --jobs 6
  python3 -m experiments.homotopy_ablation --worlds 48 --seeds 1 --scenarios blocked
"""
import argparse
import concurrent.futures
import csv
import os

import numpy as np

from controllers import AmoebaHybrid
from simulation import episode
from experiments.v31_ablation import _path_error
from experiments.v5_ablation import SCENARIOS, _make_env, _make_path


def _configs(weight):
    return {
        "off": {"homotopy_w": 0.0, "homotopy_monitor": True},
        "on": {"homotopy_w": weight},
    }


def _run_case(case):
    (world, seed, radius, max_steps, scenario, lane, name, weight,
     dynamic_seed, n_moving, moving_speed, moving_amplitude) = case
    env = _make_env(
        world, seed, radius, scenario, dynamic_seed, n_moving,
        moving_speed, moving_amplitude)
    path = _make_path(env, scenario, lane)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, fallback_share=0.25,
        mode_min_dwell=10, mode_confirm_cycles=3,
        mode_switch_margin=0.3, gate="auto", path_validity_gate=True,
        **_configs(weight)[name])

    previous_sign = 0
    sign_flips = 0
    selected_keys = []
    ess_values = []
    assist_cycles = 0
    guided_fractions = []      # per (cycle, guided mode) violating fraction
    guided_mean_violation = []
    near_fractions = []        # same, only where the mode had >= 1 plane
    wrong_side_mass = []       # per ASSIST cycle, executed update's mass
    plane_counts = []
    # T-MPC Fig. 6 analogue on the optimized means (see
    # MPPI._homotopy_mode_distinctness): guided modes whose update drifted
    # across their own reference's side, and guided-mode pairs that no
    # longer pass any obstacle on different sides.
    guided_total = own_drift_total = 0
    pairs_total = collapsed_total = 0

    def observe(_k, _state, info, _trace):
        nonlocal previous_sign, sign_flips, assist_cycles
        nonlocal guided_total, own_drift_total, pairs_total, collapsed_total
        omega = float(ctrl.last_applied[1])
        sign = int(np.sign(omega)) if abs(omega) >= 0.05 else 0
        if sign and previous_sign and sign != previous_sign:
            sign_flips += 1
        if sign:
            previous_sign = sign
        selected_keys.append(info["selected_mode"]["key"])
        ess_values.extend(item["ess"] for item in info["mode_stats"])
        if info["assist_state"] == "ASSIST":
            assist_cycles += 1
            for item in info["homotopy_stats"]:
                guided_fractions.append(item["violating_fraction"])
                guided_mean_violation.append(item["mean_violation"])
                plane_counts.append(item["n_planes"])
                if item["n_planes"] > 0:
                    near_fractions.append(item["violating_fraction"])
            if np.isfinite(info["homotopy_selected_wrong_side_mass"]):
                wrong_side_mass.append(
                    info["homotopy_selected_wrong_side_mass"])
            distinct = info.get("homotopy_modes")
            if distinct is not None:
                guided_total += distinct["guided"]
                own_drift_total += distinct["own_drift"]
                pairs_total += distinct["pairs"]
                collapsed_total += distinct["collapsed_pairs"]

    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe)
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    cycles = max(len(selected_keys), 1)
    return {
        "scenario": scenario,
        "config": name,
        "homotopy_w": weight if name == "on" else 0.0,
        "world": world,
        "seed": seed,
        **metrics,
        "omega_sign_flips": sign_flips,
        "mode_switches": int(ctrl.mode_switch_count),
        "assist_fraction": assist_cycles / cycles,
        "branch_selection_fraction": float(np.mean(
            [key != -1 for key in selected_keys])),
        "mean_ess": float(np.mean(ess_values)),
        "min_ess": float(np.min(ess_values)),
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
        "guided_violating_fraction": (
            float(np.mean(guided_fractions)) if guided_fractions else np.nan),
        "guided_mean_violation": (
            float(np.mean(guided_mean_violation))
            if guided_mean_violation else np.nan),
        "near_obstacle_violating_fraction": (
            float(np.mean(near_fractions)) if near_fractions else np.nan),
        "near_obstacle_cycle_fraction": (
            float(np.mean([n > 0 for n in plane_counts]))
            if plane_counts else np.nan),
        "selected_wrong_side_mass": (
            float(np.mean(wrong_side_mass)) if wrong_side_mass else np.nan),
        "mean_planes": (
            float(np.mean(plane_counts)) if plane_counts else 0.0),
        "own_drift_fraction": (
            own_drift_total / guided_total if guided_total else np.nan),
        "collapsed_pair_fraction": (
            collapsed_total / pairs_total if pairs_total else np.nan),
        "guided_mode_cycles": guided_total,
        "guided_pairs": pairs_total,
    }


def _summarize(rows, names):
    output = []
    for scenario in SCENARIOS:
        for name in names:
            group = [row for row in rows
                     if row["scenario"] == scenario
                     and row["config"] == name]
            if not group:
                continue
            successes = [row for row in group if row["success"]]

            def mean(field, source=group):
                values = [row[field] for row in source
                          if np.isfinite(row[field])]
                return float(np.mean(values)) if values else np.nan

            output.append({
                "scenario": scenario,
                "config": name,
                "runs": len(group),
                "successes": len(successes),
                "collisions": sum(row["result"] == "collision"
                                  for row in group),
                "timeouts": sum(row["result"] == "timeout"
                                for row in group),
                "mean_time_s": mean("time_s", successes),
                "worst_clear_m": float(np.min(
                    [row["min_clear_m"] for row in group])),
                "mean_stall_s": mean("stall_s"),
                "mean_switches": mean("mode_switches"),
                "mean_flips": mean("omega_sign_flips"),
                "assist_fraction": mean("assist_fraction"),
                "branch_fraction": mean("branch_selection_fraction"),
                "violating_fraction": mean("guided_violating_fraction"),
                "near_violating_fraction": mean(
                    "near_obstacle_violating_fraction"),
                "near_cycle_fraction": mean("near_obstacle_cycle_fraction"),
                "wrong_side_mass": mean("selected_wrong_side_mass"),
                "own_drift": mean("own_drift_fraction"),
                "collapsed_pairs": mean("collapsed_pair_fraction"),
                "mean_ess": mean("mean_ess"),
            })
    return output


def _print_summary(summaries):
    print("\nscenario        cfg success coll tout time  worst_clr stall "
          "switch flips assist branch near_viol wrong_mass own_drift "
          "collapsed ESS")
    for row in summaries:
        print(
            f"{row['scenario']:15s} {row['config']:3s} "
            f"{row['successes']:2d}/{row['runs']:<2d} "
            f"{row['collisions']:4d} {row['timeouts']:4d} "
            f"{row['mean_time_s']:5.1f} {row['worst_clear_m']:9.3f} "
            f"{row['mean_stall_s']:5.1f} {row['mean_switches']:6.1f} "
            f"{row['mean_flips']:5.1f} {row['assist_fraction']:6.2f} "
            f"{row['branch_fraction']:6.2f} "
            f"{row['near_violating_fraction']:9.3f} "
            f"{row['wrong_side_mass']:10.3f} {row['own_drift']:9.3f} "
            f"{row['collapsed_pairs']:9.3f} {row['mean_ess']:5.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="4,48,49")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--scenarios", default="blocked")
    parser.add_argument("--configs", default="off,on")
    parser.add_argument("--homotopy-w", type=float, default=20.0)
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
        "--out", default="artifacts/results/milestones/homotopy_ablation.csv")
    args = parser.parse_args()

    worlds = [int(value) for value in args.worlds.split(",")]
    scenarios = args.scenarios.split(",")
    names = args.configs.split(",")
    if set(scenarios) - set(SCENARIOS):
        parser.error("unknown scenario")
    if set(names) - set(_configs(args.homotopy_w)):
        parser.error("unknown configuration")
    cases = [
        (world, seed, args.radius, args.max_steps, scenario, args.lane,
         name, args.homotopy_w, args.dynamic_seed, args.n_moving,
         args.moving_speed, args.moving_amplitude)
        for scenario in scenarios for world in worlds
        for seed in range(args.seeds) for name in names]

    def report(row):
        print(f"{row['scenario']:15s} {row['config']:3s} "
              f"world={row['world']:2d} seed={row['seed']} "
              f"{row['result']:9s} clear={row['min_clear_m']:.3f} "
              f"wrong_mass={row['selected_wrong_side_mass']:.3f} "
              f"own_drift={row['own_drift_fraction']:.3f} "
              f"collapsed={row['collapsed_pair_fraction']:.3f}",
              flush=True)

    rows = []
    if args.jobs == 1:
        for row in map(_run_case, cases):
            rows.append(row)
            report(row)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_case, case) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                report(row)

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
    _print_summary(_summarize(rows, names))


if __name__ == "__main__":
    main()
