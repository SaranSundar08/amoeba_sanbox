#!/usr/bin/env python3
"""Horizon-scale pseudopod reference distinctness.

`docs/experiments/HOMOTOPY_ABLATION.md` found that feasible pseudopod
references pass a nearby obstacle on different sides in 0 of 742 pairs: the
shaped V2 reference crawls near obstacles, so a 56-step rollout covers the
branches' shared trunk and never reaches their split. This experiment varies
only the branch REFERENCE speed policy (`reference_*` kwargs of `MPPI`) and
measures, per ASSIST cycle, reference pair separation (the acceptance
metric), reference length, branch feasibility, and the resulting mode
selection and outcome. The path fallback proposal is unchanged in every arm.

Examples:
  python3 -m experiments.reference_distinctness --jobs 6
  python3 -m experiments.reference_distinctness --worlds 48 --seeds 1 --configs shaped,nominal
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


CONFIGS = {
    # proposals.py defaults: clearance and curvature slowdown, floor 12%.
    "shaped": {"reference_min_speed_ratio": 0.12,
               "reference_curvature_slowdown": 1.2},
    # Half-nominal floor, halved curvature slowdown.
    "floor50": {"reference_min_speed_ratio": 0.5,
                "reference_curvature_slowdown": 0.6},
    # Nominal speed: no clearance or curvature slowdown (heading factor kept).
    "nominal": {"reference_min_speed_ratio": 1.0,
                "reference_curvature_slowdown": 0.0},
    # Slow down only within 0.2 m of an obstacle instead of 0.45 m; in BARN
    # usable clearance is almost always below 0.45 m, so the default band
    # makes the reference crawl everywhere.
    "band20": {"reference_min_speed_ratio": 0.4,
               "reference_curvature_slowdown": 0.6,
               "reference_clearance_slow_band": 0.20},
    # Nominal, but an infeasible fast reference falls back to the shaped one
    # for that branch: feasible set is a superset of `shaped`'s.
    "nominal_fb": {"reference_min_speed_ratio": 1.0,
                   "reference_curvature_slowdown": 0.0,
                   "reference_infeasible_fallback": True},
}


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
        homotopy_monitor=True, **CONFIGS[name])

    previous_sign = 0
    sign_flips = 0
    selected_keys = []
    ess_values = []
    assist_cycles = 0
    guided_total = 0
    reference_length_total = 0.0
    reference_pairs = reference_separated = 0
    update_pairs = update_collapsed = 0
    own_drift_total = 0
    branches_seen = feasible_seen = 0
    coverage = []
    free_energy_gaps = []

    def observe(_k, _state, info, _trace):
        nonlocal previous_sign, sign_flips, assist_cycles, guided_total
        nonlocal reference_length_total, reference_pairs
        nonlocal reference_separated, update_pairs, update_collapsed
        nonlocal own_drift_total, branches_seen, feasible_seen
        omega = float(ctrl.last_applied[1])
        sign = int(np.sign(omega)) if abs(omega) >= 0.05 else 0
        if sign and previous_sign and sign != previous_sign:
            sign_flips += 1
        if sign:
            previous_sign = sign
        selected_keys.append(info["selected_mode"]["key"])
        ess_values.extend(item["ess"] for item in info["mode_stats"])
        if info["assist_state"] != "ASSIST":
            return
        assist_cycles += 1
        distinct = info.get("homotopy_modes")
        if distinct is not None:
            guided_total += distinct["guided"]
            reference_length_total += distinct["reference_length_sum"]
            reference_pairs += distinct["reference_pairs"]
            reference_separated += distinct["reference_separated"]
            update_pairs += distinct["pairs"]
            update_collapsed += distinct["collapsed_pairs"]
            own_drift_total += distinct["own_drift"]
        for branch, proposal in zip(ctrl.branches, ctrl.branch_proposals):
            branches_seen += 1
            if not proposal.feasible:
                continue
            feasible_seen += 1
            length = float(np.linalg.norm(
                np.diff(np.asarray(branch.centerline, float), axis=0),
                axis=1).sum())
            if length > 0.0:
                coverage.append(proposal.progress / length)
        if len(info["mode_stats"]) >= 2:
            free_energy_gaps.append(info["free_energy_gap"])

    metrics = episode(env, ctrl, max_steps=max_steps, on_step=observe)
    trace = metrics.pop("trace")
    mean_path_error, max_path_error = _path_error(trace, path)
    cycles = max(len(selected_keys), 1)
    return {
        "scenario": scenario,
        "config": name,
        "world": world,
        "seed": seed,
        **metrics,
        "omega_sign_flips": sign_flips,
        "mode_switches": int(ctrl.mode_switch_count),
        "assist_fraction": assist_cycles / cycles,
        "branch_selection_fraction": float(np.mean(
            [key != -1 for key in selected_keys])),
        "mean_ess": float(np.mean(ess_values)),
        "mean_path_error_m": round(mean_path_error, 4),
        "max_path_error_m": round(max_path_error, 4),
        "reference_separation_fraction": (
            reference_separated / reference_pairs
            if reference_pairs else np.nan),
        "reference_pairs": reference_pairs,
        "mean_reference_length_m": (
            reference_length_total / guided_total if guided_total else np.nan),
        "centreline_coverage": (
            float(np.mean(coverage)) if coverage else np.nan),
        "feasible_branch_fraction": (
            feasible_seen / branches_seen if branches_seen else np.nan),
        "mean_guided_modes": (
            guided_total / assist_cycles if assist_cycles else np.nan),
        "update_collapsed_fraction": (
            update_collapsed / update_pairs if update_pairs else np.nan),
        "own_drift_fraction": (
            own_drift_total / guided_total if guided_total else np.nan),
        "mean_free_energy_gap": (
            float(np.mean([g for g in free_energy_gaps if np.isfinite(g)]))
            if free_energy_gaps else np.nan),
        "reference_fallback_fraction": (
            ctrl.reference_fallbacks / ctrl.reference_attempts
            if ctrl.reference_attempts else np.nan),
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
                "mean_switches": mean("mode_switches"),
                "branch_fraction": mean("branch_selection_fraction"),
                "ref_separation": mean("reference_separation_fraction"),
                "ref_length": mean("mean_reference_length_m"),
                "coverage": mean("centreline_coverage"),
                "feasible": mean("feasible_branch_fraction"),
                "guided_modes": mean("mean_guided_modes"),
                "collapsed": mean("update_collapsed_fraction"),
                "own_drift": mean("own_drift_fraction"),
                "fe_gap": mean("mean_free_energy_gap"),
                "fallback": mean("reference_fallback_fraction"),
            })
    return output


def _print_summary(summaries):
    print("\nscenario        cfg     success coll tout time  worst_clr "
          "switch branch ref_sep ref_len cover feasible modes collapsed "
          "drift fe_gap fallback")
    for row in summaries:
        print(
            f"{row['scenario']:15s} {row['config']:10s} "
            f"{row['successes']:2d}/{row['runs']:<2d} "
            f"{row['collisions']:4d} {row['timeouts']:4d} "
            f"{row['mean_time_s']:5.1f} {row['worst_clear_m']:9.3f} "
            f"{row['mean_switches']:6.1f} {row['branch_fraction']:6.2f} "
            f"{row['ref_separation']:7.3f} {row['ref_length']:7.2f} "
            f"{row['coverage']:5.2f} {row['feasible']:8.2f} "
            f"{row['guided_modes']:5.2f} {row['collapsed']:9.3f} "
            f"{row['own_drift']:5.3f} {row['fe_gap']:6.2f} "
            f"{row['fallback']:8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="4,48,49")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--scenarios", default="blocked")
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
        "--out",
        default="artifacts/results/milestones/reference_distinctness.csv")
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

    def report(row):
        print(f"{row['scenario']:15s} {row['config']:7s} "
              f"world={row['world']:2d} seed={row['seed']} "
              f"{row['result']:9s} clear={row['min_clear_m']:.3f} "
              f"ref_sep={row['reference_separation_fraction']:.3f} "
              f"ref_len={row['mean_reference_length_m']:.2f} "
              f"feasible={row['feasible_branch_fraction']:.2f} "
              f"branch={row['branch_selection_fraction']:.2f}",
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
