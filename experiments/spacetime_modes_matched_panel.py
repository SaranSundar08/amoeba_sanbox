#!/usr/bin/env python3
"""Declared matched panel: `spacetime_modes` (the discrete, trigger-based
wait/detour alternative -- see `spacetime.py`) on vs off, same
worlds/seeds/dynamic scenario as `spacetime_flow_matched_panel.py`.

This is the item that was actually next on the pre-detour priority list
(see docs/PROJECT_STATUS.md, docs/experiments/SPACETIME_TOPOLOGY_PHASE0.md):
`spacetime_modes` has unit/integration test coverage but, unlike
`spacetime_flow` (validated 2026-09-08), had never been run as a declared
benchmark against real dynamic scenarios.

Only runs the `spacetime_modes=True` arm: the `spacetime_modes=False` arm
is IDENTICAL in every other parameter to `spacetime_flow_matched_panel.
py`'s `flow=False` arm (same worlds, seeds, dynamic-obstacle setup,
controller config) -- that CSV's baseline rows are reused directly rather
than re-running 18 redundant episodes.
"""
import argparse
import concurrent.futures
import csv
import os

from astar import astar_path
from controllers import AmoebaHybrid
from dynabarn import DynaBarnEnv
from simulation import episode

WORLDS = (4, 48, 49, 7, 42, 93)
SEEDS = (0, 1, 2)
DYNAMIC_SEED = 71000
BASELINE_CSV = "artifacts/results/milestones/spacetime_flow_matched_panel_w0.1.csv"


def _run(case):
    world, seed = case
    env = DynaBarnEnv(world, robot_r=0.34, n_moving=4, speed=0.4, amplitude=0.6,
                      seed=DYNAMIC_SEED + world * 1000 + seed)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, dynamic_prediction=True,
        spacetime_modes=True,
        fallback_share=0.25, gate="auto", path_validity_gate=True)
    m = episode(env, ctrl, max_steps=1200)
    return dict(
        world=world, seed=seed, spacetime_modes=True, result=m["result"],
        success=m["success"], min_clear_m=m["min_clear_m"],
        time_s=m["time_s"], controller_mean_ms=m["controller_mean_ms"],
        controller_p95_ms=m["controller_p95_ms"],
        spacetime_triggers=getattr(ctrl, "spacetime_triggers", 0),
        spacetime_modes_added=getattr(ctrl, "spacetime_modes_added", 0))


def _baseline_rows():
    rows = list(csv.DictReader(open(BASELINE_CSV)))
    out = []
    for r in rows:
        if r["spacetime_flow"] != "False":
            continue
        out.append(dict(
            world=int(r["world"]), seed=int(r["seed"]), spacetime_modes=False,
            result=r["result"], success=int(r["success"]),
            min_clear_m=float(r["min_clear_m"]), time_s=float(r["time_s"]),
            controller_mean_ms=float(r["controller_mean_ms"]),
            controller_p95_ms=float(r["controller_p95_ms"]),
            spacetime_triggers=0, spacetime_modes_added=0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--out",
        default="artifacts/results/milestones/spacetime_modes_matched_panel.csv")
    args = parser.parse_args()
    cases = [(w, s) for w in WORLDS for s in SEEDS]
    rows = _baseline_rows()
    print(f"reused {len(rows)} baseline (spacetime_modes=False) rows from "
         f"{BASELINE_CSV}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(_run, cases):
            rows.append(row)
            print(f"world={row['world']:3d} seed={row['seed']} "
                 f"modes=True  {row['result']:9s} "
                 f"clear={row['min_clear_m']:.3f} "
                 f"triggers={row['spacetime_triggers']} "
                 f"added={row['spacetime_modes_added']} "
                 f"hz={1000/row['controller_mean_ms']:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {os.path.abspath(args.out)}")

    for modes in (False, True):
        group = [r for r in rows if r["spacetime_modes"] == modes]
        successes = sum(r["success"] for r in group)
        collisions = sum(r["result"] == "collision" for r in group)
        timeouts = sum(r["result"] == "timeout" for r in group)
        worst_clear = min(r["min_clear_m"] for r in group)
        total_triggers = sum(r["spacetime_triggers"] for r in group)
        print(f"modes={modes!s:5s} {successes}/{len(group)} success, "
             f"{collisions} collision, {timeouts} timeout, "
             f"worst_clear={worst_clear:.3f}, total_triggers={total_triggers}")


if __name__ == "__main__":
    main()
