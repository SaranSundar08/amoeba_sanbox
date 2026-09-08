#!/usr/bin/env python3
"""Declared matched panel: `spacetime_flow` on vs off, identical
worlds/seeds/dynamic scenario -- see `docs/PROJECT_STATUS.md` (2026-09-08,
"Option B" wiring) and `mppi.py::spacetime_flow_cost`/`spacetime_flow.py`
for context. Deliberately small: a first-pass check of whether the full
space-time flood changes dynamic-scenario outcomes at all, not the full
generality matrix -- widen `WORLDS`/`SEEDS` for a fuller pass later if
this looks promising.
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


def _run(case):
    world, seed, flow = case
    env = DynaBarnEnv(world, robot_r=0.34, n_moving=4, speed=0.4, amplitude=0.6,
                      seed=DYNAMIC_SEED + world * 1000 + seed)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, dynamic_prediction=True,
        spacetime_flow=flow, spacetime_flow_reflood_every=5,
        fallback_share=0.25, gate="auto", path_validity_gate=True)
    m = episode(env, ctrl, max_steps=1200)
    return dict(
        world=world, seed=seed, spacetime_flow=flow, result=m["result"],
        success=m["success"], min_clear_m=m["min_clear_m"],
        time_s=m["time_s"], controller_mean_ms=m["controller_mean_ms"],
        controller_p95_ms=m["controller_p95_ms"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--out",
        default="artifacts/results/milestones/spacetime_flow_matched_panel.csv")
    args = parser.parse_args()
    cases = [(w, s, flow) for w in WORLDS for s in SEEDS for flow in (False, True)]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(_run, cases):
            rows.append(row)
            print(f"world={row['world']:3d} seed={row['seed']} "
                 f"flow={row['spacetime_flow']!s:5s} {row['result']:9s} "
                 f"clear={row['min_clear_m']:.3f} "
                 f"hz={1000/row['controller_mean_ms']:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {os.path.abspath(args.out)}")

    for flow in (False, True):
        group = [r for r in rows if r["spacetime_flow"] == flow]
        successes = sum(r["success"] for r in group)
        collisions = sum(r["result"] == "collision" for r in group)
        timeouts = sum(r["result"] == "timeout" for r in group)
        worst_clear = min(r["min_clear_m"] for r in group)
        print(f"flow={flow!s:5s} {successes}/{len(group)} success, "
             f"{collisions} collision, {timeouts} timeout, "
             f"worst_clear={worst_clear:.3f}")


if __name__ == "__main__":
    main()
