#!/usr/bin/env python3
"""Matched panel: does the space-time BLOB help as the number of random moving
obstacles grows?

Every condition sees the SAME random scene per (n_obstacles, seed): `RandomDynEnv`
is seeded, so a paired comparison is exact. Conditions (all AmoebaHybrid with
pseudopods, grouped sampling and the dynamic-prediction critic; only the space-time
part differs):

  base         no space-time mechanism
  modes        the older trigger-based wait/detour search (spacetime.py)
  blob_extras  the blob adds 2 extra modes (best route + best other homotopy class)
  blob_pods    the blob's routes replace the static pseudopod modes while moving
               obstacles matter

Metrics per episode: result (success / collision / timeout), min ground-truth
footprint clearance, time. Summary: per (N, condition) success and collision rates
and, paired against `base`, McNemar (exact) on success and Wilcoxon on min clearance.

  python3 experiments/spacetime_blob_density_panel.py --n 5 10 15 20 --seeds 20 --jobs 12
"""
import argparse
import concurrent.futures
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astar import astar_path          # noqa: E402
from controllers import AmoebaHybrid  # noqa: E402
from randdyn import RandomDynEnv      # noqa: E402
from simulation import episode        # noqa: E402

CONDITIONS = {
    "base": {},
    "modes": {"spacetime_modes": True, "spacetime_obstacle_r": 0.225},
    "blob_extras": {"spacetime_blob": "extras"},
    "blob_pods": {"spacetime_blob": "pods"},
}
SCENE_SEED = 810000


def run_one(case):
    n, seed, cond, motion, max_steps = case
    env = RandomDynEnv(n_moving=n, seed=SCENE_SEED + 1000 * n + seed, motion=motion)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = AmoebaHybrid(
        env, seed=seed, path=path, extract_branches=True, max_branches=3,
        grouped_sampling=True, dynamic_prediction=True, fallback_share=0.25,
        gate="auto", path_validity_gate=True, **CONDITIONS[cond])
    m = episode(env, ctrl, max_steps=max_steps)
    blob = getattr(ctrl, "blob", None)
    return dict(
        n=n, seed=seed, condition=cond, motion=motion, result=m["result"],
        success=m["success"], min_clear_m=m["min_clear_m"], time_s=m["time_s"],
        path_len_m=m["path_len_m"], controller_mean_ms=m["controller_mean_ms"],
        blob_active_cycles=blob.active_cycles if blob else 0,
        blob_cycles=blob.cycles if blob else 0,
        blob_routes=blob.routes_built if blob else 0,
        blob_infeasible=blob.routes_infeasible if blob else 0,
        blob_mean_ms=round(sum(blob.build_ms) / max(1, len(blob.build_ms)), 3) if blob else 0.0,
        modes_added=getattr(ctrl, "spacetime_modes_added", 0))


def summarize(rows, conds):
    from scipy.stats import binomtest, wilcoxon
    print(f"\n{'N':>3} {'condition':12s} {'success':>9s} {'collide':>8s} {'timeout':>8s}"
          f" {'mean clear':>11s} {'vs base: McNemar p':>19s} {'Wilcoxon p (clear)':>19s}")
    for n in sorted({r["n"] for r in rows}):
        base = {r["seed"]: r for r in rows if r["n"] == n and r["condition"] == "base"}
        for c in conds:
            g = {r["seed"]: r for r in rows if r["n"] == n and r["condition"] == c}
            if not g:
                continue
            k = len(g)
            succ = sum(r["success"] for r in g.values())
            col = sum(r["result"] == "collision" for r in g.values())
            tmo = sum(r["result"] == "timeout" for r in g.values())
            clr = sum(r["min_clear_m"] for r in g.values()) / k
            mcn = wil = "--"
            if c != "base" and base:
                common = sorted(set(g) & set(base))
                b01 = sum(1 for s in common if base[s]["success"] and not g[s]["success"])
                b10 = sum(1 for s in common if g[s]["success"] and not base[s]["success"])
                if b01 + b10:
                    mcn = f"{binomtest(b10, b01 + b10, 0.5).pvalue:.3f} ({b10}+/{b01}-)"
                else:
                    mcn = "1.000 (0/0)"
                diffs = [g[s]["min_clear_m"] - base[s]["min_clear_m"] for s in common]
                if any(d != 0 for d in diffs):
                    wil = f"{wilcoxon(diffs).pvalue:.3f}"
            print(f"{n:>3} {c:12s} {succ:>4d}/{k:<4d} {col:>8d} {tmo:>8d} {clr:>11.3f} "
                  f"{mcn:>19s} {wil:>19s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--motion", default="waypoint", choices=["waypoint", "bounce"])
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="artifacts/results/milestones/spacetime_blob_density_panel.csv")
    ap.add_argument("--summary-only", action="store_true", help="re-summarize an existing CSV")
    a = ap.parse_args()

    if a.summary_only:
        rows = list(csv.DictReader(open(a.out)))
        for r in rows:
            r["n"], r["seed"] = int(r["n"]), int(r["seed"])
            r["success"], r["min_clear_m"] = int(r["success"]), float(r["min_clear_m"])
        summarize(rows, [c for c in CONDITIONS if any(r["condition"] == c for r in rows)])
        return

    cases = [(n, s, c, a.motion, a.max_steps)
             for n in a.n for s in range(a.seeds) for c in a.conditions]
    print(f"{len(cases)} episodes, {a.jobs} jobs", flush=True)
    rows = []
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for i, row in enumerate(pool.map(run_one, cases, chunksize=1), 1):
            rows.append(row)
            print(f"[{i:4d}/{len(cases)}] N={row['n']:2d} seed={row['seed']:2d} "
                  f"{row['condition']:12s} {row['result']:9s} clear={row['min_clear_m']:6.3f} "
                  f"t={row['time_s']:5.1f}s active={row['blob_active_cycles']}/{row['blob_cycles']}",
                  flush=True)
            if i % 10 == 0 or i == len(cases):     # checkpoint
                with open(a.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
    print(f"\nwrote {os.path.abspath(a.out)}")
    summarize(rows, a.conditions)


if __name__ == "__main__":
    main()
