#!/usr/bin/env python3
"""Batch comparison across worlds x controllers x seeds -> table + CSV.

  python3 compare.py --worlds 42,0,138 --seeds 3
  python3 compare.py --band 0.32 0.40 --limit 8        # pick tight worlds
"""
import argparse
import csv
import os

import numpy as np

from env import BarnEnv, BARN
from controllers import CONTROLLERS
from run import episode

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_band(lo, hi, limit):
    rows = list(csv.DictReader(open(os.path.join(
        BARN, "susag_world_clearance.csv"))))
    ids = [int(r["world_id"]) for r in rows
           if lo <= float(r["bottleneck_clearance_m"]) <= hi]
    return ids[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", help="comma-separated world ids")
    p.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"),
                   help="pick worlds by bottleneck clearance range")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--controllers", default="vanilla,local")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=0.30)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument(
        "--out",
        default=os.path.join(HERE, "artifacts", "results", "comparison.csv"))
    args = p.parse_args()

    if args.worlds:
        worlds = [int(w) for w in args.worlds.split(",")]
    elif args.band:
        worlds = pick_band(*args.band, args.limit)
    else:
        worlds = [42, 0, 12, 30]
    names = args.controllers.split(",")

    rows = []
    for wid in worlds:
        for name in names:
            for seed in range(args.seeds):
                env = BarnEnv(wid, args.scale, args.radius)
                ctrl = CONTROLLERS[name](env, seed=seed)
                m = episode(env, ctrl, args.max_steps)
                m.pop("trace")
                row = {"controller": name, "world": wid, "seed": seed,
                       "scale": args.scale, "radius": args.radius, **m}
                rows.append(row)
                print(f"world {wid:3d} | {name:7s} | seed {seed} | "
                      f"{m['result']:9s} time={m['time_s']:6.1f}s "
                      f"stall={m['stall_s']:5.1f}s "
                      f"min_clear={m['min_clear_m']:6.3f}m", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    print(f"\n{'controller':10s} {'success':>8s} {'avg time':>9s} "
          f"{'avg stall':>10s} {'collisions':>11s}")
    for name in names:
        rs = [r for r in rows if r["controller"] == name]
        ok = [r for r in rs if r["success"]]
        print(f"{name:10s} {len(ok):3d}/{len(rs):<3d} "
              f"{np.mean([r['time_s'] for r in ok]) if ok else float('nan'):8.1f}s "
              f"{np.mean([r['stall_s'] for r in rs]):9.1f}s "
              f"{sum(r['result'] == 'collision' for r in rs):11d}")


if __name__ == "__main__":
    main()
