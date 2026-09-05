#!/usr/bin/env python3
"""Compare the original V6 ablation against its re-run under the exact
footprint collision check.

The original `v6_ablation.csv` was generated when the ground-truth
collision check sampled 17 points on the SLIP footprint boundary, which
can miss a BARN-radius cylinder sitting between two samples. The re-run
uses `RobotModel.exact_clearance`. This script pairs the two matrices
case by case (scenario, config, world, seed) and reports what changed.
"""
import argparse
import csv

import numpy as np


KEY = ("scenario", "config", "world", "seed")


def _load(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {tuple(row[k] for k in KEY): row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old", default="artifacts/results/milestones/v6_ablation.csv")
    parser.add_argument(
        "--new",
        default="artifacts/results/milestones/v6_ablation_exact_footprint.csv")
    args = parser.parse_args()

    old, new = _load(args.old), _load(args.new)
    keys = sorted(set(old) & set(new))
    missing = (set(old) | set(new)) - set(keys)
    if missing:
        print(f"warning: {len(missing)} cases present in only one file")

    print("\nPer-case changes (result or min clearance moved):")
    print(f"{'scenario':16s}{'model':6s}{'w':>3s}{'s':>2s}  "
          f"{'old result':10s} {'new result':10s} "
          f"{'old clr':>8s} {'new clr':>8s} {'d clr':>7s}")
    changed = 0
    for key in keys:
        o, n = old[key], new[key]
        oc, nc = float(o["min_clear_m"]), float(n["min_clear_m"])
        if o["result"] != n["result"] or abs(oc - nc) > 0.005:
            changed += 1
            flag = " <-- outcome changed" if o["result"] != n["result"] else ""
            print(f"{key[0]:16s}{key[1]:6s}{key[2]:>3s}{key[3]:>2s}  "
                  f"{o['result']:10s} {n['result']:10s} "
                  f"{oc:8.3f} {nc:8.3f} {nc - oc:7.3f}{flag}")
    print(f"\n{changed}/{len(keys)} cases moved (result or >5 mm clearance).")

    print("\nSummary per (scenario, model): successes / collisions / "
          "timeouts, worst clearance")
    print(f"{'scenario':16s}{'model':6s} {'old':^16s} {'new':^16s} "
          f"{'old worst':>10s} {'new worst':>10s}")
    scenarios, configs = [], []
    for key in keys:
        if key[0] not in scenarios:
            scenarios.append(key[0])
        if key[1] not in configs:
            configs.append(key[1])
    for scenario in scenarios:
        for config in configs:
            group = [k for k in keys if k[0] == scenario and k[1] == config]
            if not group:
                continue

            def tally(table):
                rows = [table[k] for k in group]
                s = sum(r["result"] == "success" for r in rows)
                c = sum(r["result"] == "collision" for r in rows)
                t = sum(r["result"] == "timeout" for r in rows)
                worst = min(float(r["min_clear_m"]) for r in rows)
                return f"{s}/{len(rows)} {c:2d} {t:2d}", worst

            (o_txt, o_worst), (n_txt, n_worst) = tally(old), tally(new)
            print(f"{scenario:16s}{config:6s} {o_txt:^16s} {n_txt:^16s} "
                  f"{o_worst:10.3f} {n_worst:10.3f}")


if __name__ == "__main__":
    main()
