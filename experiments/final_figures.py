#!/usr/bin/env python3
"""Generate publication figures from the frozen final benchmark artifacts."""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar import astar_path
from env import BarnEnv
from experiments.v7_benchmark import _controller
from simulation import episode
from visualization import draw_world


ROOT = "artifacts/results/milestones"
OUT = "artifacts/images/experiments/final"
ORDER = ("vanilla", "path_mppi", "path_biased", "amoeba")
LABEL = {"vanilla": "Vanilla\nMPPI", "path_mppi": "Path-cost\nMPPI",
         "path_biased": "Single-path\nBiased MPPI", "amoeba": "Amoeba\nMPPI"}
COLOR = {"vanilla": "#777777", "path_mppi": "#4a3aa7",
         "path_biased": "#e28b20", "amoeba": "#168b8f"}


def _read(name):
    with open(os.path.join(ROOT, name), newline="") as stream:
        return list(csv.DictReader(stream))


def _f(row, key):
    return float(row[key])


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def static_figures(out):
    rows = (_read("final_static_generality.csv")
            + _read("final_static_path_biased.csv"))
    groups = {name: [r for r in rows if r["config"] == name]
              for name in ORDER}
    successes = [sum(int(r["success"]) for r in groups[name])
                 for name in ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    bars = ax.bar(range(4), np.array(successes) / 50 * 100,
                  color=[COLOR[n] for n in ORDER], width=0.68)
    ax.set_xticks(range(4), [LABEL[n] for n in ORDER])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Static narrow-world success (10 worlds × 5 seeds)")
    for bar, value in zip(bars, successes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{value}/50", ha="center", fontsize=9)
    _style(ax)

    ax = axes[1]
    values = [[_f(r, "min_clear_m") for r in groups[name]] for name in ORDER]
    box = ax.boxplot(values, patch_artist=True, widths=0.62,
                     medianprops={"color": "#111111", "linewidth": 1.4})
    for patch, name in zip(box["boxes"], ORDER):
        patch.set_facecolor(COLOR[name])
        patch.set_alpha(0.78)
    ax.axhline(0, color="#b52222", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, 5), [LABEL[n] for n in ORDER])
    ax.set_ylabel("Minimum footprint clearance (m)")
    ax.set_title("Safety distribution across matched trials")
    _style(ax)
    fig.tight_layout()
    path = os.path.join(out, "static_success_clearance.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    worlds = (0, 7, 48, 1, 5, 20, 40, 42, 93, 94)
    matrix = np.array([[sum(int(r["success"]) for r in groups[name]
                            if int(r["world"]) == world) / 5
                        for world in worlds] for name in ORDER])
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(4):
        for j in range(len(worlds)):
            ax.text(j, i, f"{int(matrix[i, j] * 5)}/5", ha="center",
                    va="center", color="white" if matrix[i, j] < .35 else
                    "#151515", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(worlds)), worlds)
    ax.set_yticks(range(4), [LABEL[n].replace("\n", " ") for n in ORDER])
    ax.set_xlabel("BARN world")
    ax.set_title("World-by-world static success")
    cbar = fig.colorbar(image, ax=ax, pad=.015)
    cbar.set_label("Success fraction")
    fig.tight_layout()
    path2 = os.path.join(out, "static_world_heatmap.png")
    fig.savefig(path2, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [path, path2]


def blocked_figure(out):
    base = _read("final_blocked_flow_only.csv")
    split = _read("final_blocked_split.csv")
    extension = _read("final_blocked_attribution_seed5_9.csv")
    flow = [r for r in base if r["config"] == "flow_only"] + [
        r for r in extension if r["config"] == "flow_only"]
    amoeba = [r for r in split if r["config"] == "amoeba"] + [
        r for r in extension if r["config"] == "amoeba"]
    groups = (flow, amoeba)
    names = ("Flow only", "Full Amoeba")
    colors = ("#4a3aa7", "#168b8f")
    success = [sum(int(r["success"]) for r in rows) for rows in groups]
    peer = [np.mean([_f(r, "minimum_peer_clearance_m") for r in rows])
            for rows in groups]
    stall = [np.mean([_f(r, "stall_time_s") for r in rows])
             for rows in groups]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    metrics = ((np.array(success) * 10, "Success rate (%)", (0, 108)),
               (peer, "Mean peer clearance (m)", (0, .8)),
               (stall, "Mean stall time (s)", (0, 24)))
    for ax, (values, ylabel, ylim) in zip(axes, metrics):
        bars = ax.bar(names, values, color=colors, width=.62)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            fmt = f"{value:.0f}%" if "%" in ylabel else f"{value:.3g}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()
                    + .025 * (ylim[1] - ylim[0]), fmt, ha="center", fontsize=9)
        _style(ax)
    axes[0].set_title("Blocked-split attribution")
    fig.suptitle("Pseudopod proposals outperform geodesic flow alone (10 paired seeds)")
    fig.tight_layout()
    path = os.path.join(out, "blocked_attribution.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def timing_figure(out):
    rows = _read("final_isolated_timing_summary.csv")
    by = {r["config"]: r for r in rows}
    means = [_f(by[n], "mean_controller_ms") for n in ORDER]
    p95 = [_f(by[n], "mean_controller_p95_ms") for n in ORDER]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bars = ax.bar(range(4), means, color=[COLOR[n] for n in ORDER], width=.65)
    ax.scatter(range(4), p95, marker="D", color="#151515", s=28,
               label="Mean episode p95", zorder=4)
    ax.axhline(50, color="#b52222", linestyle="--", linewidth=1.4,
               label="20 Hz budget (50 ms)")
    ax.set_xticks(range(4), [LABEL[n] for n in ORDER])
    ax.set_ylabel("Controller update time (ms)")
    ax.set_title("Isolated serial timing: reference NumPy implementation")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + 4, f"{value:.1f}",
                ha="center", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    path = os.path.join(out, "isolated_timing.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def trajectory_figure(out, world=7, seed=0, samples=1024, horizon=56):
    records = []
    for name in ("path_biased", "amoeba"):
        env = BarnEnv(world, robot_r=.34)
        path = astar_path(env, env.start[:2], env.goal)
        ctrl = _controller(name, env, path, seed, samples, horizon,
                           "slip", False, 2.0, .03)
        metrics = episode(env, ctrl, max_steps=1200)
        records.append((name, env, path, ctrl, metrics))
        print(f"trajectory replay {name}: {metrics['result']}", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 9.2), sharex=True, sharey=True)
    for ax, (name, env, path, ctrl, metrics) in zip(axes, records):
        # path_biased reuses AmoebaHybrid as a sampling implementation and
        # therefore constructs a flow object, but its flow weight and bias are
        # both zero. Do not draw that inactive object in the baseline panel.
        inactive_flow = getattr(ctrl, "flow", None) if name != "amoeba" else None
        if name != "amoeba":
            ctrl.flow = None
        draw_world(ax, env, ctrl, density=.7)
        if name != "amoeba":
            ctrl.flow = inactive_flow
        ax.plot(path[:, 0], path[:, 1], "--", color="#4a3aa7", lw=1.5,
                label="A* path")
        trace = metrics["trace"]
        ax.plot(trace[:, 0], trace[:, 1], color="#e34948", lw=2.5,
                label="executed trajectory")
        ax.plot(trace[-1, 0], trace[-1, 1], marker="X", ms=9,
                color="#b52222" if metrics["result"] == "collision" else
                "#16853b")
        ax.set_title(f"{LABEL[name].replace(chr(10), ' ')}\n"
                     f"{metrics['result']}, clearance "
                     f"{metrics['min_clear_m']:.3f} m")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(f"Matched world-{world} trajectory replay (seed {seed})")
    fig.tight_layout()
    target = os.path.join(out, "world7_trajectory_comparison.png")
    fig.savefig(target, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--with-trajectories", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = static_figures(args.out)
    paths.append(blocked_figure(args.out))
    paths.append(timing_figure(args.out))
    if args.with_trajectories:
        paths.append(trajectory_figure(args.out))
    print("\nwrote")
    for path in paths:
        print(os.path.abspath(path))


if __name__ == "__main__":
    main()
