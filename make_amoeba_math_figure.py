#!/usr/bin/env python3
"""Generate the implementation-derived amoeba motion figure used in docs."""
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from controllers import AmoebaLocal
from env import BarnEnv, CYL_R, LocalFlowField
from pseudopods import extract_pseudopods
from simulation import episode
from visualization import (
    INK, MUTED, OBSTACLE, RED, _draw_branches, _draw_field, _style_axes)


OUT = "artifacts/images/demos/amoeba_motion_sequence.png"


def main():
    env = BarnEnv(48, robot_r=0.34)
    ctrl = AmoebaLocal(
        env, seed=0, extract_branches=True, max_branches=3,
        window=2.5, flow_res=0.05)
    metrics = episode(env, ctrl, max_steps=1200)
    trace = metrics["trace"]
    if not metrics["success"]:
        raise RuntimeError(f"reference episode did not succeed: {metrics['result']}")

    targets = (2.5, 5.5, 8.0)
    indices = [int(np.argmin(np.abs(trace[:, 1] - y))) for y in targets]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4), constrained_layout=True)

    for panel, (ax, index) in enumerate(zip(axes, indices), start=1):
        state = trace[index]
        field = LocalFlowField(
            env, state[:2], radius=2.5, res=0.05, viscosity=1.5)
        branches = [replace(branch, branch_id=branch_id)
                    for branch_id, branch in enumerate(
                        extract_pseudopods(field, max_modes=3))]

        _style_axes(ax)
        _draw_field(ax, field, mode="quiver", density=1.25)
        _draw_branches(ax, branches)
        membrane_i, membrane_j = np.nonzero(field.membrane)
        ax.scatter(
            field.x0 + membrane_i * field.res,
            field.y0 + membrane_j * field.res,
            s=5, color="#008b9a", alpha=0.65, zorder=3,
            label="geodesic membrane")

        for obstacle in env.obs:
            ax.add_patch(patches.Circle(
                obstacle, CYL_R, facecolor=OBSTACLE,
                edgecolor="#2f2e2c", linewidth=0.4, zorder=4))
        for wall_x in (env.xmin, env.xmax):
            ax.plot([wall_x, wall_x], [0.0, env.ymax],
                    color=OBSTACLE, linewidth=2.2, zorder=4)

        ax.plot(trace[:index + 1, 0], trace[:index + 1, 1],
                color=RED, linewidth=2.0, zorder=5, label="driven trace")
        ax.add_patch(patches.Circle(
            state[:2], env.robot_r, facecolor=INK, edgecolor="white",
            linewidth=1.0, zorder=7))
        heading = state[:2] + env.robot_r * np.array([
            np.cos(state[2]), np.sin(state[2])])
        ax.plot([state[0], heading[0]], [state[1], heading[1]],
                color="white", linewidth=1.5, zorder=8)

        ax.set_aspect("equal")
        ax.set_xlim(env.xmin - 0.15, env.xmax + 0.15)
        ax.set_ylim(max(0.0, state[1] - 2.8),
                    min(env.ymax, state[1] + 2.8))
        ax.set_xlabel("x [m]", color=MUTED)
        if panel == 1:
            ax.set_ylabel("y [m]", color=MUTED)
        ax.set_title(
            f"({chr(96 + panel)}) rebuild at t={index * ctrl.dt:.1f} s\n"
            f"robot y={state[1]:.2f} m, {len(branches)} pseudopod(s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False)
    fig.suptitle(
        "The geodesic amoeba is rebuilt around the moving robot (BARN world 48)",
        fontsize=14)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
