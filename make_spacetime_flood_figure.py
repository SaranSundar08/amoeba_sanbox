#!/usr/bin/env python3
"""Figures for `spacetime_flow.SpaceTimeFlood` -- see that module's
docstring for what this is and isn't (a tested standalone prototype, not
wired into `spacetime.py`/`mppi.py`; see the cost discussion there and in
the project memory dated 2026-09-08 for why not).
"""
import os

import mpl_toolkits
_LOCAL_MPL_TOOLKITS = os.path.expanduser(
    "~/.local/lib/python3.10/site-packages/mpl_toolkits")
if os.path.isdir(_LOCAL_MPL_TOOLKITS):
    mpl_toolkits.__path__.insert(0, _LOCAL_MPL_TOOLKITS)

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d  # noqa: E402  (see patch above)

from spacetime_flow import SpaceTimeFlood
from visualization import MUTED, RED, _water_cmap

OUT_FILMSTRIP = "artifacts/images/demos/spacetime_flood_filmstrip.png"
OUT_VOLUME = "artifacts/images/demos/spacetime_flood_volume.png"

ROBOT_R, OBSTACLE_R = 0.2, 0.3
START = (0.0, 0.0)
CROSS_Y, CROSS_TIME, CROSS_SPEED = 1.5, 2.0, 0.4
HORIZON, DT_LAYER, RES, WINDOW = 4.0, 0.2, 0.08, 2.0


def _crossing(t):
    return (CROSS_SPEED * (t - CROSS_TIME), CROSS_Y)


class _OpenEnvWithCrossing:
    """`SpaceTimeFlood` reads `env.clearance` (static) and, optionally,
    `env.predicted_moving_obs(times) -> (nk, N, 2)` (dynamic) -- the same
    interfaces `spacetime.py` and the rest of the sandbox already use, so
    this is the one adapter needed to point it at a single crossing
    obstacle in an otherwise open corridor."""

    def clearance(self, pts):
        return np.full(pts.shape[:-1], 10.0)

    def predicted_moving_obs(self, times):
        return np.array([_crossing(t) for t in times])[:, None, :]


def main():
    flood = SpaceTimeFlood(
        _OpenEnvWithCrossing(), START, robot_r=ROBOT_R, obstacle_r=OBSTACLE_R,
        horizon=HORIZON, dt_layer=DT_LAYER, res=RES, window=WINDOW)
    xs, ys, times = flood.xs, flood.ys, flood.times
    Xf, Yf = np.meshgrid(xs, ys, indexing="ij")
    cmap = _water_cmap()

    # --- filmstrip: the blob's shape at each time layer, side by side ---
    show_k = np.linspace(0, len(times) - 1, 8).round().astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6),
                             constrained_layout=True)
    for ax, k in zip(axes.flat, show_k):
        ax.pcolormesh(Xf, Yf, flood.depth(k), cmap=cmap, vmin=0, vmax=1,
                     shading="auto")
        ox, oy = _crossing(times[k])
        ax.add_patch(mp.Circle((ox, oy), OBSTACLE_R, facecolor="#52514e",
                               edgecolor=RED, linewidth=1.3, zorder=4))
        ax.plot(*START, marker="o", ms=6, color="black", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_title(f"t = {times[k]:.2f} s", fontsize=10, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
    fig.suptitle(
        "Flooding (x, y, t): the blob's own shape at each time layer\n"
        "(darker = closer to the robot's start; gray disc = the moving "
        "obstacle, at ITS position at that instant)", fontsize=12.5)
    os.makedirs(os.path.dirname(OUT_FILMSTRIP), exist_ok=True)
    fig.savefig(OUT_FILMSTRIP, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(OUT_FILMSTRIP)}")

    # --- volume: the same slices, stacked in one (x, y, t) figure ---
    fig2 = plt.figure(figsize=(9.0, 8.2))
    ax3 = fig2.add_subplot(111, projection="3d")
    show_k_volume = np.linspace(0, len(times) - 1, 12).round().astype(int)
    for k in show_k_volume:
        ax3.contourf(Xf, Yf, flood.depth(k), zdir="z", offset=times[k],
                    levels=12, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    tube_t = np.linspace(0.0, times[-1], 60)
    tube_xy = np.array([_crossing(t) for t in tube_t])
    ax3.plot(tube_xy[:, 0], tube_xy[:, 1], tube_t, color=RED, lw=2.2,
             ls=":", alpha=0.9, label="moving obstacle's tube")
    ax3.scatter(*START, 0.0, color="black", s=45, zorder=10, label="start")
    ax3.set_xlim(xs[0], xs[-1])
    ax3.set_ylim(ys[0], ys[-1])
    ax3.set_zlim(0.0, times[-1])
    ax3.set_xlabel("x [m]", color=MUTED, labelpad=6)
    ax3.set_ylabel("y [m]", color=MUTED, labelpad=6)
    ax3.set_zlabel("time [s]", color=MUTED, labelpad=2)
    ax3.view_init(elev=18, azim=-60)
    ax3.legend(loc="upper left", frameon=False, fontsize=9)
    ax3.set_title(
        "The flooded space-time volume: every reachable (x, y, t) cell,\n"
        "not just two candidate routes through it -- the obstacle carves\n"
        "a moving notch straight through the stack of blobs",
        fontsize=11.5)
    fig2.savefig(OUT_VOLUME, dpi=175, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {os.path.abspath(OUT_VOLUME)}")


if __name__ == "__main__":
    main()
