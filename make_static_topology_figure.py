#!/usr/bin/env python3
"""Companion to `make_spacetime_figure.py`: the STATIC case, in the same
(x, y, t) view, to make the actual argument for space-time topology
visible rather than asserted -- a static obstacle's space-time "tube" is a
perfectly vertical pillar (same x, y at every t), so any route that
dodges it in (x, y) ALONE already dodges it at every future time for
free. That's why ordinary branch extraction (`pseudopods.py`) never needed
a time axis. A moving obstacle's tube is tilted instead (see the other
figure) -- a route can be clear of its CURRENT position and still drive
straight into it later, which is the one thing plain space topology
cannot see and space-time topology exists to fix.
"""
import os
from dataclasses import replace

import mpl_toolkits
_LOCAL_MPL_TOOLKITS = os.path.expanduser(
    "~/.local/lib/python3.10/site-packages/mpl_toolkits")
if os.path.isdir(_LOCAL_MPL_TOOLKITS):
    mpl_toolkits.__path__.insert(0, _LOCAL_MPL_TOOLKITS)

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d  # noqa: E402  (see patch above)
from scipy.ndimage import gaussian_filter

from env import CYL_R, LocalFlowField
from proposals import _arc_length, _densify
from pseudopods import extract_pseudopods
from robot_model import RobotModel
from visualization import (
    AQUA, BRANCH_COLORS, INK, MUTED, RED, VIOLET, YELLOW, _water_cmap,
    draw_world)

OUT = "artifacts/images/demos/static_topology.png"
ROBOT_R = 0.34
PILLAR_R = 0.5
PILLAR_XY = np.array([0.0, 3.0])
START = np.array([0.0, 0.6, np.pi / 2])
GOAL = np.array([0.0, 5.6])
T, DT = 160, 0.05   # long enough to actually clear the pillar and diverge


class TeachingEnv:
    """One pillar in an open corridor -- the classic left/right split,
    same shape as `env.BarnEnv`'s interface (`clearance`, `obs`, bounds)."""

    def __init__(self, robot_r=ROBOT_R):
        self.robot_r = robot_r
        self.xmin, self.xmax = -2.0, 2.0
        self.ymax = 6.2
        self.start, self.goal = START, GOAL
        self.goal_tol = 0.4
        self.obs = PILLAR_XY[None, :]
        self._build_clearance_grid()

    def clearance_exact(self, pts):
        d = np.linalg.norm(pts[..., None, :] - self.obs, axis=-1).min(-1) - PILLAR_R
        walls = np.minimum(pts[..., 0] - self.xmin, self.xmax - pts[..., 0])
        return np.minimum(d, walls)

    def obstacle_centers(self):
        return self.obs, PILLAR_R

    def _build_clearance_grid(self, res=0.02):
        self._cres = res
        xs = np.arange(self.xmin, self.xmax + res, res)
        ys = np.arange(0.0, self.ymax + res, res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._cgrid = self.clearance_exact(np.stack([gx, gy], -1))

    def clearance(self, pts):
        i = np.clip(np.rint((pts[..., 0] - self.xmin) / self._cres).astype(int),
                   0, self._cgrid.shape[0] - 1)
        j = np.clip(np.rint(pts[..., 1] / self._cres).astype(int),
                   0, self._cgrid.shape[1] - 1)
        return self._cgrid[i, j]


def _timed_route(centerline, state, speed=0.35, dt=DT):
    """A branch centreline (pure geometry) walked at constant speed, giving
    it a time axis without going through the pursuit-tracking controller
    (`proposals.branch_to_control_sequence`) -- that controller is tuned
    for short, continuously-refreshed MPPI horizons (T=56, re-flooded every
    few cycles) and stalls when asked for one long, single-shot 8s
    prediction, which isn't the point of this figure anyway: the topology
    argument only needs the real, extracted geometric route, not
    low-level tracking fidelity."""
    path = _densify(np.vstack([state[:2], centerline]), spacing=0.02)
    arc = _arc_length(path)
    n = max(2, int(np.ceil(arc[-1] / speed / dt)) + 1)
    t = np.arange(n) * dt
    query_arc = np.minimum(t * speed, arc[-1])
    xy = np.column_stack([np.interp(query_arc, arc, path[:, 0]),
                          np.interp(query_arc, arc, path[:, 1])])
    return xy, t


def main():
    env = TeachingEnv()
    state = START.copy()
    robot = RobotModel(kind="ideal", radius=ROBOT_R)

    field = LocalFlowField(env, state[:2], radius=3.2, res=0.04, viscosity=1.5)
    # `extract_pseudopods` leaves every branch at its default id (-1) --
    # `make_amoeba_math_figure.py` re-numbers the same way for the same
    # reason: without it, distinct branches print identical "P-1" labels.
    branches = [replace(b, branch_id=i) for i, b in
               enumerate(extract_pseudopods(field, max_modes=2))]
    assert len(branches) == 2, f"expected a clean left/right split, got {len(branches)}"
    routes = [_timed_route(b.centerline, state) for b in branches]

    # (Not `branches = branches` inside the class body -- a class body's
    # namespace treats any name it assigns as local to the WHOLE block, so
    # a self-referential assignment like that shadows the enclosing
    # `branches` before its own right-hand side is evaluated.)
    _snap_flow, _snap_branches = field, branches

    class _Snapshot:
        flow = _snap_flow
        branches = _snap_branches
        branch_proposals = []

    fig = plt.figure(figsize=(13.4, 7.8))
    ax = fig.add_subplot(1, 2, 1)
    draw_world(ax, env, _Snapshot(), density=1.1)
    for (xy, _t), color in zip(routes, BRANCH_COLORS):
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.6, zorder=5)
    ax.plot(*GOAL, marker="*", ms=0)  # keep autoscale consistent w/ panel b
    ax.add_patch(mp.Circle(state[:2], ROBOT_R, facecolor=INK, edgecolor="white",
                           linewidth=1.0, zorder=7))
    ax.set_xlim(env.xmin - 0.1, env.xmax + 0.1)
    ax.set_ylim(-0.1, env.ymax)
    ax.set_title("(a) plan view (x, y): two branches around one fixed pillar",
                fontsize=11.5, color=INK)

    # --- (b) the same routes in (x, y, t): the obstacle is a straight
    # vertical pillar through time, unlike the moving case's tilted tube ---
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    zmax = float(max(t[-1] for _xy, t in routes)) + 0.4

    D = field.D
    xs_ = field.x0 + np.arange(D.shape[0]) * field.res
    ys_ = field.y0 + np.arange(D.shape[1]) * field.res
    Xf, Yf = np.meshgrid(xs_, ys_, indexing="ij")
    finite = np.isfinite(D)
    depth = np.where(finite, 1.0 - np.clip(D / max(field.dmax, 1e-6), 0, 1),
                     np.nan)
    ax3.contourf(Xf, Yf, depth, zdir="z", offset=0.0, levels=14,
                cmap=_water_cmap(), alpha=0.55)
    body = getattr(field, "body", None)
    if body is not None and body.any() and not body.all():
        smooth = gaussian_filter(body.astype(float), sigma=1.2)
        ax3.contour(Xf, Yf, smooth, levels=[0.5], zdir="z", offset=0.0,
                   colors=[AQUA], linewidths=1.6)

    # The pillar's space-time tube: identical (x, y) disc stacked straight
    # up through every t -- the whole point of this figure.
    for t in np.linspace(0.0, zmax, 14):
        disc = mp.Circle(PILLAR_XY, PILLAR_R, facecolor="#52514e",
                         edgecolor="#2f2e2c", linewidth=0.5, alpha=0.35)
        ax3.add_patch(disc)
        art3d.pathpatch_2d_to_3d(disc, z=t, zdir="z")
    ax3.plot([PILLAR_XY[0]] * 2, [PILLAR_XY[1]] * 2, [0.0, zmax],
             color="#2f2e2c", lw=1.4, ls=":", alpha=0.8,
             label="static obstacle's tube (vertical)")

    for branch, (xy, t), color in zip(branches, routes, BRANCH_COLORS):
        ax3.plot(xy[:, 0], xy[:, 1], t, color=color, lw=3.0,
                label=f"branch P{branch.branch_id}")
    ax3.scatter(*state[:2], 0.0, color=INK, s=45, zorder=10)
    ax3.text(state[0], state[1], 0.0, "  start", fontsize=8, color=MUTED)

    ax3.set_xlim(env.xmin, env.xmax)
    ax3.set_ylim(0.0, env.ymax)
    ax3.set_zlim(0.0, zmax)
    ax3.set_xlabel("x [m]", color=MUTED, labelpad=6)
    ax3.set_ylabel("y [m]", color=MUTED, labelpad=6)
    ax3.set_zlabel("time [s]", color=MUTED, labelpad=2)
    ax3.view_init(elev=16, azim=-55)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False,
              fontsize=8.5)
    ax3.set_title(
        "(b) same routes in x, y, AND time\n"
        "the pillar never moves, so its tube is perfectly vertical --\n"
        "dodging it once (in x, y) dodges it at every t for free",
        fontsize=10.5, color=INK)

    fig.suptitle(
        "Static topology: a fixed obstacle's space-time tube never tilts\n"
        "(compare with spacetime_topology.png's moving-obstacle tube)",
        fontsize=13, color=INK)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
