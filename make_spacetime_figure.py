#!/usr/bin/env python3
"""Generate a figure of space-time topology producing genuinely distinct
wait/detour routes, using the real production functions end to end
(`spacetime.two_route_search`, `spacetime.spacetime_path_to_proposal`,
`visualization._draw_branch_proposals`, `env.LocalFlowField` for the flood
itself) -- `make_amoeba_math_figure.py`'s companion, one axis further,
into free SPACE-TIME.

Two earlier approaches were tried and abandoned (see git history/session
log): (1) scanning real dynamic BARN episodes end to end hoping to stumble
on a moment where wait/detour diverge -- most crossings resolve to the
SAME path, so this was slow and mostly wasted; (2) transplanting a
synthetic crossing onto a REAL extracted branch's centerline in a real
BARN world -- distinctness turned out to be extremely sensitive to the
exact geometry (changing robot/obstacle radius, start/goal distance, or
horizon each independently killed it against real curvy corridors).

This uses the ONE combination already proven reliable:
`tests/test_spacetime.py::TwoRouteSearchTests.test_wait_and_detour_are_
found_and_flagged_distinct`'s exact numbers, in an open corridor -- a
deterministic, always-passing case, not a probabilistic search -- and
floods a real `LocalFlowField` around it purely for the visual "blob",
which does not participate in (and therefore cannot break) the space-time
search itself.
"""
import os

# This machine has a stale matplotlib 3.5.1 under /usr/lib/python3/
# dist-packages whose mpl_toolkits/__init__.py (an old pkg_resources
# namespace declaration) wins parent-package resolution over the correct
# pip-installed 3.10.8 one in ~/.local, even though .local comes first on
# sys.path -- a "regular" package with its own __init__.py short-circuits
# namespace-portion merging regardless of path order, and `matplotlib.
# pyplot` tries to register the 3d projection (importing mplot3d) the
# moment IT is first imported -- so this patch has to land before that
# import, not just before mplot3d's own. Forcing the real copy to the
# front of mpl_toolkits.__path__ here fixes it for this process only;
# it's a local sys-state patch, not a change to the machine's Python
# installation.
import mpl_toolkits
_LOCAL_MPL_TOOLKITS = os.path.expanduser(
    "~/.local/lib/python3.10/site-packages/mpl_toolkits")
if os.path.isdir(_LOCAL_MPL_TOOLKITS):
    mpl_toolkits.__path__.insert(0, _LOCAL_MPL_TOOLKITS)

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d  # noqa: E402  (see patch above)

from env import LocalFlowField
from robot_model import RobotModel
from spacetime import spacetime_path_to_proposal, two_route_search
from scipy.ndimage import gaussian_filter

from visualization import (
    AQUA, INK, MUTED, RED, SPACETIME_DETOUR, SPACETIME_WAIT, VIOLET,
    _water_cmap, draw_world)

OUT = "artifacts/images/demos/spacetime_topology.png"

# Exact numbers from the validated unit test -- do not tweak in isolation;
# see module docstring on how fragile "distinct" is to any one of these.
ROBOT_R, OBSTACLE_R = 0.2, 0.3
START, GOAL = (0.0, 0.0), (0.0, 3.0)
CROSS_Y, CROSS_TIME, CROSS_SPEED = 1.5, 2.0, 0.4
HORIZON, DT_LAYER, RES, WINDOW, GOAL_TOL = 8.0, 0.25, 0.2, 2.0, 0.15


class TeachingEnv:
    """Open corridor, no static obstacles -- isolates the moving-obstacle
    mechanism exactly like `tests/test_spacetime.py::_OpenEnv`, wide
    enough (side walls only) to draw as a real flooded corridor."""

    def __init__(self, robot_r=ROBOT_R):
        self.robot_r = robot_r
        self.xmin, self.xmax = -2.0, 2.0
        self.ymax = 4.2
        self.start = np.array([START[0], START[1], np.pi / 2])
        self.goal = np.array(GOAL)
        self.goal_tol = GOAL_TOL
        self.obs = np.zeros((0, 2))

    def clearance(self, pts):
        return np.full(pts.shape[:-1], 10.0)


def _crossing(t):
    return (CROSS_SPEED * (t - CROSS_TIME), CROSS_Y)


def main():
    env = TeachingEnv()
    state = np.array([START[0], START[1], np.pi / 2])
    robot = RobotModel(kind="ideal", radius=ROBOT_R)

    routes = two_route_search(
        env, START, GOAL, _crossing, OBSTACLE_R, robot.planning_radius,
        horizon=HORIZON, dt_layer=DT_LAYER, res=RES, window=WINDOW,
        goal_tol=GOAL_TOL)
    assert routes["wait"]["feasible"] and routes["detour"]["feasible"] \
        and routes["distinct"], routes
    print("distinct wait/detour pair confirmed "
          f"(wait cost={routes['wait']['cost']:.2f}, "
          f"detour cost={routes['detour']['cost']:.2f})")

    proposals = []
    for name, offset in (("wait", 100), ("detour", 200)):
        route = routes[name]
        proposal = spacetime_path_to_proposal(
            route["path"], DT_LAYER, state, np.array(GOAL), env, robot,
            _crossing, OBSTACLE_R, T=64, dt=DT_LAYER, branch_id=offset)
        proposals.append(proposal)

    field = LocalFlowField(env, state[:2], radius=2.2, res=0.04, viscosity=1.0)

    class _Snapshot:
        flow = field
        branches = []
        branch_proposals = proposals

    # The raw `route["path"]` is the coarse grid-search output (8-connected
    # moves at `res`=0.2 m spacing -- inherently blocky). `proposal.rollout`
    # is that same path resampled and pushed through acceleration/rate
    # limits and `robot_model.integrate` (`spacetime_path_to_proposal`'s
    # whole job) -- the smooth, dynamically-achievable curve already drawn
    # in panel (a). Using it here too, instead of the raw path, is what
    # makes the two panels show the same line, not two different ones.
    query_t = (np.arange(proposals[0].rollout.shape[0]) + 1.0) * DT_LAYER

    def _trim_settled(xy, tol=0.02):
        """`spacetime_path_to_proposal` holds the final position once
        arrived (see its docstring: "already arrived, waiting there"), so
        the rollout keeps going for the controller's full T -- correct,
        but drawn as-is it's a tall vertical "flagpole" of near-duplicate
        points at the end that adds nothing to look at. Trim to just past
        the last point that's still actually moving."""
        moved = np.linalg.norm(np.diff(xy, axis=0), axis=-1) > tol
        last_moving = np.nonzero(moved)[0][-1] if moved.any() else len(xy) - 1
        return slice(0, min(last_moving + 3, len(xy)))

    wait_xy = proposals[0].rollout[:, :2]
    detour_xy = proposals[1].rollout[:, :2]
    wait_sl, detour_sl = _trim_settled(wait_xy), _trim_settled(detour_xy)
    wait_xy, wait_t = wait_xy[wait_sl], query_t[wait_sl]
    detour_xy, detour_t = detour_xy[detour_sl], query_t[detour_sl]

    fig = plt.figure(figsize=(13.4, 7.6))
    ax = fig.add_subplot(1, 2, 1)
    draw_world(ax, env, _Snapshot(), density=1.1)
    ax.set_xlim(env.xmin - 0.1, env.xmax + 0.1)
    ax.set_ylim(-0.1, env.ymax)

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color=VIOLET,
            lw=1.4, alpha=0.75, zorder=3, label="direct path")
    ax.add_patch(mp.Circle(START, ROBOT_R, facecolor=INK, edgecolor="white",
                           linewidth=1.0, zorder=7))
    ax.plot([START[0], START[0]], [START[1], START[1] + ROBOT_R], color="white",
            lw=1.5, zorder=8)

    obstacle_now = _crossing(0.0)
    ax.add_patch(mp.Circle(obstacle_now, OBSTACLE_R, facecolor="#52514e",
                           edgecolor=RED, linewidth=1.6, zorder=6))
    track_t = np.linspace(0.0, HORIZON, 40)
    track = np.array([_crossing(t) for t in track_t])
    ax.plot(track[:, 0], track[:, 1], ":", color=RED, lw=1.3, alpha=0.55,
            zorder=5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=SPACETIME_WAIT, lw=2.8, ls="-.",
              label="space-time: wait it out"),
        Line2D([0], [0], color=SPACETIME_DETOUR, lw=2.8, ls="-.",
              label="space-time: detour around"),
        Line2D([0], [0], color=RED, lw=1.6, ls=":",
              label="moving obstacle's crossing path"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#52514e",
              markeredgecolor=RED, markersize=9,
              label="moving obstacle (t=0)"),
        Line2D([0], [0], color=VIOLET, lw=1.4, ls="--",
              label="direct path (no obstacle)"),
    ]
    ax.legend(handles=handles, loc="upper center",
             bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=False, fontsize=9)
    ax.set_title("(a) plan view (x, y)", fontsize=11.5, color=INK)

    # --- (b) the actual space-time volume the search flooded: x, y, t ---
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    zmax = float(max(wait_t[-1], detour_t[-1])) + 0.6

    # The blob itself, at t=0: the same LocalFlowField depth-wash + membrane
    # `_draw_field` draws in panel (a), projected flat onto the z=0 (t=0)
    # floor instead of dropped -- it's real geodesic flood data, just shown
    # as the volume's "ground floor" rather than a 2D-only backdrop.
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

    for t in np.linspace(0.0, zmax, 13):
        ox, oy = _crossing(t)
        disc = mp.Circle((ox, oy), OBSTACLE_R, facecolor=RED,
                         edgecolor="#7a1f1f", linewidth=0.6, alpha=0.28)
        ax3.add_patch(disc)
        art3d.pathpatch_2d_to_3d(disc, z=t, zdir="z")
    tube_t = np.linspace(0.0, zmax, 60)
    tube_xy = np.array([_crossing(t) for t in tube_t])
    ax3.plot(tube_xy[:, 0], tube_xy[:, 1], tube_t, color=RED, lw=2.0,
             alpha=0.85, ls=":", label="moving obstacle's tube")
    ax3.scatter(*_crossing(0.0), 0.0, color="#52514e", edgecolor=RED,
               linewidth=1.4, s=90, zorder=9, label="moving obstacle (t=0)")

    ax3.plot(wait_xy[:, 0], wait_xy[:, 1], wait_t, color=SPACETIME_WAIT,
             lw=3.2, ls="-.", label="wait it out")
    ax3.plot(detour_xy[:, 0], detour_xy[:, 1], detour_t,
             color=SPACETIME_DETOUR, lw=3.2, ls="-.", label="detour around")
    ax3.scatter(*START, 0.0, color=INK, s=45, zorder=10)
    ax3.text(START[0], START[1], 0.0, "  start", fontsize=8, color=MUTED)

    ax3.set_xlim(env.xmin, env.xmax)
    ax3.set_ylim(-0.2, env.ymax)
    ax3.set_zlim(0.0, zmax)
    ax3.set_xlabel("x [m]", color=MUTED, labelpad=6)
    ax3.set_ylabel("y [m]", color=MUTED, labelpad=6)
    ax3.set_zlabel("time [s]", color=MUTED, labelpad=2)
    ax3.view_init(elev=16, azim=-55)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False,
              fontsize=8.5)
    ax3.set_title(
        "(b) the flooded volume: x, y, AND time (blob shown at t=0)\n"
        "wait = flat run near start, then rises;\n"
        "detour = shifts in x immediately, no flat run",
        fontsize=10.5, color=INK)

    fig.suptitle(
        "Space-time topology: two genuinely distinct routes past a moving "
        "obstacle\n(flooded corridor, validated Phase-0 parameters)",
        fontsize=13, color=INK)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
