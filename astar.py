"""A* global planner over the same free-space grid FlowField uses.

Deliberately its own module: this is what the amoeba blob is meant to
ASSIST, not replace. The `--lane` straight-line paths in path_test.py are
a synthetic stand-in built for one purpose -- feeding `local`/`hybrid` a
DELIBERATELY WRONG plan to prove the amoeba still threads the true gap.
They were never meant to be a real global planner, and using them as the
default path source made `hybrid`'s "the path stays a tracked contract"
claim look hollow (a straight line cutting through a pillar). This module
is the real thing: 8-connected grid A*, Euclidean heuristic, over the
robot-radius-inflated free space, then shortcut-smoothed so the output
looks like a plan a person would draw, not a grid staircase.
"""
import heapq

import numpy as np


def _line_clear(env, a, b, robot_r, step=0.02):
    """True if the straight segment a->b stays outside robot_r of every
    obstacle the whole way, sampled at `step` spacing (fine enough that
    a discrete sample can't hide a real dip between points)."""
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
    ts = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + (b - a)[None, :] * ts[:, None]
    return bool((env.clearance(pts) - robot_r).min() > 0)


def _shortcut(path, env, robot_r):
    """Greedy string-pulling: replace any run of waypoints with a single
    straight hop whenever that hop stays clear. Turns A*'s blocky grid
    staircase into a small number of straight legs -- what the path
    SHOULD look like, not an artifact of the grid resolution."""
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _line_clear(env, out[-1], path[j], robot_r):
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def astar_path(env, start_xy, goal_xy, res=0.1, densify_step=0.05):
    """8-connected A* from start_xy to goal_xy through env's robot-radius-
    inflated free space (the same `clearance > robot_r` test FlowField
    uses, so a path this planner approves of is a gap the amoeba's own
    flood would also call wet). Returns a dense (M, 2) polyline in world
    coordinates, shortcut-smoothed then resampled at `densify_step`.
    Raises RuntimeError if the goal is unreachable in the inflated grid.
    """
    x0, y0 = env.xmin, 0.0
    nx = int(np.ceil((env.xmax - env.xmin) / res)) + 1
    ny = int(np.ceil((env.ymax - y0) / res)) + 1
    gx, gy = np.meshgrid(x0 + np.arange(nx) * res, y0 + np.arange(ny) * res,
                         indexing="ij")
    clear = env.clearance(np.stack([gx, gy], -1))
    free = clear > env.robot_r

    def to_cell(xy):
        i = int(np.clip(round((xy[0] - x0) / res), 0, nx - 1))
        j = int(np.clip(round((xy[1] - y0) / res), 0, ny - 1))
        return i, j

    def nearest_free(cell):
        if free[cell]:
            return cell
        fi, fj = np.nonzero(free)
        k = np.argmin((fi - cell[0]) ** 2 + (fj - cell[1]) ** 2)
        return int(fi[k]), int(fj[k])

    start_c = nearest_free(to_cell(start_xy))
    goal_c = nearest_free(to_cell(goal_xy))

    nbrs = [(di, dj, res * np.hypot(di, dj))
            for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

    def h(cell):
        return res * np.hypot(cell[0] - goal_c[0], cell[1] - goal_c[1])

    g = {start_c: 0.0}
    came = {}
    pq = [(h(start_c), start_c)]
    visited = set()
    while pq:
        _, cell = heapq.heappop(pq)
        if cell in visited:
            continue
        visited.add(cell)
        if cell == goal_c:
            break
        i, j = cell
        for di, dj, w in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not free[ni, nj]:
                continue
            if di != 0 and dj != 0:
                # Diagonal moves must not cut a corner: both flanking
                # orthogonal cells have to be free too, or the segment
                # between two free diagonal cells can clip an obstacle
                # corner that neither cell's own occupancy would show.
                if not free[i + di, j] or not free[i, j + dj]:
                    continue
            ncell = (ni, nj)
            ng = g[cell] + w
            if ng < g.get(ncell, np.inf):
                g[ncell] = ng
                came[ncell] = cell
                heapq.heappush(pq, (ng + h(ncell), ncell))

    if goal_c not in visited:
        raise RuntimeError(
            "A*: goal unreachable in the robot-radius-inflated free space")

    cell = goal_c
    cells = [cell]
    while cell != start_c:
        cell = came[cell]
        cells.append(cell)
    cells.reverse()
    raw = np.array([[x0 + i * res, y0 + j * res] for i, j in cells], float)
    raw[0] = start_xy
    raw[-1] = goal_xy

    smooth = _shortcut(raw, env, env.robot_r)

    seg = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return smooth
    s = np.arange(0.0, total, densify_step)
    dense = np.stack([np.interp(s, cum, smooth[:, 0]),
                      np.interp(s, cum, smooth[:, 1])], 1)
    return np.vstack([dense, goal_xy])
