"""Full space-time flood: a real Dijkstra distance value at every reachable
(x, y, t) cell in a local window, not just a search for one or two
candidate routes through it.

`spacetime.time_expanded_search` already builds the exact (x, y, t) grid
this needs; it just stops the instant it reaches its goal, discarding
every other cell's distance. This module is that same grid with the
early exit removed, so `G[i, j, k]` -- the direct time-axis analogue of
`env.LocalFlowField.D` -- is a real number (or `inf`, if unreached) at
EVERY cell, not just along the one or two paths a trigger-based search
happens to construct.

This is deliberately NOT wired into `env.LocalFlowField`, `pseudopods.py`,
or `mppi.py`. A full flood is `nx * ny * nk` cells versus the static
flood's `nx * ny`, and unlike the static flood it cannot be cached for
many cycles -- the obstacle predictions it floods against are only valid
until the next re-prediction, which is close to every cycle. Whether that
cost is worth paying is an integration decision for later; this module
exists so that decision has real, tested code to start from instead of
just a diagram (see `make_spacetime_flood_figure.py` for the figures this
was built to produce, and the project memory/PROJECT_STATUS.md entry
dated 2026-09-08 for the cost discussion).

Handles both static obstacles (`env.clearance`, exactly like every other
flood/search in this codebase) and however many moving obstacles `env`
predicts (`env.predicted_moving_obs`, if present) -- more than
`spacetime.time_expanded_search` manages (see its own docstring: "only
ever selects ONE obstacle"), purely as a byproduct of computing free
space directly: floodig a full grid's "free" test is just "clear of
everything predicted to be here at this time", so it costs nothing extra
to check N obstacles instead of one, unlike a point-to-point search built
around a single predicted crossing.
"""
import heapq

import numpy as np


class SpaceTimeFlood:
    """Dijkstra flood of free `(x, y, t)` from `start_xy` at t=0, through a
    local window, avoiding static obstacles (`env.clearance`) and every
    moving obstacle `env` predicts. `env` needs only `.clearance(pts)`;
    `.predicted_moving_obs(times) -> (nk, N, 2)` is optional -- omitted
    (or reporting zero obstacles) degrades gracefully to a purely static
    flood, the same free-space definition `env.LocalFlowField` uses.

    `.G` is `(nx, ny, nk)`, `np.inf` where unreached. Layer `k` is
    `k * dt_layer` seconds from now. Grid/window conventions match
    `spacetime.time_expanded_search` exactly (asymmetric: `window` behind
    and to the sides of `start_xy`, `2 * window` ahead, since a robot
    already has a heading of travel) so results can be sanity-checked
    against it directly (see `tests/test_spacetime_flow.py`).
    """

    def __init__(self, env, start_xy, robot_r, obstacle_r=0.075,
                horizon=3.0, dt_layer=0.25, res=0.10, window=2.5,
                wait_cost=None):
        self.res, self.dt_layer = float(res), float(dt_layer)
        self.robot_r, self.obstacle_r = float(robot_r), float(obstacle_r)
        start_xy = np.asarray(start_xy, dtype=float)
        pad = 3 * res
        self.x0 = start_xy[0] - window - pad
        self.y0 = start_xy[1] - pad
        x1 = start_xy[0] + window + pad
        y1 = start_xy[1] + 2 * window + pad
        self.nx = int(np.ceil((x1 - self.x0) / res)) + 1
        self.ny = int(np.ceil((y1 - self.y0) / res)) + 1
        self.nk = int(round(horizon / dt_layer)) + 1
        self.times = np.arange(self.nk) * self.dt_layer
        wait_cost = res if wait_cost is None else float(wait_cost)

        gx, gy = np.meshgrid(self.x0 + np.arange(self.nx) * res,
                             self.y0 + np.arange(self.ny) * res, indexing="ij")
        pts = np.stack((gx, gy), axis=-1)
        static_free = env.clearance(pts) > robot_r

        predicted = None
        if hasattr(env, "predicted_moving_obs"):
            predicted = env.predicted_moving_obs(self.times)  # (nk, N, 2)

        dynamic_free_cache = {}

        def dynamic_free(k):
            if k in dynamic_free_cache:
                return dynamic_free_cache[k]
            if predicted is None or predicted.shape[1] == 0:
                result = np.ones_like(static_free)
            else:
                obstacles_k = predicted[k]  # (N, 2)
                d = np.linalg.norm(
                    pts[:, :, None, :] - obstacles_k[None, None, :, :],
                    axis=-1)  # (nx, ny, N)
                result = (d > (obstacle_r + robot_r)).all(axis=-1)
            dynamic_free_cache[k] = result
            return result

        def free(i, j, k):
            return bool(static_free[i, j] and dynamic_free(k)[i, j])

        def to_cell(xy):
            i = int(np.clip(round((xy[0] - self.x0) / res), 0, self.nx - 1))
            j = int(np.clip(round((xy[1] - self.y0) / res), 0, self.ny - 1))
            return i, j

        self.start_cell = to_cell(start_xy)
        nbrs = [(di, dj, res * np.hypot(di, dj))
                for di in (-1, 0, 1) for dj in (-1, 0, 1)]  # (0,0) is "wait"

        G = np.full((self.nx, self.ny, self.nk), np.inf)
        start_state = (*self.start_cell, 0)
        if free(*start_state):
            G[start_state] = 0.0
            pq = [(0.0, start_state)]
            visited = set()
            while pq:
                cost, state = heapq.heappop(pq)
                if state in visited:
                    continue
                visited.add(state)
                i, j, k = state
                if k >= self.nk - 1:
                    continue
                for di, dj, w in nbrs:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                        continue
                    if di != 0 and dj != 0 and not (
                            free(i + di, j, k + 1) and free(i, j + dj, k + 1)):
                        continue  # no corner-cutting, matches astar.py's rule
                    if not free(ni, nj, k + 1):
                        continue
                    step_cost = wait_cost if (di == 0 and dj == 0) else w
                    ncost = cost + step_cost
                    nstate = (ni, nj, k + 1)
                    if ncost < G[nstate]:
                        G[nstate] = ncost
                        heapq.heappush(pq, (ncost, nstate))
        self.G = G
        finite = np.isfinite(G)
        self.dmax = float(G[finite].max()) if finite.any() else 0.0

    @property
    def xs(self):
        return self.x0 + np.arange(self.nx) * self.res

    @property
    def ys(self):
        return self.y0 + np.arange(self.ny) * self.res

    def _cell(self, xy):
        i = int(np.clip(round((xy[0] - self.x0) / self.res), 0, self.nx - 1))
        j = int(np.clip(round((xy[1] - self.y0) / self.res), 0, self.ny - 1))
        return i, j

    def _layer(self, t):
        return int(np.clip(round(t / self.dt_layer), 0, self.nk - 1))

    def slice(self, k):
        """The `(nx, ny)` distance grid at time layer `k` -- one horizontal
        cut through the flood, the direct analogue of a single instant's
        `env.LocalFlowField.D`."""
        return self.G[:, :, k]

    def depth(self, k):
        """`slice(k)` normalized to `[0, 1]` (1 = at the robot's own
        start, `nan` where unreached) -- ready to hand to the same
        `_water_cmap()` colouring `visualization._draw_field` uses."""
        d = self.slice(k)
        return np.where(np.isfinite(d), 1.0 - np.clip(d / max(self.dmax, 1e-9), 0, 1),
                        np.nan)

    def reachable(self, k):
        """Boolean `(nx, ny)` mask at layer `k` -- the blob's own shape at
        that instant, the time-sliced analogue of `LocalFlowField.body`."""
        return np.isfinite(self.slice(k))

    def distance(self, xy, t):
        """Flooded distance from `start_xy` to `xy` at time `t` (nearest
        cell/layer); `inf` if unreached."""
        i, j = self._cell(xy)
        return float(self.G[i, j, self._layer(t)])

    def dist_batch(self, pts, times):
        """Vectorized `distance()` for a whole batch at once -- the form
        MPPI actually needs: `pts` is `(..., 2)`, `times` is `(...,)`
        broadcastable against `pts[..., 0]` (e.g. `pts` shaped `(K, T, 2)`
        against `times` shaped `(T,)`, one absolute time per horizon step,
        shared across all K samples). A per-point Python-loop call to
        `distance()` is not an option here -- a single MPPI cycle queries
        on the order of K*T points (tens of thousands).

        Unreached points return `dmax + 2.0`, matching
        `env.LocalFlowField.dist`'s convention: a large-but-finite penalty
        that stays usable inside a cost sum instead of poisoning it with
        `inf`.
        """
        pts = np.asarray(pts, dtype=float)
        times = np.asarray(times, dtype=float)
        i = np.clip(np.rint((pts[..., 0] - self.x0) / self.res).astype(int),
                   0, self.nx - 1)
        j = np.clip(np.rint((pts[..., 1] - self.y0) / self.res).astype(int),
                   0, self.ny - 1)
        k = np.clip(np.rint(times / self.dt_layer).astype(int), 0, self.nk - 1)
        d = self.G[i, j, k]
        return np.where(np.isfinite(d), d, self.dmax + 2.0)
