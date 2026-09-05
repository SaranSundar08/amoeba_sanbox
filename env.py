"""BARN worlds as pure-numpy 2D environments.

Same geometry as the Gazebo pipeline (verified transform):
    obstacle cell (row, col) -> x = row*0.15 - 4.425, y = col*0.15 + 5.175
    cylinders r = 0.075, corridor x in [-4.5, 0], travel +y,
    spawn (-2.25, 1.0, pi/2), field exit y ~ 9.53.
Scaling law matches scale_barn.py: positions * s, cylinder radius constant.
"""
import heapq
import os

import numpy as np

BARN = "/home/saran/robohouse_ws/src/BARN_dataset"
RES, X_OFF, Y_OFF, CYL_R = 0.15, -4.425, 5.175, 0.075


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BarnEnv:
    def __init__(self, world, scale=1.0, robot_r=0.30):
        self.world, self.scale, self.robot_r = world, scale, robot_r
        grid = np.load(os.path.join(BARN, "grid_files", f"grid_{world}.npy"))
        r, c = np.nonzero(grid)
        self.obs = np.stack([r * RES + X_OFF, c * RES + Y_OFF], 1) * scale
        self.xmin, self.xmax = -4.5 * scale, 0.0
        self.ymax = 10.6 * scale
        self.start = np.array([-2.25 * scale, 1.0 * scale, np.pi / 2])
        self.goal = np.array([-2.25 * scale, 10.0 * scale])
        self.goal_tol = 0.5
        self._build_clearance_grid()

    def clearance_exact(self, pts):
        """Distance from pts (..., 2) to the nearest obstacle *surface*
        (cylinders + the two side walls). Robot collides when < robot_r."""
        d = np.linalg.norm(pts[..., None, :] - self.obs, axis=-1).min(-1) - CYL_R
        walls = np.minimum(pts[..., 0] - self.xmin, self.xmax - pts[..., 0])
        return np.minimum(d, walls)

    def obstacle_centers(self):
        """(N, 2) cylinder centers plus their shared surface radius.

        Lets `RobotModel.exact_clearance` compute an analytic oriented-
        footprint-to-obstacle distance for ground-truth pass/fail checks,
        instead of the sampled/quantized approximations (`clearance()`)
        used inside the K x T MPPI rollout where speed matters more than
        exactness. `DynaBarnEnv` overrides this to add moving obstacles.
        """
        return self.obs, CYL_R

    def _build_clearance_grid(self, res=0.025):
        """Precomputed clearance field: rollouts do lookups instead of N-body
        distance math (error <= res/2 ~ 1 cm, plenty for the sandbox)."""
        self._cres = res
        xs = np.arange(self.xmin, self.xmax + res, res)
        ys = np.arange(0.0, self.ymax + res, res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._cgrid = self.clearance_exact(np.stack([gx, gy], -1))

    def clearance(self, pts):
        """Fast grid-lookup clearance for pts (..., 2)."""
        i = np.clip(np.rint((pts[..., 0] - self.xmin) / self._cres).astype(int),
                    0, self._cgrid.shape[0] - 1)
        j = np.clip(np.rint(pts[..., 1] / self._cres).astype(int),
                    0, self._cgrid.shape[1] - 1)
        return self._cgrid[i, j]


class FlowField:
    """The 'water level': geodesic distance-to-goal flooded through free space
    inflated by the robot radius, so the water only flows through gaps the
    robot actually fits in. Its descent direction has no local minima on any
    cell connected to the goal — this is the amoeba/water mechanism."""

    def __init__(self, env, res=0.05, viscosity=3.0, visc_band=0.4):
        self.res = res
        self.x0, self.y0 = env.xmin, 0.0
        nx = int(np.ceil((env.xmax - env.xmin) / res)) + 1
        ny = int(np.ceil((env.ymax - self.y0) / res)) + 1
        gx, gy = np.meshgrid(self.x0 + np.arange(nx) * res,
                             self.y0 + np.arange(ny) * res, indexing="ij")
        clear = env.clearance(np.stack([gx, gy], -1))
        self.free = clear > env.robot_r
        self._set_viscosity(clear, env.robot_r, viscosity, visc_band)

        D = np.full((nx, ny), np.inf)
        gi = (int(round((env.goal[0] - self.x0) / res)),
              int(round((env.goal[1] - self.y0) / res)))
        gi = self._nearest_free(gi)
        D[gi] = 0.0
        self._dijkstra(D, [(0.0, gi)])
        self.D = D
        self.dmax = np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 0.0
        self._build_dirs()

    def _set_viscosity(self, clear, robot_r, viscosity, visc_band):
        """Water physics: traversal resistance rises as the channel narrows.
        Multiplier is 1 in open space and (1 + viscosity) at robot-width;
        viscosity=0 recovers the old pure-geodesic flood."""
        self.visc = 1.0 + viscosity * np.clip(
            1.0 - (clear - robot_r) / visc_band, 0.0, 1.0)

    def _dijkstra(self, D, seeds):
        nx, ny = D.shape
        nbrs = [(di, dj, self.res * np.hypot(di, dj))
                for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        pq = list(seeds)
        heapq.heapify(pq)
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > D[i, j]:
                continue
            for di, dj, w in nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and self.free[ni, nj]:
                    nd = d + w * self.visc[ni, nj]
                    if nd < D[ni, nj]:
                        D[ni, nj] = nd
                        heapq.heappush(pq, (nd, (ni, nj)))

    def _nearest_free(self, gi, mask=None):
        m = self.free if mask is None else mask
        if m[gi]:
            return gi
        fi, fj = np.nonzero(m)
        k = np.argmin((fi - gi[0]) ** 2 + (fj - gi[1]) ** 2)
        return (fi[k], fj[k])

    def _geodesic_ball(self, mask, center_ij, cap):
        """Cells reachable from `center_ij` through `mask` within PLAIN
        (unweighted) geodesic distance `cap` -- the amoeba's actual body
        shape: it bulges into open rooms, pinches through narrow gaps, and
        can split into separate pseudopods around a pillar, unlike a
        Euclidean-radius disc or a rectangular window.

        Deliberately NOT viscosity-weighted, even though viscosity lives
        on this same instance: viscosity is a cost-shaping PREFERENCE for
        the goal-seeking flood (it should hug the middle of wide channels
        once inside the body), not a claim about how far the body can
        physically reach. Weighting the ball by viscosity compresses its
        effective forward reach in exactly the tight, high-resistance
        corridors where foresight matters most -- confirmed by regression:
        with viscosity-weighted reach, `local` timed out on world 48's
        S-pinch (previously 9/9); switching to plain distance for the
        ball alone restored it."""
        return self._geodesic_distances(mask, center_ij, cap) <= cap

    def _geodesic_distances(self, mask, center_ij, cap=np.inf):
        """Plain 8-connected distances from ``center_ij`` through ``mask``.

        LocalFlowField retains this array for V1 branch centreline tracing.
        Keeping it separate from the viscosity-weighted navigation value is
        important: this is physical reach, not route preference.
        """
        nx, ny = mask.shape
        c = self._nearest_free(center_ij, mask)
        Db = np.full((nx, ny), np.inf)
        Db[c] = 0.0
        nbrs = [(di, dj, self.res * np.hypot(di, dj))
                for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        pq = [(0.0, c)]
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > Db[i, j] or d > cap:
                continue
            for di, dj, w in nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    nd = d + w
                    if nd < Db[ni, nj] and nd <= cap:
                        Db[ni, nj] = nd
                        heapq.heappush(pq, (nd, (ni, nj)))
        return Db

    def grad_field(self):
        """Continuous downhill unit-vector field (nx, ny, 2): a central-
        difference gradient of the water level, smoother than the
        8-quantized `dirs` table -- meant for streamline-style rendering.
        Zero where dry/undefined. Cached per instance (the field is
        immutable once built)."""
        if getattr(self, "_gfield", None) is not None:
            return self._gfield
        Dm = np.where(np.isfinite(self.D), self.D, self.dmax + 2.0)
        gx = np.zeros_like(Dm)
        gy = np.zeros_like(Dm)
        gx[1:-1, :] = Dm[2:, :] - Dm[:-2, :]
        gx[0, :] = Dm[1, :] - Dm[0, :]
        gx[-1, :] = Dm[-1, :] - Dm[-2, :]
        gy[:, 1:-1] = Dm[:, 2:] - Dm[:, :-2]
        gy[:, 0] = Dm[:, 1] - Dm[:, 0]
        gy[:, -1] = Dm[:, -1] - Dm[:, -2]
        n = np.hypot(gx, gy)
        wet = np.isfinite(self.D) & (n > 1e-9)
        safe = np.where(n > 1e-9, n, 1.0)
        u = np.where(wet, -gx / safe, 0.0)
        v = np.where(wet, -gy / safe, 0.0)
        self._gfield = (u, v)
        return self._gfield

    def _build_dirs(self):
        """Per cell: unit vector toward the steepest-descent neighbor."""
        D = np.pad(self.D, 1, constant_values=np.inf)
        offs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)]
        stack = np.stack([D[1 + di:D.shape[0] - 1 + di,
                            1 + dj:D.shape[1] - 1 + dj] for di, dj in offs])
        best = np.argmin(stack, 0)
        vecs = np.array([(di / np.hypot(di, dj), dj / np.hypot(di, dj))
                         for di, dj in offs])
        self.dirs = vecs[best]
        dead = ~np.isfinite(self.D) | ~np.isfinite(stack.min(0))
        self.dirs[dead] = 0.0

    def _idx(self, pts):
        i = np.clip(np.rint((pts[..., 0] - self.x0) / self.res).astype(int),
                    0, self.D.shape[0] - 1)
        j = np.clip(np.rint((pts[..., 1] - self.y0) / self.res).astype(int),
                    0, self.D.shape[1] - 1)
        return i, j

    def dist(self, pts):
        """Water level at pts (..., 2); unreachable cells cost dmax + 2."""
        d = self.D[self._idx(pts)]
        return np.where(np.isfinite(d), d, self.dmax + 2.0)

    def direction(self, pts):
        """Downhill unit vectors (..., 2); zero where undefined. Quantized
        to the 8-neighbor steepest-descent table -- prefer
        `direction_smooth()` for steering; this flips in 45-degree steps
        cell to cell, which is exactly what causes bang-bang chatter when
        used as a continuous bias."""
        return self.dirs[self._idx(pts)]

    def _wet(self, pts):
        """True where pts land on a reachable (wet) cell of THIS field
        (in-body and connected to the seed), as opposed to `free` on the
        raw costmap -- a cell can be physically free but still outside
        the current window/body's reach."""
        i, j = self._idx(pts)
        return np.isfinite(self.D[i, j])

    def direction_smooth(self, pts):
        """Continuous downhill unit vector via central difference of the
        water level (..., 2); zero where undefined. Smooth in angle,
        unlike `direction()`'s 8-quantized table, so a bias built from
        this doesn't flip between 45-degree steps cell to cell -- the
        same fix already applied in the C++ port's gradAt().

        Only differences across a PAIR of wet cells: `dist()` returns a
        fixed dmax+2 penalty for anything outside the current body/
        window, so a naive central difference that happens to straddle
        that boundary sees a huge artificial cliff instead of the true
        field -- confirmed empirically as the actual cause of violent
        near-obstacle steering oscillation in `local`/`hybrid` (gradient
        magnitude spiking ~8x, from ~0.6 to ~4.8, exactly when the +-h
        stencil crossed the body's edge). `flow`'s field covers the whole
        map, so its query points are rarely this close to its own edge --
        which is exactly why the same bug never showed up there. An axis
        where either flanking sample is dry contributes zero rather than
        that cliff; if neither axis has two wet samples, there's no
        direction here at all."""
        h = 3.0 * self.res
        hx = np.zeros_like(pts); hx[..., 0] = h
        hy = np.zeros_like(pts); hy[..., 1] = h
        wet_xp, wet_xm = self._wet(pts + hx), self._wet(pts - hx)
        wet_yp, wet_ym = self._wet(pts + hy), self._wet(pts - hy)
        gx = np.where(wet_xp & wet_xm,
                      self.dist(pts + hx) - self.dist(pts - hx), 0.0)
        gy = np.where(wet_yp & wet_ym,
                      self.dist(pts + hy) - self.dist(pts - hy), 0.0)
        n = np.hypot(gx, gy)
        m = n > 1e-6
        safe = np.where(m, n, 1.0)
        out = np.zeros_like(pts)
        out[..., 0] = np.where(m, -gx / safe, 0.0)
        out[..., 1] = np.where(m, -gy / safe, 0.0)
        return out


class LocalFlowField(FlowField):
    """The amoeba *body*: a real geodesic blob around the robot, not a
    rectangular window. Body membership = cells reachable from the robot
    within `radius` of viscosity-weighted geodesic distance through free
    space (`_geodesic_ball`) — it bulges into open rooms, pinches through
    narrow gaps, and can split into separate pseudopods around a pillar,
    all from the same graph search used everywhere else in this file. The
    backing array is just the computational domain: `radius` + a small pad
    so the geodesic cap, not the array edge, is what limits the body.

    The water level (goal-seeking flood) then runs *inside* the body only.
    Boundary condition when the goal lies outside it: every body cell that
    borders further free space — the body's true *membrane*, where the
    geodesic budget ran out, not where a wall got in the way — is seeded
    with a promise: straight-line distance to the goal (path_seed=False),
    or dist-to-plan + remaining-plan-length (path_seed=True, a global
    planner demoted to a boundary hint). A wrong path only mis-ranks where
    the water enters; the flood itself still runs over the true local free
    space and never routes through a dry gap. The no-local-minima
    guarantee holds *within* the body; a cul-de-sac deeper than its reach
    can still look promising (radius = foresight knob)."""

    def __init__(self, env, center, radius=2.5, res=0.05,
                 path=None, path_rem=None, viscosity=1.5, visc_band=0.4):
        self.res = res
        pad = 3 * res
        self.x0 = max(env.xmin, center[0] - radius - pad)
        self.y0 = max(0.0, center[1] - radius - pad)
        x1 = min(env.xmax, center[0] + radius + pad)
        y1 = min(env.ymax, center[1] + radius + pad)
        nx = int(np.ceil((x1 - self.x0) / res)) + 1
        ny = int(np.ceil((y1 - self.y0) / res)) + 1
        gx, gy = np.meshgrid(self.x0 + np.arange(nx) * res,
                             self.y0 + np.arange(ny) * res, indexing="ij")
        clear = env.clearance(np.stack([gx, gy], -1))
        free_full = clear > env.robot_r
        self._set_viscosity(clear, env.robot_r, viscosity, visc_band)

        ri = int(np.clip(round((center[0] - self.x0) / res), 0, nx - 1))
        rj = int(np.clip(round((center[1] - self.y0) / res), 0, ny - 1))
        if free_full.any():
            self.reach_dist = self._geodesic_distances(
                free_full, (ri, rj), radius)
            body = self.reach_dist <= radius
        else:
            self.reach_dist = np.full_like(clear, np.inf, dtype=float)
            body = np.zeros_like(free_full)
        self.center_ij = self._nearest_free((ri, rj), body) if body.any() else (
            ri, rj)
        domain = free_full & body
        self.body = domain          # exposed for visualization (the blob)
        self.free = domain          # everything below floods only inside it

        D = np.full((nx, ny), np.inf)
        goal_inside = (self.x0 <= env.goal[0] <= self.x0 + (nx - 1) * res
                       and self.y0 <= env.goal[1] <= self.y0 + (ny - 1) * res)
        if goal_inside and domain.any():
            gi = (int(round((env.goal[0] - self.x0) / res)),
                  int(round((env.goal[1] - self.y0) / res)))
            gi = (int(np.clip(gi[0], 0, nx - 1)), int(np.clip(gi[1], 0, ny - 1)))
            gi = self._nearest_free(gi, domain)
            D[gi] = 0.0
            seeds = [(0.0, gi)]
        else:
            # True membrane: domain cells bordering free space the geodesic
            # budget didn't reach (not cells that merely border a wall).
            outside = free_full & ~domain
            padm = np.pad(outside, 1, constant_values=False)
            frontier = np.zeros_like(domain)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    frontier |= padm[1 + di:1 + di + nx, 1 + dj:1 + dj + ny]
            membrane = domain & frontier
            if not membrane.any():
                # body reached the array edge before its radius cap (rare)
                # -- fall back to the true boundary so the flood has an inlet
                edge = np.zeros((nx, ny), bool)
                edge[[0, -1], :] = True
                edge[:, [0, -1]] = True
                membrane = domain & edge
            self.membrane = membrane
            bi, bj = np.nonzero(membrane)
            bc = np.stack([self.x0 + bi * res, self.y0 + bj * res], 1)
            if path is None:
                promise = np.hypot(bc[:, 0] - env.goal[0],
                                   bc[:, 1] - env.goal[1])
            else:
                d = np.linalg.norm(bc[:, None, :] - path[None], axis=-1)
                promise = (d + path_rem[None]).min(1)
            seeds = []
            for i, j, d0 in zip(bi, bj, promise):
                D[i, j] = d0
                seeds.append((float(d0), (int(i), int(j))))
        if goal_inside and domain.any():
            self.membrane = np.zeros_like(domain)
        self._dijkstra(D, seeds)
        self.D = D
        self.dmax = np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 0.0
        self._build_dirs()
