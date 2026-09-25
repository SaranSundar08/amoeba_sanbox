"""The geodesic blob lifted into (x, y, t): sandbox reference for the C++
`nav2_tgmppi_controller/tools/space_time_body.{hpp,cpp}` (branch space-time-blob).

BODY      free (x, y, t) states reachable from (robot, t=0) at grid speed
          res / dt_layer (one cell per layer), avoiding the static map AND every
          predicted moving obstacle -- ALL of them in one flood, no trigger and no
          single-crossing budget (contrast spacetime.py's two-route search).
MEMBRANE  the reachable states on the last time layer (the light-cone rim).
PROMISE   at a membrane exit: space-time cost G + w * D_static(exit), where
          D_static is the static water level (`LocalFlowField.dist`) -- the
          cost-to-go beyond the horizon, so G + D is the exact cost-to-go through
          the body.
PSEUDOPODS min-cost (x, y, t) routes to exits, distinct by HOMOTOPY CLASS (a
          winding-number signature per obstacle, T-MPC Appendix B), then by exit
          separation.

Every edge advances one layer, so the flood is a layered DP over a DAG: one
shifted-array minimum per layer, no priority queue. Unlike `spacetime_flow.
SpaceTimeFlood` (a cost field over the whole window, opt-in, neutral in the
matched panel) this returns ROUTES, and it is what the C++ port implements.
"""
import time

import numpy as np

INF = 1.0e9
PASS_THRESHOLD = 1.0 / (4.0 * np.pi)     # T-MPC's default: a quarter turn
_OFFS = [(o // 3 - 1, o % 3 - 1) for o in range(9)]     # index 4 = wait


class SpaceTimeBody:
    def __init__(self, static_free, terminal, start_xy, obstacles=None,
                 horizon=3.0, dt_layer=0.25, res=0.10, robot_r=0.34,
                 terminal_weight=1.0, wait_epsilon=0.01,
                 class_min_separation=0.3, exit_separation=0.65,
                 max_regret=2.0, max_pods=3):
        """static_free(pts (M, 2)) -> bool (M,); terminal(pts (M, 2)) -> (M,);
        obstacles: (pos (N, 2), vel (N, 2), radius (N,) or scalar) or None.
        `robot_r` is the clearance radius kept from each obstacle: it must match
        what the rollout critic will accept (see the C++ header), not just the
        inscribed radius."""
        t0 = time.perf_counter()
        self.horizon, self.dt_layer, self.res = float(horizon), float(dt_layer), float(res)
        K = max(1, int(round(horizon / dt_layer)))
        n = 2 * K + 1                        # the light cone
        self.K, self.n = K, n
        start = np.asarray(start_xy, dtype=float)
        self.x0, self.y0 = start[0] - K * res, start[1] - K * res
        ax = self.x0 + np.arange(n) * res
        ay = self.y0 + np.arange(n) * res
        self.X, self.Y = np.meshgrid(ax, ay, indexing="ij")
        cells = np.stack([self.X, self.Y], -1).reshape(-1, 2)

        pos, vel, rad = self._obstacles(obstacles)
        self.pos, self.vel, self.rad = pos, vel, rad
        # obstacles that can block any cell of the grid at any layer
        reach = K * res * 1.4143 + rad + robot_r + res
        times = np.arange(K + 1) * dt_layer
        pred = pos[None] + times[:, None, None] * vel[None]       # (K+1, N, 2)
        keep = (np.linalg.norm(pred - start, axis=-1) <= reach[None]).any(0) \
            if len(pos) else np.zeros(0, bool)
        self.pos, self.vel, self.rad = pos[keep], vel[keep], rad[keep]
        pred = pred[:, keep]                                       # (K+1, R, 2)
        # per-obstacle clearance, relaxed for one already inside it at t = 0
        d0 = np.linalg.norm(self.pos - start, axis=-1) if len(self.pos) else np.zeros(0)
        self.clear = np.minimum(self.rad + robot_r, np.maximum(self.rad, d0 - 0.02))

        sfree = np.asarray(static_free(cells), bool).reshape(n, n)

        G = np.full((K + 1, n, n), INF)
        parent = np.full((K + 1, n, n), -1, dtype=np.int8)
        G[0, K, K] = 0.0                     # the robot's own state is always the root
        steps = np.array([wait_epsilon if o == (0, 0) else res * np.hypot(*o) for o in _OFFS])
        for k in range(K):
            dfree = sfree.copy()
            if len(self.pos):
                d = np.linalg.norm(
                    np.stack([self.X, self.Y], -1)[:, :, None, :] - pred[k + 1][None, None],
                    axis=-1)                                       # (n, n, R)
                dfree &= ~(d <= self.clear[None, None]).any(-1)
            Gp = np.pad(G[k], 1, constant_values=INF)
            cand = np.stack([Gp[1 - di:1 - di + n, 1 - dj:1 - dj + n] + steps[o]
                             for o, (di, dj) in enumerate(_OFFS)])   # (9, n, n)
            best = cand.argmin(0)
            g = np.take_along_axis(cand, best[None], 0)[0]
            ok = dfree & (g < INF)
            G[k + 1] = np.where(ok, g, INF)
            parent[k + 1] = np.where(ok, best, -1).astype(np.int8)
        self.G, self.parent = G, parent
        self.body_states = int((G < INF).sum())

        self.pods, self.promises, self.signatures = [], [], []
        self.classes_found = 0
        self.membrane_cells = 0
        gK = G[K]
        ii, jj = np.nonzero(gK < INF)
        self.membrane_cells = len(ii)
        self.ready = len(ii) > 0
        if self.ready:
            term = np.asarray(terminal(np.stack([self.X[ii, jj], self.Y[ii, jj]], -1)), float)
            prom = gK[ii, jj] + terminal_weight * term
            order = np.argsort(prom, kind="stable")
            best_p = prom[order[0]]
            pool = []
            for idx in order:
                if prom[idx] > best_p + max_regret or len(pool) >= 256:
                    break
                path = self._reconstruct(int(ii[idx]), int(jj[idx]))
                pool.append((float(prom[idx]), int(ii[idx]), int(jj[idx]), path,
                             self._signature(path, pred)))
            self.classes_found = len({tuple(p[4]) for p in pool})
            self._select(pool, class_min_separation, exit_separation, max_pods)
        self.build_ms = (time.perf_counter() - t0) * 1e3

    @staticmethod
    def _obstacles(obstacles):
        if obstacles is None:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        pos, vel, rad = obstacles
        pos = np.asarray(pos, float).reshape(-1, 2)
        vel = np.asarray(vel, float).reshape(-1, 2)
        rad = np.broadcast_to(np.asarray(rad, float), (len(pos),)).copy()
        return pos, vel, rad

    def _reconstruct(self, i, j):
        path = np.empty((self.K + 1, 2))
        for k in range(self.K, -1, -1):
            path[k] = (self.X[i, j], self.Y[i, j])
            if k == 0:
                break
            di, dj = _OFFS[int(self.parent[k, i, j])]
            i, j = i - di, j - dj
        return path

    def _signature(self, path, pred):
        """Per obstacle: -1/0/+1 = sign of the winding number if |lambda| >= 1/(4 pi)."""
        if not len(self.pos):
            return ()
        rel = path[:, None, :] - pred                              # (K+1, R, 2)
        th = np.arctan2(rel[..., 1], rel[..., 0])
        d = np.diff(th, axis=0)
        d = (d + np.pi) % (2 * np.pi) - np.pi                      # wrap to [-pi, pi)
        lam = d.sum(0) / (2 * np.pi)
        return tuple(np.where(np.abs(lam) < PASS_THRESHOLD, 0, np.sign(lam)).astype(int))

    def _select(self, pool, class_sep, exit_sep, max_pods):
        accepted, taken, used = [], set(), set()

        def sep(c):
            return min((np.hypot(c[1] - a[1], c[2] - a[2]) * self.res for a in accepted),
                       default=INF)

        def accept(idx):
            taken.add(idx)
            c = pool[idx]
            accepted.append(c)
            self.pods.append(c[3])
            self.promises.append(c[0])
            self.signatures.append(c[4])

        for idx, c in enumerate(pool):                 # best exit, then new classes
            if len(accepted) >= max_pods:
                break
            if c[4] in used or (accepted and sep(c) < class_sep):
                continue
            used.add(c[4])
            accept(idx)
        for idx, c in enumerate(pool):                 # fill by exit separation
            if len(accepted) >= max_pods:
                break
            if idx not in taken and sep(c) >= exit_sep:
                accept(idx)
