"""Space-time blob as a guidance layer of the sandbox MPPI (mirror of the C++
Phase 1 / Phase 2 in nav2_tgmppi_controller, branch space-time-blob).

`BlobPlanner.propose()` runs once per control step, from `MPPI.build_branch_
proposals`. It floods (x, y, t) with ALL predicted obstacles (spacetime_body.py) and
also without obstacles; the difference in the best route's cost is how much the
moving obstacles matter right now. While it is at least `gate` metres (hysteresis:
until it falls below gate/2):

  mode "extras"  the best route and the best route of a DIFFERENT homotopy class are
                 added as 2 extra modes next to the static pseudopods (branch ids
                 1000, 1001)                                            [Phase 1]
  mode "pods"    the blob's routes REPLACE the static pseudopod modes (branch id
                 2000 + a tracked id: time-aligned route overlap)       [Phase 2]

Otherwise the static pseudopods are returned unchanged. Routes become control
sequences through `spacetime.spacetime_path_to_proposal` (resample in time, project
through the accel limits, roll out the real robot model), so what MPPI samples around
is what the robot can actually execute; feasibility against the MOVING obstacles is
then checked here against ALL of them at their predicted positions (the older helper
takes a single obstacle) and against the static map only (the exact-footprint check
must not see the obstacles' current positions: a route may legitimately pass where an
obstacle is now, just not when it is there).
"""
import dataclasses

import numpy as np

from env import BarnEnv, CYL_R
from spacetime import spacetime_path_to_proposal
from spacetime_body import SpaceTimeBody

EXTRAS_ID, PODS_ID = 1000, 2000


class _StaticView:
    """`env` as a global costmap sees it: no moving obstacles."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def obstacle_centers(self):
        return np.zeros((0, 2)), getattr(self._env, "obstacle_r", CYL_R)

    def clearance(self, pts):
        f = getattr(self._env, "static_clearance", None)
        return f(pts) if f is not None else BarnEnv.clearance(self._env, pts)


class _RouteTracker:
    """Stable ids for routes across cycles: greedy best-unused match by 0.5 * mean
    time-aligned distance + 0.5 * end distance, under `match_distance`."""

    def __init__(self, match_distance=0.5):
        self.match_distance = match_distance
        self.prev = []            # [(id, path)]
        self.next_id = 0

    def reset(self):
        self.prev = []

    def update(self, paths):
        used, out = set(), []
        for path in paths:
            best, best_m = None, np.inf
            for j, (pid, old) in enumerate(self.prev):
                if j in used or old.shape != path.shape:
                    continue
                m = (0.5 * np.linalg.norm(path - old, axis=1).mean()
                     + 0.5 * np.linalg.norm(path[-1] - old[-1]))
                if m < best_m:
                    best, best_m = j, m
            if best is not None and best_m <= self.match_distance:
                used.add(best)
                out.append(self.prev[best][0])
            else:
                out.append(self.next_id)
                self.next_id += 1
        self.prev = list(zip(out, paths))
        return out


class BlobPlanner:
    def __init__(self, mode="pods", gate=0.2, margin=0.10, max_regret=1.0,
                 horizon=3.0, dt_layer=0.25, res=0.10, match_distance=0.5):
        if mode not in ("extras", "pods"):
            raise ValueError("mode must be 'extras' or 'pods'")
        self.mode, self.gate, self.margin, self.max_regret = mode, gate, margin, max_regret
        self.horizon, self.dt_layer, self.res = horizon, dt_layer, res
        self.tracker = _RouteTracker(match_distance)
        self.active = False
        # diagnostics
        self.cycles = 0
        self.active_cycles = 0
        self.routes_built = 0
        self.routes_infeasible = 0
        self.build_ms = []
        self.last_delay = 0.0
        self.last_classes = 0
        # presentation-only snapshot of the most recent cycle (read by the animator; never
        # used by the controller): {'body', 'active', 'delay', 'chosen': [(branch_id, path)]}
        self.last = None

    # ------------------------------------------------------------------
    def propose(self, ctrl, state, base):
        env, rm = ctrl.env, ctrl.robot_model
        if not hasattr(env, "moving_obs") or ctrl.flow is None:
            return base
        self.cycles += 1
        self.last = None
        pos = env.moving_obs()
        if len(pos) == 0:
            self._deactivate()
            return base
        vel = env.moving_velocity()
        rad = getattr(env, "obstacle_r", CYL_R)
        static = getattr(env, "static_clearance", None) or (lambda p: BarnEnv.clearance(env, p))
        plan_r = rm.planning_radius
        static_free = lambda pts: static(pts) > plan_r
        terminal = lambda pts: ctrl.flow.dist(pts)
        kw = dict(horizon=self.horizon, dt_layer=self.dt_layer, res=self.res)

        body = SpaceTimeBody(static_free, terminal, state[:2], (pos, vel, rad),
                             robot_r=plan_r + self.margin, max_regret=self.max_regret,
                             max_pods=3, **kw)
        free = SpaceTimeBody(static_free, terminal, state[:2], None,
                             robot_r=plan_r + self.margin, max_pods=1, **kw)
        self.build_ms.append(body.build_ms + free.build_ms)
        self.last_classes = body.classes_found
        if not body.ready or not body.pods:
            self._deactivate()
            return base
        self.last_delay = (body.promises[0] - free.promises[0]) if free.ready else 0.0
        self.active = (self.last_delay >= (0.5 if self.active else 1.0) * self.gate)
        self.last = dict(body=body, active=self.active, delay=self.last_delay, chosen=[])
        if not self.active:
            self.tracker.reset()
            return base
        self.active_cycles += 1

        if self.mode == "extras":
            alt = next((i for i in range(1, len(body.pods))
                        if body.signatures[i] != body.signatures[0]), None)
            if alt is None:
                return base
            chosen = [(EXTRAS_ID, body.pods[0]), (EXTRAS_ID + 1, body.pods[alt])]
            keep_base = True
            self.last["chosen"] = chosen
        else:
            ids = self.tracker.update(body.pods)
            chosen = [(PODS_ID + i, p) for i, p in zip(ids, body.pods)]
            keep_base = False
            self.last["chosen"] = chosen

        props = []
        for bid, path in chosen:
            self.routes_built += 1
            prop = self._to_proposal(ctrl, state, path, bid)
            if prop.feasible:
                props.append(prop)
            else:
                self.routes_infeasible += 1
                self.last.setdefault("rejected", []).append(bid)
        if keep_base:
            return list(base) + props
        return props if props else base

    def _deactivate(self):
        self.active = False
        self.tracker.reset()

    def _to_proposal(self, ctrl, state, path, branch_id):
        env, rm = ctrl.env, ctrl.robot_model
        far = np.array([1.0e6, 1.0e6])
        prop = spacetime_path_to_proposal(
            path, self.dt_layer, state, env.goal, _StaticView(env), rm,
            lambda t: far, 0.0, T=ctrl.T, dt=ctrl.dt, v_max=ctrl.v_max, w_max=ctrl.w_max,
            v_accel_max=ctrl.v_accel_max, w_accel_max=ctrl.w_accel_max,
            initial_control=ctrl.last_applied, branch_id=branch_id,
            min_progress=-1.0e9)          # a wait is a legitimate route: no progress demand
        # all moving obstacles, at their predicted positions, along the ACHIEVED rollout
        times = (np.arange(ctrl.T) + 1.0) * ctrl.dt
        pred = env.predicted_moving_obs(times, horizon=self.horizon)        # (T, N, 2)
        d = np.linalg.norm(prop.rollout[:, None, :2] - pred, axis=-1)
        dyn = float((d - getattr(env, "obstacle_r", CYL_R) - rm.radius).min())
        feasible = prop.feasible and dyn >= 0.0
        reason = prop.reason if dyn >= 0.0 else "moving-obstacle clearance"
        return dataclasses.replace(
            prop, feasible=feasible, min_clearance=min(prop.min_clearance, dyn),
            reason=reason if not feasible else "ok")
