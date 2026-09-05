"""A simple DynaBARN-style dynamic testbed: a BarnEnv with a handful of
obstacles that oscillate back and forth, layered on top of the static map.

Deliberately minimal, and deliberately built so the static/dynamic split
matters architecturally, not just as "more obstacles": `BarnEnv`'s
precomputed clearance grid (`_cgrid`) is built once from the STATIC
obstacles only and never touched here, matching a real nav2 stack's
*global* costmap. `clearance()` -- the query every controller actually
calls at runtime -- layers a cheap on-the-fly check against the moving
obstacles' CURRENT position on top of that static lookup, matching a
*local/dynamic* costmap layer refreshed from live sensing.

The consequence is real, not cosmetic: `flow` builds its FlowField ONCE
at construction and never rebuilds it, so it's flying on a stale snapshot
the instant anything moves -- it can end up scoring a route through where
an obstacle no longer is, or isn't, blind either way. `local`/`hybrid`
reflood every `reflood_every` steps from whatever `clearance()` reports
*right now*, so they see the obstacles' live position each time they
rebuild. That gap is the actual thing worth testing in a dynamic
environment, not just whether a controller can dodge a moving circle.
"""
import numpy as np

from env import BarnEnv, CYL_R


class DynaBarnEnv(BarnEnv):
    def __init__(self, world, scale=1.0, robot_r=0.30, n_moving=4,
                 speed=0.4, amplitude=0.6, seed=0):
        super().__init__(world, scale, robot_r)
        self.mov_speed_scalar = speed
        self.mov_amp_scalar = amplitude
        self.mov_center = np.zeros((0, 2))   # placement below checks the
        self.t = 0.0                         # static map only (none placed yet)
        rng = np.random.default_rng(seed)

        # candidate spots: open enough that oscillating doesn't immediately
        # clip a static obstacle or wall, AND far enough from the start and
        # goal that no obstacle can ever reach them even at the far end of
        # its swing -- the point of a dynamic-avoidance test is a robot
        # that successfully dodges a moving obstacle, not one that's
        # already in contact with it before it takes a single step.
        margin = robot_r + amplitude + 0.25
        safe_dist = amplitude + CYL_R + robot_r + 0.25
        ys = np.linspace(2.0, self.ymax - 2.0, 40) * 1.0
        xs = np.linspace(self.xmin + margin, self.xmax - margin, 12)
        cand = []
        for y in ys:
            for x in xs:
                pt = np.array([x, y])
                if np.linalg.norm(pt - self.start[:2]) < safe_dist:
                    continue
                if np.linalg.norm(pt - self.goal) < safe_dist:
                    continue
                if float(self.clearance(pt[None])[0]) > margin:
                    cand.append((x, y))
        cand = np.array(cand) if cand else np.zeros((0, 2))
        if len(cand) == 0:
            n_moving = 0
        else:
            idx = rng.choice(len(cand), size=min(n_moving, len(cand)),
                             replace=False)
            cand = cand[idx]

        self.mov_center = cand
        n = len(cand)
        self.mov_axis = rng.integers(0, 2, size=n)          # 0=x, 1=y sweep
        self.mov_amp = np.full(n, amplitude)
        self.mov_phase = rng.uniform(0, 2 * np.pi, size=n)
        self.mov_omega = np.full(n, speed)                   # rad/s

    def step_time(self, dt):
        self.t += dt

    def moving_obs(self, t=None):
        """(N, 2) current world positions of the moving obstacles."""
        if len(self.mov_center) == 0:
            return self.mov_center
        t = self.t if t is None else t
        off = self.mov_amp * np.sin(self.mov_omega * t + self.mov_phase)
        pos = self.mov_center.copy()
        pos[np.arange(len(pos)), self.mov_axis] += off
        return pos

    def moving_velocity(self, t=None):
        """Instantaneous measured velocity of each moving obstacle."""
        if len(self.mov_center) == 0:
            return self.mov_center.copy()
        t = self.t if t is None else t
        speed = (self.mov_amp * self.mov_omega
                 * np.cos(self.mov_omega * t + self.mov_phase))
        velocity = np.zeros_like(self.mov_center)
        velocity[np.arange(len(velocity)), self.mov_axis] = speed
        return velocity

    def predicted_moving_obs(self, time_offsets, horizon=2.0):
        """Constant-velocity prediction from the current observation.

        The extrapolation is deliberately not given the oscillator's future
        phase.  It therefore represents a controller-side tracker rather than
        an oracle.  Beyond ``horizon`` the last prediction is held.
        """
        tau = np.clip(np.asarray(time_offsets, dtype=float), 0.0, horizon)
        return (self.moving_obs()[None, :, :]
                + tau[:, None, None] * self.moving_velocity()[None, :, :])

    def predicted_clearance(self, pts, time_offsets, horizon=2.0,
                            uncertainty_rate=0.03):
        """Clearance against time-indexed constant-velocity predictions."""
        pts = np.asarray(pts, dtype=float)
        tau = np.asarray(time_offsets, dtype=float)
        if pts.shape[-3] != len(tau):
            raise ValueError("time_offsets must match the trajectory axis")
        d = super().clearance(pts)
        if not len(self.mov_center):
            return d
        predicted = self.predicted_moving_obs(tau, horizon=horizon)
        lead = (1,) * (pts.ndim - 3)
        centers = predicted.reshape(lead + (len(tau), 1)
                                    + predicted.shape[1:])
        radii = (CYL_R + uncertainty_rate * tau).reshape(
            lead + (len(tau), 1, 1))
        dm = np.linalg.norm(pts[..., None, :] - centers, axis=-1)
        return np.minimum(d, (dm - radii).min(axis=-1))

    def clearance(self, pts):
        """Static-grid lookup (unaware of motion) + a live check against
        the moving obstacles' CURRENT position -- the two-layer split a
        real local/global costmap pair has."""
        d = super().clearance(pts)
        mo = self.moving_obs()
        if len(mo):
            dm = np.linalg.norm(pts[..., None, :] - mo, axis=-1).min(-1) - CYL_R
            d = np.minimum(d, dm)
        return d

    def obstacle_centers(self):
        """Static cylinders plus the moving obstacles' CURRENT position,
        for an exact ground-truth footprint check (see `BarnEnv`)."""
        centers, radius = super().obstacle_centers()
        mo = self.moving_obs()
        if len(mo):
            centers = np.vstack((centers, mo))
        return centers, radius
