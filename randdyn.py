"""An open room with a CONTROLLABLE number of RANDOMLY moving obstacles.

Sandbox counterpart of the Gazebo `dyn*` / `d15_*` / `d20_*` worlds
(make_open_world.py): an 8 x 12.5 m room, the robot crosses it from (0, 1) to
(2.5, 11.3), N moving cylinders. `DynaBarnEnv` (dynabarn.py) is the other
dynamic testbed here and is very different: 4 slow obstacles oscillating on the
BARN corridors, radius 0.075. Every earlier sandbox result on space-time
topology (spacetime_flow / spacetime_modes matched panels) used that one.

Random motion, seeded (same seed -> identical scene, so paired comparisons of
controllers see exactly the same obstacles):

  motion="waypoint" (default) random-waypoint mobility. Each obstacle heads in a
      straight line to a random target in the room at a random speed, and picks a
      new target and speed on arrival. Velocity changes at every waypoint, so the
      controller's constant-velocity prediction is wrong for a while after each
      change -- the realistic hard case for a tracker-based predictor.
  motion="bounce" constant velocity in a random direction, reflecting off the
      walls. The constant-velocity predictor is exact between bounces.

Obstacles start and travel outside 1.2 m keep-out discs around the start and the
goal (same rule as make_open_world.py), so the start and goal are never blocked
by construction, but obstacles freely cross the route between them.

API-compatible with `DynaBarnEnv` where the controllers look (`moving_obs`,
`moving_velocity`, `predicted_moving_obs`, `predicted_clearance`, `clearance`,
`obstacle_centers`, `step_time`), plus `static_clearance` (the room walls only:
the static map a global costmap would hold) and `obstacle_r` (a real radius; all
moving obstacles share it, as the ground-truth footprint check takes one scalar).
"""
import numpy as np

from env import BarnEnv

KEEPOUT_R = 1.2


class RandomDynEnv(BarnEnv):
    def __init__(self, n_moving=10, seed=0, motion="waypoint",
                 speed_range=(0.15, 0.40), obstacle_r=0.225,
                 robot_r=0.34, room=(-4.0, 4.0, 12.5),
                 start=(0.0, 1.0), goal=(2.5, 11.3)):
        if motion not in ("waypoint", "bounce"):
            raise ValueError("motion must be 'waypoint' or 'bounce'")
        # Deliberately NOT BarnEnv.__init__: no BARN grid; an empty static map
        # (the room walls only).
        self.world, self.scale, self.robot_r = "open", 1.0, robot_r
        self.obs = np.zeros((0, 2))
        self.xmin, self.xmax = float(room[0]), float(room[1])
        self.ymin, self.ymax = 0.0, float(room[2])   # FlowField assumes y0 = 0
        self.start = np.array([start[0], start[1], np.pi / 2])
        self.goal = np.array(goal, dtype=float)
        self.goal_tol = 0.5
        self.obstacle_r = float(obstacle_r)
        self.motion = motion
        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.n_moving = int(n_moving)
        self.t = 0.0
        self._rng = np.random.default_rng(seed)
        self._keepouts = (self.start[:2].copy(), self.goal.copy())
        self._build_clearance_grid()
        self._spawn()

    # ---- static map: the room walls ------------------------------------
    def clearance_exact(self, pts):
        pts = np.asarray(pts, dtype=float)
        return np.minimum.reduce([
            pts[..., 0] - self.xmin, self.xmax - pts[..., 0],
            pts[..., 1] - self.ymin, self.ymax - pts[..., 1]])

    def static_clearance(self, pts):
        """Clearance to the static map only (BarnEnv's precomputed grid)."""
        return BarnEnv.clearance(self, pts)

    # ---- random motion ---------------------------------------------------
    def _lo_hi(self):
        m = self.obstacle_r + 0.05
        return (np.array([self.xmin + m, self.ymin + m]),
                np.array([self.xmax - m, self.ymax - m]))

    def _clear_of_keepouts(self, a, b=None):
        """True if the point `a` (or the segment a-b) stays outside every keep-out."""
        for c in self._keepouts:
            if b is None:
                d = float(np.linalg.norm(a - c))
            else:
                ab = b - a
                den = float(ab @ ab)
                s = 0.0 if den < 1e-12 else float(np.clip((c - a) @ ab / den, 0, 1))
                d = float(np.linalg.norm(a + s * ab - c))
            if d < KEEPOUT_R:
                return False
        return True

    def _new_leg(self, i):
        """Random target + speed for obstacle i (waypoint motion)."""
        lo, hi = self._lo_hi()
        a = self.pos[i]
        for _ in range(50):
            b = self._rng.uniform(lo, hi)
            if self._clear_of_keepouts(a, b) and np.linalg.norm(b - a) > 1.0:
                break
        else:
            b = a.copy()          # could not find a legal leg: stay put this leg
        speed = float(self._rng.uniform(*self.speed_range))
        self.target[i] = b
        self.speed[i] = speed
        d = b - a
        n = np.linalg.norm(d)
        self.vel[i] = (d / n * speed) if n > 1e-9 else 0.0

    def _spawn(self):
        n = self.n_moving
        lo, hi = self._lo_hi()
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.target = np.zeros((n, 2))
        self.speed = np.zeros(n)
        for i in range(n):
            for _ in range(200):
                p = self._rng.uniform(lo, hi)
                if self._clear_of_keepouts(p):
                    break
            self.pos[i] = p
            if self.motion == "waypoint":
                self._new_leg(i)
            else:
                th = self._rng.uniform(0, 2 * np.pi)
                self.speed[i] = self._rng.uniform(*self.speed_range)
                self.vel[i] = self.speed[i] * np.array([np.cos(th), np.sin(th)])

    def step_time(self, dt):
        self.t += dt
        if self.n_moving == 0:
            return
        lo, hi = self._lo_hi()
        if self.motion == "waypoint":
            for i in range(self.n_moving):
                step = self.speed[i] * dt
                to = self.target[i] - self.pos[i]
                dist = float(np.linalg.norm(to))
                if dist <= step:                 # arrived: pick the next leg
                    self.pos[i] = self.target[i]
                    self._new_leg(i)
                else:
                    self.pos[i] += to / dist * step
        else:
            self.pos += self.vel * dt
            for ax in (0, 1):
                low = self.pos[:, ax] < lo[ax]
                high = self.pos[:, ax] > hi[ax]
                self.pos[low, ax] = 2 * lo[ax] - self.pos[low, ax]
                self.pos[high, ax] = 2 * hi[ax] - self.pos[high, ax]
                self.vel[low | high, ax] *= -1.0

    # ---- DynaBarnEnv-compatible API -----------------------------------------
    def moving_obs(self, t=None):
        return self.pos.copy()

    def moving_velocity(self, t=None):
        return self.vel.copy()

    def predicted_moving_obs(self, time_offsets, horizon=2.0):
        """Constant-velocity prediction from the CURRENT observation (a tracker,
        not an oracle). Beyond `horizon` the last prediction is held."""
        tau = np.clip(np.asarray(time_offsets, dtype=float), 0.0, horizon)
        return self.pos[None, :, :] + tau[:, None, None] * self.vel[None, :, :]

    def predicted_clearance(self, pts, time_offsets, horizon=2.0,
                            uncertainty_rate=0.03):
        pts = np.asarray(pts, dtype=float)
        tau = np.asarray(time_offsets, dtype=float)
        if pts.shape[-3] != len(tau):
            raise ValueError("time_offsets must match the trajectory axis")
        d = self.static_clearance(pts)
        if self.n_moving == 0:
            return d
        predicted = self.predicted_moving_obs(tau, horizon=horizon)
        lead = (1,) * (pts.ndim - 3)
        centers = predicted.reshape(lead + (len(tau), 1) + predicted.shape[1:])
        radii = (self.obstacle_r + uncertainty_rate * tau).reshape(
            lead + (len(tau), 1, 1))
        dm = np.linalg.norm(pts[..., None, :] - centers, axis=-1)
        return np.minimum(d, (dm - radii).min(axis=-1))

    def clearance(self, pts):
        """Static map + a live check against the moving obstacles' CURRENT
        position (the local-costmap layer)."""
        d = self.static_clearance(pts)
        if self.n_moving:
            pts = np.asarray(pts, dtype=float)
            dm = np.linalg.norm(pts[..., None, :] - self.pos, axis=-1).min(-1)
            d = np.minimum(d, dm - self.obstacle_r)
        return d

    def obstacle_centers(self):
        """Moving obstacles' CURRENT centres + their shared radius, for the exact
        ground-truth footprint check."""
        return self.pos.copy(), self.obstacle_r
