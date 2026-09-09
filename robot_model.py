"""Shared ideal and analytical skid-steer robot models for V6."""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class RobotModel:
    """Motion, footprint, and safety model used by planning and simulation.

    The skid-steer parameters are analytical placeholders until identified
    from SLIP command/odometry data. ``kind='ideal'`` preserves the original
    circular differential-drive sandbox behavior.
    """

    kind: str = "ideal"
    radius: float = 0.30
    # Matches the Nav2 costmap footprint in susag_nav2/param/navigation_
    # amoeba*.yaml ([[0.42, 0.34], ...] = 0.84 x 0.68 m), so the sandbox
    # and the Gazebo/Nav2 port check collisions against the same rectangle.
    # Banked SLIP results before 2026-09-09 were produced at the earlier
    # provisional 0.90 x 0.65 m (see docs/PROJECT_STATUS.md, that date).
    footprint_length: float = 0.84
    footprint_width: float = 0.68
    yaw_gain: float = 0.82
    speed_yaw_loss: float = 0.55
    turn_margin: float = 0.08
    speed_margin: float = 0.03
    v_max: float = 0.5
    w_max: float = 1.9

    def __post_init__(self):
        if self.kind not in ("ideal", "slip"):
            raise ValueError("robot model must be 'ideal' or 'slip'")
        positive = (self.radius, self.footprint_length,
                    self.footprint_width, self.yaw_gain,
                    self.v_max, self.w_max)
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("robot dimensions and gains must be positive")
        if self.speed_yaw_loss < 0.0 or self.turn_margin < 0.0:
            raise ValueError("slip loss and margins must be nonnegative")

    @property
    def is_slip(self):
        return self.kind == "slip"

    @property
    def planning_radius(self):
        if not self.is_slip:
            return self.radius
        # An orientation-independent circumcircle is safe but makes all
        # current BARN worlds unreachable for a long skid-steer chassis.
        # Width plus turn margin is the conservative 2-D planning proxy;
        # oriented rollout checks still enforce the full rectangle.
        return float(0.5 * self.footprint_width + self.turn_margin)

    def effective_omega(self, controls):
        controls = np.asarray(controls, dtype=float)
        omega = controls[..., 1]
        if not self.is_slip:
            return omega
        speed_fraction = np.abs(controls[..., 0]) / self.v_max
        return (self.yaw_gain * omega
                / (1.0 + self.speed_yaw_loss * speed_fraction))

    def integrate(self, states, controls, dt):
        """Advance arbitrary matching state/control batches by one step."""
        x = np.asarray(states, dtype=float).copy()
        controls = np.asarray(controls, dtype=float)
        omega = self.effective_omega(controls)
        x[..., 2] += omega * dt
        x[..., 0] += controls[..., 0] * np.cos(x[..., 2]) * dt
        x[..., 1] += controls[..., 0] * np.sin(x[..., 2]) * dt
        return x

    def footprint_points(self, states):
        """Return oriented footprint samples with shape ``(..., P, 2)``."""
        states = np.asarray(states, dtype=float)
        if not self.is_slip:
            return states[..., None, :2]
        half_l = 0.5 * self.footprint_length
        half_w = 0.5 * self.footprint_width
        # Boundary plus centre samples catch walls, corners, and cylinders
        # inside the rectangle without an expensive polygon distance kernel.
        longitudinal = np.linspace(-half_l, half_l, 5)
        lateral = np.linspace(-half_w, half_w, 5)
        offsets = np.unique(np.vstack((
            np.column_stack((longitudinal, np.full(5, -half_w))),
            np.column_stack((longitudinal, np.full(5, half_w))),
            np.column_stack((np.full(3, -half_l), lateral[1:-1])),
            np.column_stack((np.full(3, half_l), lateral[1:-1])),
            np.array([[0.0, 0.0]]),
        )), axis=0)
        c, s = np.cos(states[..., 2]), np.sin(states[..., 2])
        ox, oy = offsets[:, 0], offsets[:, 1]
        world_x = (states[..., 0, None] + c[..., None] * ox
                   - s[..., None] * oy)
        world_y = (states[..., 1, None] + s[..., None] * ox
                   + c[..., None] * oy)
        return np.stack((world_x, world_y), axis=-1)

    def clearance(self, env, states, time_offsets=None,
                  prediction_horizon=2.0, uncertainty_rate=0.03):
        """Signed footprint clearance; collision occurs below zero."""
        points = self.footprint_points(states)
        if time_offsets is not None and hasattr(env, "predicted_clearance"):
            clearance = np.asarray(env.predicted_clearance(
                points, time_offsets, horizon=prediction_horizon,
                uncertainty_rate=uncertainty_rate), dtype=float)
        else:
            clearance = np.asarray(env.clearance(points), dtype=float)
        if self.is_slip:
            return clearance.min(axis=-1)
        return clearance[..., 0] - self.radius

    def exact_clearance(self, env, states):
        """Authoritative, non-sampled footprint clearance for pass/fail
        checks -- executed-trajectory collision (`simulation.episode`) and
        branch feasibility (`proposals.branch_to_control_sequence`) -- not
        for the K x T MPPI rollout, where `clearance()`'s fast grid/sampled
        approximation is a deliberate, cheaper trade for a soft cost signal.

        `clearance()`'s SLIP path samples a fixed 17 points on the oriented
        footprint boundary. On the 0.84 m long edges those samples are
        0.21 m apart, so a BARN cylinder (radius 0.075 m) can sit up to
        about 0.06 m inside the true rectangle, exactly between two
        samples, and still read as clear. This instead computes the exact
        analytic point-to-oriented-rectangle distance against every real
        obstacle, so nothing can hide between samples.

        Falls back to `clearance()` when `env` doesn't expose
        `obstacle_centers()` (e.g. the synthetic single-circle test
        environments), so it degrades to the previous behavior rather than
        silently claiming an exactness it can't deliver.
        """
        if not hasattr(env, "obstacle_centers"):
            return self.clearance(env, states)
        states = np.asarray(states, dtype=float)
        centers, obstacle_r = env.obstacle_centers()
        obstacle_clear = self.circle_set_clearance(states, centers, obstacle_r)

        if not self.is_slip:
            wall_clear = np.minimum(
                states[..., 0] - env.xmin, env.xmax - states[..., 0])
            return np.minimum(obstacle_clear, wall_clear) - self.radius

        half_l, half_w = 0.5 * self.footprint_length, 0.5 * self.footprint_width
        c, s = np.cos(states[..., 2]), np.sin(states[..., 2])
        corner_x = np.stack([
            states[..., 0] + c * half_l - s * half_w,
            states[..., 0] + c * half_l + s * half_w,
            states[..., 0] - c * half_l - s * half_w,
            states[..., 0] - c * half_l + s * half_w,
        ], axis=-1)
        rect_wall_clear = np.minimum(
            corner_x.min(-1) - env.xmin, env.xmax - corner_x.max(-1))
        return np.minimum(obstacle_clear, rect_wall_clear)

    def circle_set_clearance(self, states, centers, obstacle_r):
        """Exact footprint-to-circle-set clearance, no walls.

        The obstacle-only half of `exact_clearance`, factored out so callers
        with their own (e.g. non-circular) static geometry -- such as
        `junction_env.ScriptedJunctionEnv`'s rectangular corridor walls --
        can still get an exact check against a set of circular obstacles
        (typically a scripted peer robot) without going through `env`.
        `centers` is `(N, 2)`; an empty set returns `+inf` everywhere.
        """
        states = np.asarray(states, dtype=float)
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        if not len(centers):
            return np.full(states.shape[:-1], np.inf)

        if not self.is_slip:
            d = np.linalg.norm(
                states[..., None, :2] - centers, axis=-1).min(-1)
            return d - obstacle_r

        half_l, half_w = 0.5 * self.footprint_length, 0.5 * self.footprint_width
        c, s = np.cos(states[..., 2]), np.sin(states[..., 2])
        # Exact point-to-oriented-rectangle distance for every obstacle
        # centre, in the rectangle's own frame: clamp into the box, measure
        # the miss. A centre already inside the box (clamp = itself) is a
        # true overlap, reported as the negative distance to the nearest
        # edge rather than the misleading 0 a plain clamp-distance gives.
        dx = centers[:, 0] - states[..., 0, None]
        dy = centers[:, 1] - states[..., 1, None]
        lx = dx * c[..., None] + dy * s[..., None]
        ly = -dx * s[..., None] + dy * c[..., None]
        clx = np.clip(lx, -half_l, half_l)
        cly = np.clip(ly, -half_w, half_w)
        dist = np.hypot(lx - clx, ly - cly)
        inside = (np.abs(lx) <= half_l) & (np.abs(ly) <= half_w)
        penetration = np.minimum(half_l - np.abs(lx), half_w - np.abs(ly))
        signed = np.where(inside, -penetration, dist)
        return signed.min(-1) - obstacle_r

    def safety_margin(self, controls, base_margin):
        """Turn/speed-dependent soft clearance margin."""
        controls = np.asarray(controls, dtype=float)
        if not self.is_slip:
            return np.full(controls.shape[:-1], float(base_margin))
        return float(base_margin) + self.extra_safety_margin(controls)

    def extra_safety_margin(self, controls):
        """Hard V6 allowance for slip/turn tracking uncertainty."""
        controls = np.asarray(controls, dtype=float)
        if not self.is_slip:
            return np.zeros(controls.shape[:-1], dtype=float)
        turn = np.abs(self.effective_omega(controls)) / self.w_max
        speed = np.abs(controls[..., 0]) / self.v_max
        return self.turn_margin * turn + self.speed_margin * speed


def sample_plant_mismatch(base, rng, yaw_gain_range=(0.7, 1.3),
                          speed_yaw_loss_range=(0.7, 1.3)):
    """Draw a plant model with unmodeled skid-steer behavior the
    controller does NOT know about, for `simulation.episode(...,
    plant_model=...)`.

    `base` (normally `ctrl.robot_model`) stays the controller's planning
    assumption; the returned model independently scales the two slip
    parameters the controller cannot observe or calibrate online --
    `yaw_gain` and `speed_yaw_loss` -- by a factor drawn uniformly from
    each range. `ctrl.robot_model` itself is left untouched (frozen
    dataclasses are immutable), so a matched pair of episodes can compare
    `plant_model=None` (self-consistent, the previous default) against
    `plant_model=sample_plant_mismatch(ctrl.robot_model, rng)` (genuine
    plan/plant mismatch) with everything else identical. Only meaningful
    for `kind='slip'`; an 'ideal' base has no slip parameters to jitter and
    is returned unchanged.
    """
    if base.kind != "slip":
        return base
    yaw_gain = base.yaw_gain * rng.uniform(*yaw_gain_range)
    speed_yaw_loss = base.speed_yaw_loss * rng.uniform(*speed_yaw_loss_range)
    return replace(base, yaw_gain=yaw_gain, speed_yaw_loss=speed_yaw_loss)
