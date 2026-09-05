"""Synthetic V7.1 junctions with one reproducible scripted peer robot."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JunctionScenario:
    name: str
    peer_path: np.ndarray
    peer_speed: float
    peer_delay: float
    geometry: str = "plus"


SCENARIOS = {
    "crossing": JunctionScenario(
        "crossing", np.array([[-3.0, 4.0], [3.0, 4.0]]), 0.60, 1.0),
    "blocked_split": JunctionScenario(
        "blocked_split", np.array([[-0.90, 7.0], [-0.90, 4.0]]),
        0.50, 0.0, "split"),
    "merge": JunctionScenario(
        "merge", np.array([[-3.0, 4.0], [0.0, 4.0], [0.0, 9.0]]),
        0.50, 0.0, "tee"),
}


class ScriptedJunctionEnv:
    """Plus/T junction whose dynamic obstacle has robot-scale geometry."""

    def __init__(self, scenario="crossing", robot_r=0.30, peer_r=0.30,
                 corridor_width=1.50, clearance_resolution=0.025):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown junction scenario: {scenario}")
        self.scenario = SCENARIOS[scenario]
        self.world = f"junction_{scenario}"
        self.scale = 1.0
        self.robot_r, self.peer_r = robot_r, peer_r
        self.xmin, self.xmax = -4.0, 4.0
        self.ymax = 8.0
        self.start = np.array([0.0, 0.8, np.pi / 2])
        self.goal = np.array([0.0, 7.2])
        self.goal_tol = 0.35
        self.t = 0.0
        self.dynamic_enabled = True
        self._cres = clearance_resolution
        h, yc = 0.5 * corridor_width, 4.0
        if self.scenario.geometry == "split":
            split_half = 1.55
            self.rectangles = [
                (self.xmin, -split_half, 0.0, self.ymax),
                (split_half, self.xmax, 0.0, self.ymax),
                (-0.35, 0.35, 3.15, 4.85),
            ]
        else:
            self.rectangles = [
                (self.xmin, -h, 0.0, yc - h),
                (h, self.xmax, 0.0, yc - h),
                (self.xmin, -h, yc + h, self.ymax),
                (h, self.xmax, yc + h, self.ymax),
            ]
            if self.scenario.geometry == "tee":
                self.rectangles.append((h, self.xmax, yc - h, yc + h))
        self._build_clearance_grid()
        path = self.scenario.peer_path
        self._segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self._cumulative = np.concatenate(([0.0], np.cumsum(self._segments)))

    @staticmethod
    def _rectangle_clearance(points, rectangle):
        xmin, xmax, ymin, ymax = rectangle
        x, y = points[..., 0], points[..., 1]
        dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
        dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
        outside = np.hypot(dx, dy)
        inside = ((x >= xmin) & (x <= xmax)
                  & (y >= ymin) & (y <= ymax))
        depth = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))
        return np.where(inside, -depth, outside)

    def clearance_exact(self, points):
        points = np.asarray(points, dtype=float)
        walls = np.minimum.reduce((
            points[..., 0] - self.xmin, self.xmax - points[..., 0],
            points[..., 1], self.ymax - points[..., 1]))
        result = walls
        for rectangle in self.rectangles:
            result = np.minimum(
                result, self._rectangle_clearance(points, rectangle))
        return result

    def _build_clearance_grid(self):
        xs = np.arange(self.xmin, self.xmax + self._cres, self._cres)
        ys = np.arange(0.0, self.ymax + self._cres, self._cres)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._cgrid = self.clearance_exact(np.stack((gx, gy), axis=-1))

    def static_clearance(self, points):
        points = np.asarray(points, dtype=float)
        i = np.clip(np.rint((points[..., 0] - self.xmin) / self._cres).astype(int),
                    0, self._cgrid.shape[0] - 1)
        j = np.clip(np.rint(points[..., 1] / self._cres).astype(int),
                    0, self._cgrid.shape[1] - 1)
        return self._cgrid[i, j]

    def _peer_state(self, time_s):
        path = self.scenario.peer_path
        travel = max(0.0, time_s - self.scenario.peer_delay) * self.scenario.peer_speed
        if travel <= 0.0:
            return path[0].copy(), np.zeros(2)
        if travel >= self._cumulative[-1]:
            return path[-1].copy(), np.zeros(2)
        index = int(np.searchsorted(self._cumulative, travel, side="right") - 1)
        direction = (path[index + 1] - path[index]) / self._segments[index]
        position = path[index] + (travel - self._cumulative[index]) * direction
        return position, self.scenario.peer_speed * direction

    def moving_obs(self, t=None):
        position, _ = self._peer_state(self.t if t is None else float(t))
        return position[None]

    def moving_velocity(self, t=None):
        _, velocity = self._peer_state(self.t if t is None else float(t))
        return velocity[None]

    def step_time(self, dt):
        self.t += dt

    def clearance(self, points):
        points = np.asarray(points, dtype=float)
        if not self.dynamic_enabled:
            return self.static_clearance(points)
        dynamic = np.linalg.norm(
            points - self.moving_obs()[0], axis=-1) - self.peer_r
        return np.minimum(self.static_clearance(points), dynamic)

    def predicted_moving_obs(self, time_offsets, horizon=2.0):
        tau = np.clip(np.asarray(time_offsets, dtype=float), 0.0, horizon)
        return (self.moving_obs()[None]
                + tau[:, None, None] * self.moving_velocity()[None])

    def predicted_clearance(self, points, time_offsets, horizon=2.0,
                            uncertainty_rate=0.03):
        points = np.asarray(points, dtype=float)
        tau = np.asarray(time_offsets, dtype=float)
        if points.shape[-3] != len(tau):
            raise ValueError("time_offsets must match the trajectory axis")
        static = self.static_clearance(points)
        predicted = self.predicted_moving_obs(tau, horizon=horizon)
        lead = (1,) * (points.ndim - 3)
        centers = predicted.reshape(lead + (len(tau), 1) + predicted.shape[1:])
        radii = (self.peer_r + uncertainty_rate * tau).reshape(
            lead + (len(tau), 1, 1))
        distance = np.linalg.norm(points[..., None, :] - centers, axis=-1)
        return np.minimum(static, (distance - radii).min(axis=-1))

    def inter_robot_clearance(self, controlled_xy):
        return float(np.linalg.norm(
            np.asarray(controlled_xy) - self.moving_obs()[0])
            - self.robot_r - self.peer_r)
