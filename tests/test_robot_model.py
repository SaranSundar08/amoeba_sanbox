"""Tests for the shared V6 footprint and skid-steer model."""
import unittest

import numpy as np

from robot_model import RobotModel, sample_plant_mismatch
from tests.test_pseudopods import GridEnv


class _SingleObstacleEnv:
    """Minimal env exposing only what `exact_clearance` needs: walls plus
    one real obstacle set (no grid, no `clearance()` quantization) -- lets
    a test target an exact geometric worst case precisely."""

    def __init__(self, obstacle_center, obstacle_radius, xmin=-5.0, xmax=5.0):
        self.xmin, self.xmax = xmin, xmax
        self._centers = np.array([obstacle_center])
        self._radius = obstacle_radius

    def obstacle_centers(self):
        return self._centers, self._radius

    def clearance(self, pts):
        d = np.linalg.norm(pts - self._centers[0], axis=-1) - self._radius
        walls = np.minimum(pts[..., 0] - self.xmin, self.xmax - pts[..., 0])
        return np.minimum(d, walls)


class RobotModelTests(unittest.TestCase):
    def test_ideal_integration_matches_original_dynamics(self):
        model = RobotModel(kind="ideal", radius=0.2)
        state = np.array([0.0, 0.0, 0.2])
        control = np.array([0.4, 0.6])
        actual = model.integrate(state, control, 0.05)
        heading = state[2] + control[1] * 0.05
        expected = np.array([
            state[0] + control[0] * np.cos(heading) * 0.05,
            state[1] + control[0] * np.sin(heading) * 0.05,
            heading,
        ])
        np.testing.assert_allclose(actual, expected)

    def test_skid_steer_loses_yaw_at_speed(self):
        model = RobotModel(kind="slip", radius=0.2, yaw_gain=0.8,
                           speed_yaw_loss=1.0)
        slow = model.effective_omega(np.array([0.0, 1.0]))
        fast = model.effective_omega(np.array([0.5, 1.0]))
        self.assertAlmostEqual(float(slow), 0.8)
        self.assertAlmostEqual(float(fast), 0.4)

    def test_oriented_rectangle_detects_corner_collision(self):
        env = GridEnv(obstacle=(np.array([0.4, 0.3]), 0.12))
        circle = RobotModel(kind="ideal", radius=0.2)
        rectangle = RobotModel(
            kind="slip", radius=0.2,
            footprint_length=0.9, footprint_width=0.65)
        state = np.array([0.0, 0.5, 0.0])
        self.assertGreater(float(circle.clearance(env, state)), 0.0)
        self.assertLess(float(rectangle.clearance(env, state)), 0.0)

    def test_turning_increases_soft_margin(self):
        model = RobotModel(kind="slip", radius=0.2, turn_margin=0.1)
        straight = model.safety_margin(np.array([0.2, 0.0]), 0.1)
        turning = model.safety_margin(np.array([0.2, 1.0]), 0.1)
        self.assertGreater(float(turning), float(straight))
        self.assertGreater(
            float(model.extra_safety_margin(np.array([0.2, 1.0]))), 0.0)

    def test_slip_planning_radius_uses_width_and_turn_margin(self):
        model = RobotModel(
            kind="slip", radius=0.34, footprint_width=0.65,
            turn_margin=0.08)
        self.assertAlmostEqual(model.planning_radius, 0.405)

    def test_exact_clearance_catches_obstacle_hidden_between_boundary_samples(
            self):
        # A BARN-radius cylinder placed exactly tangent to the long edge's
        # true boundary, at the worst-case midpoint between two of the
        # sampled `clearance()` boundary points (0.225 m apart).
        model = RobotModel(
            kind="slip", radius=0.2, footprint_length=0.9,
            footprint_width=0.65)
        env = _SingleObstacleEnv(
            obstacle_center=np.array([0.1125, 0.4]), obstacle_radius=0.075)
        state = np.array([0.0, 0.0, 0.0])
        sampled = float(model.clearance(env, state))
        exact = float(model.exact_clearance(env, state))
        self.assertGreater(sampled, 0.05)   # sampled check misses it
        self.assertLessEqual(exact, 1e-9)   # exact check finds true contact

    def test_exact_clearance_falls_back_without_obstacle_centers(self):
        env = GridEnv(obstacle=(np.array([0.4, 0.3]), 0.12))
        model = RobotModel(
            kind="slip", radius=0.2, footprint_length=0.9,
            footprint_width=0.65)
        state = np.array([0.0, 0.5, 0.0])
        self.assertEqual(
            float(model.exact_clearance(env, state)),
            float(model.clearance(env, state)))

    def test_exact_clearance_never_exceeds_sampled_clearance_on_barn(self):
        from env import BarnEnv
        env = BarnEnv(48)
        model = RobotModel(
            kind="slip", radius=env.robot_r, footprint_length=0.9,
            footprint_width=0.65)
        rng = np.random.default_rng(0)
        states = np.column_stack([
            rng.uniform(env.xmin, env.xmax, 300),
            rng.uniform(0.0, env.ymax, 300),
            rng.uniform(-np.pi, np.pi, 300),
        ])
        sampled = model.clearance(env, states)
        exact = model.exact_clearance(env, states)
        # `clearance()` samples a subset of the boundary AND reads it off
        # the quantized clearance grid (documented error <= 0.0125 m), so
        # exact can exceed it by at most that quantization noise -- never
        # by the much larger sampling-gap amounts this fix targets.
        self.assertTrue(np.all(exact <= sampled + 0.02))
        # and on a real map the sampling gap itself is not just noise.
        self.assertTrue(np.any(sampled - exact > 0.03))

    def test_sample_plant_mismatch_jitters_only_slip_parameters(self):
        base = RobotModel(
            kind="slip", radius=0.2, footprint_length=0.9,
            footprint_width=0.65, yaw_gain=0.82, speed_yaw_loss=0.55)
        plant = sample_plant_mismatch(
            base, np.random.default_rng(3),
            yaw_gain_range=(1.4, 1.4), speed_yaw_loss_range=(1.4, 1.4))
        self.assertAlmostEqual(plant.yaw_gain, base.yaw_gain * 1.4)
        self.assertAlmostEqual(plant.speed_yaw_loss, base.speed_yaw_loss * 1.4)
        self.assertEqual(plant.footprint_length, base.footprint_length)
        self.assertEqual(plant.radius, base.radius)

    def test_circle_set_clearance_catches_peer_hidden_between_samples(self):
        # Same worst-case geometry as the BarnEnv obstacle test, but called
        # directly against a single circle the way v71_junction_benchmark
        # calls it for the scripted peer robot, with no `env` involved.
        model = RobotModel(
            kind="slip", radius=0.2, footprint_length=0.9,
            footprint_width=0.65)
        state = np.array([0.0, 0.0, 0.0])
        peer = np.array([[0.1125, 0.4]])
        sampled = float(np.min(np.linalg.norm(
            model.footprint_points(state) - peer, axis=-1) - 0.075))
        exact = float(model.circle_set_clearance(state, peer, 0.075))
        self.assertGreater(sampled, 0.05)
        self.assertLessEqual(exact, 1e-9)

    def test_circle_set_clearance_empty_set_is_infinite(self):
        model = RobotModel(kind="slip", radius=0.2)
        state = np.array([0.0, 0.0, 0.0])
        out = model.circle_set_clearance(state, np.zeros((0, 2)), 0.075)
        self.assertTrue(np.isinf(float(out)))

    def test_circle_set_clearance_matches_exact_clearance_via_env(self):
        # circle_set_clearance is exact_clearance's obstacle term, factored
        # out -- confirm they agree exactly on a wall-free query.
        model = RobotModel(
            kind="slip", radius=0.2, footprint_length=0.9,
            footprint_width=0.65)
        env = _SingleObstacleEnv(
            obstacle_center=np.array([0.1125, 0.4]), obstacle_radius=0.075,
            xmin=-100.0, xmax=100.0)   # walls pushed out of the way
        state = np.array([0.0, 0.0, 0.0])
        via_env = float(model.exact_clearance(env, state))
        direct = float(model.circle_set_clearance(
            state, np.array([[0.1125, 0.4]]), 0.075))
        self.assertAlmostEqual(via_env, direct)

    def test_sample_plant_mismatch_is_noop_for_ideal(self):
        base = RobotModel(kind="ideal", radius=0.2)
        self.assertIs(
            sample_plant_mismatch(base, np.random.default_rng(3)), base)


if __name__ == "__main__":
    unittest.main()
