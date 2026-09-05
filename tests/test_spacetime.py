"""Phase 0 tests for the standalone space-time topology search.

See `spacetime.py`'s module docstring for the motivation: dynamic obstacles
currently only ever reshape MPPI rollout costs, never which candidate
routes exist. These tests validate the core mechanism in isolation, on
synthetic scenarios, before any integration into `LocalFlowField`,
`pseudopods.py`, or `mppi.py`.
"""
import unittest

import numpy as np

from robot_model import RobotModel
from spacetime import nearest_crossing_obstacle, routes_are_distinct, \
    spacetime_path_to_proposal, time_expanded_search, two_route_search


class _OpenEnv:
    """No static obstacles anywhere -- isolates the moving-obstacle case."""

    def clearance(self, pts):
        return np.full(pts.shape[:-1], 10.0)


class _FarAway:
    """A 'predicted obstacle' that never comes near anything."""

    def __call__(self, t):
        return (100.0, 100.0)


class _CrossingObstacle:
    """Crosses the robot's straight-line corridor once, perpendicular to
    its direction of travel -- the canonical "pedestrian crossing the
    corridor" case the paradigm example (pass in front / pass behind)
    is built on."""

    def __init__(self, cross_y=1.5, cross_time=2.0, speed=0.4):
        self.cross_y, self.cross_time, self.speed = cross_y, cross_time, speed

    def __call__(self, t):
        return (self.speed * (t - self.cross_time), self.cross_y)


class TimeExpandedSearchTests(unittest.TestCase):
    def setUp(self):
        self.env = _OpenEnv()
        self.start, self.goal = (0.0, 0.0), (0.0, 3.0)
        self.robot_r, self.obstacle_r = 0.2, 0.3
        self.kw = dict(horizon=8.0, dt_layer=0.25, res=0.2, window=2.0,
                       goal_tol=0.15)

    def _assert_never_collides(self, path, predict, label=""):
        for k, (x, y) in enumerate(path):
            ox, oy = predict(k * self.kw["dt_layer"])
            d = np.hypot(x - ox, y - oy)
            self.assertGreater(
                d, self.robot_r + self.obstacle_r,
                f"{label} collides at layer {k}: d={d:.3f}")

    def test_crossing_obstacle_is_never_hit_regardless_of_wait_cost(self):
        predict = _CrossingObstacle()
        for wait_cost, label in ((0.05, "cheap-wait"), (1.0, "expensive-wait")):
            path, cost = time_expanded_search(
                self.env, self.start, self.goal, predict, self.obstacle_r,
                self.robot_r, wait_cost=wait_cost, **self.kw)
            self.assertIsNotNone(path, label)
            self.assertTrue(np.isfinite(cost))
            self._assert_never_collides(path, predict, label)

    def test_cheap_waiting_produces_a_real_wait_and_little_detour(self):
        predict = _CrossingObstacle()
        path, _ = time_expanded_search(
            self.env, self.start, self.goal, predict, self.obstacle_r,
            self.robot_r, wait_cost=0.05, **self.kw)
        waits = sum(1 for a, b in zip(path, path[1:]) if a == b)
        max_x = max(abs(x) for x, _ in path)
        self.assertGreater(waits, 0)
        self.assertLess(max_x, 0.5)   # stays near the centreline

    def test_expensive_waiting_produces_a_real_detour_and_no_waits(self):
        predict = _CrossingObstacle()
        path, _ = time_expanded_search(
            self.env, self.start, self.goal, predict, self.obstacle_r,
            self.robot_r, wait_cost=1.0, **self.kw)
        waits = sum(1 for a, b in zip(path, path[1:]) if a == b)
        max_x = max(abs(x) for x, _ in path)
        self.assertEqual(waits, 0)
        self.assertGreater(max_x, 0.5)   # clears the exclusion radius sideways

    def test_no_obstacle_gives_identical_cost_regardless_of_wait_cost(self):
        far = _FarAway()
        _, cost_cheap = time_expanded_search(
            self.env, self.start, self.goal, far, self.obstacle_r,
            self.robot_r, wait_cost=0.05, **self.kw)
        _, cost_expensive = time_expanded_search(
            self.env, self.start, self.goal, far, self.obstacle_r,
            self.robot_r, wait_cost=1.0, **self.kw)
        self.assertAlmostEqual(cost_cheap, cost_expensive)

    def test_infeasible_when_the_goal_stays_permanently_occupied(self):
        stay_at_goal = lambda t: self.goal
        path, cost = time_expanded_search(
            self.env, self.start, self.goal, stay_at_goal, self.obstacle_r,
            self.robot_r, **self.kw)
        self.assertIsNone(path)
        self.assertEqual(cost, np.inf)

    def test_infeasible_start_returns_none_not_a_crash(self):
        sitting_on_robot = lambda t: self.start
        path, cost = time_expanded_search(
            self.env, self.start, self.goal, sitting_on_robot,
            self.obstacle_r, self.robot_r, **self.kw)
        self.assertIsNone(path)
        self.assertEqual(cost, np.inf)


class TwoRouteSearchTests(unittest.TestCase):
    def test_wait_and_detour_are_found_and_flagged_distinct(self):
        env = _OpenEnv()
        result = two_route_search(
            env, (0.0, 0.0), (0.0, 3.0), _CrossingObstacle(), 0.3, 0.2,
            horizon=8.0, dt_layer=0.25, res=0.2, window=2.0, goal_tol=0.15)
        self.assertTrue(result["wait"]["feasible"])
        self.assertTrue(result["detour"]["feasible"])
        self.assertNotEqual(result["wait"]["path"], result["detour"]["path"])
        self.assertTrue(result["distinct"])

    def test_no_obstacle_is_not_falsely_flagged_distinct(self):
        env = _OpenEnv()
        result = two_route_search(
            env, (0.0, 0.0), (0.0, 3.0), _FarAway(), 0.3, 0.2,
            horizon=8.0, dt_layer=0.25, res=0.2, window=2.0, goal_tol=0.15)
        self.assertFalse(result["distinct"])

    def test_infeasible_route_makes_distinct_false_not_an_error(self):
        env = _OpenEnv()
        result = two_route_search(
            env, (0.0, 0.0), (0.0, 3.0), lambda t: (0.0, 3.0), 0.3, 0.2,
            horizon=8.0, dt_layer=0.25, res=0.2, window=2.0, goal_tol=0.15)
        self.assertFalse(result["wait"]["feasible"])
        self.assertFalse(result["detour"]["feasible"])
        self.assertFalse(result["distinct"])


class RoutesAreDistinctTests(unittest.TestCase):
    """Direct unit tests of the distinctness test on hand-built
    trajectories, independent of whether a search happens to produce a
    non-degenerate pair (see `two_route_search`'s docstring)."""

    def test_opposite_sides_at_a_shared_relevant_time_are_distinct(self):
        # a is ahead of the obstacle, b is behind it, at the same instant.
        obstacle = lambda t: (0.0, 0.0)
        path_a = [(0.0, 0.3)]
        path_b = [(0.0, -0.3)]
        self.assertTrue(routes_are_distinct(path_a, path_b, obstacle))

    def test_same_side_is_not_distinct(self):
        obstacle = lambda t: (0.0, 0.0)
        path_a = [(0.0, 0.3)]
        path_b = [(0.0, 0.4)]
        self.assertFalse(routes_are_distinct(path_a, path_b, obstacle))

    def test_far_from_the_obstacle_is_not_distinct_even_if_opposite_signs(self):
        # both points are outside `relevance`, so the "side" is meaningless.
        obstacle = lambda t: (0.0, 0.0)
        path_a = [(0.0, 5.0)]
        path_b = [(0.0, -5.0)]
        self.assertFalse(
            routes_are_distinct(path_a, path_b, obstacle, relevance=0.8))

    def test_shorter_path_is_held_at_its_final_position(self):
        # The obstacle is far away at k=0 (both paths irrelevant there) and
        # at (0, 0) at k=1. `short` arrived at k=0 and holds at (0, 0.3);
        # `long` is still transiting and reaches (0, -0.3) at k=1 -- opposite
        # sides of the obstacle once it arrives. This only works if `short`
        # (length 1) is correctly padded/held through k=1, not truncated.
        obstacle_positions = {0: (100.0, 100.0), 1: (0.0, 0.0)}
        obstacle = lambda t: obstacle_positions[int(round(t))]
        short = [(0.0, 0.3)]
        long = [(0.0, 0.3), (0.0, -0.3)]
        self.assertTrue(
            routes_are_distinct(short, long, obstacle, dt_layer=1.0))


class SpacetimePathToProposalTests(unittest.TestCase):
    """`spacetime_path_to_proposal` is the glue between Phase 0's raw
    (x, y)-per-time-layer output and the `[T, 2]` control sequences the rest
    of the pipeline consumes -- these pin its two load-bearing claims: a
    wait segment survives resampling (unlike handing the same points to
    `proposals.branch_to_control_sequence`, which only understands
    geometry and would discard a repeated point as a zero-length segment),
    and feasibility is judged on the ACHIEVED rollout, not the request."""

    def setUp(self):
        self.env = _OpenEnv()
        self.model = RobotModel(kind="ideal", radius=0.2)
        self.far = _FarAway()

    def test_straight_path_is_tracked_and_feasible(self):
        state = np.array([0.0, 0.0, np.pi / 2])
        goal = np.array([0.0, 1.0])
        path = [(0.0, 0.5 * t) for t in np.arange(9) * 0.25]
        proposal = spacetime_path_to_proposal(
            path, 0.25, state, goal, self.env, self.model, self.far, 0.3,
            T=56, dt=0.05, v_max=0.5, w_max=1.9,
            v_accel_max=1.0, w_accel_max=3.0)
        self.assertTrue(proposal.feasible)
        self.assertAlmostEqual(proposal.progress, 1.0, places=2)
        self.assertLess(proposal.mean_tracking_error, 0.15)
        self.assertEqual(proposal.controls.shape, (56, 2))
        self.assertEqual(proposal.rollout.shape, (56, 3))

    def test_wait_segment_survives_resampling_as_zero_velocity(self):
        state = np.array([0.0, 0.0, np.pi / 2])
        goal = np.array([0.0, 1.0])
        # Hold at the start for 1.0 s (5 points at dt_layer=0.25), then move.
        path = ([(0.0, 0.0)] * 5
               + [(0.0, 0.5 * t) for t in np.arange(1, 9) * 0.25 - 1.0])
        proposal = spacetime_path_to_proposal(
            path, 0.25, state, goal, self.env, self.model, self.far, 0.3,
            T=56, dt=0.05, v_max=0.5, w_max=1.9,
            v_accel_max=1.0, w_accel_max=3.0)
        wait_steps = int(round(1.0 / 0.05))
        self.assertTrue(
            np.allclose(proposal.controls[:wait_steps, 0], 0.0, atol=1e-9),
            "commanded speed during the wait segment should be exactly "
            "zero, not a discarded/collapsed segment")
        self.assertTrue(proposal.feasible)

    def test_feasibility_checks_the_achieved_rollout_not_the_request(self):
        # A raw path that swerves away almost immediately -- safe on paper
        # against an obstacle sitting on the original straight-ahead line --
        # but a real robot with a very tight turn-rate limit cannot follow
        # that swerve and keeps heading toward the obstacle instead.
        state = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 0.5])
        model = RobotModel(kind="ideal", radius=0.1)
        obstacle_xy = (0.5, 0.0)
        obstacle_r = 0.15
        predict = lambda t: obstacle_xy
        path = [(0.05 * i, 0.5 if i >= 2 else 0.0) for i in range(20)]

        raw_clear = np.min(np.linalg.norm(
            np.asarray(path) - np.asarray(obstacle_xy), axis=-1)
        ) - (obstacle_r + model.radius)
        self.assertGreater(raw_clear, 0.0, "test setup: raw path must look "
                           "safe, or this isn't testing what it claims to")

        proposal = spacetime_path_to_proposal(
            path, 0.05, state, goal, self.env, model, predict, obstacle_r,
            T=40, dt=0.05, v_max=0.5, w_max=1.9, v_accel_max=1.0,
            w_accel_max=0.3, initial_control=[0.3, 0.0])
        self.assertFalse(proposal.feasible)
        self.assertIn("clearance", proposal.reason)
        self.assertLess(proposal.min_clearance, 0.0)

    def test_prefers_exact_clearance_when_env_supports_it(self):
        class _ObstacleEnv:
            def __init__(self):
                self.xmin, self.xmax = -5.0, 5.0

            def clearance(self, pts):
                return np.full(pts.shape[:-1], 10.0)

            def obstacle_centers(self):
                return np.zeros((0, 2)), 0.075

        state = np.array([0.0, 0.0, np.pi / 2])
        goal = np.array([0.0, 1.0])
        path = [(0.0, 0.5 * t) for t in np.arange(9) * 0.25]
        proposal = spacetime_path_to_proposal(
            path, 0.25, state, goal, _ObstacleEnv(), self.model, self.far,
            0.3, T=56, dt=0.05, v_max=0.5, w_max=1.9,
            v_accel_max=1.0, w_accel_max=3.0)
        self.assertTrue(proposal.feasible)


class _PredictingEnv:
    """Minimal env exposing only `predicted_moving_obs`, matching
    `dynabarn.DynaBarnEnv`'s signature -- lets `nearest_crossing_obstacle`
    be tested without a real BARN world."""

    def __init__(self, predict_fn, n_obstacles=1):
        self._predict = predict_fn
        self.n_obstacles = n_obstacles

    def predicted_moving_obs(self, time_offsets, horizon=2.0):
        times = np.asarray(time_offsets, dtype=float)
        if self.n_obstacles == 0:
            return np.zeros((len(times), 0, 2))
        return self._predict(times)


class NearestCrossingObstacleTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.column_stack(
            (np.zeros(20), np.linspace(0.0, 2.0, 20)))
        self.times = (np.arange(20) + 1.0) * 0.05

    def test_no_prediction_support_returns_none(self):
        class _StaticEnv:
            def clearance(self, pts):
                return np.full(pts.shape[:-1], 10.0)

        self.assertIsNone(nearest_crossing_obstacle(
            _StaticEnv(), self.reference, self.times, 0.2, 0.15))

    def test_zero_obstacles_returns_none(self):
        env = _PredictingEnv(None, n_obstacles=0)
        self.assertIsNone(nearest_crossing_obstacle(
            env, self.reference, self.times, 0.2, 0.15))

    def test_far_obstacle_returns_none(self):
        env = _PredictingEnv(
            lambda t: np.tile([[100.0, 100.0]], (len(t), 1, 1)))
        self.assertIsNone(nearest_crossing_obstacle(
            env, self.reference, self.times, 0.2, 0.15))

    def test_crossing_obstacle_is_found_with_correct_margin(self):
        # A stationary obstacle sitting exactly on the reference's
        # midpoint -- co-located with the reference at every queried time,
        # so it necessarily conflicts at the reference's own midpoint time.
        mid_k = 9
        mid_xy = self.reference[mid_k]
        mid_t = self.times[mid_k]

        def predict(t):
            t = np.asarray(t)
            return np.tile(mid_xy, (len(t), 1, 1))

        env = _PredictingEnv(predict)
        result = nearest_crossing_obstacle(
            env, self.reference, self.times, 0.2, 0.15)
        self.assertIsNotNone(result)
        predict_fn, margin = result
        self.assertLessEqual(margin, 0.0)
        np.testing.assert_allclose(predict_fn(mid_t), mid_xy)

    def test_selects_the_conflicting_obstacle_among_several(self):
        far = np.array([50.0, 50.0])
        mid_k = 9
        conflicting = self.reference[mid_k]

        def predict(t):
            t = np.asarray(t)
            return np.stack([
                np.tile(far, (len(t), 1)),
                np.tile(conflicting, (len(t), 1)),
            ], axis=1)   # (T, 2 obstacles, 2)

        env = _PredictingEnv(predict, n_obstacles=2)
        result = nearest_crossing_obstacle(
            env, self.reference, self.times, 0.2, 0.15)
        self.assertIsNotNone(result)
        predict_fn, _ = result
        np.testing.assert_allclose(
            predict_fn(self.times[mid_k]), conflicting)


if __name__ == "__main__":
    unittest.main()
