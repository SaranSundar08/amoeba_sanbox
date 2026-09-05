"""Phase 0 tests for the standalone space-time topology search.

See `spacetime.py`'s module docstring for the motivation: dynamic obstacles
currently only ever reshape MPPI rollout costs, never which candidate
routes exist. These tests validate the core mechanism in isolation, on
synthetic scenarios, before any integration into `LocalFlowField`,
`pseudopods.py`, or `mppi.py`.
"""
import unittest

import numpy as np

from spacetime import routes_are_distinct, time_expanded_search, \
    two_route_search


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


if __name__ == "__main__":
    unittest.main()
