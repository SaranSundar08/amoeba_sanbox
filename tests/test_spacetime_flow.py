"""Tests for `spacetime_flow.SpaceTimeFlood` -- the full (x, y, t) flood
prototype, standalone and not wired into anything else (see that module's
docstring). Mirrors `tests/test_spacetime.py`'s fixtures/style since this
is the same domain, just computing every reachable cell's distance instead
of stopping at the first path to one goal.
"""
import unittest

import numpy as np

from spacetime import two_route_search
from spacetime_flow import SpaceTimeFlood


class _OpenEnv:
    """No static obstacles -- isolates the moving-obstacle case, same as
    `tests/test_spacetime.py::_OpenEnv`."""

    def clearance(self, pts):
        return np.full(pts.shape[:-1], 10.0)


class _WallEnv:
    """Free everywhere except a vertical strip at x=0 -- a static obstacle
    that should block every time layer identically."""

    def clearance(self, pts):
        return np.where(np.abs(pts[..., 0]) < 0.05, -1.0, 10.0)


def _crossing(cross_y=1.5, cross_time=2.0, speed=0.4):
    def predict(t):
        return (speed * (t - cross_time), cross_y)
    return predict


def _appears_only_during(xy, t0, t1, far=(100.0, 100.0)):
    """A "moving" obstacle that's only ever AT `xy` during `[t0, t1]`, and
    far away otherwise -- deliberately not a continuous trajectory (unlike
    `_crossing`): this isolates the free-space-at-each-layer computation
    from any path-timing subtlety, since `SpaceTimeFlood` only ever reads
    a predicted position layer by layer, never assumes continuity."""
    def predict(t):
        return xy if t0 <= t <= t1 else far
    return predict


class _EnvWithMovingObstacles:
    """Adapts one or more `predict(t) -> (x, y)` callables (the convention
    `spacetime.py`'s own tests use) to `env.predicted_moving_obs(times) ->
    (nk, N, 2)`, the interface `SpaceTimeFlood` actually reads -- letting
    the same crossing scenarios validate both modules directly against
    each other."""

    def __init__(self, predict_fns, static=None):
        self.predict_fns = predict_fns
        self._static = static

    def clearance(self, pts):
        if self._static is not None:
            return self._static(pts)
        return np.full(pts.shape[:-1], 10.0)

    def predicted_moving_obs(self, times):
        cols = [np.array([fn(t) for t in times]) for fn in self.predict_fns]
        return np.stack(cols, axis=1) if cols else np.zeros((len(times), 0, 2))


class OpenSpaceTests(unittest.TestCase):
    def test_start_cell_has_zero_cost(self):
        flood = SpaceTimeFlood(_OpenEnv(), (0.0, 0.0), robot_r=0.2,
                               horizon=2.0, dt_layer=0.25, res=0.2, window=1.5)
        i, j = flood.start_cell
        self.assertEqual(flood.G[i, j, 0], 0.0)

    def test_best_arrival_cost_grows_with_distance(self):
        # NOTE: querying `distance(xy, t)` at a fixed, later-than-necessary
        # `t` is the wrong metric here when `wait_cost == res` (the
        # default): waiting and stepping then cost exactly the same per
        # layer, so the cost to be at ANY reachable cell at a fixed later
        # layer degenerates to `layer_index * res` regardless of distance
        # (the early arrival just "pads" with equal-cost waits). The best
        # (earliest-available) cost across all layers is the metric that
        # actually reflects distance.
        flood = SpaceTimeFlood(_OpenEnv(), (0.0, 0.0), robot_r=0.2,
                               horizon=3.0, dt_layer=0.25, res=0.2, window=2.0)
        i_near, j_near = flood._cell((0.0, 0.2))
        i_far, j_far = flood._cell((0.0, 1.4))
        near = np.min(flood.G[i_near, j_near, :])
        far = np.min(flood.G[i_far, j_far, :])
        self.assertTrue(np.isfinite(near) and np.isfinite(far))
        self.assertLess(near, far)

    def test_unreachable_point_is_infinite(self):
        flood = SpaceTimeFlood(_OpenEnv(), (0.0, 0.0), robot_r=0.2,
                               horizon=1.0, dt_layer=0.25, res=0.2, window=2.0)
        # Far outside anything reachable in 1 second at grid speed.
        self.assertEqual(flood.distance((0.0, 1.9), 1.0), np.inf)

    def test_reachable_and_depth_shapes_and_values(self):
        flood = SpaceTimeFlood(_OpenEnv(), (0.0, 0.0), robot_r=0.2,
                               horizon=2.0, dt_layer=0.25, res=0.2, window=1.5)
        mask = flood.reachable(0)
        depth = flood.depth(0)
        self.assertEqual(mask.shape, (flood.nx, flood.ny))
        self.assertEqual(depth.shape, mask.shape)
        i, j = flood.start_cell
        self.assertTrue(mask[i, j])
        self.assertAlmostEqual(depth[i, j], 1.0, places=6)
        self.assertTrue(np.isnan(depth[~mask]).all())

    def test_no_predicted_moving_obs_degrades_to_static_only(self):
        # `_OpenEnv` has no `predicted_moving_obs` at all -- must not crash,
        # and must behave exactly like the fully-static case.
        flood = SpaceTimeFlood(_OpenEnv(), (0.0, 0.0), robot_r=0.2,
                               horizon=1.5, dt_layer=0.25, res=0.2, window=1.5)
        self.assertTrue(np.isfinite(flood.distance((0.0, 0.4), 1.0)))


class StaticObstacleTests(unittest.TestCase):
    def test_wall_blocks_every_time_layer(self):
        flood = SpaceTimeFlood(_WallEnv(), (-1.0, 0.0), robot_r=0.2,
                               horizon=2.0, dt_layer=0.25, res=0.1, window=2.0)
        for k in range(flood.nk):
            self.assertFalse(flood.reachable(k)[flood._cell((0.5, 0.0))],
                             f"far side of the wall reachable at layer {k}")


class MovingObstacleTests(unittest.TestCase):
    def test_crossing_point_blocked_only_during_its_window(self):
        # Target 0.2 m from start (two straight grid steps away, so it's
        # reachable well before the obstacle's [1.0, 1.5]s window and
        # still reachable after it, by waiting at `start` -- which stays
        # free throughout -- until the target opens back up).
        # Radii small relative to `res` so the obstacle only ever blocks
        # its OWN cell, never a neighbour: otherwise it also blocks the
        # one-step-away retreat cell the robot needs during the window,
        # forcing an extra layer of delay after the window closes that
        # has nothing to do with what this test is checking.
        target = (0.0, 0.2)
        predict = _appears_only_during(target, 1.0, 1.5)
        env = _EnvWithMovingObstacles([predict])
        flood = SpaceTimeFlood(env, (0.0, 0.0), robot_r=0.03, obstacle_r=0.03,
                               horizon=2.0, dt_layer=0.25, res=0.1, window=1.0)
        cell = flood._cell(target)
        self.assertTrue(flood.reachable(flood._layer(0.75))[cell])   # before
        self.assertFalse(flood.reachable(flood._layer(1.25))[cell])  # during
        self.assertTrue(flood.reachable(flood._layer(1.75))[cell])   # after

    def test_two_stationary_obstacles_each_block_their_own_cell(self):
        left, right = (-0.5, 0.3), (0.5, 0.3)
        env = _EnvWithMovingObstacles(
            [_appears_only_during(left, 0.0, 10.0),
             _appears_only_during(right, 0.0, 10.0)])
        flood = SpaceTimeFlood(env, (0.0, 0.0), robot_r=0.15, obstacle_r=0.2,
                               horizon=1.0, dt_layer=0.25, res=0.1, window=1.5)
        # 0.3 m from `start` needs 3 grid steps (res=0.1) at minimum --
        # querying reachability any earlier fails for lack of time, not
        # because of the obstacles.
        k = flood._layer(0.75)
        self.assertFalse(flood.reachable(k)[flood._cell(left)])
        self.assertFalse(flood.reachable(k)[flood._cell(right)])
        # Directly between the two, clear of both -- must stay reachable.
        self.assertTrue(flood.reachable(k)[flood._cell((0.0, 0.3))])


class ConsistencyWithTwoRouteSearchTests(unittest.TestCase):
    """The flood and `two_route_search` run the identical Dijkstra
    mechanics over the identical grid/cost function -- the only
    difference is the flood doesn't stop at the first goal cell. So the
    flood's cost to the wait-route's own final cell, at its own arrival
    layer, must equal that route's reported cost."""

    def test_flood_reproduces_wait_route_cost_exactly(self):
        predict = _crossing(cross_y=1.5, cross_time=2.0, speed=0.4)
        kw = dict(horizon=8.0, dt_layer=0.25, res=0.2, window=2.0,
                 goal_tol=0.15)
        routes = two_route_search(_OpenEnv(), (0.0, 0.0), (0.0, 3.0), predict,
                                  0.3, 0.2, wait_cost_cheap=0.05, **kw)
        self.assertTrue(routes["wait"]["feasible"])
        path = routes["wait"]["path"]
        arrival_t = (len(path) - 1) * kw["dt_layer"]
        env = _EnvWithMovingObstacles([predict])
        flood = SpaceTimeFlood(env, (0.0, 0.0), robot_r=0.2, obstacle_r=0.3,
                               horizon=kw["horizon"], dt_layer=kw["dt_layer"],
                               res=kw["res"], window=kw["window"],
                               wait_cost=0.05)
        got = flood.distance(path[-1], arrival_t)
        self.assertAlmostEqual(got, routes["wait"]["cost"], places=6)


if __name__ == "__main__":
    unittest.main()
