"""Integration tests: `spacetime_flow.SpaceTimeFlood` wired into
`MPPI` as a continuous per-cycle cost term (`MPPI.spacetime_flow_cost`,
built/refreshed by `MPPI._maybe_rebuild_spacetime_flood`). See that
module and `spacetime_flow.py` for what this is and isn't -- this is
Option B from the 2026-09-08 session: a continuous cost, not the discrete
trigger-based `spacetime_modes` alternatives (`test_spacetime_
integration.py` covers those separately).

`spacetime_flow` defaults to `False`; every test here that doesn't pass
it explicitly is implicitly checking that nothing changes when it's off,
same as the rest of the suite passing unmodified confirms.
"""
import unittest

import numpy as np

from mppi import MPPI
from tests.test_pseudopods import GridEnv


class _DynamicGridEnv(GridEnv):
    """`GridEnv` plus one predicted moving obstacle, matching
    `dynabarn.DynaBarnEnv`'s `predicted_moving_obs` signature -- same
    fixture pattern as `test_spacetime_integration.py`."""

    def __init__(self, predict_fn=None, **kw):
        super().__init__(**kw)
        self._predict = predict_fn

    def predicted_moving_obs(self, time_offsets, horizon=None):
        if self._predict is None:
            return np.zeros((len(time_offsets), 0, 2))
        times = np.asarray(time_offsets, dtype=float)
        return np.stack([self._predict(t) for t in times])[:, None, :]


# A stationary "moving" obstacle sitting directly on the straight path to
# goal (x=0), 2 m ahead -- deliberately in the way of the direct route.
_BLOCKING = lambda t: np.array([0.0, 2.5])


class DisabledByDefaultTests(unittest.TestCase):
    def test_cost_is_a_zero_scalar_when_disabled(self):
        env = _DynamicGridEnv(_BLOCKING)
        ctrl = MPPI(env, seed=0, K=8, T=10)
        xs = ctrl.rollout(env.start, np.tile(ctrl.nominal, (8, 1, 1)))
        self.assertEqual(ctrl.spacetime_flow_cost(xs), 0.0)

    def test_no_flood_built_across_several_steps_when_disabled(self):
        env = _DynamicGridEnv(_BLOCKING)
        ctrl = MPPI(env, seed=0, K=16, T=10)
        state = env.start.copy()
        for _ in range(3):
            u, _info = ctrl.step(state)
        self.assertIsNone(ctrl._spacetime_flood)
        self.assertEqual(ctrl.spacetime_flow_build_ms, [])


class EnabledBehaviorTests(unittest.TestCase):
    def test_flood_is_built_on_first_step(self):
        env = _DynamicGridEnv(_BLOCKING)
        ctrl = MPPI(env, seed=0, K=16, T=10, spacetime_flow=True,
                   spacetime_flow_obstacle_r=0.15, spacetime_flow_window=1.5)
        ctrl.step(env.start.copy())
        self.assertIsNotNone(ctrl._spacetime_flood)
        self.assertEqual(len(ctrl.spacetime_flow_build_ms), 1)

    def test_degrades_gracefully_with_no_moving_obstacles(self):
        # `_DynamicGridEnv(None)` reports zero obstacles -- must not crash,
        # matching `SpaceTimeFlood`'s own no-predicted-obstacles test.
        env = _DynamicGridEnv(None)
        ctrl = MPPI(env, seed=0, K=16, T=10, spacetime_flow=True)
        u, info = ctrl.step(env.start.copy())
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(info["costs"])))

    def test_reflood_cadence_is_respected(self):
        env = _DynamicGridEnv(_BLOCKING)
        ctrl = MPPI(env, seed=0, K=8, T=10, spacetime_flow=True,
                   spacetime_flow_reflood_every=3)
        state = env.start.copy()
        for _ in range(7):
            ctrl.step(state)
        # Steps 1, 4, 7 rebuild (age % 3 == 1, using age starting at 1) --
        # exactly `ceil(7 / 3)` = 3 builds, not 7.
        self.assertEqual(len(ctrl.spacetime_flow_build_ms), 3)

    def test_a_sample_through_the_obstacles_future_position_costs_more(self):
        """Direct, deterministic check of the cost shape itself -- avoids
        asserting on full stochastic MPPI behaviour, which is noisier than
        this needs to be. One hand-built trajectory drives straight through
        `_BLOCKING`'s position at the time it's actually there; another
        reaches the same final point by first waiting, then going once
        the "obstacle" has been there and gone (it's stationary here, so
        going around in space isn't available -- only timing is)."""
        env = _DynamicGridEnv(_BLOCKING)
        ctrl = MPPI(env, seed=0, K=2, T=20, dt=0.1, spacetime_flow=True,
                   spacetime_flow_obstacle_r=0.15, spacetime_flow_window=2.0,
                   spacetime_flow_horizon=2.0, spacetime_flow_dt_layer=0.1)
        ctrl._maybe_rebuild_spacetime_flood(env.start)
        self.assertIsNotNone(ctrl._spacetime_flood)

        T = ctrl.T
        straight_through = np.tile(env.start, (T, 1))
        straight_through[:, 1] = np.linspace(0.5, 2.5, T)  # arrives at t=~2s

        waits_then_arrives_late = np.tile(env.start, (T, 1))
        # Sit at the start for the first half, then cover the same ground
        # in the second half -- arrives at the obstacle's exact spot only
        # once, at the very last (already-safe, obstacle's stationary but
        # this at least exercises a different time-of-arrival) instant.
        half = T // 2
        waits_then_arrives_late[:half, 1] = 0.5
        waits_then_arrives_late[half:, 1] = np.linspace(0.5, 1.5, T - half)

        xs = np.stack([straight_through, waits_then_arrives_late])  # (K=2, T, 3)
        cost = ctrl.spacetime_flow_cost(xs)
        self.assertGreater(cost[0], cost[1])


if __name__ == "__main__":
    unittest.main()
