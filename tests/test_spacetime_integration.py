"""Integration tests: space-time alternatives wired into
`MPPI.build_branch_proposals`. See `spacetime.py` and
`docs/experiments/SPACETIME_TOPOLOGY_PHASE0.md` for the standalone
mechanism these depend on; these tests cover the DECISION layer -- when a
predicted crossing triggers a "wait"/"detour" alternative and how it
surfaces as an ordinary `SamplingMode`.

`spacetime_horizon` deliberately looks further ahead than the short test
`T` used here (a real controller would use its normal T=56, dt=0.05, whose
horizon is already close to typical `spacetime_horizon` values) -- getting
this pair of numbers wrong silently made the very first version of this
integration report zero crossings ever, for two different reasons (see the
comments in `MPPI._spacetime_alternatives`): a goal picked using `v_max`
that the search's own coarser grid speed couldn't reach in time, and a
crossing check limited to the base proposal's short MPPI-horizon rollout
instead of the full `spacetime_horizon` window. Both are exactly the
"looks fine, silently checks the wrong window" failure class the rest of
this week's work has been about -- worth knowing if these parameters are
retuned later.
"""
import unittest

import numpy as np

from mppi import MPPI
from pseudopods import Pseudopod
from tests.test_pseudopods import GridEnv


class _DynamicGridEnv(GridEnv):
    """`GridEnv` plus one predicted moving obstacle, matching
    `dynabarn.DynaBarnEnv`'s `predicted_moving_obs` signature."""

    def __init__(self, predict_fn, horizon=6.0, **kw):
        super().__init__(**kw)
        self._predict = predict_fn
        self._horizon = horizon

    def predicted_moving_obs(self, time_offsets, horizon=None):
        horizon = self._horizon if horizon is None else horizon
        times = np.clip(np.asarray(time_offsets, dtype=float), 0.0, horizon)
        return np.stack([self._predict(t) for t in times])[:, None, :]


def _branch(points, branch_id=0):
    points = np.asarray(points, dtype=float)
    return Pseudopod(points[-1], points, 0.0, branch_id)


# Crosses the branch's straight-line path (x=0) at y=1.5, t=2.0s.
_CROSSING = lambda t: np.array([-0.6 + 0.3 * t, 1.5])
_FAR_AWAY = lambda t: np.array([100.0, 100.0])
_BRANCH_POINTS = [[0.0, 0.5], [0.0, 4.0]]
_SPACETIME_KW = dict(spacetime_horizon=6.0, spacetime_window=1.0,
                     spacetime_obstacle_r=0.15)


class SpacetimeAlternativesTests(unittest.TestCase):
    def _controller(self, predict, **kw):
        env = _DynamicGridEnv(predict)
        ctrl = MPPI(env, K=8, T=30, **kw)
        ctrl.branches = [_branch(_BRANCH_POINTS)]
        ctrl.build_branch_proposals(env.start)
        return env, ctrl

    def test_disabled_by_default_ignores_a_genuine_crossing(self):
        _, ctrl = self._controller(_CROSSING)   # spacetime_modes not passed
        self.assertEqual(len(ctrl.branch_proposals), 1)
        self.assertEqual(ctrl.spacetime_triggers, 0)
        self.assertEqual(ctrl.spacetime_modes_added, 0)

    def test_enabled_with_no_crossing_adds_nothing(self):
        _, ctrl = self._controller(
            _FAR_AWAY, spacetime_modes=True, **_SPACETIME_KW)
        self.assertEqual(len(ctrl.branch_proposals), 1)
        self.assertEqual(ctrl.spacetime_triggers, 0)

    def test_genuine_crossing_adds_feasible_wait_and_detour_modes(self):
        env, ctrl = self._controller(
            _CROSSING, spacetime_modes=True, **_SPACETIME_KW)
        self.assertEqual(ctrl.spacetime_triggers, 1)
        ids = sorted(p.branch_id for p in ctrl.branch_proposals)
        self.assertEqual(ids, [0, 100, 200])
        for proposal in ctrl.branch_proposals:
            self.assertTrue(proposal.feasible, proposal.reason)
            self.assertEqual(proposal.controls.shape, (30, 2))

    def test_extra_modes_are_picked_up_as_ordinary_sampling_modes(self):
        env, ctrl = self._controller(
            _CROSSING, spacetime_modes=True, **_SPACETIME_KW)
        modes = ctrl._sampling_modes(env.start)
        self.assertEqual(sorted(m.key for m in modes), [-1, 0, 100, 200])
        # Each mode's reference is its own rollout -- required for the
        # homotopy-consistency cost (mppi.py, 2026-09-04) to treat a
        # space-time mode exactly like any other guided mode, with no
        # special-casing.
        for mode in modes:
            if mode.key != -1:
                self.assertIsNotNone(mode.reference)

    def test_a_short_episode_with_spacetime_modes_runs_without_crashing(self):
        env, ctrl = self._controller(
            _CROSSING, spacetime_modes=True, grouped_sampling=True,
            **_SPACETIME_KW)
        state = env.start.copy()
        for _ in range(10):
            u, info = ctrl.step(state)
            self.assertTrue(np.all(np.isfinite(u)))
            state = ctrl.robot_model.integrate(state, u, ctrl.dt)


if __name__ == "__main__":
    unittest.main()
