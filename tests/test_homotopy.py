"""Tests for the T-MPC-style homotopy-consistency (mode-purity) cost."""
import unittest

import numpy as np

from controllers import AmoebaHybrid
from env import BarnEnv
from homotopy import mode_side_planes, side_violations
from path_test import lane_path


def _straight(x, y0=0.0, y1=4.0, n=41):
    return np.column_stack((np.full(n, x), np.linspace(y0, y1, n)))


class HomotopyPlaneTests(unittest.TestCase):
    def test_plane_faces_from_reference_toward_obstacle(self):
        # Reference passes the pillar at (0, 2) on its left (x = -0.5).
        A, b = mode_side_planes(
            _straight(-0.5), np.array([[0.0, 2.0]]), 0.3, 0.075)
        self.assertEqual(A.shape, (1, 2))
        np.testing.assert_allclose(A[0], [1.0, 0.0])
        self.assertAlmostEqual(float(b[0]), 0.0)   # beta = 0: through centre

    def test_wrong_side_sample_is_penalized_and_same_side_is_free(self):
        A, b = mode_side_planes(
            _straight(-0.5), np.array([[0.0, 2.0]]), 0.3, 0.075)
        same_side = _straight(-0.5)[None]          # (1, T, 2)
        wrong_side = _straight(0.5)[None]
        self.assertEqual(float(side_violations(same_side, A, b)[0]), 0.0)
        self.assertGreater(float(side_violations(wrong_side, A, b)[0]), 0.0)

    def test_far_obstacle_is_not_relevant(self):
        A, b = mode_side_planes(
            _straight(-0.5), np.array([[3.0, 2.0]]), 0.3, 0.075,
            relevance=0.8)
        self.assertEqual(len(A), 0)
        self.assertEqual(
            float(side_violations(_straight(0.5)[None], A, b)[0]), 0.0)

    def test_beta_moves_plane_toward_reference(self):
        A, b0 = mode_side_planes(
            _straight(-0.5), np.array([[0.0, 2.0]]), 0.3, 0.075, beta=0.0)
        _, b1 = mode_side_planes(
            _straight(-0.5), np.array([[0.0, 2.0]]), 0.3, 0.075, beta=1.0)
        self.assertAlmostEqual(float(b0[0] - b1[0]), 0.375)  # r + r_o


class _PillarEnv:
    """One cylinder ahead of the robot; only what the metric needs."""

    def __init__(self):
        self.robot_r = 0.30
        self.xmin, self.xmax = -5.0, 5.0

    def obstacle_centers(self):
        return np.array([[0.0, 1.0]]), 0.075


class HomotopyDistinctnessTests(unittest.TestCase):
    """Pin `MPPI._homotopy_mode_distinctness` on known geometry: two arcs
    from the same start that pass a pillar on opposite sides."""

    def setUp(self):
        from grouped_sampling import SamplingMode
        from mppi import MPPI
        self.ctrl = MPPI(_PillarEnv(), T=56, dt=0.05)
        self.state = np.array([0.0, 0.0, np.pi / 2])
        self.left = np.tile([0.5, 0.5], (56, 1))     # curves left of pillar
        self.right = np.tile([0.5, -0.5], (56, 1))   # curves right of pillar
        ref = lambda u: self.ctrl.rollout(self.state, u[None])[0][:, :2]
        self.modes = [
            SamplingMode(1, "P1", self.left, reference=ref(self.left)),
            SamplingMode(2, "P2", self.right, reference=ref(self.right)),
        ]

    def test_updates_matching_references_are_distinct(self):
        out = self.ctrl._homotopy_mode_distinctness(
            self.state, self.modes, {1: self.left, 2: self.right})
        self.assertEqual(out["guided"], 2)
        self.assertEqual(out["pairs"], 1)
        self.assertEqual(out["collapsed_pairs"], 0)
        self.assertEqual(out["own_drift"], 0)

    def test_updates_that_swapped_sides_show_drift_and_collapse(self):
        out = self.ctrl._homotopy_mode_distinctness(
            self.state, self.modes, {1: self.right, 2: self.left})
        self.assertEqual(out["own_drift"], 2)
        # Both updates still pass the pillar on different sides from each
        # other, just the wrong ones -- so the pair is still separated.
        self.assertEqual(out["collapsed_pairs"], 0)

    def test_identical_updates_are_a_collapsed_pair(self):
        out = self.ctrl._homotopy_mode_distinctness(
            self.state, self.modes, {1: self.left, 2: self.left})
        self.assertEqual(out["own_drift"], 1)      # P2 crossed to the left
        self.assertEqual(out["pairs"], 1)
        self.assertEqual(out["collapsed_pairs"], 1)


class HomotopyControllerTests(unittest.TestCase):
    def _controller(self, gate="auto", **kw):
        env = BarnEnv(48, robot_r=0.34)
        path = lane_path(env, -2.25)          # deliberately blocked lane
        return env, AmoebaHybrid(
            env, seed=0, path=path, K=128, T=24, extract_branches=True,
            max_branches=3, grouped_sampling=True, gate=gate,
            path_validity_gate=True, **kw)

    def _drive(self, env, ctrl, steps, start=None):
        state = env.start.copy() if start is None else np.asarray(start, float)
        infos = []
        for _ in range(steps):
            u, info = ctrl.step(state)
            state = ctrl.robot_model.integrate(state, u, ctrl.dt)
            infos.append(info)
        return state, infos

    def test_monitor_does_not_change_commands(self):
        env_a, base = self._controller()
        env_b, monitored = self._controller(homotopy_monitor=True)
        state_a, _ = self._drive(env_a, base, 30)
        state_b, infos = self._drive(env_b, monitored, 30)
        np.testing.assert_allclose(state_a, state_b)
        self.assertTrue(all("homotopy_stats" in info for info in infos))
        self.assertTrue(np.isfinite(
            infos[-1]["homotopy_selected_wrong_side_mass"]))

    def test_weighted_cost_runs_and_reports_planes_in_assist(self):
        # gate="always" keeps ASSIST on from the first cycle; the auto gate
        # only opens once the robot reaches the blockage, which is the gate's
        # business, not this cost's.
        env, ctrl = self._controller(gate="always", homotopy_w=20.0)
        # Start just short of the cylinder field (which begins at y ~ 5.2);
        # from the spawn, ~4 m of open corridor keeps every obstacle beyond
        # the relevance distance of a short reference rollout.
        _, infos = self._drive(
            env, ctrl, 60, start=[env.start[0], 4.6, np.pi / 2])
        assist = [info for info in infos if info["assist_state"] == "ASSIST"]
        self.assertTrue(assist)
        guided = [s for info in assist for s in info["homotopy_stats"]
                  if s["key"] != -1]
        self.assertTrue(guided, "ASSIST cycles should carry guided modes")
        self.assertTrue(any(s["n_planes"] > 0 for s in guided))
        self.assertTrue(all(np.isfinite(ctrl.last_applied)))
        distinct = [info["homotopy_modes"] for info in assist]
        self.assertTrue(all(d is not None for d in distinct))
        self.assertTrue(all(
            d["own_drift"] <= d["guided"]
            and d["collapsed_pairs"] <= d["pairs"] for d in distinct))

    def test_reference_policy_defaults_reproduce_previous_commands(self):
        env_a, base = self._controller()
        env_b, explicit = self._controller(
            reference_min_speed_ratio=0.12, reference_curvature_slowdown=1.2,
            reference_clearance_slow_band=0.45)
        start = [env_a.start[0], 4.6, np.pi / 2]
        state_a, _ = self._drive(env_a, base, 30, start=start)
        state_b, _ = self._drive(env_b, explicit, 30, start=start)
        np.testing.assert_allclose(state_a, state_b)

    def test_nominal_reference_reaches_further_and_reports_separation(self):
        start = None
        lengths = {}
        for name, kw in (("shaped", {}),
                         ("nominal", {"reference_min_speed_ratio": 1.0,
                                      "reference_curvature_slowdown": 0.0})):
            env, ctrl = self._controller(
                gate="always", homotopy_monitor=True, **kw)
            start = [env.start[0], 4.6, np.pi / 2]
            _, infos = self._drive(env, ctrl, 40, start=start)
            stats = [info["homotopy_modes"] for info in infos
                     if info["homotopy_modes"]
                     and info["homotopy_modes"]["guided"] > 0]
            self.assertTrue(stats)
            self.assertTrue(all(
                s["reference_separated"] <= s["reference_pairs"] <= s["pairs"]
                for s in stats))
            lengths[name] = (sum(s["reference_length_sum"] for s in stats)
                             / sum(s["guided"] for s in stats))
        self.assertGreater(lengths["nominal"], lengths["shaped"])

    def test_infeasible_fallback_keeps_feasible_set_a_superset(self):
        # At one identical state, nominal-with-fallback must keep at least
        # as many feasible branches as either nominal or shaped alone.
        counts = {}
        nominal = {"reference_min_speed_ratio": 1.0,
                   "reference_curvature_slowdown": 0.0}
        for name, kw in (("shaped", {}), ("nominal", nominal),
                         ("nominal_fb", {**nominal,
                                         "reference_infeasible_fallback": True})):
            env, ctrl = self._controller(gate="always", **kw)
            ctrl.prepare(np.array([env.start[0], 5.4, np.pi / 2]))
            counts[name] = sum(p.feasible for p in ctrl.branch_proposals)
            self.assertEqual(len(ctrl.branch_proposals), len(ctrl.branches))
        self.assertGreaterEqual(counts["nominal_fb"],
                                max(counts["nominal"], counts["shaped"]))

    def test_fallback_is_unconstrained_by_default(self):
        env, ctrl = self._controller(homotopy_monitor=True)
        _, infos = self._drive(env, ctrl, 10)
        self.assertTrue(all(
            s["key"] != -1 for info in infos for s in info["homotopy_stats"]))


if __name__ == "__main__":
    unittest.main()
