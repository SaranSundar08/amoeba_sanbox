"""Focused V2 tests; runnable with ``python3 -m unittest -v``."""
import unittest

import numpy as np

from controllers import AmoebaLocal, MPPI
from proposals import branch_to_control_sequence
from pseudopods import Pseudopod
from tests.test_pseudopods import GridEnv


def branch(points, branch_id=0):
    points = np.asarray(points, dtype=float)
    return Pseudopod(points[-1], points, 0.0, branch_id)


class ProposalTests(unittest.TestCase):
    def test_straight_branch_has_complete_bounded_sequence(self):
        env = GridEnv()
        proposal = branch_to_control_sequence(
            branch([[0.0, 0.5], [0.0, 4.0]]), env.start, env)
        self.assertEqual(proposal.controls.shape, (56, 2))
        self.assertEqual(proposal.rollout.shape, (56, 3))
        self.assertTrue(proposal.feasible)
        self.assertTrue(np.all(proposal.controls[:, 0] >= 0.0))
        self.assertTrue(np.all(proposal.controls[:, 0] <= 0.5))
        self.assertLess(np.max(np.abs(proposal.controls[:, 1])), 1e-6)

    def test_acceleration_and_angular_acceleration_are_enforced(self):
        env = GridEnv()
        proposal = branch_to_control_sequence(
            branch([[0.0, 0.5], [1.5, 0.5], [1.5, 3.0]]),
            env.start, env, initial_control=np.zeros(2))
        u = np.vstack((np.zeros((1, 2)), proposal.controls))
        self.assertLessEqual(np.max(np.abs(np.diff(u[:, 0]))),
                             1.0 * 0.05 + 1e-12)
        self.assertLessEqual(np.max(np.abs(np.diff(u[:, 1]))),
                             3.0 * 0.05 + 1e-12)
        self.assertGreater(np.max(np.abs(proposal.controls[:, 1])), 0.1)

    def test_collision_marks_proposal_infeasible(self):
        env = GridEnv(obstacle=(np.array([0.0, 0.75]), 0.15))
        proposal = branch_to_control_sequence(
            branch([[0.0, 0.5], [0.0, 3.5]]), env.start, env)
        self.assertFalse(proposal.feasible)
        self.assertLess(proposal.min_clearance, 0.0)
        self.assertIn("clearance", proposal.reason)

    def test_local_controller_exposes_one_proposal_per_branch(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, K=8, T=20, flow_res=0.1, window=3.8,
            extract_branches=True, max_branches=3)
        ctrl.prepare(env.start)
        self.assertEqual(len(ctrl.branch_proposals), len(ctrl.branches))
        self.assertGreaterEqual(len(ctrl.branch_proposals), 2)
        self.assertTrue(all(p.controls.shape == (20, 2)
                            for p in ctrl.branch_proposals))

    def test_rollout_matches_mppi_prediction_model(self):
        env = GridEnv()
        proposal = branch_to_control_sequence(
            branch([[0.0, 0.5], [0.8, 2.0], [0.0, 3.5]]),
            env.start, env, T=24)
        ctrl = MPPI(env, K=2, T=24)
        expected = ctrl.rollout(env.start, proposal.controls[None])[0]
        np.testing.assert_allclose(proposal.rollout, expected)


if __name__ == "__main__":
    unittest.main()
