"""Bookkeeping tests for the proposal attribution experiment."""
import unittest

from experiments.attribution_ablation import CONFIGS, _summarize
from controllers import AmoebaHybrid
from tests.test_pseudopods import GridEnv

import numpy as np


class AttributionTests(unittest.TestCase):
    def test_matrix_isolates_path_branches_and_commitment(self):
        self.assertFalse(CONFIGS["v0"]["grouped_sampling"])
        self.assertFalse(CONFIGS["path_only"]["use_branches"])
        self.assertTrue(CONFIGS["path_branches"]["use_branches"])
        self.assertEqual(CONFIGS["no_commit"]["mode_min_dwell"], 0)
        self.assertEqual(CONFIGS["no_commit"]["mode_confirm_cycles"], 1)

    def test_summary_keeps_scenarios_separate(self):
        base = dict(
            config="path_branches", success=1, result="success", time_s=10.0,
            path_len_m=4.0, min_clear_m=0.1, omega_sign_flips=2,
            mode_switches=1, branch_selection_fraction=0.2,
            mean_path_error_m=0.1)
        rows = [dict(base, scenario="normal"),
                dict(base, scenario="blocked")]
        summary = _summarize(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual({item["scenario"] for item in summary},
                         {"normal", "blocked"})

    def test_nav2_like_gate_marks_blocked_path_points_invalid(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.5))
        path = np.column_stack((
            np.zeros(80), np.linspace(0.5, 5.8, 80)))
        ctrl = AmoebaHybrid(
            env, path=path, K=8, T=10, flow_res=0.1,
            path_validity_gate=True)
        self.assertTrue(ctrl.track_valid.any())
        self.assertFalse(ctrl.track_valid.all())


if __name__ == "__main__":
    unittest.main()
