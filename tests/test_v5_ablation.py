"""Tests for V5 covariance-ablation bookkeeping."""
import unittest

from experiments.v5_ablation import CONFIGS, _summarize


class V5AblationTests(unittest.TestCase):
    def test_configs_isolate_adaptive_covariance(self):
        self.assertEqual(CONFIGS["fixed"], {"adaptive_covariance": False})
        self.assertEqual(CONFIGS["adaptive"], {"adaptive_covariance": True})

    def test_summary_separates_scenarios_and_covariances(self):
        base = dict(
            world=48, seed=0, success=1, result="success", time_s=20.0,
            path_len_m=9.0, min_clear_m=0.1, stall_s=0.1,
            omega_sign_flips=10, mode_switches=2, mean_ess=20.0,
            mean_path_error_m=0.1)
        rows = [
            dict(base, scenario="normal", config="fixed"),
            dict(base, scenario="normal", config="adaptive",
                 min_clear_m=0.2),
            dict(base, scenario="dynamic", config="adaptive", success=0,
                 result="timeout"),
        ]
        summary = _summarize(rows)
        self.assertEqual(len(summary), 3)
        normal_adaptive = next(
            row for row in summary
            if row["scenario"] == "normal" and row["config"] == "adaptive")
        self.assertEqual(normal_adaptive["successes"], 1)
        self.assertEqual(normal_adaptive["worst_clear_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
