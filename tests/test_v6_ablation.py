"""Tests for V6 model-ablation bookkeeping."""
import unittest

from experiments.v6_ablation import CONFIGS, _summarize


class V6AblationTests(unittest.TestCase):
    def test_configs_isolate_robot_model(self):
        self.assertEqual(CONFIGS["ideal"], {"robot_model": "ideal"})
        self.assertEqual(CONFIGS["slip"]["robot_model"], "slip")

    def test_summary_separates_models(self):
        base = dict(
            scenario="normal", world=48, seed=0, success=1,
            result="success", time_s=20.0, path_len_m=9.0,
            min_clear_m=0.1, stall_s=0.1, omega_sign_flips=10,
            mode_switches=2, mean_ess=20.0, mean_path_error_m=0.1)
        summary = _summarize([
            dict(base, config="ideal"),
            dict(base, config="slip", min_clear_m=0.2),
        ])
        self.assertEqual(len(summary), 2)
        slip = next(row for row in summary if row["config"] == "slip")
        self.assertEqual(slip["worst_clear_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
