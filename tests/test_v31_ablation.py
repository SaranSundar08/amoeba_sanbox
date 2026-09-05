"""Tests for V3.1 ablation bookkeeping (not the expensive BARN episodes)."""
import unittest

from experiments.v31_ablation import CONFIGS, _summarize


class AblationTests(unittest.TestCase):
    def test_matrix_contains_baseline_allocation_and_switch_variants(self):
        self.assertIn("baseline", CONFIGS)
        self.assertEqual(CONFIGS["path25"]["fallback_share"], 0.25)
        self.assertEqual(CONFIGS["default"]["fallback_share"], 0.50)
        self.assertEqual(CONFIGS["path70"]["fallback_share"], 0.70)
        self.assertGreater(
            CONFIGS["conservative"]["mode_min_dwell"],
            CONFIGS["default"]["mode_min_dwell"])

    def test_summary_aggregates_safety_and_stability_metrics(self):
        rows = [
            dict(config="default", success=1, result="success", time_s=10.0,
                 path_len_m=4.0, min_clear_m=0.12, omega_sign_flips=2,
                 mode_switches=1, fallback_fraction=0.75, mean_ess=20.0,
                 mean_path_error_m=0.1),
            dict(config="default", success=0, result="collision", time_s=5.0,
                 path_len_m=2.0, min_clear_m=-0.01, omega_sign_flips=4,
                 mode_switches=3, fallback_fraction=0.50, mean_ess=10.0,
                 mean_path_error_m=0.2),
        ]
        summary = _summarize(rows)[0]
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["collisions"], 1)
        self.assertEqual(summary["worst_clear_m"], -0.01)
        self.assertEqual(summary["mean_sign_flips"], 3.0)


if __name__ == "__main__":
    unittest.main()
