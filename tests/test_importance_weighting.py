"""Mathematical tests for V4 Gaussian-mixture importance weighting."""

import unittest

import numpy as np

from controllers import AmoebaLocal
from grouped_sampling import (
    SamplingMode, diagonal_gaussian_logpdf, gaussian_mixture_logpdf,
    guard_importance_statistics, mixture_importance_weights)
from tests.test_pseudopods import GridEnv


class ImportanceWeightingTests(unittest.TestCase):
    def test_standard_gaussian_matches_closed_form(self):
        samples = np.array([[0.0, 0.0], [1.0, -2.0]])
        actual = diagonal_gaussian_logpdf(
            samples, np.zeros(2), np.ones(2))
        expected = -0.5 * (
            np.sum(samples ** 2, axis=1) + 2.0 * np.log(2.0 * np.pi))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_diagonal_gaussian_sums_complete_control_sequence(self):
        sample = np.zeros((1, 3, 2))
        mean = np.zeros((3, 2))
        std = np.array([0.2, 0.6])
        actual = diagonal_gaussian_logpdf(sample, mean, std)[0]
        per_step = -np.log(2.0 * np.pi * std[0] * std[1])
        self.assertAlmostEqual(actual, 3.0 * per_step)

    def test_mixture_matches_manual_two_component_density(self):
        samples = np.array([[-0.5], [0.25], [2.0]])
        means = np.array([[-1.0], [1.0]])
        weights = np.array([0.25, 0.75])
        actual = gaussian_mixture_logpdf(
            samples, means, np.ones(1), weights)
        normalizer = np.sqrt(2.0 * np.pi)
        manual = np.log(
            weights[0] * np.exp(-0.5 * (samples[:, 0] + 1.0) ** 2)
            / normalizer
            + weights[1] * np.exp(-0.5 * (samples[:, 0] - 1.0) ** 2)
            / normalizer)
        np.testing.assert_allclose(actual, manual, rtol=1e-13, atol=1e-13)

    def test_mixture_supports_component_specific_standard_deviations(self):
        samples = np.array([[0.0], [1.0]])
        means = np.array([[0.0], [1.0]])
        stds = np.array([[0.25], [2.0]])
        weights = np.array([0.4, 0.6])
        actual = gaussian_mixture_logpdf(samples, means, stds, weights)
        manual = []
        for sample in samples[:, 0]:
            densities = [
                np.exp(-0.5 * ((sample - mean[0]) / std[0]) ** 2)
                / (np.sqrt(2.0 * np.pi) * std[0])
                for mean, std in zip(means, stds)
            ]
            manual.append(np.log(np.dot(weights, densities)))
        np.testing.assert_allclose(actual, manual, rtol=1e-13, atol=1e-13)

    def test_mixture_log_density_is_stable_in_far_tail(self):
        value = gaussian_mixture_logpdf(
            np.array([[1.0e6]]), np.array([[-1.0], [1.0]]),
            np.ones(1), np.array([0.5, 0.5]))
        self.assertTrue(np.isfinite(value[0]))

    def test_p_equals_q_reduces_to_vanilla_cost_weights(self):
        rng = np.random.default_rng(2)
        mean = np.zeros((4, 2))
        mode = SamplingMode(-1, "fallback", mean)
        raw = rng.normal(size=(12, 4, 2)) * np.array([0.2, 0.6])
        costs = np.linspace(0.0, 3.0, len(raw))
        weights, _, log_p, log_q, _ = mixture_importance_weights(
            raw, costs, mean, [mode], np.array([0.2, 0.6]), [1.0], 0.3)
        expected = np.exp(-(costs - costs.min()) / 0.3)
        expected /= expected.sum()
        np.testing.assert_allclose(log_p, log_q, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(weights, expected, rtol=1e-13, atol=1e-13)

    def test_controller_reports_complete_v4_diagnostics(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, seed=3, K=60, T=20, flow_res=0.1, window=3.8,
            extract_branches=True, grouped_sampling=True,
            importance_weighting=True)
        _, info = ctrl.step(env.start)
        self.assertEqual(info["weighting"], "v4_mixture_importance")
        self.assertAlmostEqual(float(info["importance_weights"].sum()), 1.0)
        self.assertEqual(info["log_p"].shape, (ctrl.K,))
        self.assertEqual(info["log_q"].shape, (ctrl.K,))
        self.assertEqual(len(info["mixture_weights"]),
                         len(info["mode_stats"]))
        self.assertTrue(np.all(np.isfinite(info["log_importance_weights"])))
        selected = info["selected_mode"]["index"]
        self.assertTrue(np.all(
            info["weights"][info["mode_labels"] != selected] == 0.0))
        self.assertAlmostEqual(float(info["weights"].sum()), 1.0)

    def test_ess_guard_falls_back_complete_cycle(self):
        corrected_stats = [
            dict(index=0, key=10, name="branch", count=100,
                 free_energy=3.0, min_cost=np.nan, ess=1.5),
            dict(index=1, key=-1, name="path", count=100,
                 free_energy=2.0, min_cost=np.nan, ess=20.0),
        ]
        uncorrected_stats = [
            dict(index=0, key=10, name="branch", count=100,
                 free_energy=1.0, min_cost=0.5, ess=30.0),
            dict(index=1, key=-1, name="path", count=100,
                 free_energy=4.0, min_cost=1.0, ess=40.0),
        ]
        labels = np.repeat([0, 1], 100)
        corrected_weights = np.full(200, 0.01)
        uncorrected_weights = np.concatenate((
            np.full(100, 0.01),
            np.linspace(1.0, 2.0, 100)))
        uncorrected_weights[100:] /= uncorrected_weights[100:].sum()
        stats, weights, fallback_count = guard_importance_statistics(
            corrected_stats, corrected_weights,
            uncorrected_stats, uncorrected_weights, labels)
        self.assertEqual(fallback_count, 1)
        self.assertTrue(stats[0]["ess_guard_fallback"])
        self.assertTrue(stats[0]["ess_guard_trigger"])
        self.assertEqual(stats[0]["weight_source"], "v3_cycle_fallback")
        self.assertEqual(stats[0]["free_energy"], 1.0)
        self.assertTrue(stats[1]["ess_guard_fallback"])
        self.assertFalse(stats[1]["ess_guard_trigger"])
        self.assertEqual(stats[1]["weight_source"], "v3_cycle_fallback")
        self.assertEqual(stats[1]["free_energy"], 4.0)
        np.testing.assert_allclose(weights[:100], uncorrected_weights[:100])
        np.testing.assert_allclose(weights[100:], uncorrected_weights[100:])

    def test_controller_reports_v4_1_guard_diagnostics(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, seed=3, K=60, T=20, flow_res=0.1, window=3.8,
            extract_branches=True, grouped_sampling=True,
            importance_weighting=True, importance_ess_guard=True,
            importance_min_ess=1.0e6)
        _, info = ctrl.step(env.start)
        self.assertEqual(
            info["weighting"], "v4_1_ess_guarded_mixture_importance")
        self.assertTrue(info["importance_guard_enabled"])
        self.assertEqual(info["importance_guard_cycle_fallbacks"],
                         1)
        self.assertEqual(info["importance_guard_mode_fallbacks"],
                         len(info["mode_stats"]))
        self.assertTrue(all(
            item["weight_source"] == "v3_cycle_fallback"
            for item in info["mode_stats"]))
        self.assertTrue(all(
            "corrected_ess" in item for item in info["mode_stats"]))

    def test_v5_covariances_are_used_with_v4_density(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, seed=3, K=60, T=20, flow_res=0.1, window=3.8,
            extract_branches=True, grouped_sampling=True,
            importance_weighting=True, importance_ess_guard=True,
            adaptive_covariance=True)
        _, info = ctrl.step(env.start)
        self.assertEqual(
            info["weighting"],
            "v5_geometry_adaptive_ess_guarded_importance")
        self.assertEqual(info["mode_stds"].shape,
                         (len(info["mode_stats"]), ctrl.T, 2))
        self.assertTrue(np.all(np.isfinite(info["log_p"])))
        self.assertTrue(np.all(np.isfinite(info["log_q"])))


if __name__ == "__main__":
    unittest.main()
