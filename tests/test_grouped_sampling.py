"""Focused V3 tests; runnable with ``python3 -m unittest -v``."""
import unittest

import numpy as np

from controllers import AmoebaHybrid, AmoebaLocal
from grouped_sampling import (
    SamplingMode, allocate_group_counts, allocate_weighted_counts,
    project_control_sequences,
    sample_fixed_groups, select_mode)
from tests.test_pseudopods import GridEnv


class GroupedSamplingTests(unittest.TestCase):
    def test_allocation_is_complete_and_nearly_equal(self):
        counts = allocate_group_counts(1024, 3)
        self.assertEqual(int(counts.sum()), 1024)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_weighted_allocation_protects_fallback_capacity(self):
        counts = allocate_weighted_counts(100, [0.25, 0.25, 0.50])
        self.assertEqual(list(counts), [25, 25, 50])

    def test_fixed_groups_preserve_labels_and_means(self):
        means = [np.full((4, 2), value) for value in (1.0, 2.0, 3.0)]
        modes = [SamplingMode(i, f"P{i}", mean)
                 for i, mean in enumerate(means)]
        raw, labels, counts = sample_fixed_groups(
            np.random.default_rng(4), modes, 10, np.zeros(2))
        self.assertEqual(list(counts), [4, 3, 3])
        for index, mean in enumerate(means):
            expected = np.broadcast_to(
                mean, (int(counts[index]),) + mean.shape)
            np.testing.assert_allclose(raw[labels == index], expected)

    def test_fixed_groups_use_mode_specific_covariance(self):
        mean = np.zeros((100, 2))
        modes = [
            SamplingMode(0, "narrow", mean,
                         std=np.full_like(mean, 0.01)),
            SamplingMode(1, "wide", mean,
                         std=np.full_like(mean, 1.0)),
        ]
        raw, labels, _ = sample_fixed_groups(
            np.random.default_rng(4), modes, 400, np.ones(2))
        narrow = float(raw[labels == 0].std())
        wide = float(raw[labels == 1].std())
        self.assertLess(narrow, 0.02)
        self.assertGreater(wide, 0.9)

    def test_v5_covariance_tracks_clearance_and_turning(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, seed=2, K=60, T=24, flow_res=0.1, window=3.8,
            extract_branches=True, grouped_sampling=True,
            adaptive_covariance=True)
        _, info = ctrl.step(env.start)
        self.assertEqual(info["weighting"],
                         "v5_geometry_adaptive_v3_weights")
        self.assertEqual(info["mode_stds"].shape,
                         (len(info["mode_stats"]), ctrl.T, 2))
        self.assertTrue(np.all(info["mode_stds"] > 0.0))
        self.assertGreater(float(np.ptp(info["mode_stds"][..., 0])), 0.01)
        self.assertGreater(float(np.ptp(info["mode_stds"][..., 1])), 0.01)

    def test_projection_enforces_box_and_rate_limits(self):
        raw = np.zeros((5, 8, 2))
        raw[..., 0] = 4.0
        raw[..., 1] = -8.0
        projected = project_control_sequences(
            raw, np.zeros(2), dt=0.05, v_max=0.5, w_max=1.9,
            v_accel_max=1.0, w_accel_max=3.0)
        previous = np.concatenate(
            (np.zeros((5, 1, 2)), projected), axis=1)
        self.assertLessEqual(float(projected[..., 0].max()), 0.5)
        self.assertLessEqual(float(np.abs(projected[..., 1]).max()), 1.9)
        self.assertLessEqual(float(np.abs(np.diff(
            previous[..., 0], axis=1)).max()), 0.05 + 1e-12)
        self.assertLessEqual(float(np.abs(np.diff(
            previous[..., 1], axis=1)).max()), 0.15 + 1e-12)

    def test_mode_selection_has_hysteresis(self):
        stats = [
            {"key": 1, "free_energy": 10.0},
            {"key": 2, "free_energy": 9.8},
        ]
        self.assertEqual(
            select_mode(stats, previous_key=1, switch_margin=0.3)["key"], 1)
        self.assertEqual(
            select_mode(stats, previous_key=1, switch_margin=0.1)["key"], 2)

    def test_controller_keeps_modes_separate_and_fallback_present(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        batch_size = 90
        ctrl = AmoebaLocal(
            env, seed=2, K=batch_size, T=24, flow_res=0.1, window=3.8,
            extract_branches=True, max_branches=3, grouped_sampling=True,
            fallback_share=0.5)
        _, info = ctrl.step(env.start)
        self.assertEqual(info["weighting"], "v3_uncorrected_within_mode")
        self.assertEqual(info["raw_controls"].shape, (batch_size, 24, 2))
        self.assertEqual(int(info["mode_counts"].sum()), batch_size)
        self.assertIn(-1, [item["key"] for item in info["mode_stats"]])
        selected_index = info["selected_mode"]["index"]
        self.assertTrue(np.all(
            info["weights"][info["mode_labels"] != selected_index] == 0.0))
        self.assertAlmostEqual(float(info["weights"].sum()), 1.0)
        fallback = next(item for item in info["mode_stats"]
                        if item["key"] == -1)
        self.assertGreaterEqual(fallback["count"], batch_size // 2 - 1)
        self.assertIn(info["selection_reason"], {
            "initial", "best", "minimum_dwell", "switch_margin",
            "awaiting_confirmation", "confirmed_improvement",
            "current_unavailable"})

    def test_each_active_mode_gets_a_persistent_warm_start(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, seed=3, K=60, T=20, flow_res=0.1, window=3.8,
            extract_branches=True, grouped_sampling=True)
        ctrl.step(env.start)
        first_keys = set(ctrl.mode_nominals)
        self.assertIn(-1, first_keys)
        self.assertGreaterEqual(len(first_keys), 3)
        moved = env.start.copy()
        moved[1] += 0.02
        ctrl.step(moved)
        self.assertTrue(first_keys.intersection(ctrl.mode_nominals))

    def test_hybrid_uses_global_path_as_protected_fallback(self):
        env = GridEnv(obstacle=(np.array([1.5, 2.5]), 0.3))
        path = np.column_stack((
            np.zeros(80), np.linspace(0.5, 5.8, 80)))
        ctrl = AmoebaHybrid(
            env, path=path, seed=1, K=60, T=20, flow_res=0.1,
            window=3.0, extract_branches=True, grouped_sampling=True)
        _, info = ctrl.step(env.start)
        fallback = next(item for item in info["mode_stats"]
                        if item["key"] == -1)
        self.assertEqual(fallback["name"], "path")
        self.assertIsNotNone(ctrl.global_path_proposal)
        self.assertTrue(ctrl.global_path_proposal.feasible)

    def test_selected_commands_remain_rate_limited_across_cycles(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        ctrl = AmoebaLocal(
            env, seed=5, K=60, T=20, flow_res=0.1, window=3.8,
            extract_branches=True, grouped_sampling=True,
            mode_min_dwell=0, mode_confirm_cycles=1)
        state = env.start.copy()
        commands = [np.zeros(2)]
        for _ in range(8):
            u, _ = ctrl.step(state)
            commands.append(u)
            state[2] += u[1] * ctrl.dt
            state[0] += u[0] * np.cos(state[2]) * ctrl.dt
            state[1] += u[0] * np.sin(state[2]) * ctrl.dt
        du = np.diff(np.asarray(commands), axis=0)
        self.assertLessEqual(float(np.abs(du[:, 0]).max()),
                             ctrl.v_accel_max * ctrl.dt + 1e-12)
        self.assertLessEqual(float(np.abs(du[:, 1]).max()),
                             ctrl.w_accel_max * ctrl.dt + 1e-12)

    def test_auto_assist_uses_path_only_when_plan_is_valid(self):
        env = GridEnv()
        path = np.column_stack((
            np.zeros(80), np.linspace(0.5, 5.8, 80)))
        ctrl = AmoebaHybrid(
            env, path=path, seed=1, K=60, T=20, flow_res=0.1,
            window=3.0, extract_branches=True, grouped_sampling=True,
            gate="auto", path_validity_gate=True)
        _, info = ctrl.step(env.start)
        self.assertEqual(info["assist_state"], "NORMAL")
        self.assertEqual([item["key"] for item in info["mode_stats"]], [-1])

    def test_auto_assist_activates_for_blocked_path_and_resets_on_replan(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        blocked = np.column_stack((
            np.zeros(80), np.linspace(0.5, 5.8, 80)))
        ctrl = AmoebaHybrid(
            env, path=blocked, seed=1, K=60, T=20, flow_res=0.1,
            window=3.8, extract_branches=True, grouped_sampling=True,
            gate="auto", path_validity_gate=True)
        _, info = ctrl.step(env.start)
        self.assertEqual(info["assist_state"], "ASSIST")
        self.assertTrue(any(item["key"] != -1 for item in info["mode_stats"]))

        detour = np.array([
            [0.0, 0.5], [-1.2, 1.6], [-1.2, 3.5], [0.0, 5.8]])
        ctrl.set_path(detour)
        self.assertEqual(ctrl.plan_version, 1)
        self.assertEqual(ctrl.assist_state, "NORMAL")


if __name__ == "__main__":
    unittest.main()
