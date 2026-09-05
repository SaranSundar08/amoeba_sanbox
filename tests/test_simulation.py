"""Tests for the plan/plant separation in `simulation.episode`."""
import unittest

import numpy as np

from env import BarnEnv
from mppi import MPPI
from robot_model import sample_plant_mismatch
from simulation import episode


def _slip_controller():
    env = BarnEnv(48)
    ctrl = MPPI(env, seed=0, K=64, T=20, robot_model="slip",
                footprint_length=0.90, footprint_width=0.65)
    return env, ctrl


class SimulationPlantMismatchTests(unittest.TestCase):
    def test_default_plant_model_is_the_controller_model(self):
        env, ctrl = _slip_controller()
        result = episode(env, ctrl, max_steps=10)
        self.assertIn("trace", result)
        self.assertEqual(result["trace"].shape[1], 3)

    def test_mismatched_plant_model_changes_the_executed_trace(self):
        env, ctrl = _slip_controller()
        baseline = episode(env, ctrl, max_steps=15)

        env2, ctrl2 = _slip_controller()
        plant = sample_plant_mismatch(
            ctrl2.robot_model, np.random.default_rng(5),
            yaw_gain_range=(1.6, 1.6), speed_yaw_loss_range=(1.6, 1.6))
        mismatched = episode(env2, ctrl2, max_steps=15, plant_model=plant)

        # Same seed, same starting conditions, same planner -- the only
        # difference is what the plant actually does with the commanded
        # (v, omega). A real yaw-response mismatch has to show up in the
        # executed trace once the controller commands any turning.
        self.assertFalse(np.allclose(baseline["trace"], mismatched["trace"]))

    def test_matched_plant_model_reproduces_the_default_result(self):
        env, ctrl = _slip_controller()
        baseline = episode(env, ctrl, max_steps=15)

        env2, ctrl2 = _slip_controller()
        explicit_match = episode(
            env2, ctrl2, max_steps=15, plant_model=ctrl2.robot_model)

        np.testing.assert_allclose(
            baseline["trace"], explicit_match["trace"])


if __name__ == "__main__":
    unittest.main()
