"""V6.1 dynamic-obstacle prediction tests."""
import numpy as np

from dynabarn import DynaBarnEnv
from env import CYL_R
from mppi import MPPI


def _one_obstacle_env():
    env = DynaBarnEnv(42, n_moving=1, speed=0.4, amplitude=0.6, seed=7)
    assert len(env.mov_center) == 1
    return env


def test_constant_velocity_prediction_starts_at_current_observation():
    env = _one_obstacle_env()
    env.t = 0.37
    prediction = env.predicted_moving_obs(np.array([0.0, 0.5]))
    np.testing.assert_allclose(prediction[0], env.moving_obs())
    np.testing.assert_allclose(
        prediction[1], env.moving_obs() + 0.5 * env.moving_velocity())


def test_prediction_horizon_holds_last_extrapolated_position():
    env = _one_obstacle_env()
    prediction = env.predicted_moving_obs(
        np.array([1.0, 3.0]), horizon=1.0)
    np.testing.assert_allclose(prediction[0], prediction[1])


def test_time_indexed_clearance_tracks_predicted_centres():
    env = _one_obstacle_env()
    tau = np.array([0.0, 0.5])
    centres = env.predicted_moving_obs(tau)[:, 0]
    points = centres[:, None, :]
    clearance = env.predicted_clearance(
        points, tau, uncertainty_rate=0.0)
    np.testing.assert_allclose(clearance[:, 0], -CYL_R, atol=1e-12)


def test_mppi_prediction_is_opt_in_and_shape_preserving():
    env = _one_obstacle_env()
    states = np.tile(env.start, (3, 8, 1))
    current = MPPI(env, K=3, T=8, dynamic_prediction=False)
    predicted = MPPI(env, K=3, T=8, dynamic_prediction=True)
    assert current.rollout_clearance(states).shape == (3, 8)
    assert predicted.rollout_clearance(states).shape == (3, 8)
