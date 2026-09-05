import numpy as np

from junction_env import ScriptedJunctionEnv


def test_crossing_peer_reaches_junction_at_six_seconds():
    env = ScriptedJunctionEnv("crossing")
    np.testing.assert_allclose(env.moving_obs(6.0)[0], [0.0, 4.0])


def test_junction_corridors_are_free_and_quadrants_are_occupied():
    env = ScriptedJunctionEnv("crossing")
    assert env.static_clearance(np.array([[0.0, 4.0]]))[0] > env.robot_r
    assert env.static_clearance(np.array([[2.0, 2.0]]))[0] < 0.0


def test_prediction_is_time_indexed_and_robot_sized():
    env = ScriptedJunctionEnv("crossing")
    env.t = 1.0
    tau = np.array([0.0, 1.0])
    centers = env.predicted_moving_obs(tau)[:, 0]
    points = centers[:, None]
    clear = env.predicted_clearance(points, tau, uncertainty_rate=0.0)
    np.testing.assert_allclose(clear[:, 0], -env.peer_r, atol=1e-12)


def test_blocked_split_retains_an_open_alternative_route():
    env = ScriptedJunctionEnv("blocked_split")
    assert env.static_clearance(np.array([[-0.9, 4.0]]))[0] > env.robot_r
    assert env.static_clearance(np.array([[0.9, 4.0]]))[0] > env.robot_r
