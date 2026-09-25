"""RandomDynEnv: controllable count, seeded randomness, keep-outs, DynaBarn-compatible API."""
import numpy as np
import pytest

from randdyn import KEEPOUT_R, RandomDynEnv


def _run(env, steps, dt=0.05):
    for _ in range(steps):
        env.step_time(dt)


@pytest.mark.parametrize("n", [0, 1, 10, 20, 35])
def test_count_is_controllable(n):
    env = RandomDynEnv(n_moving=n, seed=3)
    assert env.moving_obs().shape == (n, 2)
    assert env.moving_velocity().shape == (n, 2)


@pytest.mark.parametrize("motion", ["waypoint", "bounce"])
def test_same_seed_same_scene_different_seed_differs(motion):
    a, b, c = (RandomDynEnv(10, seed=s, motion=motion) for s in (5, 5, 6))
    _run(a, 300), _run(b, 300), _run(c, 300)
    assert np.allclose(a.moving_obs(), b.moving_obs())
    assert not np.allclose(a.moving_obs(), c.moving_obs())


@pytest.mark.parametrize("motion", ["waypoint", "bounce"])
def test_obstacles_stay_in_the_room_and_speeds_are_in_range(motion):
    env = RandomDynEnv(20, seed=1, motion=motion, speed_range=(0.2, 0.6))
    for _ in range(40):
        _run(env, 50)
        p = env.moving_obs()
        assert (p[:, 0] > env.xmin).all() and (p[:, 0] < env.xmax).all()
        assert (p[:, 1] > env.ymin).all() and (p[:, 1] < env.ymax).all()
        s = np.linalg.norm(env.moving_velocity(), axis=1)
        assert ((s > 0.2 - 1e-6) & (s < 0.6 + 1e-6)).all() or motion == "waypoint" and (s < 0.6 + 1e-6).all()


def test_initial_positions_respect_keepouts():
    for seed in range(10):
        env = RandomDynEnv(25, seed=seed)
        for c in (env.start[:2], env.goal):
            assert (np.linalg.norm(env.moving_obs() - c, axis=1) >= KEEPOUT_R - 1e-9).all()


def test_waypoint_velocity_actually_changes_over_time():
    env = RandomDynEnv(10, seed=2, motion="waypoint")
    v0 = env.moving_velocity().copy()
    _run(env, 2000)                       # 100 s: every obstacle has reached a waypoint
    assert not np.allclose(env.moving_velocity(), v0)


def test_prediction_is_constant_velocity_from_now():
    env = RandomDynEnv(6, seed=4)
    t = np.array([0.0, 0.5, 1.0])
    pred = env.predicted_moving_obs(t, horizon=2.0)
    assert pred.shape == (3, 6, 2)
    assert np.allclose(pred[0], env.moving_obs())
    assert np.allclose(pred[2], env.moving_obs() + 1.0 * env.moving_velocity())


def test_clearance_sees_static_walls_and_live_obstacles():
    env = RandomDynEnv(1, seed=0)
    o = env.moving_obs()[0]
    assert env.clearance(o[None]) [0] == pytest.approx(-env.obstacle_r, abs=1e-6)
    assert env.static_clearance(o[None])[0] > 0
    assert env.clearance(np.array([[env.xmin, 5.0]]))[0] == pytest.approx(0.0, abs=env._cres)


def test_zero_obstacles_is_a_valid_empty_room():
    env = RandomDynEnv(0)
    env.step_time(0.05)
    assert env.clearance(np.array([[0.0, 5.0]]))[0] > 3.0
    c, r = env.obstacle_centers()
    assert c.shape == (0, 2)


def test_full_episode_runs_with_the_real_controller():
    from astar import astar_path
    from controllers import AmoebaHybrid
    from simulation import episode
    env = RandomDynEnv(6, seed=11)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = AmoebaHybrid(env, seed=0, path=path, extract_branches=True, max_branches=3,
                        grouped_sampling=True, dynamic_prediction=True,
                        fallback_share=0.25, gate="auto", path_validity_gate=True)
    m = episode(env, ctrl, max_steps=60)
    assert m["result"] in ("timeout", "collision", "success")
    assert np.isfinite(m["min_clear_m"])
