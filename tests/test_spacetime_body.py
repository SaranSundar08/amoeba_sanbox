"""SpaceTimeBody (sandbox reference for the C++ space_time_body). Mirrors
nav2_tgmppi_controller/test/space_time_body_test.cpp."""
import numpy as np
import pytest

from spacetime_body import SpaceTimeBody

ALL_FREE = lambda pts: np.ones(len(pts), bool)


def goal_dist(gx, gy):
    return lambda pts: np.hypot(pts[:, 0] - gx, pts[:, 1] - gy)


def obstacles(*rows):
    """rows of (x, y, vx, vy, r)"""
    a = np.array(rows, float).reshape(-1, 5)
    return a[:, :2], a[:, 2:4], a[:, 4]


def check_routes(b, sf, obs, robot_r, start=(0.0, 0.0)):
    for path in b.pods:
        assert path.shape == (b.K + 1, 2)
        assert np.allclose(path[0], start, atol=1e-6)
        assert (np.linalg.norm(np.diff(path, axis=0), axis=1) <= b.res * 1.4143 + 1e-6).all()
        assert sf(path[1:]).all()
        if obs is not None:
            pos, vel, rad = obs
            for k in range(1, b.K + 1):
                q = pos + min(k * b.dt_layer, b.horizon) * vel
                assert (np.linalg.norm(path[k] - q, axis=1) > rad + robot_r - 1e-6).all()


def test_free_space():
    b = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0))
    assert b.ready and b.K == 12
    assert b.promises[0] == pytest.approx(4.0, abs=0.1)
    assert b.pods[0][-1] == pytest.approx([1.2, 0.0], abs=0.06)
    assert len(b.pods) == 3 and b.classes_found == 1
    check_routes(b, ALL_FREE, None, 0.34)


def test_static_obstacle_gives_above_and_below_classes():
    obs = obstacles((0.9, 0, 0, 0, 0.25))
    b0 = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0))
    b = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0), obs)
    assert b.classes_found >= 2 and len(b.pods) >= 2
    s = [sig[0] for sig in b.signatures]
    assert any(x * y == -1 for x in s for y in s)          # opposite sides
    assert b.promises[0] > b0.promises[0] + 0.01
    check_routes(b, ALL_FREE, obs, 0.34)


def test_moving_obstacle_routes_are_valid_and_cannot_improve_the_best():
    obs = obstacles((0.8, 1.6, 0, -1.0, 0.25))
    b0 = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0))
    b = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0), obs)
    assert b.ready and b.pods
    assert b.promises[0] >= b0.promises[0] - 1e-6
    check_routes(b, ALL_FREE, obs, 0.34)


def test_blocked_root_still_routes_out():
    sf = lambda pts: np.hypot(pts[:, 0], pts[:, 1]) > 0.05
    assert SpaceTimeBody(sf, goal_dist(4, 0), (0, 0)).ready


def test_enclosed_is_not_ready_and_does_not_crash():
    b = SpaceTimeBody(lambda pts: np.zeros(len(pts), bool), goal_dist(4, 0), (0, 0))
    assert not b.ready and b.pods == []


def test_static_wall_is_never_entered():
    wall = lambda pts: ~((pts[:, 0] > 0.5) & (pts[:, 0] < 0.7) & (pts[:, 1] < 0.4))
    b = SpaceTimeBody(wall, goal_dist(4, 0), (0, 0))
    assert b.ready and b.pods
    check_routes(b, wall, None, 0.34)


def test_obstacle_already_inside_the_clearance_is_relaxed_not_fatal():
    obs = obstacles((0.5, 0, 0, 0, 0.25))                   # 0.5 m away, wants 0.75
    b = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0), obs, robot_r=0.5)
    assert b.ready and b.pods
    for path in b.pods:
        assert (np.hypot(path[1:, 0] - 0.5, path[1:, 1]) > 0.48 - 1e-6).all()


def test_larger_clearance_radius_is_respected():
    obs = obstacles((0.9, 0, 0, 0, 0.25))
    b = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0), obs, robot_r=0.5)
    assert b.ready and b.pods
    check_routes(b, ALL_FREE, obs, 0.5)


def test_scattered_scene_is_fast_and_valid():
    rng = np.random.default_rng(12345)
    pos = []
    while len(pos) < 20:
        p = rng.uniform(-3, 3, 2)
        if np.hypot(*p) >= 1.0:
            pos.append(p)
    obs = (np.array(pos), rng.uniform(-0.5, 0.5, (20, 2)), 0.25)
    b = SpaceTimeBody(ALL_FREE, goal_dist(4, 0), (0, 0), obs)
    assert b.ready and b.pods
    check_routes(b, ALL_FREE, (obs[0], obs[1], np.full(20, 0.25)), 0.34)
    assert b.build_ms < 50.0
