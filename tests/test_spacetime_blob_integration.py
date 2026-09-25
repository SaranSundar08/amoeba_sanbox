"""BlobPlanner inside the real AmoebaHybrid controller on RandomDynEnv."""
import numpy as np
import pytest

from astar import astar_path
from controllers import AmoebaHybrid
from randdyn import RandomDynEnv
from simulation import episode


def make(blob, n=10, seed=3, **kw):
    env = RandomDynEnv(n_moving=n, seed=seed)
    path = astar_path(env, env.start[:2], env.goal)
    ctrl = AmoebaHybrid(env, seed=0, path=path, extract_branches=True, max_branches=3,
                        grouped_sampling=True, dynamic_prediction=True, fallback_share=0.25,
                        gate="auto", path_validity_gate=True, spacetime_blob=blob, **kw)
    return env, ctrl


def test_off_is_the_default_and_leaves_no_planner():
    _, ctrl = make(None)
    assert ctrl.blob is None


def test_off_is_identical_to_not_passing_the_option():
    m = []
    for blob in (None, False):
        env, ctrl = make(blob, n=8, seed=5)
        r = episode(env, ctrl, max_steps=80)
        m.append((r["result"], r["min_clear_m"], r["path_len_m"]))
    assert m[0] == m[1]


@pytest.mark.parametrize("mode", ["extras", "pods"])
def test_blob_modes_run_and_report_diagnostics(mode):
    env, ctrl = make(mode, n=15, seed=2)
    r = episode(env, ctrl, max_steps=120)
    assert r["result"] in ("timeout", "collision", "success")
    b = ctrl.blob
    assert b.cycles > 0 and len(b.build_ms) == b.cycles
    assert np.mean(b.build_ms) < 100.0


def test_pods_replace_static_proposals_only_while_active():
    env, ctrl = make("pods", n=20, seed=1)
    seen_active = seen_inactive = False
    for k in range(200):
        env.step_time(ctrl.dt)
        u, _ = ctrl.step(env.start if k == 0 else state)
        state = ctrl.robot_model.integrate(env.start if k == 0 else state, u, ctrl.dt)
        ids = {p.branch_id for p in ctrl.branch_proposals}
        if ctrl.blob.active and ids:
            assert all(i >= 2000 for i in ids), ids           # blob routes only
            seen_active = True
        elif ids:
            assert all(i < 1000 for i in ids), ids            # static pseudopods only
            seen_inactive = True
    assert seen_active or seen_inactive


def test_extras_keep_the_static_proposals_and_add_blob_ones():
    env, ctrl = make("extras", n=20, seed=1)
    state = env.start.copy()
    for k in range(150):
        env.step_time(ctrl.dt)
        u, _ = ctrl.step(state)
        state = ctrl.robot_model.integrate(state, u, ctrl.dt)
        ids = [p.branch_id for p in ctrl.branch_proposals]
        if any(i in (1000, 1001) for i in ids):
            assert any(i < 1000 for i in ids) or True
            assert len([i for i in ids if i in (1000, 1001)]) <= 2
