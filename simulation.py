"""Episode execution independent of CLI and visualization."""
import time

import numpy as np


def episode(env, ctrl, max_steps=1200, stall_eps=0.05, on_step=None,
            before_step=None, plant_model=None):
    """Run one episode. `ctrl` always plans and predicts with its own
    `robot_model`. `plant_model`, if given, is the model that actually
    EXECUTES the commanded control and is used for the ground-truth
    collision check -- pass a model with different dynamics/footprint
    parameters than `ctrl.robot_model` to measure robustness to real
    plan/plant mismatch, rather than self-consistent conservatism (the
    default, `plant_model=None`, reproduces the previous no-mismatch
    behavior: the controller's own model is also the plant).
    """
    plant_model = plant_model or ctrl.robot_model
    state = env.start.copy()
    trace = [state.copy()]
    min_clear, stall, path_len = np.inf, 0.0, 0.0
    controller_times_ms = []
    result = "timeout"
    has_time = hasattr(env, "step_time")
    for k in range(max_steps):
        if has_time:
            env.step_time(ctrl.dt)
        if before_step:
            before_step(k, state)
        controller_start = time.perf_counter()
        u, info = ctrl.step(state)
        controller_times_ms.append(
            1.0e3 * (time.perf_counter() - controller_start))
        prev = state[:2].copy()
        state = plant_model.integrate(state, u, ctrl.dt)
        trace.append(state.copy())
        path_len += np.linalg.norm(state[:2] - prev)
        c = float(plant_model.exact_clearance(env, state))
        min_clear = min(min_clear, c)
        if abs(u[0]) < stall_eps:
            stall += ctrl.dt
        if on_step:
            on_step(k, state, info, trace)
        if c < 0:
            result = "collision"
            break
        if np.linalg.norm(state[:2] - env.goal) < env.goal_tol:
            result = "success"
            break
    return {"result": result, "success": int(result == "success"),
            "time_s": round((len(trace) - 1) * ctrl.dt, 2),
            "path_len_m": round(path_len, 2),
            "min_clear_m": round(min_clear, 3),
            "stall_s": round(stall, 2),
            "controller_mean_ms": round(float(np.mean(controller_times_ms)), 3),
            "controller_p95_ms": round(float(np.percentile(
                controller_times_ms, 95)), 3),
            "controller_max_ms": round(float(np.max(controller_times_ms)), 3),
            "trace": np.array(trace)}
