"""V2 conversion from pseudopod geometry to feasible control-sequence means."""
from dataclasses import dataclass

import numpy as np

from env import wrap


@dataclass(frozen=True)
class BranchProposal:
    """A complete branch-conditioned MPPI proposal mean and its diagnostics."""

    branch_id: int
    controls: np.ndarray
    rollout: np.ndarray
    feasible: bool
    min_clearance: float
    progress: float
    mean_tracking_error: float
    reason: str


def _arc_length(path):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _interpolate(path, arc, query):
    q = np.clip(query, 0.0, arc[-1])
    return np.array([np.interp(q, arc, path[:, 0]),
                     np.interp(q, arc, path[:, 1])])


def _densify(path, spacing=0.05):
    """Uniform arc samples make progress independent of V1 grid sparsity."""
    arc = _arc_length(path)
    if arc[-1] < 1e-12:
        return path
    query = np.arange(0.0, arc[-1], spacing)
    dense = np.column_stack((np.interp(query, arc, path[:, 0]),
                             np.interp(query, arc, path[:, 1])))
    return np.vstack((dense, path[-1]))


def _nearest_progress(position, path, arc, previous):
    """Monotone closest-point progress, protected against branch crossings."""
    start = max(0, int(np.searchsorted(arc, previous)) - 2)
    d = np.linalg.norm(path[start:] - position, axis=1)
    index = start + int(np.argmin(d))
    return max(previous, float(arc[index]))


def branch_to_control_sequence(
        branch, state, env, T=56, dt=0.05, v_max=0.5, w_max=1.9,
        v_accel_max=1.0, w_accel_max=3.0, initial_control=None,
        lookahead=0.30, heading_gain=2.2, curvature_gain=1.0,
        curvature_slowdown=1.2, clearance_slow_band=0.45,
        min_speed_ratio=0.12, feasibility_margin=0.0,
        min_progress=0.05, robot_model=None):
    """Convert one centreline into a constrained ``[T, 2]`` mean.

    A receding pure-pursuit-like target supplies heading feedback. A second
    lookahead point estimates signed curvature for angular feedforward and
    speed reduction. The sequence is rolled forward as it is constructed, so
    every later command is conditioned on the same dynamics used by MPPI.
    """
    path = np.asarray(branch.centerline, dtype=float)
    controls = np.zeros((T, 2), dtype=float)
    rollout = np.empty((T, 3), dtype=float)
    if len(path) < 2:
        return BranchProposal(
            branch.branch_id, controls, np.tile(state, (T, 1)), False,
            -np.inf, 0.0, np.inf, "centreline has fewer than two points")

    # Remove repeated grid points, which otherwise create zero-length arcs.
    keep = np.concatenate(([True],
                           np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9))
    path = path[keep]
    if len(path) < 2:
        return BranchProposal(
            branch.branch_id, controls, np.tile(state, (T, 1)), False,
            -np.inf, 0.0, np.inf, "centreline has zero length")

    path = _densify(path)
    arc = _arc_length(path)
    x = np.asarray(state, dtype=float).copy()
    previous_u = (np.zeros(2) if initial_control is None
                  else np.asarray(initial_control, dtype=float).copy())
    previous_u[0] = np.clip(previous_u[0], 0.0, v_max)
    previous_u[1] = np.clip(previous_u[1], -w_max, w_max)
    start_progress = _nearest_progress(x[:2], path, arc, 0.0)
    progress = start_progress

    for t in range(T):
        progress = _nearest_progress(x[:2], path, arc, progress)
        p1 = _interpolate(path, arc, progress + lookahead)
        p2 = _interpolate(path, arc, progress + 2.0 * lookahead)
        delta1, delta2 = p1 - x[:2], p2 - p1
        heading1 = np.arctan2(delta1[1], delta1[0])
        heading2 = (heading1 if np.linalg.norm(delta2) < 1e-9
                    else np.arctan2(delta2[1], delta2[0]))
        heading_error = float(wrap(heading1 - x[2]))
        curvature = float(wrap(heading2 - heading1)) / max(lookahead, 1e-6)

        if robot_model is None:
            query_clearance = float(np.min(
                env.clearance(np.stack((x[:2], p1)))))
            usable = query_clearance - env.robot_r
        else:
            target_state = np.array([p1[0], p1[1], heading1])
            usable = float(np.min(robot_model.clearance(
                env, np.stack((x, target_state)))))
        clearance_factor = np.clip(
            usable / max(clearance_slow_band, 1e-6),
            min_speed_ratio, 1.0)
        curvature_factor = 1.0 / (
            1.0 + curvature_slowdown * abs(curvature))
        heading_factor = np.clip(np.cos(heading_error), min_speed_ratio, 1.0)
        desired_v = v_max * clearance_factor * curvature_factor * heading_factor
        desired_w = heading_gain * heading_error + (
            curvature_gain * desired_v * curvature)
        desired_w = np.clip(desired_w, -w_max, w_max)

        dv = np.clip(desired_v - previous_u[0],
                     -v_accel_max * dt, v_accel_max * dt)
        dw = np.clip(desired_w - previous_u[1],
                     -w_accel_max * dt, w_accel_max * dt)
        u = previous_u + np.array([dv, dw])
        u[0] = np.clip(u[0], 0.0, v_max)
        u[1] = np.clip(u[1], -w_max, w_max)
        controls[t] = u

        if robot_model is None:
            x[2] += u[1] * dt
            x[0] += u[0] * np.cos(x[2]) * dt
            x[1] += u[0] * np.sin(x[2]) * dt
        else:
            x = robot_model.integrate(x, u, dt)
        rollout[t] = x
        previous_u = u

    surface_clearance = (
        env.clearance(rollout[:, :2]) - env.robot_r
        if robot_model is None else
        robot_model.exact_clearance(env, rollout)
        - robot_model.extra_safety_margin(controls))
    min_clearance = float(np.min(surface_clearance))
    final_progress = _nearest_progress(
        rollout[-1, :2], path, arc, progress)
    achieved = final_progress - start_progress
    tracking = np.linalg.norm(
        rollout[:, None, :2] - path[None, :, :], axis=-1).min(axis=1)
    mean_tracking_error = float(tracking.mean())

    reasons = []
    if min_clearance < feasibility_margin:
        reasons.append("footprint clearance")
    if achieved < min(min_progress, arc[-1] - start_progress):
        reasons.append("insufficient progress")
    feasible = not reasons
    return BranchProposal(
        branch.branch_id, controls, rollout, feasible, min_clearance,
        float(achieved), mean_tracking_error,
        "ok" if feasible else ", ".join(reasons))
