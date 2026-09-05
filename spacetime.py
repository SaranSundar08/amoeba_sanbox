"""Space-time (time-expanded) topology search -- Phase 0 standalone core.

The static amoeba floods free SPACE (`env.LocalFlowField`) and reads
topologically distinct routes off it (`pseudopods.extract_pseudopods`):
left of the pillar, right of the pillar. Dynamic obstacles never enter that
picture -- they only reshape MPPI rollout COSTS afterward
(`mppi.MPPI.rollout_clearance`'s prediction layer). So against a moving
obstacle there is exactly one geometric branch, and "wait" versus "go now"
has to be stumbled into through cost-shaped sampling noise around that one
mean, rather than offered as a genuine second mode the way left/right are.

This module extends the flood one axis further: free SPACE-TIME. A moving
obstacle's predicted future position traces a tube through (x, y, t); a
route is a curve through that space, and two routes are topologically
distinct if the tube passes between them. Concretely this gives "pass
before the obstacle arrives" and "wait for it to pass" as two literally
constructed candidate routes -- the direct temporal analogue of how
`pseudopods.py` literally constructs left/right from the geodesic body,
rather than a generic k-shortest-paths search.

Deliberately NOT wired into `LocalFlowField`, `pseudopods.py`, or `mppi.py`
yet. This proves the mechanism against synthetic single-obstacle scenarios
first; integrating it into the controller (so these routes become real
sampling modes) is the next phase.
"""
import heapq

import numpy as np

from env import wrap
from grouped_sampling import project_control_sequences
from proposals import BranchProposal


def _grid_origin(start_xy, goal_xy, window, res):
    """Grid bounds covering both endpoints plus `window` of lateral room
    around each -- NOT just `window` around `start_xy` alone. A grid sized
    only around the start silently clips a distant goal to the window edge
    (`to_cell` clamps out-of-range indices), which looks like a normal
    search but is quietly solving for the wrong goal. Sized from both
    endpoints, `window` is guaranteed to mean "room to detour," never
    "how far away the goal is allowed to be."
    """
    pad = 3 * res
    x0 = min(start_xy[0], goal_xy[0]) - window - pad
    x1 = max(start_xy[0], goal_xy[0]) + window + pad
    y0 = min(start_xy[1], goal_xy[1]) - window - pad
    y1 = max(start_xy[1], goal_xy[1]) + window + pad
    nx = int(np.ceil((x1 - x0) / res)) + 1
    ny = int(np.ceil((y1 - y0) / res)) + 1
    return x0, y0, nx, ny


def time_expanded_search(env, start_xy, goal_xy, predict_obstacle,
                         obstacle_r, robot_r, horizon=3.0, dt_layer=0.25,
                         res=0.10, window=2.5, goal_tol=0.15,
                         wait_cost=None, min_arrival_k=0, max_arrival_k=None):
    """Cheapest space-time path from `start_xy` at t=0 to within `goal_tol`
    of `goal_xy`, through a local window, avoiding static obstacles
    (`env.clearance`) and one predicted moving obstacle
    (`predict_obstacle(t) -> (x, y)`, e.g. `dynabarn.DynaBarnEnv`'s
    constant-velocity `predicted_moving_obs`).

    `min_arrival_k`/`max_arrival_k` bound which discrete time layers may
    count as "arrived" -- the mechanism `two_route_search` uses to build
    the two literal route classes (forced-before, forced-after) rather than
    a generic distinct-path search. Time layers are `dt_layer` seconds
    apart; 8-connected spatial moves plus a same-cell "wait" step always
    advance exactly one layer, so waiting costs real time the same way
    moving does.

    Returns `(path, cost)`: `path` is a list of `(x, y)` one per time layer
    from t=0 to arrival (implicitly `(x, y, k * dt_layer)`), or
    `(None, inf)` if no feasible path exists within the given bounds.
    """
    x0, y0, nx, ny = _grid_origin(start_xy, goal_xy, window, res)
    nk = int(round(horizon / dt_layer)) + 1
    times = np.arange(nk) * dt_layer
    max_arrival_k = (nk - 1) if max_arrival_k is None else int(max_arrival_k)
    if max_arrival_k < 0:
        return None, np.inf
    max_arrival_k = min(max_arrival_k, nk - 1)
    wait_cost = res if wait_cost is None else float(wait_cost)

    gx, gy = np.meshgrid(x0 + np.arange(nx) * res, y0 + np.arange(ny) * res,
                         indexing="ij")
    pts = np.stack((gx, gy), axis=-1)
    static_free = env.clearance(pts) > robot_r

    def to_cell(xy):
        i = int(np.clip(round((xy[0] - x0) / res), 0, nx - 1))
        j = int(np.clip(round((xy[1] - y0) / res), 0, ny - 1))
        return i, j

    start_cell = to_cell(start_xy)
    goal_cell = to_cell(goal_xy)
    goal_radius_cells = max(1, int(round(goal_tol / res)))

    def is_goal(cell):
        return (abs(cell[0] - goal_cell[0]) <= goal_radius_cells
                and abs(cell[1] - goal_cell[1]) <= goal_radius_cells)

    dynamic_free_cache = {}

    def dynamic_free(k):
        if k not in dynamic_free_cache:
            ox, oy = predict_obstacle(times[k])
            d = np.hypot(gx - ox, gy - oy)
            dynamic_free_cache[k] = d > (obstacle_r + robot_r)
        return dynamic_free_cache[k]

    def free(i, j, k):
        return bool(static_free[i, j] and dynamic_free(k)[i, j])

    nbrs = [(di, dj, res * np.hypot(di, dj))
            for di in (-1, 0, 1) for dj in (-1, 0, 1)]  # (0, 0) is "wait"

    start_state = (start_cell[0], start_cell[1], 0)
    if not free(*start_state):
        return None, np.inf
    g = {start_state: 0.0}
    came = {}
    pq = [(0.0, start_state)]
    visited = set()
    goal_state = None
    while pq:
        cost, state = heapq.heappop(pq)
        if state in visited:
            continue
        visited.add(state)
        i, j, k = state
        if is_goal((i, j)) and k >= min_arrival_k:
            goal_state = state
            break
        if k >= max_arrival_k:
            continue
        for di, dj, w in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if di != 0 and dj != 0 and not (
                    free(i + di, j, k + 1) and free(i, j + dj, k + 1)):
                continue  # no corner-cutting, matches astar.py's rule
            if not free(ni, nj, k + 1):
                continue
            step_cost = wait_cost if (di == 0 and dj == 0) else w
            ncost = cost + step_cost
            nstate = (ni, nj, k + 1)
            if ncost < g.get(nstate, np.inf):
                g[nstate] = ncost
                came[nstate] = state
                heapq.heappush(pq, (ncost, nstate))

    if goal_state is None:
        return None, np.inf

    cells = [goal_state]
    state = goal_state
    while state != start_state:
        state = came[state]
        cells.append(state)
    cells.reverse()
    path = [(x0 + i * res, y0 + j * res) for i, j, _ in cells]
    return path, g[goal_state]


def two_route_search(env, start_xy, goal_xy, predict_obstacle, obstacle_r,
                     robot_r, horizon=3.0, dt_layer=0.25, res=0.10,
                     window=2.5, goal_tol=0.15, wait_cost_cheap=None,
                     wait_cost_expensive=None):
    """Construct "wait it out" and "go around" as two genuinely distinct
    space-time routes past one crossing obstacle.

    An earlier version of this function tried to force the two classes by
    bounding which time layer counted as "arrived" (arrive-before-deadline
    vs. arrive-after). That degenerates: if reaching the goal safely before
    the obstacle is not too much sooner than the deadline, the cheapest way
    to satisfy a LATE deadline is simply to go early anyway and idle at the
    goal -- physically the same route, wrongly labelled distinct. Deadline
    bookkeeping alone does not force a different physical strategy.

    Varying the relative cost of waiting versus detouring does, because it
    changes which strategy is the TRUE minimum-cost path, not just which
    label a fixed path qualifies for. `wait_cost_cheap` (default: cheaper
    than any real move, `0.25 * res`) makes waiting in place the global
    optimum; `wait_cost_expensive` (default: `5 * res`, so five wait-layers
    cost as much as detouring five cells sideways) makes a lateral detour
    around the obstacle the global optimum instead. Verified empirically
    (not just asserted) on a crossing-obstacle scenario: this reliably
    yields a genuine wait-then-cross path and a genuine detour-around path,
    confirmed distinct by `routes_are_distinct` -- see
    `tests/test_spacetime.py`.

    Returns `{"wait": {...}, "detour": {...}, "distinct": bool}`, each
    route a `{"path", "cost", "feasible"}` dict from `time_expanded_search`.
    `"distinct"` is `False` (not None) when either route is infeasible.
    """
    wait_cost_cheap = 0.25 * res if wait_cost_cheap is None else wait_cost_cheap
    wait_cost_expensive = (5.0 * res if wait_cost_expensive is None
                           else wait_cost_expensive)

    def run(wait_cost):
        path, cost = time_expanded_search(
            env, start_xy, goal_xy, predict_obstacle, obstacle_r, robot_r,
            horizon=horizon, dt_layer=dt_layer, res=res, window=window,
            goal_tol=goal_tol, wait_cost=wait_cost)
        return {"path": path, "cost": cost, "feasible": path is not None}

    wait_route = run(wait_cost_cheap)
    detour_route = run(wait_cost_expensive)
    distinct = bool(
        wait_route["feasible"] and detour_route["feasible"]
        and routes_are_distinct(
            wait_route["path"], detour_route["path"], predict_obstacle,
            dt_layer=dt_layer))
    return {"wait": wait_route, "detour": detour_route, "distinct": distinct}


def routes_are_distinct(path_a, path_b, predict_obstacle, dt_layer=0.25,
                        relevance=0.8):
    """True if some synchronized time puts the obstacle between the routes.

    The space-time analogue of `homotopy.mode_side_planes`'s side test,
    evaluated at synchronized TIME rather than closest spatial approach:
    a moving obstacle's tube only separates two routes that pass its
    location at genuinely different times, so the comparison has to line
    the routes up in time, not just in space. `path_a`/`path_b` are
    `(x, y)` lists, one entry per `dt_layer`-spaced time layer starting at
    t=0 (as returned by `time_expanded_search`); a shorter path (arrived
    earlier) is held at its final position for the comparison, exactly the
    physical situation of a robot waiting at its goal.
    """
    n = max(len(path_a), len(path_b))

    def pad(path):
        arr = np.asarray(path, dtype=float)
        if len(arr) < n:
            arr = np.vstack([arr, np.tile(arr[-1], (n - len(arr), 1))])
        return arr

    a, b = pad(path_a), pad(path_b)
    for k in range(n):
        o = np.asarray(predict_obstacle(k * dt_layer), dtype=float)
        da, db = a[k] - o, b[k] - o
        if np.linalg.norm(da) > relevance or np.linalg.norm(db) > relevance:
            continue
        if np.dot(da, db) < 0.0:
            return True
    return False


def spacetime_path_to_proposal(path, dt_layer, state, goal_xy, env,
                               robot_model, predict_obstacle, obstacle_r,
                               T=56, dt=0.05, v_max=0.5, w_max=1.9,
                               v_accel_max=1.0, w_accel_max=3.0,
                               initial_control=None, branch_id=-2,
                               feasibility_margin=0.0, min_progress=0.05):
    """Turn a `time_expanded_search` path into a `proposals.BranchProposal`
    -- the missing piece connecting Phase 0's raw `(x, y)`-per-time-layer
    output to the concrete `[T, 2]` velocity commands the rest of the
    pipeline (`SamplingMode`, grouped sampling, MPPI cost) already consumes,
    so a "wait" or "detour" space-time route can slot in as an ordinary
    mode exactly like a static pseudopod branch.

    `path` is spaced `dt_layer` apart starting at t=0 (as returned by
    `time_expanded_search`/`two_route_search`); it is resampled onto the
    controller's own `(T, dt)` grid by linear interpolation (holding the
    final position beyond the path's own duration -- "already arrived,
    waiting there"), which is exactly what preserves a WAIT segment
    (repeated identical points interpolate to the same held position)
    instead of collapsing it the way `proposals.branch_to_control_sequence`
    would if handed the same points as a bare polyline: that function only
    understands geometry, not timing, and treats repeated points as
    zero-length segments to be discarded.

    Velocity commands are the finite difference between consecutive
    resampled points, then projected through the same box/rate limits
    (`grouped_sampling.project_control_sequences`) every other proposal
    uses, and rolled out through `robot_model.integrate` -- so, like
    `branch_to_control_sequence`, the returned `rollout` is the ACHIEVED
    trajectory under real dynamics and limits, not the raw requested path;
    acceleration limits can make the two disagree, which is exactly why
    feasibility is checked against the achieved rollout, not the request.

    Feasibility checks the achieved rollout against static clearance
    (`robot_model.exact_clearance` if `env` supports it, else
    `robot_model.clearance`) AND the predicted moving obstacle at each
    step's own absolute time -- a route that looked collision-free as a
    plan can still clip the obstacle once acceleration limits distort it,
    and this is what would catch that.
    """
    state = np.asarray(state, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)
    path_xy = np.asarray(path, dtype=float)
    query_times = (np.arange(T) + 1.0) * dt
    source_times = np.arange(len(path_xy)) * dt_layer
    desired_xy = np.column_stack((
        np.interp(query_times, source_times, path_xy[:, 0]),
        np.interp(query_times, source_times, path_xy[:, 1])))

    previous_u = (np.zeros(2) if initial_control is None
                 else np.asarray(initial_control, dtype=float).copy())
    previous_u[0] = np.clip(previous_u[0], 0.0, v_max)
    previous_u[1] = np.clip(previous_u[1], -w_max, w_max)

    raw = np.zeros((T, 2))
    prev_point, prev_heading = state[:2].copy(), float(state[2])
    for k in range(T):
        delta = desired_xy[k] - prev_point
        dist = float(np.linalg.norm(delta))
        heading = float(np.arctan2(delta[1], delta[0])) if dist > 1e-9 \
            else prev_heading
        raw[k, 0] = dist / dt
        raw[k, 1] = wrap(heading - prev_heading) / dt
        prev_point, prev_heading = desired_xy[k], heading

    controls = project_control_sequences(
        raw[None], previous_u, dt, v_max, w_max, v_accel_max, w_accel_max)[0]

    rollout = np.empty((T, 3))
    x = state.copy()
    for k in range(T):
        x = robot_model.integrate(x, controls[k], dt)
        rollout[k] = x

    if hasattr(robot_model, "exact_clearance") and hasattr(
            env, "obstacle_centers"):
        static_clear = (robot_model.exact_clearance(env, rollout)
                        - robot_model.extra_safety_margin(controls))
    else:
        static_clear = (robot_model.clearance(env, rollout)
                        - robot_model.extra_safety_margin(controls))
    obstacle_positions = np.array([predict_obstacle(t) for t in query_times])
    dynamic_clear = (np.linalg.norm(rollout[:, :2] - obstacle_positions,
                                    axis=-1)
                     - obstacle_r - robot_model.radius)
    min_clearance = float(np.min(np.minimum(static_clear, dynamic_clear)))

    achieved = float(np.linalg.norm(state[:2] - goal_xy)
                     - np.linalg.norm(rollout[-1, :2] - goal_xy))
    tracking = float(np.mean(
        np.linalg.norm(rollout[:, :2] - desired_xy, axis=-1)))

    reasons = []
    if min_clearance < feasibility_margin:
        reasons.append("clearance")
    if achieved < min_progress:
        reasons.append("insufficient progress")
    feasible = not reasons
    return BranchProposal(
        branch_id, controls, rollout, feasible, min_clearance, achieved,
        tracking, "ok" if feasible else ", ".join(reasons))
