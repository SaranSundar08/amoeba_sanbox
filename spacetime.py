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
