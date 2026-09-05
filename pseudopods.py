"""V1 topology-aware pseudopod extraction from a LocalFlowField.

This module deliberately stops at geometry. Each result is a scored membrane
endpoint and a robot-to-frontier centreline; ``proposals.py`` performs the V2
conversion without altering MPPI sampling.
"""
from dataclasses import dataclass, replace

import numpy as np


_NBR8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
         if (di, dj) != (0, 0)]


@dataclass(frozen=True)
class Pseudopod:
    """One distinct local free-space branch."""

    endpoint: np.ndarray
    centerline: np.ndarray
    score: float
    branch_id: int = -1


def _world(field, cells):
    cells = np.asarray(cells)
    return np.column_stack((field.x0 + cells[:, 0] * field.res,
                            field.y0 + cells[:, 1] * field.res))


def _trace_to_center(field, endpoint):
    """Trace monotonically down the unweighted reach field to the robot."""
    cur = tuple(int(x) for x in endpoint)
    root = tuple(int(x) for x in field.center_ij)
    rev = [cur]
    limit = field.reach_dist.size
    for _ in range(limit):
        if cur == root:
            break
        i, j = cur
        choices = []
        for di, dj in _NBR8:
            ni, nj = i + di, j + dj
            if not (0 <= ni < field.reach_dist.shape[0]
                    and 0 <= nj < field.reach_dist.shape[1]):
                continue
            d = field.reach_dist[ni, nj]
            if np.isfinite(d) and d < field.reach_dist[i, j] - 1e-9:
                # Prefer open cells only as a tie-breaker; monotone reach
                # distance is what guarantees termination at the robot.
                choices.append((d, float(field.visc[ni, nj]), ni, nj))
        if not choices:
            break
        _, _, ni, nj = min(choices)
        cur = (ni, nj)
        rev.append(cur)
    rev.reverse()
    return _world(field, rev)


def _resample(path, n=32):
    if len(path) <= 1:
        return np.repeat(path[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] < 1e-12:
        return np.repeat(path[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.column_stack((np.interp(q, s, path[:, 0]),
                            np.interp(q, s, path[:, 1])))


def _cell_is_dry(field, pts):
    i = np.rint((pts[:, 0] - field.x0) / field.res).astype(int)
    j = np.rint((pts[:, 1] - field.y0) / field.res).astype(int)
    inside = ((i >= 0) & (i < field.body.shape[0])
              & (j >= 0) & (j < field.body.shape[1]))
    wet = np.zeros(len(pts), dtype=bool)
    wet[inside] = field.body[i[inside], j[inside]]
    return ~wet


def _topologically_distinct(field, a, b, separation):
    """True when paths diverge and dry space separates their interiors.

    This rejects several arbitrary rays across one open room while retaining
    left/right paths split by an obstacle or wall.  It is a local, grid-based
    homotopy proxy rather than a formal homotopy proof.
    """
    pa, pb = _resample(a), _resample(b)
    distances = np.linalg.norm(pa - pb, axis=1)
    for k in np.flatnonzero(
            (np.arange(len(pa)) >= len(pa) // 4) & (distances >= separation)):
        n = max(3, int(np.ceil(distances[k] / (0.5 * field.res))))
        connector = np.linspace(pa[k], pb[k], n)
        if _cell_is_dry(field, connector).any():
            return True
    return False


def extract_pseudopods(field, max_modes=3, endpoint_spacing=0.35,
                       branch_separation=0.45, score_slack=3.0):
    """Extract up to ``max_modes`` distinct promising membrane branches.

    Candidates are ranked by local travel distance plus the membrane's
    navigation promise. Spatial non-maximum suppression removes neighboring
    frontier cells, and a dry-space separator test removes geometrically
    equivalent paths through the same open region.
    """
    if max_modes < 1:
        return []
    membrane = getattr(field, "membrane", None)
    if membrane is None or not membrane.any():
        return []
    ci, cj = np.nonzero(membrane & np.isfinite(field.D)
                        & np.isfinite(field.reach_dist))
    if not len(ci):
        return []
    scores = field.D[ci, cj] + field.reach_dist[ci, cj]
    order = np.argsort(scores, kind="stable")
    best = float(scores[order[0]])

    candidates = []
    for k in order:
        if scores[k] > best + score_slack:
            break
        cell = np.array([ci[k], cj[k]])
        xy = _world(field, cell[None])[0]
        if any(np.linalg.norm(xy - old[1]) < endpoint_spacing
               for old in candidates):
            continue
        candidates.append((cell, xy, float(scores[k])))

    selected = []
    for cell, endpoint, score in candidates:
        path = _trace_to_center(field, cell)
        if len(path) < 2:
            continue
        if selected and not all(
                _topologically_distinct(
                    field, path, old.centerline, branch_separation)
                for old in selected):
            continue
        selected.append(Pseudopod(endpoint, path, score))
        if len(selected) == max_modes:
            break
    return selected


class BranchTracker:
    """Greedy small-N identity matching across field rebuilds."""

    def __init__(self, match_distance=1.0):
        self.match_distance = match_distance
        self.previous = []
        self.next_id = 0

    def update(self, branches):
        remaining = {b.branch_id: b for b in self.previous}
        tracked = []
        for branch in branches:
            best_id, best_d = None, np.inf
            sample = _resample(branch.centerline)
            for branch_id, old in remaining.items():
                d = np.linalg.norm(sample - _resample(old.centerline), axis=1)
                metric = 0.5 * float(d.mean()) + 0.5 * float(
                    np.linalg.norm(branch.endpoint - old.endpoint))
                if metric < best_d:
                    best_id, best_d = branch_id, metric
            if best_id is not None and best_d <= self.match_distance:
                remaining.pop(best_id)
            else:
                best_id = self.next_id
                self.next_id += 1
            tracked.append(replace(branch, branch_id=best_id))
        self.previous = tracked
        return tracked
