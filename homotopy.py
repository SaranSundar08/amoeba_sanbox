"""Homotopy-consistency (mode-purity) cost for grouped sampling.

After de Groot, Ferranti, Gavrila, Alonso-Mora, "Topology-Driven Parallel
Trajectory Optimization in Dynamic Environments" (T-MPC): initializing each
parallel local planner from a distinct guidance trajectory is not enough to
keep the optimized trajectories distinct -- without an explicit constraint
they collapse back into the same homotopy class (their Fig. 6). T-MPC holds
each planner to its guidance trajectory's side of every obstacle with a
linear half-plane per obstacle and time step (their Eq. 8).

Amoeba's analogue of a guidance trajectory is a pseudopod proposal's rolled
mean; its analogue of the local planner is the mode's fixed sample group and
within-mode MPPI update. Nothing pins that update to the branch's side of an
obstacle between refloods except the ordinary collision cost, so the group's
weighted mean can random-walk across a pillar into a neighbouring mode's
corridor. This module supplies the missing pressure as a SOFT cost (MPPI
samples; there is no NLP to constrain):

    violation(x) = max(0, A . x - b)     summed over horizon and planes

with, for each obstacle the reference passes near, the plane through the
obstacle centre perpendicular to the reference's closest-approach offset
(``beta = 0``, T-MPC's inactive-at-the-boundary setting). A sample is
penalized only when its centre is on the far side of an obstacle from where
the reference passes it -- i.e. when it changes homotopy class -- and by how
far it strayed.

Static obstacles use one plane each from the reference's closest approach
(side of passage), rather than T-MPC's time-synchronized plane per step.
The per-step form penalizes samples merely for being ahead of a slower
reference; the closest-approach form does not, and for a fixed cylinder the
side of passage is the entire homotopy content. Moving obstacles, when an
environment reports them, are taken at their current position; a
time-synchronized variant is future work.
"""
import numpy as np


def obstacle_set(env):
    """(centers (N, 2), surface radius) if the environment exposes it."""
    if not hasattr(env, "obstacle_centers"):
        return None
    centers, radius = env.obstacle_centers()
    return np.asarray(centers, dtype=float), float(radius)


def mode_side_planes(reference_xy, obstacles, robot_radius, obstacle_radius,
                     relevance=0.8, beta=0.0):
    """Half-planes ``A x <= b`` holding samples to the reference's side.

    One plane per obstacle within ``relevance`` of the reference polyline,
    built at the reference's closest approach: ``A`` is the unit vector from
    that reference point to the obstacle centre, ``b = A . o - beta (r + r_o)``
    so ``beta = 0`` puts the plane through the obstacle centre and ``beta = 1``
    on the inflated obstacle surface. Returns ``A (M, 2)``, ``b (M,)``.
    """
    reference = np.asarray(reference_xy, dtype=float)
    obstacles = np.asarray(obstacles, dtype=float)
    empty = np.zeros((0, 2)), np.zeros(0)
    if len(reference) == 0 or len(obstacles) == 0:
        return empty
    d = np.linalg.norm(
        reference[:, None, :] - obstacles[None, :, :], axis=-1)   # (T, N)
    closest = d.argmin(axis=0)
    dmin = d[closest, np.arange(len(obstacles))]
    keep = (dmin <= relevance) & (dmin > 1e-9)
    if not keep.any():
        return empty
    centres = obstacles[keep]
    anchors = reference[closest[keep]]
    A = (centres - anchors) / dmin[keep][:, None]
    b = np.einsum("mi,mi->m", A, centres) - beta * (
        robot_radius + obstacle_radius)
    return A, b


def side_violations(xs_xy, A, b):
    """Per-sample total violation ``sum_t sum_m max(0, A_m . x_t - b_m)``.

    ``xs_xy`` is ``(K, T, 2)``; returns ``(K,)``. Zero when there are no
    planes.
    """
    xs_xy = np.asarray(xs_xy, dtype=float)
    if len(A) == 0:
        return np.zeros(len(xs_xy))
    signed = xs_xy @ np.asarray(A, dtype=float).T - np.asarray(b, dtype=float)
    return np.clip(signed, 0.0, None).sum(axis=(1, 2))
