"""Focused V1 tests; runnable with ``python3 -m unittest -v``."""
import unittest

import numpy as np

from env import LocalFlowField
from pseudopods import BranchTracker, extract_pseudopods


class GridEnv:
    """Small analytic environment implementing the sandbox env contract."""

    def __init__(self, obstacle=None):
        self.xmin, self.xmax, self.ymax = -3.0, 3.0, 6.0
        self.robot_r = 0.20
        self.start = np.array([0.0, 0.5, np.pi / 2])
        self.goal = np.array([0.0, 5.8])
        self.obstacle = obstacle

    def clearance(self, pts):
        walls = np.minimum(pts[..., 0] - self.xmin,
                           self.xmax - pts[..., 0])
        floor = pts[..., 1]
        ceiling = self.ymax - pts[..., 1]
        clear = np.minimum(np.minimum(walls, floor), ceiling)
        if self.obstacle is not None:
            center, radius = self.obstacle
            clear = np.minimum(
                clear, np.linalg.norm(pts - center, axis=-1) - radius)
        return clear


class PseudopodTests(unittest.TestCase):
    def test_open_space_collapses_to_one_geometric_branch(self):
        env = GridEnv()
        field = LocalFlowField(env, env.start[:2], radius=3.5, res=0.1)
        branches = extract_pseudopods(field, max_modes=3)
        self.assertEqual(len(branches), 1)
        self.assertGreater(branches[0].endpoint[1], env.start[1])

    def test_pillar_produces_left_and_right_branches(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        field = LocalFlowField(env, env.start[:2], radius=3.8, res=0.1)
        branches = extract_pseudopods(
            field, max_modes=3, branch_separation=0.35, score_slack=5.0)
        self.assertEqual(len(branches), 2)
        sides = sorted(np.sign(b.endpoint[0]) for b in branches)
        self.assertEqual(sides, [-1.0, 1.0])

    def test_tracker_preserves_ids_after_small_field_motion(self):
        env = GridEnv(obstacle=(np.array([0.0, 2.5]), 0.65))
        tracker = BranchTracker()
        f1 = LocalFlowField(env, env.start[:2], radius=3.8, res=0.1)
        first = tracker.update(extract_pseudopods(
            f1, branch_separation=0.35, score_slack=5.0))
        f2 = LocalFlowField(
            env, env.start[:2] + np.array([0.03, 0.05]),
            radius=3.8, res=0.1)
        second = tracker.update(extract_pseudopods(
            f2, branch_separation=0.35, score_slack=5.0))
        by_side_1 = {np.sign(b.endpoint[0]): b.branch_id for b in first}
        by_side_2 = {np.sign(b.endpoint[0]): b.branch_id for b in second}
        self.assertEqual(by_side_1, by_side_2)


if __name__ == "__main__":
    unittest.main()
