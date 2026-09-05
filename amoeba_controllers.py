"""Controller variants built on the reusable :class:`mppi.MPPI` core.

Defaults mirror the Nav2 setup where it matters for transfer:
T=56, dt=0.05 (same horizon), v/w noise like navigation_sim.yaml.
The sampling-bias hook (`sample_bias`) and the goal-cost hook (`goal_cost`)
are the two extension points — same split as critics vs. noise generator
in the nav2 plugin, so whatever wins here ports cleanly.
"""
import time

import numpy as np

from env import LocalFlowField, wrap
from pseudopods import BranchTracker, Pseudopod, extract_pseudopods
from proposals import branch_to_control_sequence
from mppi import MPPI
from astar import _line_clear, astar_path


class _FlowGuidedMPPI(MPPI):
    """Shared local-field cost and steering hooks.

    This is an implementation detail, not a separately runnable controller.
    """

    def goal_cost(self, xs):
        if not self.use_flow_cost:
            return super().goal_cost(xs)
        d = self.flow.dist(xs[..., :2])
        return self.goal_w * d[:, -1] + self.run_w * d.mean(1)

    def sample_bias(self, state):
        if self.bias_strength == 0.0:
            return 0.0
        xs = self.rollout(state, self.nominal[None])[0]         # (T, 3)
        vec = self.flow.direction_smooth(xs[:, :2])
        want = np.arctan2(vec[:, 1], vec[:, 0])
        err = np.where((vec ** 2).sum(1) > 0, wrap(want - xs[:, 2]), 0.0)
        b = np.zeros((self.T, 2))
        b[:, 1] = self.bias_strength * np.clip(self.bias_gain * err,
                                               -self.w_max, self.w_max)
        return b


class AmoebaLocal(_FlowGuidedMPPI):
    """The amoeba as a finite *body*: the water field lives only inside a
    window around the robot (LocalFlowField) and is re-flooded every
    `reflood_every` steps as the robot moves — no global knowledge, same
    information a rolling local costmap holds on the real robot. Cost and
    bias work exactly as in `flow`, just against the windowed field.
    `build_ms` records each re-flood's wall time (the Orin-budget number)."""
    name = "local"

    def __init__(self, env, seed=0, bias_gain=1.5, bias_strength=0.6,
                 flow_res=0.05, window=2.5, reflood_every=5, path=None,
                 viscosity=1.5, extract_branches=False, max_branches=3, **kw):
        MPPI.__init__(self, env, seed, **kw)
        self.bias_gain, self.bias_strength = bias_gain, bias_strength
        self.use_flow_cost = True
        self.flow_res, self.window = flow_res, window
        self.viscosity = viscosity
        self.reflood_every = reflood_every
        self.flow, self._k, self.build_ms = None, 0, []
        self.extract_branches = extract_branches
        self.max_branches = max_branches
        self.branches = []
        self.branch_proposals = []
        self.proposal_ms = []
        self.branch_tracker = BranchTracker()
        # optional global path -> boundary condition (planner demoted):
        # boundary promise = dist-to-path + remaining path length
        self.path = None if path is None else np.asarray(path, float)
        self.path_rem = None
        if self.path is not None:
            seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
            self.path_rem = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

    def prepare(self, state):
        if self.flow is None or self._k % self.reflood_every == 0:
            t0 = time.perf_counter()
            self.flow = LocalFlowField(self.env, state[:2],
                                       radius=self.window, res=self.flow_res,
                                       path=self.path, path_rem=self.path_rem,
                                       viscosity=self.viscosity)
            self.build_ms.append((time.perf_counter() - t0) * 1e3)
            if self.extract_branches:
                found = extract_pseudopods(
                    self.flow, max_modes=self.max_branches)
                self.branches = self.branch_tracker.update(found)
            elif not self.extract_branches:
                self.branches = []
        if self.extract_branches:
            self.build_branch_proposals(state)
        self._k += 1


class AmoebaHybrid(MPPI):
    """The amoeba body as a genuine ADDITIONAL guidance layer, not a path
    replacement. Unlike `local` (which demotes the plan to a boundary
    condition), here the global path stays a tracked *preference* in the
    COST only — distance-to-path + remaining-length, like nav2's
    PathAlign/PathFollow — while the local geodesic-blob water field
    (LocalFlowField) is the sole author of the sampling BIAS, exactly as
    in `local`. Steering is deliberately single-authority: an earlier
    version also blended a path-tangent vector into the bias, and it
    actively made things worse -- when the given path is wrong (heads
    into a pillar) and the flow correctly curves around it, averaging two
    disagreeing unit vectors cancels into a compromise that clips the
    obstacle either one alone would have missed (the APF superposition
    failure, reproduced in miniature). Cost, unlike bias, only re-ranks
    trajectories the flow-biased sampling already made safe, so path
    preference can never fight the flow field's steering.
    `flow_w=0` recovers pure path tracking; `path_w=0` recovers the
    planner-free `local` controller. Same information budget as `local`:
    a rolling geodesic body re-flooded every `reflood_every` steps.

    `gate="always"` (validated 8/9 above) blends flow guidance every
    step, everywhere. `gate="auto"` instead keeps the global planner as
    the only voice on the open road and only lets the amoeba's cost/bias
    weight rise when it's actually needed: local clearance drops below
    `gate_clear_mul * robot_r` (a confined/narrow spot), or the planned
    forward speed has stayed near-zero for `gate_stall_steps` (stuck at
    what looks like a local minimum). The gate is a smoothed scalar in
    [0, 1] (an exponential approach to the binary trigger, not a hard
    flip) so the transition doesn't chatter; it scales `flow_w` alone --
    `path_w` (the A* contract) never turns off."""
    name = "hybrid"

    def __init__(self, env, seed=0, path=None, bias_gain=1.5,
                 bias_strength=0.6, flow_res=0.05, window=2.5,
                 reflood_every=5, viscosity=1.5, path_w=1.0, flow_w=1.0,
                 align_w=4.0, gate="always", gate_clear_mul=1.6,
                 gate_stall_v=0.05, gate_stall_steps=15, gate_rate=0.2,
                 bias_deadband=0.08, bias_smooth=0.35,
                 extract_branches=False, max_branches=3,
                 path_validity_gate=False, max_path_occupancy_ratio=0.07,
                 assist_progress_window=20, assist_progress_distance=0.10,
                 assist_min_cycles=10,
                 **kw):
        super().__init__(env, seed, **kw)
        if path is None:
            raise ValueError("hybrid needs a global path (the A* stand-in)")
        self.bias_gain, self.bias_strength = bias_gain, bias_strength
        self.flow_res, self.window = flow_res, window
        self.viscosity, self.reflood_every = viscosity, reflood_every
        self.path_w, self.flow_w = path_w, flow_w
        self.align_w = align_w
        self.gate = gate
        self.gate_clear_mul, self.gate_rate = gate_clear_mul, gate_rate
        self.gate_stall_v, self.gate_stall_steps = gate_stall_v, gate_stall_steps
        self.bias_deadband, self.bias_smooth = bias_deadband, bias_smooth
        self.path_validity_gate = path_validity_gate
        self.max_path_occupancy_ratio = max_path_occupancy_ratio
        self.assist_progress_window = assist_progress_window
        self.assist_progress_distance = assist_progress_distance
        self.assist_min_cycles = assist_min_cycles
        self._bias_dir_ema = None
        self._stall = 0
        self.active = 1.0 if gate == "always" else 0.0   # exposed for viz

        self.plan_version = 0
        self._set_path_data(path)
        self.assist_state = "ASSIST" if gate == "always" else "NORMAL"
        self.assist_reason = "always" if gate == "always" else "path_valid"
        self._assist_anchor = self.env.start[:2].copy()
        self._assist_progress_cycles = 0
        self._assist_age = 0

        self.flow, self._k, self.build_ms = None, 0, []
        self.extract_branches = extract_branches
        self.max_branches = max_branches
        self.branches = []
        self.branch_proposals = []
        self.proposal_ms = []
        self.global_path_proposal = None
        self.path_proposal_ms = []
        self.branch_tracker = BranchTracker()

    def _set_path_data(self, path):
        self.path = np.asarray(path, float)
        if self.path.ndim != 2 or self.path.shape[1] != 2 or len(self.path) < 2:
            raise ValueError("hybrid path must contain at least two XY points")
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.path_rem = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.track_path = self.path[::4]
        if not np.array_equal(self.track_path[-1], self.path[-1]):
            self.track_path = np.vstack((self.track_path, self.path[-1]))
        seg2 = np.linalg.norm(np.diff(self.track_path, axis=0), axis=1)
        self.track_rem = np.concatenate([np.cumsum(seg2[::-1])[::-1], [0.0]])
        self.track_valid = self._path_footprint_clearance() > 0.0

    def _path_states(self):
        delta = np.diff(self.track_path, axis=0)
        headings = np.arctan2(delta[:, 1], delta[:, 0])
        headings = np.concatenate((headings, headings[-1:]))
        return np.column_stack((self.track_path, headings))

    def _path_footprint_clearance(self):
        # Gates NORMAL/ASSIST off a small per-cycle set of path points, not
        # the K x T rollout, so the exact (non-sampled) check is cheap here
        # and matters: a blockage the sampled footprint check missed would
        # otherwise read as `track_valid`, keeping the gate in NORMAL and
        # protecting a path proposal that isn't actually clear.
        return self.robot_model.exact_clearance(self.env, self._path_states())

    def set_path(self, path):
        """Accept a newly replanned global path without rebuilding the controller."""
        self._set_path_data(path)
        self.plan_version += 1
        self.global_path_proposal = None
        self.mode_nominals.pop(-1, None)
        # Force the next prepare() to rebuild path-conditioned field/proposals.
        self.flow = None
        self.branches = []
        self.branch_proposals = []
        if self.gate == "auto" and self.track_valid.all():
            self.assist_state = "NORMAL"
            self.assist_reason = "new_valid_plan"

    def _update_assist_state(self, state):
        if self.gate == "always":
            self.assist_state, self.assist_reason = "ASSIST", "always"
            return
        self.track_valid = self._path_footprint_clearance() > 0.0
        blocked_ratio = 1.0 - float(np.mean(self.track_valid))
        path_blocked = blocked_ratio > self.max_path_occupancy_ratio

        if np.linalg.norm(state[:2] - self._assist_anchor) >= (
                self.assist_progress_distance):
            self._assist_anchor = state[:2].copy()
            self._assist_progress_cycles = 0
            made_progress = True
        else:
            self._assist_progress_cycles += 1
            made_progress = False
        stalled = self._assist_progress_cycles >= self.assist_progress_window

        if self.assist_state == "NORMAL" and (path_blocked or stalled):
            self.assist_state = "ASSIST"
            self.assist_reason = "path_blocked" if path_blocked else "no_progress"
            self._assist_age = 0
        elif self.assist_state == "ASSIST":
            self._assist_age += 1
            if (not path_blocked and made_progress
                    and self._assist_age >= self.assist_min_cycles):
                self.assist_state = "NORMAL"
                self.assist_reason = "valid_path_progress"

    def build_global_path_proposal(self, state):
        """Nav2-oriented fallback mean from the current local path segment."""
        t0 = time.perf_counter()
        nearest = int(np.argmin(
            np.linalg.norm(self.path - state[:2], axis=1)))
        start = max(0, nearest - 2)
        centerline = self.path[start:]
        if len(centerline) < 2:
            self.global_path_proposal = None
        else:
            geometry = Pseudopod(
                endpoint=centerline[-1].copy(),
                centerline=centerline,
                score=0.0,
                branch_id=-1)
            self.global_path_proposal = branch_to_control_sequence(
                geometry, state, self.env, T=self.T, dt=self.dt,
                v_max=self.v_max, w_max=self.w_max,
                v_accel_max=self.v_accel_max,
                w_accel_max=self.w_accel_max,
                initial_control=self.last_applied,
                robot_model=self.robot_model)
        self.path_proposal_ms.append((time.perf_counter() - t0) * 1e3)

    def prepare(self, state):
        self._update_assist_state(state)
        if self.flow is None or self._k % self.reflood_every == 0:
            t0 = time.perf_counter()
            self.flow = LocalFlowField(self.env, state[:2],
                                       radius=self.window, res=self.flow_res,
                                       path=self.path, path_rem=self.path_rem,
                                       viscosity=self.viscosity)
            self.build_ms.append((time.perf_counter() - t0) * 1e3)
            if self.extract_branches:
                found = extract_pseudopods(
                    self.flow, max_modes=self.max_branches)
                self.branches = self.branch_tracker.update(found)
            elif not self.extract_branches:
                self.branches = []
        if self.extract_branches:
            self.build_branch_proposals(state)
        if self.grouped_sampling:
            self.build_global_path_proposal(state)
        self._k += 1

        if self.gate == "auto":
            clear = float(self.robot_model.clearance(self.env, state))
            confined = clear < self.gate_clear_mul * self.env.robot_r
            slow = abs(self.nominal[0, 0]) < self.gate_stall_v
            self._stall = self._stall + 1 if slow else 0
            stuck = self._stall >= self.gate_stall_steps
            target = 1.0 if (confined or stuck) else 0.0
            self.active += self.gate_rate * (target - self.active)

    def goal_cost(self, xs):
        # the global path stays the contract (PathAlign/PathFollow-style)
        d = np.linalg.norm(xs[..., None, :2] - self.track_path, axis=-1)
        path_occupancy = 1.0 - float(np.mean(self.track_valid))
        if self.path_validity_gate and self.track_valid.any():
            # Approximate Nav2's PathFollow behavior: occupied path points are
            # not valid tracking targets, so attraction advances to reachable
            # points beyond a local blockage instead of pulling through it.
            valid_d = np.where(self.track_valid, d, np.inf)
            nearest = valid_d.argmin(-1)
            lateral = valid_d.min(-1)
        else:
            nearest = d.argmin(-1)
            lateral = d.min(-1)
        along = lateral + self.track_rem[nearest]
        align_weight = self.align_w
        if (self.path_validity_gate
                and path_occupancy > self.max_path_occupancy_ratio):
            # Nav2 PathAlignCritic returns without scoring when a significant
            # fraction of the local plan is occupied.
            align_weight = 0.0
        track_cost = (self.goal_w * along[:, -1] + self.run_w * along.mean(1)
                      + align_weight * lateral.mean(1))
        # amoeba guidance layered on top of it, not instead of it -- gated
        # in "auto" mode so it only speaks up where it's actually needed
        fd = self.flow.dist(xs[..., :2])
        flow_cost = self.goal_w * fd[:, -1] + self.run_w * fd.mean(1)
        return self.path_w * track_cost + self.flow_w * self.active * flow_cost

    def sample_bias(self, state):
        # Steering stays single-authority: the flow field is grounded in
        # the real local costmap, so it alone drives the bias (exactly as
        # in `local`). Averaging it with a path-tangent vector was tried
        # and actively made things worse -- when the given path is wrong
        # (heads into a pillar) and flow correctly curves around it,
        # summing two disagreeing unit vectors cancels into a compromise
        # direction that clips the very obstacle either one alone would
        # have missed. That is the APF superposition failure reproduced
        # in miniature; the path's pull belongs in the cost (below), which
        # only ranks already-safe sampled trajectories, never steers them.
        if self.bias_strength == 0.0 or self.flow is None or self.active <= 1e-3:
            return 0.0
        xs = self.rollout(state, self.nominal[None])[0]        # (T, 3)
        vec = self.flow.direction_smooth(xs[:, :2])

        # Temporal smoothing on the step that's actually applied (t=0):
        # even a numerically-clean gradient still has a few-degree tail of
        # jitter cycle to cycle (grid discretization + MPPI's own fresh
        # noise draw each step), and that's enough to keep flipping w's
        # sign near an obstacle where the deadband below doesn't catch it.
        # An EMA on the immediate direction rejects that tail without
        # touching how the bias curves along the rest of the horizon.
        v0 = vec[0]
        n0 = np.linalg.norm(v0)
        if n0 > 1e-9:
            v0u = v0 / n0
            if self._bias_dir_ema is None:
                self._bias_dir_ema = v0u
            else:
                blended = (self.bias_smooth * v0u
                           + (1 - self.bias_smooth) * self._bias_dir_ema)
                bn = np.linalg.norm(blended)
                self._bias_dir_ema = blended / bn if bn > 1e-9 else v0u
            vec = vec.copy()
            vec[0] = self._bias_dir_ema

        want = np.arctan2(vec[:, 1], vec[:, 0])
        err = np.where((vec ** 2).sum(1) > 0, wrap(want - xs[:, 2]), 0.0)
        # Deadband: don't react to a heading error small enough that it's
        # more likely noise than a real correction worth making.
        err = np.where(np.abs(err) > self.bias_deadband, err, 0.0)
        b = np.zeros((self.T, 2))
        b[:, 1] = (self.active * self.bias_strength
                   * np.clip(self.bias_gain * err, -self.w_max, self.w_max))
        return b


class AmoebaRepairHybrid(AmoebaHybrid):
    """V7.2 blob-guided local repair of a dynamically blocked A* path.

    The planner-owned path is retained in ``original_path``.  A feasible
    pseudopod is spliced to the first collision-free downstream rejoin point;
    that repaired polyline becomes the active MPPI reference until rejoin.
    """
    name = "repair"

    def __init__(self, *args, repair_rejoin_tolerance=0.35,
                 repair_min_advance=0.60, repair_connector_step=0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.original_path = self.path.copy()
        original_segments = np.linalg.norm(
            np.diff(self.original_path, axis=0), axis=1)
        self.original_arc = np.concatenate(
            ([0.0], np.cumsum(original_segments)))
        self.repair_rejoin_tolerance = repair_rejoin_tolerance
        self.repair_min_advance = repair_min_advance
        self.repair_connector_step = repair_connector_step
        self.repair_active = False
        self.repair_path = None
        self.repair_branch_id = None
        self.repair_rejoin_index = None
        self.repair_count = 0
        self.repair_failures = 0

    @staticmethod
    def _densify_polyline(points, spacing):
        points = np.asarray(points, dtype=float)
        output = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            length = np.linalg.norm(end - start)
            count = max(1, int(np.ceil(length / spacing)))
            output.extend(start + (end - start) * (k / count)
                          for k in range(1, count + 1))
        return np.asarray(output)

    def _restore_original_path(self, state):
        index = int(np.argmin(np.linalg.norm(
            self.original_path - state[:2], axis=1)))
        suffix = self.original_path[index:]
        if len(suffix) < 2:
            suffix = self.original_path[-2:]
        restored = np.vstack((state[:2], suffix))
        self._set_path_data(restored)
        self.repair_active = False
        self.repair_path = None
        self.repair_branch_id = None
        self.repair_rejoin_index = None
        self.mode_nominals.clear()
        self.global_path_proposal = None
        self.flow = None

    def _maybe_finish_repair(self, state):
        if not self.repair_active:
            return
        rejoin = self.original_path[self.repair_rejoin_index]
        if np.linalg.norm(state[:2] - rejoin) <= self.repair_rejoin_tolerance:
            self._restore_original_path(state)
            self.assist_state = "NORMAL"
            self.assist_reason = "repair_rejoined"

    def _repair_candidate(self, state, branch, proposal):
        if len(branch.centerline) < 2:
            return None
        nearest = int(np.argmin(np.linalg.norm(
            self.original_path - state[:2], axis=1)))
        minimum_arc = self.original_arc[nearest] + self.repair_min_advance
        endpoint = np.asarray(branch.centerline[-1], dtype=float)
        candidate_indices = np.flatnonzero(self.original_arc >= minimum_arc)
        for index in candidate_indices:
            rejoin = self.original_path[index]
            branch_path = np.asarray(branch.centerline, dtype=float)
            if _line_clear(self.env, endpoint, rejoin,
                           self.env.robot_r, step=0.02):
                connector = self._densify_polyline(
                    np.vstack((endpoint, rejoin)), self.repair_connector_step)
            else:
                try:
                    connector = astar_path(
                        self.env, endpoint, rejoin, res=0.10,
                        densify_step=self.repair_connector_step)
                except RuntimeError:
                    continue
            suffix = self.original_path[index + 1:]
            repaired = np.vstack((state[:2], branch_path[1:],
                                  connector[1:], suffix))
            clearance = float(np.min(
                self.env.clearance(repaired) - self.env.robot_r))
            if clearance <= 0.0:
                continue
            length = float(np.linalg.norm(
                np.diff(repaired, axis=0), axis=1).sum())
            proposal_penalty = 0.0 if proposal.feasible else 1.0
            score = length + 0.25 / max(clearance, 0.01) + proposal_penalty
            return score, index, repaired
        return None

    def _activate_repair(self, state):
        candidates = []
        for branch, proposal in zip(self.branches, self.branch_proposals):
            candidate = self._repair_candidate(state, branch, proposal)
            if candidate is not None:
                candidates.append((*candidate, branch.branch_id))
        if not candidates:
            self.repair_failures += 1
            return False
        _score, index, repaired, branch_id = min(
            candidates, key=lambda item: item[0])
        self.repair_active = True
        self.repair_path = repaired.copy()
        self.repair_branch_id = branch_id
        self.repair_rejoin_index = index
        self.repair_count += 1
        self._set_path_data(repaired)
        self.mode_nominals.clear()
        self.global_path_proposal = None
        self.build_global_path_proposal(state)
        self.assist_state = "NORMAL"
        self.assist_reason = "repair_committed"
        return True

    def prepare(self, state):
        self._maybe_finish_repair(state)
        super().prepare(state)
        if self.repair_active:
            # The repaired reference is the committed homotopy decision.
            # Continuing to offer competing branches recreates the very
            # oscillation that path repair is meant to remove.
            self.assist_state = "NORMAL"
            self.assist_reason = "repair_committed"
            return
        topology_refreshed = ((self._k - 1) % self.reflood_every == 0)
        if (not self.repair_active and self.assist_state == "ASSIST"
                and topology_refreshed and self.extract_branches
                and self.branch_proposals):
            self._activate_repair(state)

    def step(self, state):
        control, info = super().step(state)
        info["repair_active"] = self.repair_active
        info["repair_path"] = self.repair_path
        info["repair_branch_id"] = self.repair_branch_id
        info["repair_rejoin_index"] = self.repair_rejoin_index
        info["repair_count"] = self.repair_count
        return control, info


CONTROLLERS = {
    "vanilla": MPPI,
    "local": AmoebaLocal,
    "hybrid": AmoebaHybrid,
    "repair": AmoebaRepairHybrid,
}

__all__ = ["AmoebaLocal", "AmoebaHybrid", "AmoebaRepairHybrid",
           "CONTROLLERS"]
