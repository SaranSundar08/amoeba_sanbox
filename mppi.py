"""Pure-NumPy MPPI optimizer with V3--V5 multimodal sampling.

Defaults mirror the nav2 setup where it matters for transfer:
T=56, dt=0.05 (same horizon), v/w noise like navigation_sim.yaml.
The sampling-bias hook (`sample_bias`) and the goal-cost hook (`goal_cost`)
are the two extension points — same split as critics vs. noise generator
in the nav2 plugin, so whatever wins here ports cleanly.
"""
import time

import numpy as np

from homotopy import mode_side_planes, obstacle_set, side_violations
from proposals import _arc_length, _interpolate, branch_to_control_sequence
from robot_model import RobotModel
from spacetime import (nearest_crossing_obstacle, spacetime_path_to_proposal,
                       two_route_search)
from spacetime_flow import SpaceTimeFlood
from grouped_sampling import (
    SamplingMode, guard_importance_statistics, importance_mode_statistics,
    mixture_importance_weights, mode_statistics, project_control_sequences,
    sample_fixed_groups)


class MPPI:
    name = "vanilla"

    def __init__(self, env, seed=0, K=1024, T=56, dt=0.05, lam=0.3,
                 v_max=0.5, w_max=1.9, v_std=0.2, w_std=0.6,
                 v_accel_max=1.0, w_accel_max=3.0,
                 grouped_sampling=False, importance_weighting=False,
                 importance_ess_guard=False, importance_min_ess=8.0,
                 importance_min_ess_fraction=0.02,
                 adaptive_covariance=False,
                 robot_model="ideal", footprint_length=0.90,
                 footprint_width=0.65, skid_yaw_gain=0.82,
                 skid_speed_yaw_loss=0.55, turn_safety_margin=0.08,
                 speed_safety_margin=0.03,
                 dynamic_prediction=False, prediction_horizon=2.0,
                 prediction_uncertainty_rate=0.03,
                 homotopy_w=0.0, homotopy_beta=0.0, homotopy_relevance=0.8,
                 homotopy_constrain_fallback=False, homotopy_monitor=False,
                 reference_min_speed_ratio=0.12,
                 reference_curvature_slowdown=1.2,
                 reference_clearance_slow_band=0.45,
                 reference_infeasible_fallback=False,
                 spacetime_modes=False, spacetime_obstacle_r=0.075,
                 spacetime_horizon=None, spacetime_dt_layer=0.25,
                 spacetime_res=0.10, spacetime_window=1.0,
                 spacetime_flow=False, spacetime_flow_w=0.1,
                 spacetime_flow_reflood_every=1,
                 spacetime_flow_obstacle_r=0.075, spacetime_flow_horizon=None,
                 spacetime_flow_dt_layer=0.25, spacetime_flow_res=0.10,
                 spacetime_flow_window=1.0,
                 mode_switch_margin=0.3,
                 fallback_share=0.25, mode_warm_start=0.7,
                 mode_min_dwell=10, mode_confirm_cycles=3,
                 goal_w=5.0, run_w=2.0, repulse_w=20.0, margin=0.10,
                 collision_cost=1e4):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.K, self.T, self.dt, self.lam = K, T, dt, lam
        self.v_max, self.w_max = v_max, w_max
        self.v_accel_max, self.w_accel_max = v_accel_max, w_accel_max
        self.grouped_sampling = grouped_sampling
        self.importance_weighting = importance_weighting
        self.importance_ess_guard = importance_ess_guard
        self.importance_min_ess = importance_min_ess
        self.importance_min_ess_fraction = importance_min_ess_fraction
        self.adaptive_covariance = adaptive_covariance
        self.dynamic_prediction = bool(dynamic_prediction)
        self.prediction_horizon = float(prediction_horizon)
        self.prediction_uncertainty_rate = float(prediction_uncertainty_rate)
        self.prediction_times = (np.arange(T, dtype=float) + 1.0) * dt
        # Homotopy-consistency cost (see homotopy.py). Off by default so
        # every existing result is unchanged; `homotopy_monitor` measures
        # mode purity without altering costs or commands.
        self.homotopy_w = float(homotopy_w)
        self.homotopy_beta = float(homotopy_beta)
        self.homotopy_relevance = float(homotopy_relevance)
        self.homotopy_constrain_fallback = bool(homotopy_constrain_fallback)
        self.homotopy_monitor = bool(homotopy_monitor)
        # Speed shaping of the pseudopod REFERENCE (V2 proposal) only. The
        # defaults reproduce proposals.py exactly. Near obstacles the shaped
        # reference crawls (~0.16 m/s in ASSIST), so a 56-step rollout covers
        # ~0.46 m -- the shared trunk of every branch, not their split; see
        # docs/experiments/HOMOTOPY_ABLATION.md. The path fallback proposal
        # keeps the shaped defaults so NORMAL behaviour is unchanged.
        self.reference_kwargs = {
            "min_speed_ratio": float(reference_min_speed_ratio),
            "curvature_slowdown": float(reference_curvature_slowdown),
            "clearance_slow_band": float(reference_clearance_slow_band),
        }
        # A fast reference that reaches the branch split can clip a tight
        # gap and be rejected outright, which trades indistinct modes for
        # fewer modes. With this on, an infeasible fast reference falls back
        # to the shaped one for that branch, so the feasible set is a
        # superset of the shaped policy's.
        self.reference_infeasible_fallback = bool(reference_infeasible_fallback)
        self.reference_attempts = 0
        self.reference_fallbacks = 0
        # Space-time topology (see spacetime.py, docs/experiments/
        # SPACETIME_TOPOLOGY_PHASE0.md): when a branch's own reference would
        # predictably collide with a moving obstacle, add "wait"/"detour"
        # space-time alternatives as extra modes for that same branch
        # endpoint, instead of relying on cost-shaped sampling noise to
        # notice. Off by default -- no prior result changes.
        self.spacetime_modes = bool(spacetime_modes)
        self.spacetime_obstacle_r = float(spacetime_obstacle_r)
        self.spacetime_horizon = (
            T * dt if spacetime_horizon is None else float(spacetime_horizon))
        self.spacetime_dt_layer = float(spacetime_dt_layer)
        self.spacetime_res = float(spacetime_res)
        self.spacetime_window = float(spacetime_window)
        self.spacetime_triggers = 0
        self.spacetime_modes_added = 0
        # Space-time FLOOD (see spacetime_flow.py): a full (x, y, t)
        # distance field, evaluated as a continuous cost over every sample
        # every cycle -- unlike `spacetime_modes` above, which only ever
        # adds two extra discrete candidates when a crossing is detected
        # on one branch. Off by default -- no prior result changes.
        # `spacetime_flow_w=0.1` (not 1.0): at 1.0 the term is comparable
        # in scale to the existing flow/track guidance cost, which at
        # MPPI's fairly sharp `lam=0.3` softmax temperature was enough to
        # occasionally tip a near-tied sample selection the wrong way --
        # one matched-panel case (world 4, seed 1) turned a success into a
        # 700-step stuck-then-timeout episode at w=1.0, fixed at w=0.1.
        # First-pass matched panel at w=0.1 (18 world/seed pairs, dynamic
        # scenario): 18/18 vs 18/18 success both arms (no more flips),
        # mean clearance delta +0.014 (0.124 -> 0.138), but a paired
        # Wilcoxon signed-rank test gives p=0.316 -- directionally
        # positive, NOT statistically significant at this sample size.
        # Standalone/experimental: see docs/PROJECT_STATUS.md 2026-09-08.
        self.spacetime_flow = bool(spacetime_flow)
        self.spacetime_flow_w = float(spacetime_flow_w)
        self.spacetime_flow_reflood_every = max(
            1, int(spacetime_flow_reflood_every))
        self.spacetime_flow_obstacle_r = float(spacetime_flow_obstacle_r)
        self.spacetime_flow_horizon = (
            T * dt if spacetime_flow_horizon is None
            else float(spacetime_flow_horizon))
        self.spacetime_flow_dt_layer = float(spacetime_flow_dt_layer)
        self.spacetime_flow_res = float(spacetime_flow_res)
        self.spacetime_flow_window = float(spacetime_flow_window)
        self._spacetime_flood = None
        self._spacetime_flood_age = 0
        self.spacetime_flow_build_ms = []
        self.robot_model = RobotModel(
            kind=robot_model, radius=env.robot_r,
            footprint_length=footprint_length,
            footprint_width=footprint_width, yaw_gain=skid_yaw_gain,
            speed_yaw_loss=skid_speed_yaw_loss,
            turn_margin=turn_safety_margin,
            speed_margin=speed_safety_margin, v_max=v_max, w_max=w_max)
        self.importance_guard_cycles = 0
        self.importance_guard_mode_fallbacks = 0
        self.importance_guard_mode_evaluations = 0
        self.mode_switch_margin = mode_switch_margin
        self.fallback_share = fallback_share
        self.mode_warm_start = mode_warm_start
        self.mode_min_dwell = mode_min_dwell
        self.mode_confirm_cycles = mode_confirm_cycles
        self.selected_mode_key = None
        self.selected_mode_name = None
        self.selected_mode_age = 0
        self.pending_mode_key = None
        self.pending_mode_count = 0
        self.mode_switch_count = 0
        self.mode_nominals = {}
        self.omega_sign_flips = 0
        self._last_nonzero_omega_sign = 0
        self.std = np.array([v_std, w_std])
        self.goal_w, self.run_w = goal_w, run_w
        self.repulse_w, self.margin = repulse_w, margin
        self.collision_cost = collision_cost
        self.nominal = np.zeros((T, 2))
        self.nominal[:, 0] = 0.5 * v_max
        self.last_applied = np.zeros(2)

    # ---- extension points -------------------------------------------------
    def prepare(self, state):
        """Called once at the top of step(); variants update internal
        targets here before costs/bias are evaluated."""

    def sample_bias(self, state):
        """(T, 2) shift of the sampling mean. Zero for vanilla."""
        return 0.0

    def goal_cost(self, xs):
        """Per-trajectory attraction cost from states xs (K, T, 3)."""
        d = np.linalg.norm(xs[..., :2] - self.env.goal, axis=-1)
        return self.goal_w * d[:, -1] + self.run_w * d.mean(1)

    # ---- core -------------------------------------------------------------
    def rollout(self, x0, controls):
        K = controls.shape[0]
        xs = np.empty((K, self.T, 3))
        x = np.tile(x0, (K, 1))
        for t in range(self.T):
            x = self.robot_model.integrate(x, controls[:, t], self.dt)
            xs[:, t] = x
        return xs

    def rollout_clearance(self, xs):
        """Use V6.1 time-indexed obstacle predictions when enabled."""
        times = self.prediction_times if self.dynamic_prediction else None
        return self.robot_model.clearance(
            self.env, xs, time_offsets=times,
            prediction_horizon=self.prediction_horizon,
            uncertainty_rate=self.prediction_uncertainty_rate)

    def _maybe_rebuild_spacetime_flood(self, state):
        """Rebuild the full (x, y, t) flood every `spacetime_flow_
        reflood_every` cycles (default: every cycle) -- see
        `spacetime_flow.SpaceTimeFlood`. No-op unless `spacetime_flow` is
        on, so every existing result/config is completely unaffected by
        this method's existence."""
        if not self.spacetime_flow:
            return
        self._spacetime_flood_age += 1
        if (self._spacetime_flood is not None
                and self._spacetime_flood_age % self.spacetime_flow_reflood_every):
            return
        t0 = time.perf_counter()
        self._spacetime_flood = SpaceTimeFlood(
            self.env, state[:2], robot_r=self.robot_model.planning_radius,
            obstacle_r=self.spacetime_flow_obstacle_r,
            horizon=self.spacetime_flow_horizon,
            dt_layer=self.spacetime_flow_dt_layer, res=self.spacetime_flow_res,
            window=self.spacetime_flow_window)
        self.spacetime_flow_build_ms.append((time.perf_counter() - t0) * 1e3)

    def spacetime_flow_cost(self, xs):
        """Continuous space-time-flood attraction-away-from-danger cost,
        the same "goal_w * final + run_w * mean" shape `goal_cost` and
        `_FlowGuidedMPPI.goal_cost`'s flow term already use, just reading
        `SpaceTimeFlood.dist_batch` instead of `LocalFlowField.dist`.
        Zero when disabled or before the first flood is built (e.g. the
        very first cycle, or a `predicted_moving_obs`-free env)."""
        if not self.spacetime_flow or self._spacetime_flood is None:
            return 0.0
        d = self._spacetime_flood.dist_batch(xs[..., :2], self.prediction_times)
        return self.goal_w * d[:, -1] + self.run_w * d.mean(1)

    def step(self, state):
        self.prepare(state)
        self._maybe_rebuild_spacetime_flood(state)
        if self.grouped_sampling:
            return self._step_grouped(state)
        noise = self.rng.normal(size=(self.K, self.T, 2)) * self.std
        controls = self.nominal + self.sample_bias(state) + noise
        controls[..., 0] = np.clip(controls[..., 0], 0.0, self.v_max)
        controls[..., 1] = np.clip(controls[..., 1], -self.w_max, self.w_max)

        xs = self.rollout(state, controls)
        clear = self.rollout_clearance(xs)
        hard_margin = self.robot_model.extra_safety_margin(controls)
        collided = (clear < hard_margin).any(1)
        safety = self.robot_model.safety_margin(controls, self.margin)
        soft = np.clip(safety - clear, 0.0, None)
        costs = (collided * self.collision_cost
                 + self.repulse_w * soft.sum(1)
                 + self.goal_cost(xs)
                 + self.spacetime_flow_w * self.spacetime_flow_cost(xs))

        w = np.exp(-(costs - costs.min()) / self.lam)
        w /= w.sum()
        self.nominal = np.einsum("k,ktc->tc", w, controls)
        u = self.nominal[0].copy()
        plan = self.rollout(state, self.nominal[None])[0]
        info = {"xs": xs, "weights": w, "costs": costs, "plan": plan,
                "branch_proposals": getattr(self, "branch_proposals", [])}
        self.nominal[:-1] = self.nominal[1:]
        self.last_applied = u.copy()
        return u, info

    def _trajectory_costs(self, xs, controls):
        clear = self.rollout_clearance(xs)
        hard_margin = self.robot_model.extra_safety_margin(controls)
        collided = (clear < hard_margin).any(1)
        safety = self.robot_model.safety_margin(controls, self.margin)
        soft = np.clip(safety - clear, 0.0, None)
        return (collided * self.collision_cost
                + self.repulse_w * soft.sum(1)
                + self.goal_cost(xs)
                + self.spacetime_flow_w * self.spacetime_flow_cost(xs))

    def _sampling_modes(self, state):
        """Feasible V2 branches plus an always-present V0-style fallback."""
        path_proposal = getattr(self, "global_path_proposal", None)
        if path_proposal is not None and path_proposal.feasible:
            fallback_anchor = path_proposal.controls
            fallback_name = "path"
            fallback_reference = path_proposal.rollout[:, :2]
        else:
            fallback_anchor = self.nominal + self.sample_bias(state)
            fallback_name = "fallback"
            fallback_reference = None
        assist_active = getattr(self, "assist_state", "ASSIST") == "ASSIST"
        anchors = []
        if assist_active:
            anchors = [
                (p.branch_id, f"P{p.branch_id}", p.controls, p.rollout[:, :2])
                for p in getattr(self, "branch_proposals", []) if p.feasible
            ]
        anchors.append(
            (-1, fallback_name, fallback_anchor, fallback_reference))
        active_keys = {key for key, _, _, _ in anchors}
        self.mode_nominals = {
            key: value for key, value in self.mode_nominals.items()
            if key in active_keys}
        modes = []
        for key, name, anchor, reference in anchors:
            previous = self.mode_nominals.get(key)
            mean = (anchor if previous is None else
                    self.mode_warm_start * previous
                    + (1.0 - self.mode_warm_start) * anchor)
            mode_std = (
                self._geometry_adaptive_std(state, mean)
                if self.adaptive_covariance else None)
            modes.append(SamplingMode(
                key, name, mean, std=mode_std, reference=reference))
        return modes

    def _homotopy_costs(self, xs, labels, modes):
        """Soft T-MPC-style side-of-obstacle penalty per guided mode.

        Each mode with a reference rollout is held to that reference's side
        of every obstacle it passes near (see homotopy.py). The unguided
        fallback is left free unless `homotopy_constrain_fallback`, mirroring
        T-MPC++'s unconstrained planner running alongside the guided ones.
        Returns per-sample violation, a per-sample violated mask, and
        per-mode purity statistics.
        """
        total = np.zeros(len(labels))
        violated = np.zeros(len(labels), dtype=bool)
        stats = []
        obstacles = obstacle_set(self.env)
        if obstacles is None:
            return total, violated, stats
        centers, obstacle_r = obstacles
        for index, mode in enumerate(modes):
            if mode.reference is None:
                continue
            if mode.key == -1 and not self.homotopy_constrain_fallback:
                continue
            A, b = mode_side_planes(
                mode.reference, centers, self.robot_model.planning_radius,
                obstacle_r, relevance=self.homotopy_relevance,
                beta=self.homotopy_beta)
            group = np.flatnonzero(labels == index)
            violation = side_violations(xs[group][..., :2], A, b)
            total[group] = violation
            violated[group] = violation > 0.0
            stats.append({
                "index": index,
                "key": mode.key,
                "name": mode.name,
                "count": int(len(group)),
                "n_planes": int(len(A)),
                "violating_fraction": (
                    float(np.mean(violation > 0.0)) if len(group) else 0.0),
                "mean_violation": (
                    float(np.mean(violation)) if len(group) else 0.0),
            })
        return total, violated, stats

    def _homotopy_mode_distinctness(self, state, modes, updates):
        """T-MPC Fig. 6 analogue on the OPTIMIZED means, not the samples.

        Per-sample crossings are rare by construction (a 2.8 s horizon of
        rate-limited angular noise seldom carries a sample past an obstacle
        centre), so purity has to be read off the mode updates: does each
        guided mode's updated mean still lie on its own reference's side
        (`own_drift` counts those that don't), and does every pair of guided
        modes' UPDATES still pass at least one nearby obstacle on different
        sides from each other (`collapsed_pairs` counts those that don't)?
        The pair test compares update to update -- two updates that both
        crossed to the same side are collapsed, two that swapped sides are
        not -- using either mode's reference planes as the side definition.
        Pairs with no planes on either side (open space) are not counted.
        """
        guided = [
            (index, mode) for index, mode in enumerate(modes)
            if mode.reference is not None
            and (mode.key != -1 or self.homotopy_constrain_fallback)]
        out = {"guided": len(guided), "own_drift": 0,
               "pairs": 0, "collapsed_pairs": 0,
               # Same pair test on the REFERENCES themselves: the acceptance
               # metric for horizon-scale distinctness. A pair whose
               # references never pass a nearby obstacle on different sides
               # offers MPPI two copies of the same guidance.
               "reference_pairs": 0, "reference_separated": 0,
               "reference_length_sum": 0.0}
        obstacles = obstacle_set(self.env)
        if obstacles is None or not guided:
            return out
        centers, obstacle_r = obstacles
        planes, rollouts, crossed_own = {}, {}, {}

        def crosses(index, plane_index):
            return float(side_violations(
                rollouts[index], *planes[plane_index])[0]) > 0.0

        def reference_crosses(index, plane_index):
            return float(side_violations(
                modes[index].reference[None], *planes[plane_index])[0]) > 0.0

        for index, mode in guided:
            planes[index] = mode_side_planes(
                mode.reference, centers, self.robot_model.planning_radius,
                obstacle_r, relevance=self.homotopy_relevance,
                beta=self.homotopy_beta)
            rollouts[index] = self.rollout(
                state, updates[mode.key][None])[..., :2]
            crossed_own[index] = crosses(index, index)
            if crossed_own[index]:
                out["own_drift"] += 1
            out["reference_length_sum"] += float(np.linalg.norm(
                np.diff(mode.reference, axis=0), axis=1).sum())
        for a in range(len(guided)):
            for b in range(a + 1, len(guided)):
                i, j = guided[a][0], guided[b][0]
                if len(planes[i][0]) == 0 and len(planes[j][0]) == 0:
                    continue
                out["pairs"] += 1
                # Different sides of some obstacle: the two updates disagree
                # on whether they cross j's planes, or on whether they cross
                # i's planes.
                separated = (crosses(i, j) != crossed_own[j]
                             or crosses(j, i) != crossed_own[i])
                if not separated:
                    out["collapsed_pairs"] += 1
                out["reference_pairs"] += 1
                if reference_crosses(i, j) or reference_crosses(j, i):
                    out["reference_separated"] += 1
        return out

    def _geometry_adaptive_std(self, state, mean):
        """V5 diagonal exploration schedule from clearance and turn demand.

        Clearance is evaluated along the mode mean with the same footprint
        convention as rollout costs. Open straight motion receives wider
        exploration; confinement contracts both channels, while commanded
        turning shifts exploration from linear speed toward angular velocity.
        """
        predicted = self.rollout(state, np.asarray(mean)[None])[0]
        usable_clearance = (
            np.asarray(self.rollout_clearance(predicted),
                       dtype=float))
        clearance = np.clip(usable_clearance / 0.60, 0.0, 1.0)
        turn = np.clip(np.abs(np.asarray(mean)[:, 1])
                       / max(self.w_max, 1e-9), 0.0, 1.0)
        speed_scale = np.clip(
            0.60 + 0.60 * clearance - 0.20 * turn, 0.50, 1.20)
        turn_scale = np.clip(
            0.85 + 0.25 * (1.0 - clearance) + 0.20 * turn,
            0.85, 1.25)
        return np.column_stack((
            self.std[0] * speed_scale,
            self.std[1] * turn_scale))

    def _allocation_weights(self, modes):
        """Protect fallback/path capacity; divide the remainder over branches."""
        n_branches = sum(mode.key != -1 for mode in modes)
        if n_branches == 0:
            return np.ones(1)
        fallback = float(np.clip(self.fallback_share, 0.0, 1.0))
        branch_share = (1.0 - fallback) / n_branches
        return np.array([
            fallback if mode.key == -1 else branch_share for mode in modes])

    def _select_committed_mode(self, stats):
        """Mode choice with dwell time, improvement margin, and confirmation."""
        best = min(stats, key=lambda item: item["free_energy"])
        current = next(
            (item for item in stats
             if item["key"] == self.selected_mode_key), None)
        reason = "retained"
        if current is None:
            chosen, reason = best, (
                "initial" if self.selected_mode_key is None
                else "current_unavailable")
        elif best["key"] == current["key"]:
            chosen, reason = current, "best"
            self.pending_mode_key, self.pending_mode_count = None, 0
        elif self.selected_mode_age < self.mode_min_dwell:
            chosen, reason = current, "minimum_dwell"
        elif best["free_energy"] + self.mode_switch_margin >= current["free_energy"]:
            chosen, reason = current, "switch_margin"
            self.pending_mode_key, self.pending_mode_count = None, 0
        else:
            if self.pending_mode_key == best["key"]:
                self.pending_mode_count += 1
            else:
                self.pending_mode_key = best["key"]
                self.pending_mode_count = 1
            if self.pending_mode_count >= self.mode_confirm_cycles:
                chosen, reason = best, "confirmed_improvement"
            else:
                chosen, reason = current, "awaiting_confirmation"

        switched = (self.selected_mode_key is not None
                    and chosen["key"] != self.selected_mode_key)
        if switched:
            self.mode_switch_count += 1
            self.selected_mode_age = 0
            self.pending_mode_key, self.pending_mode_count = None, 0
        else:
            self.selected_mode_age += 1
        self.selected_mode_key = chosen["key"]
        self.selected_mode_name = chosen["name"]
        return chosen, reason, switched

    def _step_grouped(self, state):
        """V3/V4 grouped sampling with mode-conditioned policy updating."""
        modes = self._sampling_modes(state)
        raw, labels, counts = sample_fixed_groups(
            self.rng, modes, self.K, self.std,
            allocation_weights=self._allocation_weights(modes))
        controls = project_control_sequences(
            raw, self.last_applied, self.dt, self.v_max, self.w_max,
            self.v_accel_max, self.w_accel_max)
        xs = self.rollout(state, controls)
        costs = self._trajectory_costs(xs, controls)
        homotopy_stats, homotopy_violated = [], None
        if self.homotopy_w > 0.0 or self.homotopy_monitor:
            homotopy_costs, homotopy_violated, homotopy_stats = (
                self._homotopy_costs(xs, labels, modes))
            if self.homotopy_w > 0.0:
                costs = costs + self.homotopy_w * homotopy_costs
        importance = None
        if self.importance_weighting:
            # Fixed stratification is a deterministic draw allocation.  The
            # corresponding balance-heuristic mixture uses the realised
            # component proportions N_m / K.
            mixture_weights = counts / counts.sum()
            target = next(mode.mean for mode in modes if mode.key == -1)
            (global_weights, log_weights, log_p, log_q,
             log_normalizer) = mixture_importance_weights(
                raw, costs, target, modes, self.std, mixture_weights, self.lam)
            stats, within_weights = importance_mode_statistics(
                log_weights, labels, modes, self.lam)
            raw_corrected_stats = [dict(item) for item in stats]
            guard_cycle_fallback = 0
            if self.importance_ess_guard:
                uncorrected_stats, uncorrected_weights = mode_statistics(
                    costs, labels, modes, self.lam)
                stats, within_weights, guard_cycle_fallback = (
                    guard_importance_statistics(
                        stats, within_weights,
                        uncorrected_stats, uncorrected_weights, labels,
                        min_ess=self.importance_min_ess,
                        min_ess_fraction=self.importance_min_ess_fraction))
                self.importance_guard_mode_evaluations += len(stats)
                if guard_cycle_fallback:
                    self.importance_guard_cycles += 1
                    self.importance_guard_mode_fallbacks += len(stats)
            importance = {
                "importance_weights": global_weights,
                "log_importance_weights": log_weights,
                "log_p": log_p,
                "log_q": log_q,
                "log_weight_normalizer": log_normalizer,
                "mixture_weights": mixture_weights,
                "target_mean": target,
                "raw_corrected_mode_stats": raw_corrected_stats,
                "importance_guard_enabled": self.importance_ess_guard,
                "importance_guard_cycle_fallbacks": guard_cycle_fallback,
                "importance_guard_cycles": self.importance_guard_cycles,
                "importance_guard_mode_fallbacks":
                    self.importance_guard_mode_fallbacks,
                "importance_guard_mode_evaluations":
                    self.importance_guard_mode_evaluations,
            }
        else:
            stats, within_weights = mode_statistics(
                costs, labels, modes, self.lam)
        selected, selection_reason, switched = self._select_committed_mode(stats)
        chosen = labels == selected["index"]
        weights = np.where(chosen, within_weights, 0.0)
        # Weight mass of the executed update that sits on the wrong side of
        # an obstacle: the sharpest mode-purity number, since that mass is
        # what pulls the mode mean across the pillar.
        selected_wrong_side_mass = (
            float(np.sum(weights[chosen] * homotopy_violated[chosen]))
            if homotopy_violated is not None else np.nan)
        updates = {}
        for stat in stats:
            group = labels == stat["index"]
            update = np.einsum(
                "k,ktc->tc", within_weights[group], controls[group])
            updates[stat["key"]] = update
        self.nominal = updates[selected["key"]].copy()
        u = self.nominal[0].copy()
        plan = self.rollout(state, self.nominal[None])[0]
        homotopy_modes = (
            self._homotopy_mode_distinctness(state, modes, updates)
            if homotopy_violated is not None else None)
        for key, update in updates.items():
            shifted = update.copy()
            shifted[:-1] = shifted[1:]
            self.mode_nominals[key] = shifted
        omega_sign = int(np.sign(u[1])) if abs(u[1]) > 1e-3 else 0
        if (omega_sign and self._last_nonzero_omega_sign
                and omega_sign != self._last_nonzero_omega_sign):
            self.omega_sign_flips += 1
        if omega_sign:
            self._last_nonzero_omega_sign = omega_sign
        sorted_energy = sorted(item["free_energy"] for item in stats)
        energy_gap = (np.inf if len(sorted_energy) < 2
                      else float(sorted_energy[1] - sorted_energy[0]))
        info = {
            "xs": xs,
            "weights": weights,
            "costs": costs,
            "plan": plan,
            "branch_proposals": getattr(self, "branch_proposals", []),
            "raw_controls": raw,
            "controls": controls,
            "mode_labels": labels,
            "mode_counts": counts,
            "mode_stds": np.stack([
                np.broadcast_to(self.std, mode.mean.shape)
                if mode.std is None else mode.std
                for mode in modes]),
            "mode_stats": stats,
            "selected_mode": selected,
            "selection_reason": selection_reason,
            "mode_switched": switched,
            "selected_mode_age": self.selected_mode_age,
            "mode_switch_count": self.mode_switch_count,
            "omega_sign_flips": self.omega_sign_flips,
            "free_energy_gap": energy_gap,
            "mode_bank_keys": tuple(self.mode_nominals),
            "assist_state": getattr(self, "assist_state", "ASSIST"),
            "assist_reason": getattr(self, "assist_reason", "always"),
            "plan_version": getattr(self, "plan_version", 0),
            "robot_model": self.robot_model.kind,
            "weighting": (
                "v5_geometry_adaptive_ess_guarded_importance"
                if self.adaptive_covariance and self.importance_weighting
                and self.importance_ess_guard
                else "v5_geometry_adaptive_importance"
                if self.adaptive_covariance and self.importance_weighting
                else "v5_geometry_adaptive_v3_weights"
                if self.adaptive_covariance
                else
                "v4_1_ess_guarded_mixture_importance"
                if self.importance_weighting and self.importance_ess_guard
                else "v4_mixture_importance"
                if self.importance_weighting
                else "v3_uncorrected_within_mode"),
        }
        info["homotopy_w"] = self.homotopy_w
        info["homotopy_stats"] = homotopy_stats
        info["homotopy_selected_wrong_side_mass"] = selected_wrong_side_mass
        info["homotopy_modes"] = homotopy_modes
        if importance is not None:
            info.update(importance)
        self.nominal[:-1] = self.nominal[1:]
        self.last_applied = u.copy()
        return u, info

    # proposals.py defaults: the shaped reference every prior result used.
    _shaped_reference_kwargs = {
        "min_speed_ratio": 0.12,
        "curvature_slowdown": 1.2,
        "clearance_slow_band": 0.45,
    }

    def _branch_proposal(self, branch, state, **reference_kwargs):
        return branch_to_control_sequence(
            branch, state, self.env, T=self.T, dt=self.dt,
            v_max=self.v_max, w_max=self.w_max,
            v_accel_max=self.v_accel_max,
            w_accel_max=self.w_accel_max,
            initial_control=self.last_applied,
            robot_model=self.robot_model,
            **reference_kwargs)

    def build_branch_proposals(self, state):
        """V2 means for the current branches; observational until V3."""
        t0 = time.perf_counter()
        proposals = []
        for branch in getattr(self, "branches", []):
            proposal = self._branch_proposal(
                branch, state, **self.reference_kwargs)
            self.reference_attempts += 1
            if (self.reference_infeasible_fallback and not proposal.feasible
                    and self.reference_kwargs != self._shaped_reference_kwargs):
                shaped = self._branch_proposal(
                    branch, state, **self._shaped_reference_kwargs)
                if shaped.feasible:
                    proposal = shaped
                    self.reference_fallbacks += 1
            proposals.append(proposal)
            proposals.extend(
                self._spacetime_alternatives(state, branch, proposal))
        self.branch_proposals = proposals
        if hasattr(self, "proposal_ms"):
            self.proposal_ms.append((time.perf_counter() - t0) * 1e3)

    def _spacetime_alternatives(self, state, branch, base_proposal):
        """"Wait"/"detour" space-time alternatives for one branch (see
        `spacetime.py`, `docs/experiments/SPACETIME_TOPOLOGY_PHASE0.md`),
        if a moving obstacle is predicted to collide with the branch's own
        reference within the planning horizon.

        The search targets a point along the branch centreline reachable at
        NOMINAL speed within the search horizon, not the branch's full
        (possibly distant) endpoint: `time_expanded_search`'s goal is a hard
        arrival requirement, unlike the receding-horizon partial-progress
        tracking `branch_to_control_sequence` does, so handing it a
        genuinely unreachable-in-time goal makes both routes trivially
        infeasible regardless of the obstacle. Using the base proposal's
        own (possibly obstacle-slowed) rollout endpoint instead would be
        circular -- exactly the point being displaced by the obstacle it
        needs to route around.

        Returns zero, one, or two extra `BranchProposal`s, one per feasible
        route. Their `branch_id` is the original id offset by
        +100 ("wait") or +200 ("detour") -- distinct from the -1 fallback
        and the small non-negative ids `pseudopods.py` assigns, and stable
        across cycles for the same branch, so mode warm-starting and
        dwell/switch hysteresis treat them exactly like any other mode
        without special-casing. Only triggers off an already-feasible base
        proposal: if the branch itself isn't viable there is nothing for a
        space-time alternative to be an alternative TO.
        """
        if not self.spacetime_modes or not base_proposal.feasible:
            return []
        centerline = np.asarray(branch.centerline, dtype=float)
        arc = _arc_length(centerline)
        # 90% of the search grid's OWN straight-line reach (res / dt_layer,
        # not v_max -- a coarser grid than the MPPI-implied speed makes even
        # the direct, unobstructed route take longer than the horizon
        # allows, failing "reachable in principle" for a reason that has
        # nothing to do with the obstacle), leaving slack for any detour's
        # extra distance or a wait's lost time.
        search_speed = self.spacetime_res / self.spacetime_dt_layer
        local_goal = _interpolate(
            centerline, arc, 0.9 * search_speed * self.spacetime_horizon)

        # Crossing detection must look as far ahead as the search itself
        # (`spacetime_horizon`), not just the base proposal's own MPPI-
        # horizon-limited rollout (`self.prediction_times`, capped at
        # T * dt): a crossing that only matters beyond that shorter window
        # would otherwise never be seen, no matter how far `spacetime_
        # horizon` is set to look. Checked against the branch's own
        # nominal-speed continuation along its centreline, not the base
        # proposal's achieved (possibly slower) rollout, for the same
        # reason the local goal isn't tied to that rollout either.
        nk = int(round(self.spacetime_horizon / self.spacetime_dt_layer)) + 1
        check_times = np.arange(nk) * self.spacetime_dt_layer
        check_xy = _interpolate(
            centerline, arc, search_speed * check_times).T
        crossing = nearest_crossing_obstacle(
            self.env, check_xy, check_times,
            self.robot_model.planning_radius, self.spacetime_obstacle_r)
        if crossing is None:
            return []
        self.spacetime_triggers += 1
        predict, _margin = crossing
        routes = two_route_search(
            self.env, state[:2], local_goal, predict,
            self.spacetime_obstacle_r, self.robot_model.planning_radius,
            horizon=self.spacetime_horizon, dt_layer=self.spacetime_dt_layer,
            res=self.spacetime_res, window=self.spacetime_window)
        extras = []
        for name, offset in (("wait", 100), ("detour", 200)):
            route = routes[name]
            if not route["feasible"]:
                continue
            proposal = spacetime_path_to_proposal(
                route["path"], self.spacetime_dt_layer, state,
                local_goal, self.env, self.robot_model, predict,
                self.spacetime_obstacle_r, T=self.T, dt=self.dt,
                v_max=self.v_max, w_max=self.w_max,
                v_accel_max=self.v_accel_max, w_accel_max=self.w_accel_max,
                initial_control=self.last_applied,
                branch_id=branch.branch_id + offset)
            if proposal.feasible:
                extras.append(proposal)
                self.spacetime_modes_added += 1
        return extras
