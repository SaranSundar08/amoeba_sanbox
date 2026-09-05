#!/usr/bin/env python3
"""Run one episode in a BARN world and watch / score it.

  python3 run.py --world 42 --controller local --animate
  python3 run.py --world 138 --controller vanilla --png out.png
"""
import argparse
import time

import numpy as np

from env import BarnEnv, CYL_R
from controllers import CONTROLLERS
from diagnostics import _print_mode_debug
from simulation import episode
from visualization import (
    AQUA, BRANCH_COLORS, GREEN, GRIDLINE, INK, OBSTACLE, ORANGE,
    RED, SURFACE, VIOLET, _clear_contour, _clear_flowviz, _draw_field,
    draw_world)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=42)
    p.add_argument("--controller",
                   choices=list(CONTROLLERS) + ["track"], default="vanilla")
    p.add_argument("--lane", choices=["left", "center", "right"],
                   help="feed a fake straight-lane global path (see "
                        "path_test.py); works with local, track and hybrid")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--animate", action="store_true", help="live visualization")
    p.add_argument("--renderer", choices=("pyqtgraph", "matplotlib"),
                   default="pyqtgraph",
                   help="live --animate backend (default: pyqtgraph)")
    p.add_argument("--every", type=int, default=4, help="redraw every N steps")
    p.add_argument("--png", help="save final trace to this file")
    p.add_argument("--viscosity", type=float, default=None,
                   help="water viscosity override (0 = pure geodesic flood)")
    p.add_argument("--path-w", type=float, default=None,
                   help="hybrid: weight on the A*-path-tracking term")
    p.add_argument("--flow-w", type=float, default=None,
                   help="hybrid: weight on the amoeba flow-guidance term")
    p.add_argument("--window", type=float, default=None,
                   help="local/hybrid: amoeba body's geodesic reach in "
                        "meters (default 2.5) -- how far it can sense, "
                        "not how far it wants to go")
    p.add_argument("--gate", choices=["always", "auto"], default="always",
                   help="hybrid: 'always' blends flow guidance every step "
                        "(validated 8/9); 'auto' keeps the A* path as the "
                        "only voice on the open road and only engages the "
                        "amoeba in confined spots or when stuck")
    p.add_argument("--dynamic", action="store_true",
                   help="DynaBARN-style test: add oscillating obstacles on "
                        "top of the static world (see dynabarn.py)")
    p.add_argument("--n-moving", type=int, default=4,
                   help="--dynamic: number of oscillating obstacles")
    p.add_argument("--mov-speed", type=float, default=0.4,
                   help="--dynamic: oscillation angular speed (rad/s)")
    p.add_argument("--mov-amp", type=float, default=0.6,
                   help="--dynamic: oscillation amplitude (m)")
    p.add_argument("--predict-dynamic", action="store_true",
                   help="V6.1: score rollouts against constant-velocity "
                        "dynamic-obstacle predictions")
    p.add_argument("--prediction-horizon", type=float, default=2.0,
                   help="V6.1: maximum constant-velocity extrapolation (s)")
    p.add_argument("--prediction-uncertainty", type=float, default=0.03,
                   help="V6.1: obstacle-radius growth per predicted second")
    p.add_argument("--dyn-seed", type=int, default=None,
                   help="--dynamic: obstacle placement/motion seed. Default "
                        "combines --world and --seed, so different worlds "
                        "get different layouts automatically instead of "
                        "the same pattern every time; pass this to pin an "
                        "exact layout independent of --world/--seed (e.g. "
                        "for an apples-to-apples controller comparison)")
    p.add_argument("--arrow-density", type=float, default=1.0,
                   help="downhill-direction arrow/streamline density "
                        "(1.0 = default, 2.0 = ~2x as many -- render cost "
                        "barely changes, safe to push higher)")
    p.add_argument("--pseudopods", action="store_true",
                   help="V1/V2: extract branches and build/visualize their "
                        "control-sequence means; MPPI sampling is unchanged")
    p.add_argument("--max-pseudopods", type=int, choices=(1, 2, 3), default=3,
                   help="V1: maximum number of retained branches")
    p.add_argument("--grouped-sampling", action="store_true",
                   help="V3: sample fixed groups around feasible branch means "
                        "and a nominal fallback")
    p.add_argument("--importance-weighting", action="store_true",
                   help="V4: apply full-mixture log p - log q correction in "
                        "latent control space (requires grouped sampling)")
    p.add_argument("--importance-ess-guard", action="store_true",
                   help="V4.1: fall back to V3 for a complete cycle when any "
                        "mode's corrected ESS is below max(8, 2%% of its "
                        "sample count)")
    p.add_argument("--adaptive-covariance", action="store_true",
                   help="V5: use per-mode, per-step diagonal covariance from "
                        "predicted clearance and turning demand")
    p.add_argument("--reference-policy", choices=("shaped", "nominal_fb"),
                   default="shaped",
                   help="pseudopod reference speed: 'shaped' (clearance/"
                        "curvature slowdown, the frozen default) or "
                        "'nominal_fb' (nominal speed, falling back to shaped "
                        "per branch when infeasible; see "
                        "docs/experiments/REFERENCE_DISTINCTNESS.md)")
    p.add_argument("--homotopy-w", type=float, default=0.0,
                   help="T-MPC-style mode-purity cost weight (0 = off)")
    p.add_argument("--homotopy-monitor", action="store_true",
                   help="report mode-purity/distinctness statistics without "
                        "changing costs or commands")
    p.add_argument("--spacetime-modes", action="store_true",
                   help="add 'wait'/'detour' space-time alternatives "
                        "(branch id +100/+200) when a predicted moving "
                        "obstacle (--dynamic) crosses a branch's own "
                        "corridor within --spacetime-horizon; requires "
                        "--pseudopods/--grouped-sampling and --dynamic. See "
                        "docs/experiments/SPACETIME_TOPOLOGY_PHASE0.md")
    p.add_argument("--spacetime-horizon", type=float, default=None,
                   help="--spacetime-modes: how far ahead the space-time "
                        "search looks, in seconds (default: the controller's "
                        "own T*dt)")
    p.add_argument("--spacetime-dt-layer", type=float, default=0.25,
                   help="--spacetime-modes: seconds between search time "
                        "layers")
    p.add_argument("--spacetime-res", type=float, default=0.10,
                   help="--spacetime-modes: search grid resolution in "
                        "metres (with --spacetime-dt-layer, sets the "
                        "search's own implied speed)")
    p.add_argument("--spacetime-window", type=float, default=1.0,
                   help="--spacetime-modes: lateral room, in metres, for a "
                        "detour route")
    p.add_argument("--spacetime-obstacle-r", type=float, default=CYL_R,
                   help="--spacetime-modes: assumed moving-obstacle radius")
    p.add_argument("--robot-model", choices=("ideal", "slip"),
                   default="ideal",
                   help="V6: shared prediction/simulation model")
    p.add_argument("--footprint-length", type=float, default=0.90,
                   help="V6 slip footprint length in metres")
    p.add_argument("--footprint-width", type=float, default=0.65,
                   help="V6 slip footprint width in metres")
    p.add_argument("--skid-yaw-gain", type=float, default=0.82,
                   help="V6 zero-speed achieved/commanded yaw-rate ratio")
    p.add_argument("--skid-speed-yaw-loss", type=float, default=0.55,
                   help="V6 additional yaw loss as speed increases")
    p.add_argument("--turn-safety-margin", type=float, default=0.08,
                   help="V6 extra soft clearance at maximum turning")
    p.add_argument("--speed-safety-margin", type=float, default=0.03,
                   help="V6 extra soft clearance at maximum speed")
    p.add_argument("--fallback-share", type=float, default=0.25,
                   help="V3.1: fraction of samples reserved for the path/"
                        "nominal fallback (default 0.25; selected by ablation)")
    p.add_argument("--mode-dwell", type=int, default=10,
                   help="V3.1: minimum cycles before a voluntary mode switch")
    p.add_argument("--mode-confirm", type=int, default=3,
                   help="V3.1: consecutive winning cycles required to switch")
    p.add_argument("--switch-margin", type=float, default=0.3,
                   help="V3.1: free-energy improvement required to switch")
    p.add_argument("--nav2-path-validity", action="store_true",
                   help="hybrid: ignore occupied path points and disable path "
                        "alignment when blockage exceeds Nav2's 0.07 ratio")
    p.add_argument("--mode-debug-every", type=int, default=0, metavar="N",
                   help="print V2/V3 mode diagnostics every N cycles")
    p.add_argument("--replan-every", type=float, default=0.0, metavar="SECONDS",
                   help="hybrid: periodically rerun A* from the current pose; "
                        "0 disables replanning")
    args = p.parse_args()

    if args.dynamic:
        from dynabarn import DynaBarnEnv
        dyn_seed = (args.dyn_seed if args.dyn_seed is not None
                   else args.world * 7919 + args.seed)
        env = DynaBarnEnv(args.world, args.scale, args.radius,
                          n_moving=args.n_moving, speed=args.mov_speed,
                          amplitude=args.mov_amp, seed=dyn_seed)
    else:
        env = BarnEnv(args.world, args.scale, args.radius)
    if args.robot_model == "slip":
        # A*/geodesic layers are 2-D and orientation-free. Inflate them by
        # half-width plus turn margin; oriented rollout collision checks use
        # the complete rectangle later in RobotModel.
        env.robot_r = max(
            env.robot_r, 0.5 * args.footprint_width + args.turn_safety_margin)
    vkw = {}
    vkw.update(
        robot_model=args.robot_model,
        footprint_length=args.footprint_length,
        footprint_width=args.footprint_width,
        skid_yaw_gain=args.skid_yaw_gain,
        skid_speed_yaw_loss=args.skid_speed_yaw_loss,
        turn_safety_margin=args.turn_safety_margin,
        speed_safety_margin=args.speed_safety_margin,
        dynamic_prediction=args.predict_dynamic,
        prediction_horizon=args.prediction_horizon,
        prediction_uncertainty_rate=args.prediction_uncertainty)
    if args.predict_dynamic and not args.dynamic:
        p.error("--predict-dynamic requires --dynamic")
    if args.viscosity is not None and args.controller in ("local", "hybrid"):
        vkw["viscosity"] = args.viscosity
    if args.window is not None and args.controller in ("local", "hybrid"):
        vkw["window"] = args.window
    if args.pseudopods or args.grouped_sampling:
        if args.controller not in ("local", "hybrid"):
            p.error("--pseudopods/--grouped-sampling only work with "
                    "--controller local or hybrid")
        vkw["extract_branches"] = True
        vkw["max_branches"] = args.max_pseudopods
    if args.grouped_sampling:
        vkw["grouped_sampling"] = True
        vkw["importance_weighting"] = args.importance_weighting
        vkw["importance_ess_guard"] = args.importance_ess_guard
        vkw["adaptive_covariance"] = args.adaptive_covariance
        vkw["fallback_share"] = args.fallback_share
        vkw["mode_min_dwell"] = args.mode_dwell
        vkw["mode_confirm_cycles"] = args.mode_confirm
        vkw["mode_switch_margin"] = args.switch_margin
        vkw["homotopy_w"] = args.homotopy_w
        vkw["homotopy_monitor"] = args.homotopy_monitor
        if args.spacetime_modes:
            if not args.dynamic:
                p.error("--spacetime-modes requires --dynamic")
            vkw["spacetime_modes"] = True
            vkw["spacetime_dt_layer"] = args.spacetime_dt_layer
            vkw["spacetime_res"] = args.spacetime_res
            vkw["spacetime_window"] = args.spacetime_window
            vkw["spacetime_obstacle_r"] = args.spacetime_obstacle_r
            if args.spacetime_horizon is not None:
                vkw["spacetime_horizon"] = args.spacetime_horizon
    elif args.importance_weighting:
        p.error("--importance-weighting requires --grouped-sampling")
    elif args.homotopy_w > 0.0 or args.homotopy_monitor:
        p.error("--homotopy-w/--homotopy-monitor require --grouped-sampling")
    elif args.spacetime_modes:
        p.error("--spacetime-modes requires --pseudopods/--grouped-sampling")
    if args.reference_policy == "nominal_fb":
        if not (args.pseudopods or args.grouped_sampling):
            p.error("--reference-policy nominal_fb requires --pseudopods or "
                    "--grouped-sampling")
        vkw["reference_min_speed_ratio"] = 1.0
        vkw["reference_curvature_slowdown"] = 0.0
        vkw["reference_infeasible_fallback"] = True
    if args.importance_ess_guard and not args.importance_weighting:
        p.error("--importance-ess-guard requires --importance-weighting")
    if args.adaptive_covariance and not args.grouped_sampling:
        p.error("--adaptive-covariance requires --grouped-sampling")
    if args.controller == "hybrid":
        if args.path_w is not None:
            vkw["path_w"] = args.path_w
        if args.flow_w is not None:
            vkw["flow_w"] = args.flow_w
        vkw["gate"] = args.gate
        vkw["path_validity_gate"] = args.nav2_path_validity
    if args.replan_every > 0.0 and args.controller != "hybrid":
        p.error("--replan-every currently requires --controller hybrid")
    path = None
    if args.lane:
        if args.controller not in ("local", "track", "hybrid"):
            p.error("--lane only works with --controller local, track or hybrid")
        # deliberately WRONG stand-in path, for the wrong-global-plan
        # stress test (path_test.py) -- not what a real planner would emit
        from path_test import lane_path, PathTrackMPPI
        path = lane_path(env, {"left": -3.5, "center": -2.25,
                               "right": -1.0}[args.lane])
        ctrl = (PathTrackMPPI(env, seed=args.seed, path=path)
                if args.controller == "track"
                else CONTROLLERS[args.controller](env, seed=args.seed,
                                                  path=path, **vkw))
    elif args.controller in ("track", "hybrid"):
        # the real global planner: A* over the same robot-radius-inflated
        # free space the amoeba's own flood uses, then shortcut-smoothed
        from astar import astar_path
        from path_test import PathTrackMPPI
        t0 = time.time()
        path = astar_path(env, env.start[:2], env.goal)
        print(f"A* plan: {len(path)} pts, {(time.time() - t0) * 1000:.1f} ms")
        ctrl = (PathTrackMPPI(env, seed=args.seed, path=path)
                if args.controller == "track"
                else CONTROLLERS[args.controller](env, seed=args.seed,
                                                  path=path, **vkw))
    else:
        ctrl = CONTROLLERS[args.controller](env, seed=args.seed, **vkw)

    on_step = None
    animator = None
    if args.animate and args.renderer == "pyqtgraph":
        from pyqtgraph_visualization import PyQtGraphAnimator
        try:
            animator = PyQtGraphAnimator(
                env, ctrl, path, args.world, every=args.every,
                mode_debug_every=args.mode_debug_every)
        except RuntimeError as error:
            p.error(str(error))
        on_step = animator.update

    if args.animate and args.renderer == "matplotlib":
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        from matplotlib.colors import to_rgba
        from matplotlib.collections import LineCollection
        plt.ion()
        fig, ax = plt.subplots(figsize=(5.5, 9))
        draw_world(ax, env, ctrl, density=args.arrow_density)
        global_path_ln, = ax.plot(
            [], [], "--", color=VIOLET, lw=1.6, label="global plan (A*)")
        sub = np.linspace(0, ctrl.K - 1, 128).astype(int)
        # the sample fan: one hue, alpha graded by softmax weight
        fan = LineCollection([], linewidths=0.7, zorder=2)
        ax.add_collection(fan)
        branch_lines = LineCollection([], linewidths=2.2, zorder=4)
        ax.add_collection(branch_lines)
        proposal_lines = LineCollection([], linewidths=2.7, zorder=4.5)
        ax.add_collection(proposal_lines)
        branch_ends = ax.scatter([], [], marker="X", s=48, zorder=5)
        plan_ln, = ax.plot([], [], color=GREEN, lw=2.2, zorder=4,
                           label="chosen plan")
        trace_ln, = ax.plot([], [], color=RED, lw=2, zorder=5,
                            label="driven path")
        heading_ln, = ax.plot([], [], color=INK, lw=2, zorder=7)
        leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
                        edgecolor=GRIDLINE)
        leg.get_frame().set_facecolor(SURFACE)

        # Blitting: snapshot everything static (walls/obstacles/goal/legend,
        # once as a bitmap. Every frame then restores that bitmap and
        # draws only what actually moved, instead of re-rendering the whole
        # figure (obstacles included) from scratch -- ~210ms/frame measured
        # on this scene, the main source of the choppiness. `body` is added
        # AFTER the snapshot so its start-position circle isn't baked in.
        fig.canvas.draw()
        bg = fig.canvas.copy_from_bbox(ax.bbox)
        if path is not None:
            global_path_ln.set_data(path[:, 0], path[:, 1])
        if ctrl.robot_model.is_slip:
            length = ctrl.robot_model.footprint_length
            width = ctrl.robot_model.footprint_width
            body = mp.Rectangle(
                env.start[:2] - np.array([0.5 * length, 0.5 * width]),
                length, width, angle=np.degrees(env.start[2]),
                rotation_point="center", fill=False,
                color=INK, lw=2, zorder=6)
        else:
            body = mp.Circle(env.start[:2], env.robot_r, fill=False,
                             color=INK, lw=2, zorder=6)
        ax.add_patch(body)
        mov_patches = []
        if hasattr(env, "moving_obs"):
            for _ in range(len(env.mov_center)):
                mp_patch = mp.Circle((0, 0), CYL_R, facecolor=OBSTACLE,
                                     edgecolor=RED, linewidth=1.4, zorder=4)
                ax.add_patch(mp_patch)
                mov_patches.append(mp_patch)
        dynamic = ([fan, branch_lines, proposal_lines, branch_ends,
                   global_path_ln, plan_ln, trace_ln,
                   heading_ln, body, ax.title] + mov_patches)
        holder = {"im": None, "cont": None, "dirs": None}  # moving amoeba
                                                            # body (local/hybrid)

        def _draw_any(a):
            if a is None:
                return
            if isinstance(a, (list, tuple)):
                for item in a:
                    _draw_any(item)
                return
            if hasattr(a, "collections") and not hasattr(a, "get_array"):
                for c in a.collections:
                    ax.draw_artist(c)
            else:
                ax.draw_artist(a)

        def on_step(k, state, info, trace):
            if args.mode_debug_every and k % args.mode_debug_every == 0:
                _print_mode_debug(k, state, info, ctrl)
            if k % args.every:
                return
            fig.canvas.restore_region(bg)
            if ctrl.name in ("local", "hybrid") and ctrl.flow is not None:
                _clear_flowviz(holder["dirs"])
                _clear_contour(holder["cont"])
                if holder["im"] is not None:
                    holder["im"].remove()
                # "quiver" here, not "stream": streamplot integrates the
                # whole grid from scratch every call (~100ms) and can't be
                # incrementally updated, so it's reserved for the static
                # PNG / first frame; a coarse quiver is ~1ms and looks
                # almost as good once it's actually moving at speed.
                holder["im"], holder["cont"], holder["dirs"] = \
                    _draw_field(ax, ctrl.flow, mode="quiver",
                               density=args.arrow_density)
                for a in (holder["im"], holder["cont"], holder["dirs"]):
                    _draw_any(a)
            xs, w = info["xs"][sub], info["weights"][sub]
            wn = w / (w.max() + 1e-12)
            rgba = np.tile(to_rgba(ORANGE), (len(wn), 1))
            if "mode_labels" in info:
                key_by_index = {
                    item["index"]: item["key"] for item in info["mode_stats"]}
                for row, label in enumerate(info["mode_labels"][sub]):
                    key = key_by_index[int(label)]
                    if key >= 0:
                        rgba[row] = to_rgba(
                            BRANCH_COLORS[key % len(BRANCH_COLORS)])
            rgba[:, 3] = 0.05 + 0.55 * wn         # faint losers, vivid winners
            fan.set_segments(list(xs[..., :2]))
            fan.set_colors(rgba)
            branches = getattr(ctrl, "branches", [])
            branch_lines.set_segments([b.centerline for b in branches])
            colors = [BRANCH_COLORS[b.branch_id % len(BRANCH_COLORS)]
                      for b in branches]
            branch_lines.set_colors(colors)
            branch_lines.set_linestyles(":")
            branch_ends.set_offsets(
                np.array([b.endpoint for b in branches]).reshape(-1, 2))
            branch_ends.set_color(colors)
            proposals = info.get("branch_proposals", [])
            proposal_lines.set_segments([p.rollout[:, :2] for p in proposals])
            proposal_lines.set_colors([
                BRANCH_COLORS[p.branch_id % len(BRANCH_COLORS)]
                for p in proposals])
            proposal_lines.set_linestyles([
                "-" if p.feasible else "--" for p in proposals])
            proposal_lines.set_alpha(
                0.95 if all(p.feasible for p in proposals) else 0.7)
            current_path = getattr(ctrl, "path", path)
            if current_path is not None:
                global_path_ln.set_data(current_path[:, 0], current_path[:, 1])
            plan_ln.set_data(info["plan"][:, 0], info["plan"][:, 1])
            t = np.array(trace)
            trace_ln.set_data(t[:, 0], t[:, 1])
            if ctrl.robot_model.is_slip:
                body.set_xy(state[:2] - np.array([
                    0.5 * ctrl.robot_model.footprint_length,
                    0.5 * ctrl.robot_model.footprint_width]))
                body.angle = np.degrees(state[2])
            else:
                body.center = state[:2]
            if mov_patches:
                for mp_patch, xy in zip(mov_patches, env.moving_obs()):
                    mp_patch.center = xy
            # hybrid --gate auto: outline turns aqua exactly when the
            # amoeba's guidance actually engages (confined / stuck), ink
            # otherwise -- makes the gate's on/off behavior visible
            engaged = getattr(ctrl, "active", None)
            body.set_edgecolor(AQUA if (engaged is not None and engaged > 0.5)
                               else INK)
            heading_ln.set_data(
                [state[0], state[0] + 0.9 * env.robot_r * np.cos(state[2])],
                [state[1], state[1] + 0.9 * env.robot_r * np.sin(state[2])])
            selected = info.get("selected_mode")
            mode_text = "" if selected is None else (
                f" | selected {selected['name']} "
                f"({info.get('selection_reason', '')})")
            ax.set_title(f"{ctrl.name} | world {args.world} | "
                         f"t={k * ctrl.dt:.1f}s{mode_text}",
                         color=INK, fontsize=10)
            for a in dynamic:
                ax.draw_artist(a)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

    if args.mode_debug_every and on_step is None:
        def on_step(k, state, info, trace):
            del trace
            if k % args.mode_debug_every == 0:
                _print_mode_debug(k, state, info, ctrl)

    before_step = None
    if args.replan_every > 0.0:
        replan_steps = max(1, int(round(args.replan_every / ctrl.dt)))

        def before_step(k, state):
            if k == 0 or k % replan_steps:
                return
            from astar import astar_path
            try:
                new_path = astar_path(env, state[:2], env.goal)
            except RuntimeError as error:
                print(f"replan k={k}: no path ({error}); keeping plan v"
                      f"{ctrl.plan_version}", flush=True)
                return
            ctrl.set_path(new_path)
            print(f"replan k={k}: plan v{ctrl.plan_version}, "
                  f"{len(new_path)} points", flush=True)

    t0 = time.time()
    m = episode(
        env, ctrl, args.max_steps, on_step=on_step, before_step=before_step)
    wall = time.time() - t0
    print(f"[{ctrl.name} | world {args.world} x{args.scale} | seed {args.seed}] "
          f"{m['result']}  time={m['time_s']}s  path={m['path_len_m']}m  "
          f"min_clear={m['min_clear_m']}m  stall={m['stall_s']}s  "
          f"(wall {wall:.1f}s)")
    if args.grouped_sampling and getattr(ctrl, "selected_mode_key", None) is not None:
        weighting = (
            "V5 adaptive covariance with V4.1 guarded weights"
            if args.adaptive_covariance and args.importance_weighting
            and args.importance_ess_guard
            else "V5 adaptive covariance with raw V4 weights"
            if args.adaptive_covariance and args.importance_weighting
            else "V5 adaptive covariance with V3 weights"
            if args.adaptive_covariance
            else
            "V4.1 ESS-guarded mixture weights"
            if args.importance_weighting and args.importance_ess_guard
            else "V4 corrected mixture weights"
            if args.importance_weighting
            else "V3 uncorrected grouped weights")
        print(f"final selected mode: "
              f"{ctrl.selected_mode_name} "
              f"({weighting}, switches={ctrl.mode_switch_count}, "
              f"omega_sign_flips={ctrl.omega_sign_flips})")
        if args.spacetime_modes:
            print(f"space-time: {ctrl.spacetime_triggers} crossings "
                  f"detected, {ctrl.spacetime_modes_added} wait/detour "
                  f"modes added over the episode")
        if args.importance_ess_guard:
            evaluations = ctrl.importance_guard_mode_evaluations
            fallbacks = ctrl.importance_guard_mode_fallbacks
            rate = fallbacks / evaluations if evaluations else 0.0
            print(f"V4.1 ESS guard: cycles={ctrl.importance_guard_cycles}, "
                  f"mode_fallbacks={fallbacks}/{evaluations} "
                  f"({rate:.1%})")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        fig, ax = plt.subplots(figsize=(5, 9))
        draw_world(ax, env, ctrl, density=args.arrow_density)
        final_path = getattr(ctrl, "path", path)
        if final_path is not None:
            ax.plot(final_path[:, 0], final_path[:, 1], "--",
                    color=VIOLET, lw=1.6)
        if hasattr(env, "moving_obs"):
            for xy in env.moving_obs():
                ax.add_patch(mp.Circle(xy, CYL_R, facecolor=OBSTACLE,
                                       edgecolor=RED, linewidth=1.4, zorder=4))
        t = m["trace"]
        ax.plot(t[:, 0], t[:, 1], color=RED, lw=2)
        ax.set_title(f"{ctrl.name} | world {args.world} | {m['result']}",
                     color=INK, fontsize=10)
        fig.savefig(args.png, dpi=110, bbox_inches="tight")
        print(f"saved {args.png}")

    if animator is not None:
        animator.finish()
    elif args.animate:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
