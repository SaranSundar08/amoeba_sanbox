#!/usr/bin/env python3
"""Live side-by-side comparison of path-biased and Amoeba MPPI."""
import argparse
import time

import numpy as np

from astar import astar_path
from env import BarnEnv, CYL_R
from experiments.v7_benchmark import _controller


COLORS = ("#d33682", "#6c9f2f", "#8c5ac7")


def _segments(paths):
    if not paths:
        return np.empty(0), np.empty(0)
    points = np.vstack([part for path in paths
                        for part in (np.asarray(path)[:, :2],
                                     np.full((1, 2), np.nan))])
    return points[:, 0], points[:, 1]


class Panel:
    def __init__(self, pg, qtwidgets, plot, env, path, ctrl, label, nshow):
        self.pg, self.plot, self.env, self.ctrl = pg, plot, env, ctrl
        self.label, self.nshow = label, nshow
        plot.setAspectLocked(True)
        plot.setXRange(env.xmin - 0.25, env.xmax + 0.25)
        plot.setYRange(0.0, env.ymax + 0.15)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel("bottom", "x", units="m")
        plot.setLabel("left", "y", units="m")
        for xy in env.obs:
            obstacle = qtwidgets.QGraphicsEllipseItem(
                xy[0] - CYL_R, xy[1] - CYL_R, 2 * CYL_R, 2 * CYL_R)
            obstacle.setBrush(pg.mkBrush("#52514e"))
            obstacle.setPen(pg.mkPen("#52514e"))
            obstacle.setZValue(-3)
            plot.addItem(obstacle)
        for x in (env.xmin, env.xmax):
            plot.plot([x, x], [0.0, env.ymax],
                      pen=pg.mkPen("#52514e", width=3))
        plot.plot(path[:, 0], path[:, 1],
                  pen=pg.mkPen("#4a3aa7", width=2,
                               style=pg.QtCore.Qt.PenStyle.DashLine))
        plot.plot([env.goal[0]], [env.goal[1]], pen=None, symbol="star",
                  symbolSize=20, symbolBrush="#eda100", symbolPen="#151515")
        self.samples = plot.plot(pen=pg.mkPen((235, 104, 52, 55), width=1))
        self.plan = plot.plot(pen=pg.mkPen("#008300", width=3))
        self.trace = plot.plot(pen=pg.mkPen("#e34948", width=3))
        self.membrane = plot.plot(pen=None, symbol="o", symbolSize=2,
                                  symbolBrush=(34, 190, 205, 175),
                                  symbolPen=None)
        self.branches = [plot.plot(pen=pg.mkPen(
            color, width=2, style=pg.QtCore.Qt.PenStyle.DotLine))
            for color in COLORS]
        self.proposals = [plot.plot(pen=pg.mkPen(color, width=3))
                          for color in COLORS]
        model = ctrl.robot_model
        length = model.footprint_length if model.is_slip else 2 * model.radius
        width = model.footprint_width if model.is_slip else 2 * model.radius
        self.robot = qtwidgets.QGraphicsRectItem(
            -length / 2, -width / 2, length, width)
        self.robot.setTransformOriginPoint(0.0, 0.0)
        self.robot.setBrush(pg.mkBrush("#151515"))
        self.robot.setPen(pg.mkPen("#ffffff", width=2))
        self.robot.setZValue(20)
        plot.addItem(self.robot)

    def update(self, step, state, info, trace, status):
        xs, costs = info.get("xs"), info.get("costs")
        if xs is not None and costs is not None:
            chosen = np.argsort(costs)[:self.nshow]
            self.samples.setData(*_segments([xs[i] for i in chosen]))
        plan = info.get("plan")
        if plan is not None:
            self.plan.setData(plan[:, 0], plan[:, 1])
        driven = np.asarray(trace)
        self.trace.setData(driven[:, 0], driven[:, 1])
        flow = getattr(self.ctrl, "flow", None)
        if flow is not None and hasattr(flow, "membrane"):
            ii, jj = np.nonzero(flow.membrane)
            stride = max(1, int(np.ceil(len(ii) / 2500)))
            self.membrane.setData(flow.x0 + ii[::stride] * flow.res,
                                  flow.y0 + jj[::stride] * flow.res)
        else:
            self.membrane.clear()
        branches = getattr(self.ctrl, "branches", [])
        proposals = info.get("branch_proposals", [])
        for i, item in enumerate(self.branches):
            if i < len(branches):
                line = branches[i].centerline
                item.setData(line[:, 0], line[:, 1])
            else:
                item.clear()
        for i, item in enumerate(self.proposals):
            if i < len(proposals):
                line = proposals[i].rollout
                item.setData(line[:, 0], line[:, 1])
            else:
                item.clear()
        self.robot.setPos(float(state[0]), float(state[1]))
        self.robot.setRotation(float(np.degrees(state[2])))
        if status == "success":
            self.robot.setBrush(self.pg.mkBrush("#16853b"))
        elif status == "collision":
            self.robot.setBrush(self.pg.mkBrush("#c62828"))
        selected = info.get("selected_mode")
        mode = "" if selected is None else f" | mode={selected['name']}"
        assist = info.get("assist_state")
        assist = "" if assist is None else f" | {assist}"
        self.plot.setTitle(
            f"{self.label}<br>step={step} | {status}{assist}{mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", choices=("path_mppi", "path_biased"),
                        default="path_biased")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--no-hold", action="store_true")
    args = parser.parse_args()
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets
    except ImportError as error:
        raise RuntimeError("Install pyqtgraph and a Qt binding first") from error

    envs = [BarnEnv(args.world, robot_r=args.radius) for _ in range(2)]
    paths = [astar_path(env, env.start[:2], env.goal) for env in envs]
    names = (args.baseline, "amoeba")
    ctrls = [_controller(name, env, path, args.seed, args.samples,
                         args.horizon, "slip", False, 2.0, 0.03)
             for name, env, path in zip(names, envs, paths)]
    labels = (("Path-cost MPPI" if args.baseline == "path_mppi" else
               "Single-path Biased MPPI"), "Geodesic Amoeba MPPI")
    pg.setConfigOption("background", "#fcfcfb")
    pg.setConfigOption("foreground", "#343330")
    app = pg.mkQApp("Path MPPI versus Amoeba MPPI")
    window = pg.GraphicsLayoutWidget(
        title=f"World {args.world}: matched MPPI comparison")
    window.resize(1500, 850)
    panels = [Panel(pg, QtWidgets, window.addPlot(row=0, col=i), env, path,
                    ctrl, label, args.sample_count)
              for i, (env, path, ctrl, label) in enumerate(
                  zip(envs, paths, ctrls, labels))]
    window.show()
    app.processEvents()

    states = [env.start.copy() for env in envs]
    traces = [[state.copy()] for state in states]
    statuses, infos = ["running", "running"], [{}, {}]
    min_clear, finish_step = [np.inf, np.inf], [None, None]
    for step in range(args.max_steps):
        for i, (env, ctrl) in enumerate(zip(envs, ctrls)):
            if statuses[i] != "running":
                continue
            control, infos[i] = ctrl.step(states[i])
            states[i] = ctrl.robot_model.integrate(
                states[i], control, ctrl.dt)
            traces[i].append(states[i].copy())
            clearance = float(ctrl.robot_model.clearance(env, states[i]))
            min_clear[i] = min(min_clear[i], clearance)
            if clearance < 0.0:
                statuses[i], finish_step[i] = "collision", step
            elif np.linalg.norm(states[i][:2] - env.goal) < env.goal_tol:
                statuses[i], finish_step[i] = "success", step
        if step % args.every == 0 or all(s != "running" for s in statuses):
            for panel, state, info, trace, status in zip(
                    panels, states, infos, traces, statuses):
                panel.update(step, state, info, trace, status)
            app.processEvents()
        if all(status != "running" for status in statuses):
            break
    for label, status, done, clearance, ctrl in zip(
            labels, statuses, finish_step, min_clear, ctrls):
        steps = args.max_steps if done is None else done + 1
        print(f"{label}: result={status} time_s={steps * ctrl.dt:.2f} "
              f"minimum_clearance_m={clearance:.3f}")
    if not args.no_hold:
        while window.isVisible():
            app.processEvents()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
