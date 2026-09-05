#!/usr/bin/env python3
"""Live PyQtGraph viewer for one V7.1 dynamic-junction episode."""
import argparse
import time

import numpy as np

from astar import astar_path
from junction_env import SCENARIOS, ScriptedJunctionEnv
from experiments.v71_junction_benchmark import CONFIGS, _make_controller


COLORS = ("#d33682", "#6c9f2f", "#8c5ac7")


def _segments(paths):
    if not paths:
        return np.empty(0), np.empty(0)
    pieces = []
    for path in paths:
        pieces.extend((np.asarray(path)[:, :2], np.full((1, 2), np.nan)))
    points = np.vstack(pieces)
    return points[:, 0], points[:, 1]


class JunctionAnimator:
    def __init__(self, env, controller, path, config, every=1,
                 sample_count=24):
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtWidgets
        except ImportError as error:
            raise RuntimeError(
                "PyQtGraph and a Qt binding are required") from error
        self.pg, self.QtCore, self.QtWidgets = pg, QtCore, QtWidgets
        self.env, self.controller = env, controller
        self.every, self.sample_count = every, sample_count
        pg.setConfigOption("background", "#fcfcfb")
        pg.setConfigOption("foreground", "#343330")
        self.app = pg.mkQApp("V7.1 Dynamic Junction")
        self.window = pg.GraphicsLayoutWidget(
            title=f"V7.1 {env.scenario.name}: {config}")
        self.window.resize(850, 850)
        self.plot = self.window.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.setXRange(env.xmin - 0.25, env.xmax + 0.25)
        self.plot.setYRange(-0.25, env.ymax + 0.25)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")

        for xmin, xmax, ymin, ymax in env.rectangles:
            item = QtWidgets.QGraphicsRectItem(
                xmin, ymin, xmax - xmin, ymax - ymin)
            item.setBrush(pg.mkBrush("#52514e"))
            item.setPen(pg.mkPen("#2f2e2c", width=1))
            item.setZValue(-10)
            self.plot.addItem(item)
        boundary = np.array([
            [env.xmin, 0.0], [env.xmax, 0.0], [env.xmax, env.ymax],
            [env.xmin, env.ymax], [env.xmin, 0.0]])
        self.plot.plot(boundary[:, 0], boundary[:, 1],
                       pen=pg.mkPen("#2f2e2c", width=3))
        self.plot.plot(
            path[:, 0], path[:, 1],
            pen=pg.mkPen("#4a3aa7", width=2,
                         style=QtCore.Qt.PenStyle.DashLine))
        self.plot.plot([env.goal[0]], [env.goal[1]], pen=None,
                       symbol="star", symbolSize=22,
                       symbolBrush="#eda100", symbolPen="#0b0b0b")

        self.samples = self.plot.plot(
            pen=pg.mkPen((235, 104, 52, 45), width=1))
        self.plan = self.plot.plot(pen=pg.mkPen("#008300", width=3))
        self.repair = self.plot.plot(
            pen=pg.mkPen("#00a6a6", width=4,
                         style=QtCore.Qt.PenStyle.DashLine))
        self.trace = self.plot.plot(pen=pg.mkPen("#e34948", width=3))
        self.membrane = self.plot.plot(
            pen=None, symbol="o", symbolSize=3,
            symbolBrush=(34, 190, 205, 190), symbolPen=None)
        self.branches = [self.plot.plot(
            pen=pg.mkPen(color, width=2,
                         style=QtCore.Qt.PenStyle.DotLine))
            for color in COLORS]
        self.proposals = [self.plot.plot(pen=pg.mkPen(color, width=3))
                          for color in COLORS]

        diameter = 2.0 * env.robot_r
        self.robot = QtWidgets.QGraphicsEllipseItem(
            -env.robot_r, -env.robot_r, diameter, diameter)
        self.robot.setBrush(pg.mkBrush("#111111"))
        self.robot.setPen(pg.mkPen("#ffffff", width=2))
        self.robot.setZValue(20)
        self.plot.addItem(self.robot)
        self.peer = QtWidgets.QGraphicsEllipseItem(
            -env.peer_r, -env.peer_r, 2.0 * env.peer_r, 2.0 * env.peer_r)
        self.peer.setBrush(pg.mkBrush("#e34948"))
        self.peer.setPen(pg.mkPen("#7f1d1d", width=2))
        self.peer.setZValue(20)
        self.plot.addItem(self.peer)
        self.window.show()
        self.app.processEvents()

    def update(self, step, state, info, trace, peer_clearance):
        if step % self.every:
            return
        trace = np.asarray(trace)
        self.trace.setData(trace[:, 0], trace[:, 1])
        plan = info.get("plan")
        if plan is not None:
            self.plan.setData(plan[:, 0], plan[:, 1])
        repair = info.get("repair_path")
        if repair is not None:
            self.repair.setData(repair[:, 0], repair[:, 1])
        else:
            self.repair.clear()
        trajectories = info.get("xs")
        costs = info.get("costs")
        if trajectories is not None and costs is not None:
            chosen = np.argsort(costs)[:self.sample_count]
            x, y = _segments([trajectories[index] for index in chosen])
            self.samples.setData(x, y)
        else:
            self.samples.clear()

        branches = getattr(self.controller, "branches", [])
        proposals = getattr(self.controller, "branch_proposals", [])
        for index, item in enumerate(self.branches):
            if index < len(branches):
                branch = branches[index].centerline
                item.setData(branch[:, 0], branch[:, 1])
            else:
                item.clear()
        for index, item in enumerate(self.proposals):
            if index < len(proposals):
                proposal = proposals[index].rollout
                item.setData(proposal[:, 0], proposal[:, 1])
            else:
                item.clear()

        flow = getattr(self.controller, "flow", None)
        if flow is not None and hasattr(flow, "membrane"):
            ii, jj = np.nonzero(flow.membrane)
            self.membrane.setData(
                flow.x0 + ii * flow.res, flow.y0 + jj * flow.res)
        else:
            self.membrane.clear()
        self.robot.setPos(float(state[0]), float(state[1]))
        peer = self.env.moving_obs()[0]
        self.peer.setPos(float(peer[0]), float(peer[1]))
        mode = info.get("selected_mode")
        mode_name = mode["name"] if mode is not None else "single"
        assist = info.get("assist_state", "n/a")
        repair_state = " | REPAIRED PATH" if info.get("repair_active") else ""
        self.plot.setTitle(
            f"step {step} | t={self.env.t:.2f}s | {assist} | "
            f"mode={mode_name}{repair_state} | "
            f"peer clearance={peer_clearance:+.3f}m")
        self.app.processEvents()

    def finish(self, result, hold=True):
        self.plot.setTitle(f"finished: {result}")
        self.app.processEvents()
        if hold:
            while self.window.isVisible():
                self.app.processEvents()
                time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIOS),
                        default="blocked_split")
    parser.add_argument("--controller", choices=CONFIGS, default="amoeba")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--no-hold", action="store_true")
    args = parser.parse_args()
    env = ScriptedJunctionEnv(args.scenario)
    env.dynamic_enabled = False
    path = astar_path(env, env.start[:2], env.goal)
    env.dynamic_enabled = True
    controller = _make_controller(
        args.controller, env, path, args.seed, args.samples, args.horizon)
    animator = JunctionAnimator(
        env, controller, path, args.controller, every=args.every,
        sample_count=args.sample_count)
    state = env.start.copy()
    trace = [state.copy()]
    result = "timeout"
    for step in range(args.max_steps):
        env.step_time(controller.dt)
        control, info = controller.step(state)
        state = controller.robot_model.integrate(state, control, controller.dt)
        trace.append(state.copy())
        peer_clearance = env.inter_robot_clearance(state[:2])
        static_clearance = float(
            env.static_clearance(state[None, :2])[0] - env.robot_r)
        animator.update(step, state, info, trace, peer_clearance)
        if peer_clearance < 0.0:
            result = "peer collision"
            break
        if static_clearance < 0.0:
            result = "wall collision"
            break
        if np.linalg.norm(state[:2] - env.goal) < env.goal_tol:
            result = "success"
            break
    print(f"result={result} time={env.t:.2f}s "
          f"peer_clearance={env.inter_robot_clearance(state[:2]):+.3f}m")
    animator.finish(result, hold=not args.no_hold)


if __name__ == "__main__":
    main()
