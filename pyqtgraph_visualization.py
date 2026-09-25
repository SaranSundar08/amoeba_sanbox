"""Fast optional PyQtGraph live renderer.

This module is deliberately presentation-only. It consumes episode callbacks
and never mutates the environment, controller, controls, or trace.
"""
import numpy as np

from env import CYL_R
from visualization import (
    AQUA, BRANCH_COLORS, GREEN, GRIDLINE, INK, OBSTACLE, ORANGE, RED,
    SPACETIME_DETOUR, SPACETIME_WAIT, STREAM, SURFACE, VIOLET, YELLOW)


# Space-time blob overlay (spacetime_blob.py): one colour per route slot, teal for the body rim.
BLOB_BODY = "#00A6A6"
BLOB_ROUTE_COLORS = ("#E0308C", "#7B61FF", "#F28E2B")


def _hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _segments_xy(segments):
    """Join independent line segments with NaN separators."""
    arrays = [np.asarray(segment, dtype=float) for segment in segments
              if len(segment)]
    if not arrays:
        return np.empty(0), np.empty(0)
    joined = []
    for segment in arrays:
        joined.extend((segment[:, :2], np.full((1, 2), np.nan)))
    xy = np.vstack(joined)
    return xy[:, 0], xy[:, 1]


class PyQtGraphAnimator:
    """Reusable in-place renderer for the sandbox's live episode data."""

    def __init__(self, env, ctrl, path, world, every=4,
                 mode_debug_every=0, sample_count=128):
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtWidgets
        except ImportError as error:
            raise RuntimeError(
                "PyQtGraph live rendering requires `pyqtgraph` and a Qt "
                "binding. Install it or use `--renderer matplotlib`.") from error

        self.pg, self.QtCore, self.QtWidgets = pg, QtCore, QtWidgets
        self.env, self.ctrl, self.path, self.world = env, ctrl, path, world
        self.every = max(1, int(every))
        self.mode_debug_every = max(0, int(mode_debug_every))
        self.sample_indices = np.linspace(
            0, ctrl.K - 1, min(sample_count, ctrl.K)).astype(int)

        self.app = pg.mkQApp("TG-MPPI")
        self.window = pg.GraphicsLayoutWidget(
            title=f"{ctrl.name} | world {world}")
        self.window.resize(620, 980)
        self.plot = self.window.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.setXRange(env.xmin - 0.3, env.xmax + 0.3, padding=0)
        self.plot.setYRange(-0.3, env.ymax, padding=0)
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.getViewBox().setBackgroundColor(SURFACE)

        self.field_image = pg.ImageItem(axisOrder="row-major")
        self.field_image.setZValue(-10)
        self.plot.addItem(self.field_image)
        self._draw_static_world()

        self.global_path = self.plot.plot(
            pen=pg.mkPen(VIOLET, width=2, style=QtCore.Qt.DashLine))
        if path is not None:
            self.global_path.setData(path[:, 0], path[:, 1])
        # One batched curve per visual role keeps updates fast while making
        # topology modes immediately distinguishable. Index 3 is the
        # protected path/fallback (and the single-distribution fallback);
        # 4 and 5 are the space-time "wait"/"detour" alternatives
        # (spacetime.py) -- dedicated roles rather than wrapping their
        # branch_id (+100/+200) through `% len(BRANCH_COLORS)`, which would
        # otherwise collide with an unrelated ordinary branch's fan colour.
        self.fan_curves = [
            self.plot.plot(pen=pg.mkPen(
                (*_hex_rgb(color), 85), width=1))
            for color in (*BRANCH_COLORS, ORANGE, SPACETIME_WAIT,
                         SPACETIME_DETOUR, BLOB_ROUTE_COLORS[0])
        ]
        self.branch_lines = [
            self.plot.plot(pen=pg.mkPen(
                BRANCH_COLORS[index], width=2,
                style=QtCore.Qt.DotLine))
            for index in range(3)]
        self.proposal_lines = [
            self.plot.plot(pen=pg.mkPen(BRANCH_COLORS[index], width=3))
            for index in range(3)]
        # Space-time alternatives (spacetime.py; branch_id offset by
        # +100 "wait" / +200 "detour") get dedicated curves rather than
        # competing with ordinary branches for the 3 proposal_lines slots
        # -- sharing those would either silently drop a real branch or
        # colour a wait/detour route the same as an unrelated branch.
        # Only the single most prominent wait/detour is shown; several
        # simultaneous ones (multiple crossing branches at once) collapse
        # onto one curve each, a known display limitation.
        self.spacetime_wait_line = self.plot.plot(pen=pg.mkPen(
            SPACETIME_WAIT, width=3, style=QtCore.Qt.DashDotLine))
        self.spacetime_detour_line = self.plot.plot(pen=pg.mkPen(
            SPACETIME_DETOUR, width=3, style=QtCore.Qt.DashDotLine))
        self.plan = self.plot.plot(pen=pg.mkPen(GREEN, width=3))
        self.trace = self.plot.plot(pen=pg.mkPen(RED, width=3))
        self.heading = self.plot.plot(pen=pg.mkPen(INK, width=3))
        if ctrl.robot_model.is_slip:
            length = ctrl.robot_model.footprint_length
            width = ctrl.robot_model.footprint_width
            self.robot = QtWidgets.QGraphicsRectItem(
                -0.5 * length, -0.5 * width, length, width)
        else:
            self.robot = QtWidgets.QGraphicsEllipseItem(
                -env.robot_r, -env.robot_r, 2 * env.robot_r, 2 * env.robot_r)
        self.robot.setPen(pg.mkPen(INK, width=2))
        self.robot.setBrush(pg.mkBrush(None))
        self.robot.setZValue(20)
        self.plot.addItem(self.robot)
        self.moving_items = []
        if hasattr(env, "moving_obs"):
            obstacle_r = getattr(env, "obstacle_r", CYL_R)
            for _ in env.moving_obs():
                item = QtWidgets.QGraphicsEllipseItem(
                    -obstacle_r, -obstacle_r, 2 * obstacle_r, 2 * obstacle_r)
                item.setPen(pg.mkPen(RED, width=2))
                item.setBrush(pg.mkBrush(OBSTACLE))
                item.setZValue(12)
                self.plot.addItem(item)
                self.moving_items.append(item)
        self._make_blob_items()
        self.window.show()
        self.app.processEvents()

    def _make_blob_items(self):
        """Space-time blob overlay; created only when the controller has a BlobPlanner."""
        pg = self.pg
        self.blob = getattr(self.ctrl, "blob", None)
        if self.blob is None:
            return
        # the body's membrane: reachable cells on the last time layer, brighter = lower promise
        self.blob_rim = pg.ScatterPlotItem(size=5, pen=None)
        self.blob_rim.setZValue(-5)
        self.plot.addItem(self.blob_rim)
        # constant-velocity prediction the flood uses: current position -> position at horizon
        self.blob_pred = self.plot.plot(pen=pg.mkPen((*_hex_rgb(RED), 120), width=1,
                                                     style=self.QtCore.Qt.DashLine))
        self.blob_pred.setZValue(11)
        # up to 3 routes, one colour each; dots mark the time layers (spacing = speed)
        self.blob_routes = [self.plot.plot(pen=pg.mkPen(c, width=2)) for c in BLOB_ROUTE_COLORS]
        self.blob_dots = [pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(c))
                          for c in BLOB_ROUTE_COLORS]
        for line, dots in zip(self.blob_routes, self.blob_dots):
            line.setZValue(14)
            dots.setZValue(15)
            self.plot.addItem(dots)
        self.blob_label = pg.TextItem(anchor=(0, 0), color=INK)
        self.blob_label.setZValue(30)
        self.plot.addItem(self.blob_label)

    def _update_blob(self, state, info):
        if getattr(self, "blob", None) is None:
            return
        snap = self.blob.last
        for line, dots in zip(self.blob_routes, self.blob_dots):
            line.setData([], [])
            dots.setData([], [])
        if snap is None:
            self.blob_rim.setData([], [])
            self.blob_pred.setData([], [])
            self.blob_label.setText("blob: no obstacles")
            self.blob_label.setPos(self.env.xmin + 0.1, self.env.ymax - 0.1)
            return
        body = snap["body"]
        # body rim
        gk = body.G[body.K]
        ii, jj = np.nonzero(gk < 1.0e8)
        if len(ii):
            xy = np.stack([body.X[ii, jj], body.Y[ii, jj]], 1)
            term = self.ctrl.flow.dist(xy)
            score = gk[ii, jj] + term
            span = max(float(np.ptp(score)), 1e-6)
            rel = 1.0 - (score - score.min()) / span            # 1 = best exit
            base = np.asarray(_hex_rgb(BLOB_BODY))
            brushes = [pg_brush for pg_brush in (
                self.pg.mkBrush(int(base[0]), int(base[1]), int(base[2]), int(25 + 130 * r))
                for r in rel)]
            self.blob_rim.setData(pos=xy, brush=brushes)
        else:
            self.blob_rim.setData([], [])
        # predicted obstacle motion over the horizon
        if len(body.pos):
            end = body.pos + body.vel * body.horizon
            seg = [np.array([a, b]) for a, b in zip(body.pos, end)]
            px, py = _segments_xy(seg)
            self.blob_pred.setData(px, py)
        else:
            self.blob_pred.setData([], [])
        # routes (colour = slot); the selected mode's route is drawn thick
        selected = info.get("selected_mode")
        selected_key = None if selected is None else selected.get("key")
        rejected = set(snap.get("rejected", []))
        for slot, (bid, path) in enumerate(snap["chosen"][:3]):
            width = 4 if bid == selected_key else 2
            style = self.QtCore.Qt.DotLine if bid in rejected else self.QtCore.Qt.SolidLine
            self.blob_routes[slot].setPen(self.pg.mkPen(
                BLOB_ROUTE_COLORS[slot], width=width, style=style))
            self.blob_routes[slot].setData(path[:, 0], path[:, 1])
            self.blob_dots[slot].setData(pos=path)
        status = "ACTIVE" if snap["active"] else "idle"
        mode = self.blob.mode
        self.blob_label.setText(
            f"space-time blob [{mode}] {status}\n"
            f"obstacles cost {snap['delay']:.2f} m | classes {body.classes_found} | "
            f"build {body.build_ms:.0f} ms")
        self.blob_label.setPos(self.env.xmin + 0.1, self.env.ymax - 0.1)

    def _circle(self, xy, radius, fill, edge=INK, width=1, z=5):
        item = self.QtWidgets.QGraphicsEllipseItem(
            xy[0] - radius, xy[1] - radius, 2 * radius, 2 * radius)
        item.setPen(self.pg.mkPen(edge, width=width))
        item.setBrush(self.pg.mkBrush(fill))
        item.setZValue(z)
        self.plot.addItem(item)

    def _draw_static_world(self):
        pg = self.pg
        for x in (self.env.xmin, self.env.xmax):
            self.plot.plot(
                [x, x], [-0.3, self.env.ymax],
                pen=pg.mkPen(OBSTACLE, width=4))
        for xy in self.env.obs:
            self._circle(xy, CYL_R, OBSTACLE, edge=INK, z=5)
        self._circle(self.env.goal, 0.35, (*_hex_rgb(YELLOW), 45),
                     edge=YELLOW, z=4)
        goal = pg.ScatterPlotItem(
            [self.env.goal[0]], [self.env.goal[1]], symbol="star",
            size=20, pen=pg.mkPen(INK), brush=pg.mkBrush(YELLOW))
        goal.setZValue(15)
        self.plot.addItem(goal)
        if hasattr(self.env, "mov_center"):
            for center, axis, amplitude in zip(
                    self.env.mov_center, self.env.mov_axis, self.env.mov_amp):
                delta = np.array(
                    [amplitude, 0.0] if axis == 0 else [0.0, amplitude])
                ends = np.stack((center - delta, center + delta))
                self.plot.plot(
                    ends[:, 0], ends[:, 1],
                    pen=pg.mkPen((*_hex_rgb(RED), 90), width=1,
                                 style=self.QtCore.Qt.DotLine))

    def _update_field(self):
        flow = getattr(self.ctrl, "flow", None)
        if flow is None:
            self.field_image.clear()
            return
        finite = np.isfinite(flow.D)
        depth = np.where(
            finite, 1.0 - np.clip(flow.D / max(flow.dmax, 1e-6), 0, 1), 0)
        rgba = np.zeros(flow.D.T.shape + (4,), dtype=np.uint8)
        stream = np.asarray(_hex_rgb(STREAM))
        surface = np.asarray(_hex_rgb(SURFACE))
        blend = depth.T[..., None]
        rgba[..., :3] = surface + blend * (stream - surface)
        rgba[..., 3] = np.where(finite.T, 105, 0)
        self.field_image.setImage(rgba, autoLevels=False)
        self.field_image.setRect(self.QtCore.QRectF(
            flow.x0, flow.y0,
            flow.D.shape[0] * flow.res, flow.D.shape[1] * flow.res))

    def update(self, k, state, info, trace):
        if self.mode_debug_every and k % self.mode_debug_every == 0:
            from diagnostics import _print_mode_debug
            _print_mode_debug(k, state, info, self.ctrl)
        if k % self.every:
            return
        self._update_field()
        xs = info["xs"][self.sample_indices, :, :2]
        for curve in self.fan_curves:
            curve.setData([], [])
        labels = info.get("mode_labels")
        if labels is None:
            fan_x, fan_y = _segments_xy(xs)
            self.fan_curves[3].setData(fan_x, fan_y)   # generic/fallback role
        else:
            sampled_labels = np.asarray(labels)[self.sample_indices]
            key_by_index = {
                item["index"]: item["key"]
                for item in info.get("mode_stats", [])}
            sampled_weights = np.asarray(info["weights"])[self.sample_indices]
            role_segments = [[] for _ in self.fan_curves]
            role_strength = np.zeros(len(self.fan_curves))
            global_peak = float(sampled_weights.max(initial=0.0))
            for index in np.unique(sampled_labels):
                key = key_by_index.get(int(index), -1)
                if key >= 1000:
                    role = 6              # space-time blob mode (ids 1000+ / 2000+)
                elif key >= 200:
                    role = 5
                elif key >= 100:
                    role = 4
                elif key >= 0:
                    role = key % len(BRANCH_COLORS)
                else:
                    role = 3
                group = sampled_labels == index
                role_segments[role].extend(xs[group])
                # Selected/high-weight modes are vivid; alternatives remain
                # visible but recede. One pen per mode avoids 128 graphics
                # objects and preserves PyQtGraph's fast batched update.
                peak = float(sampled_weights[group].max(initial=0.0))
                role_strength[role] = max(
                    role_strength[role], peak / (global_peak + 1e-12))
            for role, segments in enumerate(role_segments):
                if not segments:
                    continue
                fan_x, fan_y = _segments_xy(segments)
                strength = role_strength[role]
                alpha = int(45 + 150 * np.clip(strength, 0.0, 1.0))
                role_colors = (*BRANCH_COLORS, ORANGE, SPACETIME_WAIT,
                              SPACETIME_DETOUR, BLOB_ROUTE_COLORS[0])
                color = role_colors[role]
                self.fan_curves[role].setPen(
                    self.pg.mkPen((*_hex_rgb(color), alpha), width=1))
                self.fan_curves[role].setData(fan_x, fan_y)

        for item in (self.branch_lines + self.proposal_lines
                    + [self.spacetime_wait_line, self.spacetime_detour_line]):
            item.setData([], [])
        for row, branch in enumerate(getattr(self.ctrl, "branches", [])[:3]):
            self.branch_lines[row].setData(
                branch.centerline[:, 0], branch.centerline[:, 1])
        proposals = info.get("branch_proposals", [])
        ordinary = [p for p in proposals if p.branch_id < 100]
        for row, proposal in enumerate(ordinary[:3]):
            self.proposal_lines[row].setData(
                proposal.rollout[:, 0], proposal.rollout[:, 1])
        for proposal in proposals:
            if 100 <= proposal.branch_id < 200:
                self.spacetime_wait_line.setData(
                    proposal.rollout[:, 0], proposal.rollout[:, 1])
            elif 200 <= proposal.branch_id < 1000:
                self.spacetime_detour_line.setData(
                    proposal.rollout[:, 0], proposal.rollout[:, 1])

        current_path = getattr(self.ctrl, "path", self.path)
        if current_path is not None:
            self.global_path.setData(current_path[:, 0], current_path[:, 1])
        self.plan.setData(info["plan"][:, 0], info["plan"][:, 1])
        driven = np.asarray(trace)
        self.trace.setData(driven[:, 0], driven[:, 1])
        self.robot.setPos(float(state[0]), float(state[1]))
        if self.ctrl.robot_model.is_slip:
            self.robot.setRotation(float(np.degrees(state[2])))
        engaged = getattr(self.ctrl, "active", None)
        self.robot.setPen(self.pg.mkPen(
            AQUA if engaged is not None and engaged > 0.5 else INK, width=2))
        end = state[:2] + 0.9 * self.env.robot_r * np.array([
            np.cos(state[2]), np.sin(state[2])])
        self.heading.setData([state[0], end[0]], [state[1], end[1]])
        if self.moving_items:
            for item, xy in zip(self.moving_items, self.env.moving_obs()):
                item.setPos(float(xy[0]), float(xy[1]))
        self._update_blob(state, info)
        selected = info.get("selected_mode")
        mode = "" if selected is None else (
            f" | {selected['name']} ({info.get('selection_reason', '')})")
        self.plot.setTitle(
            f"{self.ctrl.name} | world {self.world} | "
            f"t={k * self.ctrl.dt:.1f}s{mode}")
        self.app.processEvents()

    def finish(self):
        """Flush the last frame without blocking automated/headless runs."""
        self.app.processEvents()
