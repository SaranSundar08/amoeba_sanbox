#!/usr/bin/env python3
"""Thesis figure: benchmark start pose, goal poses and obstacle tracks per world.

Reads the real occupancy map (map_files_fine/yaml_open_dynamic.yaml) and the
generated dyn worlds (make_open_world.py --motion polynomial), and draws the
route that benchmark_dynamic.py drives: start (0, 1) facing +y -> A -> B -> A -> B.
Straight segments are the start-to-goal reference, not the planned paths.

    python3 make_benchmark_route_figure.py   ->  benchmark_route_dyn.{pdf,png}
"""
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Circle, Polygon       # noqa: E402
import numpy as np                                    # noqa: E402
from PIL import Image                                 # noqa: E402
import yaml                                           # noqa: E402

WS = os.path.expanduser("~/robohouse_ws")
MAP_YAML = f"{WS}/src/BARN_dataset/scaled_1/map_files_fine/yaml_open_dynamic.yaml"
WORLD_DIR = f"{WS}/src/BARN_dataset/scaled_1/world_files"
PARAMS = f"{WS}/src/susag_nav2/param/navigation_tgmppi_tight.yaml"
WORLDS = ["dyn1", "dyn2", "dyn3", "dyn4", "dyn5"]
OUT = os.path.dirname(os.path.abspath(__file__))

# same values benchmark_dynamic.py / make_open_world.py use
START = (0.0, 1.0, math.pi / 2)
A, B = (2.5, 11.3), (-2.8, 0.5)
KEEPOUT_R = 1.2

# reference palette (dataviz skill, light mode) -- validated: slots 1-2 pass all-pairs
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ROUTE, GOAL = "#2a78d6", "#eb6834"


def heading(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def load_map():
    meta = yaml.safe_load(open(MAP_YAML))
    img = np.asarray(Image.open(os.path.join(os.path.dirname(MAP_YAML), meta["image"])))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    h, w = img.shape
    return img, (ox, ox + w * res, oy, oy + h * res)


def load_world(name):
    txt = open(os.path.join(WORLD_DIR, f"world_{name}.world")).read()
    obs = []
    for m in re.finditer(r"<model name='moving_obstacle_(\d+)'>(.*?)</model>", txt, re.S):
        body = m.group(2)
        wp = re.search(r"<waypoints>([^<]+)</waypoints>", body)
        if not wp:
            continue
        v = [float(x) for x in wp.group(1).split()]
        obs.append(dict(idx=int(m.group(1)), xy=list(zip(v[0::2], v[1::2])),
                        speed=float(re.search(r"<speed>([0-9.]+)</speed>", body).group(1)),
                        radius=float(re.search(r"<radius>([0-9.]+)</radius>", body).group(1))))
    return obs


def footprint():
    fp = yaml.safe_load(open(PARAMS))["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"]
    return yaml.safe_load(fp) if isinstance(fp, str) else fp


def arrow(ax, x, y, yaw, color, length=0.9, lw=1.4):
    ax.annotate("", xy=(x + length * math.cos(yaw), y + length * math.sin(yaw)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=9), zorder=6)


def draw_base(ax, mp, extent, foot, obstacles=None):
    ax.set_facecolor(SURFACE)
    walls = np.ma.masked_where(mp >= 128, np.ones_like(mp, dtype=float))
    ax.imshow(walls, extent=extent, origin="upper", cmap=matplotlib.colors.ListedColormap([INK2]),
              interpolation="nearest", zorder=1)
    for gx in range(-4, 5, 2):
        ax.axvline(gx, color=GRID, lw=0.5, zorder=0)
    for gy in range(0, 13, 2):
        ax.axhline(gy, color=GRID, lw=0.5, zorder=0)
    for c in (START[:2], A, B):
        ax.add_patch(Circle(c, KEEPOUT_R, fill=False, ls=(0, (3, 2)), lw=0.8, ec=MUTED, zorder=2))
    if obstacles:
        for o in obstacles:
            xs, ys = zip(*o["xy"])
            ax.plot(xs, ys, color=MUTED, lw=1.0, solid_capstyle="round", zorder=3)
            ax.add_patch(Circle(o["xy"][0], o["radius"], fc=GRID, ec=INK2, lw=0.7, zorder=4))
    # route: leg 1 start -> A, legs 2-4 shuttle A <-> B (straight-line reference)
    ax.annotate("", xy=A, xytext=START[:2], arrowprops=dict(arrowstyle="-|>", color=ROUTE, lw=1.6,
                shrinkA=4, shrinkB=7, mutation_scale=10), zorder=5)
    ax.annotate("", xy=B, xytext=A, arrowprops=dict(arrowstyle="<|-|>", color=ROUTE, lw=1.6,
                shrinkA=7, shrinkB=7, mutation_scale=10), zorder=5)
    # start pose: footprint at true size + heading
    c, s = math.cos(START[2]), math.sin(START[2])
    pts = [(START[0] + c * p[0] - s * p[1], START[1] + s * p[0] + c * p[1]) for p in foot]
    ax.add_patch(Polygon(pts, closed=True, fc="none", ec=ROUTE, lw=1.4, zorder=6))
    arrow(ax, START[0], START[1], START[2], ROUTE)
    # goals + arrival headings
    for g, yaw in ((A, heading(START[:2], A)), (B, heading(A, B))):
        ax.plot(*g, marker="D", ms=6.5, color=GOAL, mec=SURFACE, mew=1.0, zorder=7)
        arrow(ax, g[0], g[1], yaw, GOAL, length=0.8, lw=1.3)
    ax.set_xlim(-4.35, 4.35)
    ax.set_ylim(-0.85, 12.85)
    ax.set_aspect("equal")
    ax.set_xticks(range(-4, 5, 2))
    ax.set_yticks(range(0, 13, 2))
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    mp, extent = load_map()
    foot = footprint()
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 7.6), facecolor=SURFACE)
    axes = axes.ravel()

    ax = axes[0]
    draw_base(ax, mp, extent, foot)
    ax.set_title("(a) Start pose and goals", fontsize=9, loc="left", color=INK)
    # labels sit outside the keep-out circles and clear of the route lines
    ax.text(1.4, START[1], "Start\n(0, 1)\nfacing +y", fontsize=7, color=INK, va="center", ha="left")
    ax.text(1.15, A[1], "A\n(2.5, 11.3)", fontsize=7, color=INK, va="center", ha="right")
    ax.text(-3.85, 3.0, "B\n(−2.8, 0.5)", fontsize=7, color=INK, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"), zorder=8)
    mid1 = ((START[0] + A[0]) / 2, (START[1] + A[1]) / 2)
    mid2 = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
    for (mx, my), txt, dx in ((mid1, "leg 1", 0.35), (mid2, "legs 2–4", -0.35)):
        ax.text(mx + dx, my, txt, fontsize=7, color=INK2, ha="left" if dx > 0 else "right", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    for k, w in enumerate(WORLDS, start=1):
        obs = load_world(w)
        ax = axes[k]
        draw_base(ax, mp, extent, foot, obs)
        sp = [o["speed"] for o in obs]
        ax.set_title(f"({chr(ord('a') + k)}) {w}: {len(obs)} obstacles\n"
                     f"peak speed {min(sp):.2f}–{max(sp):.2f} m/s", fontsize=8, loc="left", color=INK)

    for i, ax in enumerate(axes):
        if i % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y (m)", fontsize=7)
        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x (m)", fontsize=7)

    handles = [
        Line2D([], [], color=ROUTE, lw=1.6, marker=">", ms=5, label="Route (straight-line reference)"),
        Line2D([], [], color=ROUTE, lw=0, marker="s", ms=7, mfc="none", mew=1.4,
               label="Robot footprint at start"),
        Line2D([], [], color=GOAL, lw=0, marker="D", ms=6, label="Goal pose (arrow: arrival heading)"),
        Line2D([], [], color=MUTED, lw=1.0, label="Obstacle track (back and forth)"),
        Line2D([], [], color=INK2, lw=0, marker="o", ms=6, mfc=GRID, mew=0.7, label="Obstacle at t = 0 (true size)"),
        Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label=f"Keep-out, {KEEPOUT_R:g} m"),
        Line2D([], [], color=INK2, lw=4, label="Walls (occupancy map)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=2.2, w_pad=0.6)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(os.path.join(OUT, f"benchmark_route_dyn.{ext}"), facecolor=SURFACE, **kw)
    print("wrote", os.path.join(OUT, "benchmark_route_dyn.{pdf,png}"))
    print("start %s, A %s (arrival %.1f deg on leg 1, %.1f deg on leg 3), B %s (arrival %.1f deg)"
          % (START[:2], A, math.degrees(heading(START[:2], A)), math.degrees(heading(B, A)), B,
             math.degrees(heading(A, B))))


if __name__ == "__main__":
    main()
