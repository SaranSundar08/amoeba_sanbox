#!/usr/bin/env python3
"""Generate the thesis TG-MPPI architecture diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("tgmppi_pipeline.pdf")


def box(ax, xy, width, height, color, title, lines):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.7, edgecolor=color, facecolor=color + "18")
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - 0.07, title, ha="center", va="center",
            fontsize=9.5, weight="bold", color=color)
    line_y = y + height - 0.17
    for line in lines:
        ax.text(x + 0.028, line_y, line, ha="left", va="top", fontsize=7.25)
        line_y -= 0.079
    return patch


def arrow(ax, start, end, label="", color="#30343b", rad=0.0):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                          linewidth=1.45, color=color,
                          connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arr)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, label, ha="center", va="bottom", fontsize=7.7,
                color=color, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


fig, ax = plt.subplots(figsize=(10.6, 4.45))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.02, 0.83, "Inputs", fontsize=10, weight="bold", color="#343a40")
for i, text in enumerate(("robot state $x_k$", "local costmap $C_k$",
                          "global path $\\mathcal{P}$", "tracked obstacles $\\hat{O}_{0:T}$")):
    ax.text(0.02, 0.72 - 0.105 * i, text, fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f2f3f5", edgecolor="#b7bcc5"))

box(ax, (0.185, 0.16), 0.225, 0.71, "#2474b5", "1  Topology proposals", [
    "Dijkstra geodesic field $d_r$",
    "body $\\mathcal{B}_\\rho$ + membrane",
    "pseudopod backtraces $P_m$",
    "space--time wait / detour",
    "winding distinctness filter",
    "constraint projection + footprint gate",
    "outputs: $\\mu_0,\\ldots,\\mu_M$",
])

box(ax, (0.455, 0.16), 0.225, 0.71, "#318a52", "2  Grouped MPPI", [
    "allocate $N_m$ samples per mode",
    "$U^{(i)} \\sim \\mathcal{N}(\\mu_m,\\Sigma_m)$",
    "parallel dynamics rollout",
    "Nav2 static/path critics",
    "predicted-obstacle critic",
    "conditional weights $w_{i\\mid m}$",
    "free energy $F_m$",
])

box(ax, (0.725, 0.16), 0.225, 0.71, "#bd3f3f", "3  Decision \& execution", [
    "$m^*=\\arg\\min_m F_m$",
    "dwell + switch margin",
    "multi-cycle confirmation",
    "selected-group weighted update",
    "apply first command $u_k^*$",
    "shift warm start",
    "repeat at next control cycle",
])

arrow(ax, (0.135, 0.52), (0.182, 0.52), "$x_k,C_k,\\mathcal{P},\\hat O$")
arrow(ax, (0.412, 0.52), (0.452, 0.52), "$\\{\\mu_m\\}$")
arrow(ax, (0.682, 0.52), (0.722, 0.52), "$\\{F_m,w_{i\\mid m}\\}$")
arrow(ax, (0.952, 0.52), (0.995, 0.52), "$u_k^*$", color="#bd3f3f")
arrow(ax, (0.89, 0.145), (0.30, 0.095), "receding-horizon feedback", color="#6a6f78", rad=0.10)

ax.text(0.50, 0.955, "TG-MPPI control cycle", ha="center", va="center",
        fontsize=14, weight="bold")
ax.text(0.50, 0.915,
        "environment-derived ancillary controllers + mode-conditioned path-integral control",
        ha="center", va="center", fontsize=9, color="#555b63")

fig.savefig(OUT)
plt.close(fig)
print(OUT)
