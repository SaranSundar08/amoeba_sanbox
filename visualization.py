"""Matplotlib rendering for environments, fields, branches, and proposals."""
import numpy as np
from scipy.ndimage import gaussian_filter

from env import CYL_R


# --- palette (dataviz skill's validated categorical + sequential steps) ---
# Fixed roles, not ranked series -- one hex per meaning, reused everywhere.
INK      = "#0b0b0b"    # primary ink: robot body, goal star edge
MUTED    = "#898781"    # axes/ticks
GRIDLINE = "#e1e0d9"    # hairline grid + "outside corridor" wash
SURFACE  = "#fcfcfb"    # chart surface
OBSTACLE = "#52514e"    # cylinders + walls (secondary ink)
AQUA     = "#7fe3ee"    # amoeba body membrane outline (pale cyan)
VIOLET   = "#4a3aa7"    # given global path (the A*/plan contract)
GREEN    = "#008300"    # chosen MPPI plan (mean rollout)
RED      = "#e34948"    # driven trace
YELLOW   = "#eda100"    # goal beacon
ORANGE   = "#eb6834"    # sample fan (single hue, weight-graded alpha)
STREAM   = "#1c5cab"    # water streamlines
BRANCH_COLORS = ("#d33682", "#6c9f2f", "#8c5ac7")

_BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
               "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
               "#184f95", "#104281", "#0d366b"]


def _water_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("water_depth", _BLUE_STEPS)
    cmap.set_bad((0, 0, 0, 0))  # dry/off-body cells: fully transparent
    return cmap


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRIDLINE, linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _draw_field(ax, f, mode="stream", density=1.0):
    """Water-depth wash + amoeba membrane outline + a downhill-direction
    layer. `mode="stream"` gives continuous streamlines (the "cool" static
    look -- costs ~100ms/call, numerical integration over the whole grid,
    fine for a one-off PNG or the initial frame). `mode="quiver"` gives a
    coarse arrow field instead (~1ms/call, and barely grows with count --
    measured ~0.7ms from 120 up to 1700 arrows -- so `density` is free to
    push up as far as it looks good): the only cheap option once the body
    moves every frame, since streamplot can't be incrementally updated --
    every redraw is a from-scratch integration. `density` scales arrow/
    streamline spacing (1.0 = default, 2.0 = roughly double the arrows).
    Returns (image, contour, direction-artist) so a caller can remove and
    replace them as the body moves with the robot."""
    D = f.D
    xs_ = f.x0 + np.arange(D.shape[0]) * f.res
    ys_ = f.y0 + np.arange(D.shape[1]) * f.res
    ext = [xs_[0], xs_[-1], ys_[0], ys_[-1]]
    finite = np.isfinite(D)
    # depth -> [0,1], inverted so "close to goal" (the salient part) is the
    # dark end of the ramp and "far" recedes toward the surface
    wn = np.where(finite, 1.0 - np.clip(D / max(f.dmax, 1e-6), 0, 1), 0.0)

    # Per-pixel alpha built the same way the membrane contour is smoothed
    # below (a Gaussian-blurred mask), instead of a hard NaN cutoff at the
    # exact wet/dry boundary: `interpolation="bilinear"` only smooths
    # between real values, it can't smooth across a cell that's NaN, so a
    # scalar-alpha image with a NaN cutoff shows the raw grid resolution
    # as a blocky edge sitting just outside the already-smooth contour
    # line -- reading as a second, jagged "outline" of its own.
    alpha_mask = np.clip(gaussian_filter(finite.astype(float), sigma=1.2), 0, 1)
    rgba = _water_cmap()(wn.T)
    rgba[..., 3] = 0.6 * alpha_mask.T
    im = ax.imshow(rgba, origin="lower", extent=ext, zorder=1,
                   interpolation="bilinear")

    body = getattr(f, "body", None)
    cont = None
    if body is not None and body.any() and not body.all():
        # Gaussian-blur the mask before contouring: a raw 0/1 grid gives a
        # blocky staircase outline (matplotlib is faithfully tracing grid
        # cell edges). Blurring first and still contouring at 0.5 turns
        # that into a smooth, organic membrane -- purely cosmetic, doesn't
        # touch the underlying body/flood used for cost or steering.
        smooth = gaussian_filter(body.astype(float), sigma=1.2)
        # Soft aqua glow bracketing the crisp membrane, fading outward --
        # the same "glow" treatment already used on the goal marker, so
        # the blob reads as a translucent organic membrane rather than a
        # single hard outline.
        halo_outer = ax.contourf(smooth.T, levels=[0.10, 0.5], colors=[AQUA],
                                 alpha=0.10, extent=ext, zorder=1.8)
        halo_inner = ax.contourf(smooth.T, levels=[0.28, 0.5], colors=[AQUA],
                                 alpha=0.16, extent=ext, zorder=1.9)
        line = ax.contour(smooth.T, levels=[0.5], colors=[AQUA],
                          linewidths=1.4, extent=ext, zorder=2, alpha=0.95)
        cont = [halo_outer, halo_inner, line]

    u, v = f.grad_field()
    if mode == "stream":
        dirs = ax.streamplot(xs_, ys_, u.T, v.T, density=0.85 * density,
                             linewidth=0.7, color=STREAM, arrowsize=0.75,
                             zorder=1.5)
    else:
        st = max(1, int(0.4 / f.res / density))
        ii = np.arange(0, D.shape[0], st)
        jj = np.arange(0, D.shape[1], st)
        II, JJ = np.meshgrid(ii, jj, indexing="ij")
        uu, vv = u[II, JJ], v[II, JJ]
        m = (uu ** 2 + vv ** 2) > 1e-9
        dirs = ax.quiver(xs_[II][m], ys_[JJ][m], uu[m], vv[m], color=STREAM,
                         alpha=0.6, scale=22, width=0.0045, zorder=1.5)
    return im, cont, dirs


def _clear_flowviz(a):
    """Removes whatever `_draw_field` returned as the direction artist --
    a StreamplotSet (lines + arrows, no single .remove()) in "stream" mode
    or a plain Quiver (a normal Artist) in "quiver" mode."""
    if a is None:
        return
    if hasattr(a, "lines") and hasattr(a, "arrows"):
        for coll in (a.lines, a.arrows):
            try:
                coll.remove()
            except (NotImplementedError, ValueError):
                pass
    else:
        try:
            a.remove()
        except (NotImplementedError, ValueError):
            pass


def _clear_contour(c):
    if c is None:
        return
    if isinstance(c, (list, tuple)):
        for item in c:
            _clear_contour(item)
        return
    try:
        c.remove()
    except (AttributeError, NotImplementedError):
        for coll in getattr(c, "collections", []):
            coll.remove()


def draw_world(ax, env, ctrl, density=1.0):
    import matplotlib.patches as mp
    ax.set_aspect("equal")
    ax.set_xlim(env.xmin - 0.3, env.xmax + 0.3)
    ax.set_ylim(-0.3, env.ymax)
    _style_axes(ax)

    ax.axvspan(env.xmin - 0.3, env.xmin, color=GRIDLINE, zorder=0)
    ax.axvspan(env.xmax, env.xmax + 0.3, color=GRIDLINE, zorder=0)
    for x in (env.xmin, env.xmax):
        ax.plot([x, x], [-0.3, env.ymax], color=OBSTACLE, lw=2.5,
                solid_capstyle="round", zorder=2)

    if getattr(ctrl, "flow", None) is not None:
        _draw_field(ax, ctrl.flow, density=density)
    _draw_branches(ax, getattr(ctrl, "branches", []))
    _draw_branch_proposals(ax, getattr(ctrl, "branch_proposals", []))

    for xy in env.obs:
        ax.add_patch(mp.Circle(xy, CYL_R, facecolor=OBSTACLE,
                                edgecolor="#2f2e2c", linewidth=0.6, zorder=3))
    _draw_moving_tracks(ax, env)

    ax.add_patch(mp.Circle(env.goal, 0.35, color=YELLOW, alpha=0.15, zorder=2))
    ax.plot(*env.goal, marker="*", ms=18, mfc=YELLOW, mec=INK, mew=0.6,
            zorder=6)


def _draw_branches(ax, branches):
    """Draw V1 centrelines and stable branch IDs without affecting control."""
    for branch in branches:
        color = BRANCH_COLORS[branch.branch_id % len(BRANCH_COLORS)]
        ax.plot(branch.centerline[:, 0], branch.centerline[:, 1], color=color,
                lw=1.4, ls=":", alpha=0.8, zorder=4)
        ax.plot(*branch.endpoint, marker="X", ms=7, color=color, zorder=5)
        ax.annotate(f"P{branch.branch_id}", branch.endpoint, xytext=(4, 4),
                    textcoords="offset points", color=color, fontsize=8,
                    zorder=6)


def _draw_branch_proposals(ax, proposals):
    """Draw V2 dynamically rolled means; dashed means failed feasibility."""
    for proposal in proposals:
        color = BRANCH_COLORS[proposal.branch_id % len(BRANCH_COLORS)]
        ax.plot(proposal.rollout[:, 0], proposal.rollout[:, 1], color=color,
                lw=2.5, ls="-" if proposal.feasible else "--",
                alpha=0.95 if proposal.feasible else 0.55, zorder=4.5)


def _draw_moving_tracks(ax, env):
    """Static (drawn once) dotted line showing each moving obstacle's
    sweep range -- the obstacles themselves are drawn live, every frame,
    since they actually move; this is just the track they move along."""
    if not hasattr(env, "moving_obs") or len(env.mov_center) == 0:
        return
    for c, axis, amp in zip(env.mov_center, env.mov_axis, env.mov_amp):
        if axis == 0:
            ax.plot([c[0] - amp, c[0] + amp], [c[1], c[1]], ls=":",
                    color=RED, lw=1.0, alpha=0.35, zorder=2)
        else:
            ax.plot([c[0], c[0]], [c[1] - amp, c[1] + amp], ls=":",
                    color=RED, lw=1.0, alpha=0.35, zorder=2)

