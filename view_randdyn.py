#!/usr/bin/env python3
"""Watch ONLY the random moving obstacles (randdyn.RandomDynEnv) in matplotlib.

No controller, no robot, no space-time blob: just the scene the controllers are tested on.
Fast enough to run in real time (or faster), so it is the quick way to see what N obstacles,
a motion model and a seed look like before running anything expensive.

  python3 view_randdyn.py --n 20 --seed 3                  # live window
  python3 view_randdyn.py --n 30 --motion bounce --speed 0.3 0.6
  python3 view_randdyn.py --n 20 --seed 3 --predict        # + constant-velocity prediction rays
  python3 view_randdyn.py --n 20 --seed 3 --save scene.gif --seconds 20   # no window, writes a file
"""
import argparse
import os

import numpy as np

from randdyn import KEEPOUT_R, RandomDynEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of moving obstacles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion", choices=("waypoint", "bounce"), default="waypoint")
    ap.add_argument("--speed", type=float, nargs=2, default=(0.15, 0.40), metavar=("MIN", "MAX"),
                    help="obstacle speed range (m/s)")
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to run")
    ap.add_argument("--rate", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--trail", type=float, default=3.0, help="seconds of trail behind each obstacle")
    ap.add_argument("--predict", action="store_true",
                    help="draw the constant-velocity prediction (3 s) a tracker-based controller uses")
    ap.add_argument("--save", help="write a .gif or .mp4 instead of opening a window")
    a = ap.parse_args()
    if a.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from matplotlib.animation import FuncAnimation

    env = RandomDynEnv(n_moving=a.n, seed=a.seed, motion=a.motion, speed_range=tuple(a.speed))
    dt = 0.05
    fps = 20
    steps_per_frame = max(1, int(round(a.rate / (fps * dt))))
    frames = int(a.seconds / (steps_per_frame * dt))
    trail_len = max(2, int(a.trail / (steps_per_frame * dt)))

    fig, ax = plt.subplots(figsize=(5.2, 8.4))
    ax.set_xlim(env.xmin - 0.3, env.xmax + 0.3)
    ax.set_ylim(-0.3, env.ymax + 0.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.plot([env.xmin, env.xmax, env.xmax, env.xmin, env.xmin],
            [env.ymin, env.ymin, env.ymax, env.ymax, env.ymin], color="0.35", lw=3)
    for c, label in ((env.start[:2], "start"), (env.goal, "goal")):
        ax.add_patch(mp.Circle(c, KEEPOUT_R, fc="#f2c94c", alpha=0.18, ec="#f2c94c", ls="--"))
        ax.plot(*c, marker="*", ms=14, color="#f2c94c", mec="k")
        ax.annotate(label, c, xytext=(6, 6), textcoords="offset points", fontsize=8)
    r = env.obstacle_r
    discs = [mp.Circle(p, r, fc="#4a4a4a", ec="#d64545", lw=1.5, zorder=5) for p in env.moving_obs()]
    for d in discs:
        ax.add_patch(d)
    trails = [ax.plot([], [], color="#d64545", alpha=0.35, lw=1)[0] for _ in discs]
    arrows = ax.quiver(env.moving_obs()[:, 0], env.moving_obs()[:, 1],
                       env.moving_velocity()[:, 0], env.moving_velocity()[:, 1],
                       angles="xy", scale_units="xy", scale=1.0 / 1.2, color="#d64545",
                       width=0.004, zorder=6)
    pred = ax.plot([], [], color="#d64545", ls="--", lw=1, alpha=0.6)[0] if a.predict else None
    text = ax.text(0.02, 0.985, "", transform=ax.transAxes, va="top", fontsize=9)
    history = [[p.copy()] for p in env.moving_obs()]

    def update(_):
        for _s in range(steps_per_frame):
            env.step_time(dt)
        pos, vel = env.moving_obs(), env.moving_velocity()
        for i, d in enumerate(discs):
            d.center = pos[i]
            history[i].append(pos[i].copy())
            h = np.array(history[i][-trail_len:])
            trails[i].set_data(h[:, 0], h[:, 1])
        arrows.set_offsets(pos)
        arrows.set_UVC(vel[:, 0], vel[:, 1])
        if pred is not None:
            seg = []
            for p0, v0 in zip(pos, vel):
                seg += [(p0[0], p0[1]), (p0[0] + 3 * v0[0], p0[1] + 3 * v0[1]), (np.nan, np.nan)]
            seg = np.array(seg)
            pred.set_data(seg[:, 0], seg[:, 1])
        sp = np.linalg.norm(vel, axis=1)
        text.set_text(f"t = {env.t:5.1f} s   N = {a.n}   {a.motion}   seed {a.seed}\n"
                      f"speed {sp.min():.2f}-{sp.max():.2f} m/s (arrow = 1.2 s of motion)")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    if a.save:
        anim.save(a.save, fps=fps, dpi=70)
        print("wrote", os.path.abspath(a.save))
    else:
        plt.show()


if __name__ == "__main__":
    main()
