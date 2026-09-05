#!/usr/bin/env python3
"""Feed `local` a global path — including a deliberately wrong one — and
check it still threads the true gap. Contrast: a nav2-style path-TRACKING
MPPI that treats the same path as a reference to follow.

Three fake global plans per world: straight lanes at left / center / right
of the corridor. In tight worlds most lanes are BLOCKED (they cross
pillars) — exactly a stale/wrong global plan. The claim under test:
  * track  (nav2 paradigm: path is a contract)  -> fails on blocked lanes
  * local  (amoeba: path is a boundary hint)    -> threads the true gap

  python3 path_test.py --worlds 4,48,49 --png path_test_w48.png
"""
import argparse

import numpy as np

from env import BarnEnv
from controllers import MPPI, AmoebaLocal
from run import episode, draw_world


def densify(wps, step=0.05):
    wps = np.asarray(wps, float)
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def lane_path(env, lane_x):
    s, g = env.start[:2], env.goal
    return densify([s, [lane_x, s[1] + 1.0], [lane_x, g[1] - 1.0], g])


class PathTrackMPPI(MPPI):
    """Stand-in for the nav2 paradigm: the goal cost tracks the given
    global path (distance-to-path + remaining-length-to-its-end), like
    PathAlign/PathFollow enslaving MPPI to the planner's one polyline."""
    name = "track"

    def __init__(self, env, seed=0, path=None, align_w=4.0, **kw):
        super().__init__(env, seed, **kw)
        self.path = np.asarray(path, float)[::4]     # coarse for (K,T,M)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.rem = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.align_w = align_w

    def goal_cost(self, xs):
        d = np.linalg.norm(xs[..., None, :2] - self.path, axis=-1)  # K,T,M
        lateral = d.min(-1)
        along = lateral + self.rem[d.argmin(-1)]
        return (self.goal_w * along[:, -1] + self.run_w * along.mean(1)
                + self.align_w * lateral.mean(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", default="4,48,49")
    p.add_argument("--radius", type=float, default=0.34)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--png", help="save lane x controller trace grid "
                                 "for the FIRST world to this file")
    args = p.parse_args()
    worlds = [int(w) for w in args.worlds.split(",")]

    lanes = [("left", -3.5), ("center", -2.25), ("right", -1.0)]
    rows, panels = [], {}
    for wid in worlds:
        env = BarnEnv(wid, 1.0, args.radius)
        for lane, lx in lanes:
            path = lane_path(env, lx)
            feas = float((env.clearance(path) - env.robot_r).min())
            tag = "feasible" if feas > 0 else "BLOCKED "
            for cls, kw in ((AmoebaLocal, dict(path=path)),
                            (PathTrackMPPI, dict(path=path))):
                ctrl = cls(env, seed=args.seed, **kw)
                m = episode(env, ctrl, args.max_steps)
                rows.append({"world": wid, "lane": lane, "feas": feas > 0,
                             "ctrl": ctrl.name, **m})
                if wid == worlds[0]:
                    panels[(ctrl.name, lane)] = (env, ctrl, path, m)
                print(f"world {wid:3d} | {lane:6s} lane ({tag}) | "
                      f"{ctrl.name:5s} | {m['result']:9s} "
                      f"time={m['time_s']:5.1f}s stall={m['stall_s']:4.1f}s",
                      flush=True)

    print(f"\n{'controller':10s} {'blocked-lane success':>21s} "
          f"{'feasible-lane success':>22s}")
    for name in ("local", "track"):
        for feas in (False, True):
            rs = [r for r in rows if r["ctrl"] == name and r["feas"] == feas]
            ok = sum(r["success"] for r in rs)
            if feas:
                print(f"{ok:3d}/{len(rs)}")
            else:
                print(f"{name:10s} {ok:14d}/{len(rs):<3d} ", end="")

    if args.png and panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(lanes), figsize=(4 * len(lanes), 13))
        for r, name in enumerate(("local", "track")):
            for c, (lane, _) in enumerate(lanes):
                env, ctrl, path, m = panels[(name, lane)]
                ax = axes[r, c]
                draw_world(ax, env, ctrl)
                ax.plot(path[:, 0], path[:, 1], "--", color="0.4", lw=1.5,
                        label="given global path")
                t = m["trace"]
                ax.plot(t[:, 0], t[:, 1], color="tab:red", lw=2,
                        label="driven")
                ax.set_title(f"{name} | {lane} lane | {m['result']}",
                             fontsize=10)
                if r == 0 and c == 0:
                    ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"world {worlds[0]}: path as boundary hint (top) "
                     f"vs path as contract (bottom)")
        fig.tight_layout()
        fig.savefig(args.png, dpi=100, bbox_inches="tight")
        print(f"\nsaved {args.png}")


if __name__ == "__main__":
    main()
