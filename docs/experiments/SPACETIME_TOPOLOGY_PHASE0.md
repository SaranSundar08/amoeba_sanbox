# Space-time topology: Phase 0 (standalone mechanism)

Date: 2026-09-05 (or 06, depending on timezone at write time).

## Motivation

Pseudopods are extracted from a static snapshot of the local field
(`env.LocalFlowField`, `pseudopods.extract_pseudopods`). A moving obstacle
never changes which branches get proposed -- it only reshapes MPPI rollout
COSTS afterward (`mppi.MPPI.rollout_clearance`'s prediction layer). Against a
crossing obstacle there is exactly one geometric branch, and "wait" versus
"go now" has to be stumbled into through cost-shaped sampling noise around
that single mean, rather than offered as a genuine second mode the way
left-of-pillar and right-of-pillar are. This is consistent with where the
sandbox's dynamic results are weakest (SLIP dynamic 1/6, dynamic-blocked 0/6
after the 2026-09-04 exact-footprint re-validation) and explains why the
existing dynamic machinery (`dynamic_wait`, the yield/reverse recovery state
machine in `torch_port`) is a set of hand-built special cases: each is
patching around the absence of a general mechanism.

After de Groot et al. (T-MPC, RA-L; see the earlier lit-review analysis),
space-time topology generalizes this: flood free SPACE-TIME instead of free
space. A moving obstacle's predicted path traces a tube through (x, y, t); a
route is a curve through that space, and two routes are distinct if the tube
separates them. This is a multi-week integration effort (redefining what a
pseudopod is, re-validating every existing dynamic result against the new
definition), explicitly scoped as a background investment (roughly an hour
or two a day) alongside report writing and the Nav2 port, not a green-light
blocker. Phase 0 proves the core mechanism stands on its own, against
synthetic scenarios, before touching `LocalFlowField`, `pseudopods.py`, or
`mppi.py` at all.

## What Phase 0 builds (`spacetime.py`)

- `time_expanded_search`: Dijkstra over `(x, y, t)` -- 8-connected spatial
  moves plus a same-cell "wait" step, each advancing exactly one discrete
  time layer, avoiding static obstacles (`env.clearance`) and one predicted
  moving obstacle (`predict_obstacle(t) -> (x, y)`, the same signature as
  `dynabarn.DynaBarnEnv.predicted_moving_obs`). Returns the cheapest
  collision-free path or `(None, inf)` if none exists in the horizon.
- `routes_are_distinct`: the space-time analogue of `homotopy.py`'s
  side-of-obstacle test, evaluated at SYNCHRONIZED time rather than closest
  spatial approach -- a moving obstacle's tube only separates two routes
  that pass its location at genuinely different times, so the comparison
  has to line the routes up in time. True when some shared instant (with
  the obstacle within `relevance` of both routes) puts them on opposite
  sides of it.
- `two_route_search`: constructs "wait it out" and "go around" explicitly,
  the same way `pseudopods.py` constructs left/right explicitly from
  geometry rather than a generic k-shortest-paths search.

## A real design mistake, found and fixed before it shipped

The first version of `two_route_search` tried to force two classes by
bounding which time layer counts as "arrived" (`min_arrival_k`/
`max_arrival_k`: arrive-before-deadline vs. arrive-after). Tracing it by
hand showed this degenerates: if a safe crossing is available well before a
"late" deadline, the cheapest way to satisfy that deadline is to cross
early anyway and idle at the goal -- physically the identical route,
wrongly labelled distinct. Deadline bookkeeping alone doesn't force a
different physical strategy; it only relabels whichever one was already
cheapest.

The fix, verified empirically before committing to it (not just asserted):
vary the relative cost of waiting versus detouring, so the search's own
minimum-cost path genuinely changes. On a corridor crossed once by a moving
obstacle (an open area, `start=(0,0)`, `goal=(0,3)`, obstacle crossing
`x=0` at `y=1.5, t=2.0s` at 0.4 m/s):

| wait cost | result | waits | max lateral offset |
|---|---|---:|---:|
| cheap (0.05) | waits near the centreline, then proceeds | 5 | 0.20 m |
| expensive (1.0) | detours around, never stops | 0 | 0.60 m |

Both are collision-free at every time layer (checked exhaustively, not
sampled) and `routes_are_distinct` correctly flags them as topologically
distinct. A no-obstacle control case gives identical cost regardless of
wait cost and is correctly *not* flagged distinct. This is now the
mechanism `two_route_search` uses; the deadline-based version was replaced,
not kept as an option, since it silently produces false positives.

A second, unrelated bug surfaced during testing: the search grid was
originally sized as `window` around `start_xy` alone. A goal farther than
`window` away fell outside the grid, and `to_cell`'s index clamping quietly
solved for the clamped edge instead of erroring -- a search that runs,
returns a plausible-looking answer, and is silently wrong about what it
solved. Fixed by sizing the grid from both endpoints
(`_grid_origin(start_xy, goal_xy, window, res)`); `window` now means "room
to detour," never "how far the goal is allowed to be." This exact failure
mode (a check that runs and looks fine while quietly checking the wrong
thing) is the same class of bug behind the exact-footprint fix, the plant/
plan mismatch harness, and the peer-collision fix earlier this week --
worth remembering as a standing hazard in this codebase's style of grid-
indexed geometry.

## Tests

`tests/test_spacetime.py`, 13 tests: collision-avoidance correctness under
both wait-cost regimes (exhaustive per-layer check, not spot checks); the
wait-vs-detour emergent behavior above; a no-obstacle degenerate control;
infeasibility handling (permanently-occupied goal, occupied start) returns
`(None, inf)` rather than crashing or returning a colliding path; and direct
unit tests of `routes_are_distinct` on hand-built trajectories (opposite
sides, same side, too-far-to-matter, and a short/held-position path
correctly padded against a longer one). Full suite: 82/82.

## Explicitly not yet done

Not wired into `LocalFlowField`, `pseudopods.py`, `BranchTracker`, or
`mppi.py`. No multi-obstacle support (`predict_obstacle` is single-obstacle;
extending to several is straightforward -- take the max over per-obstacle
exclusion checks -- but untested). No viscosity weighting (plain Euclidean
step cost, unlike the static flood's viscosity-weighted distance). The
`window` parameter still needs enough lateral room for a detour to be
geometrically possible; a corridor narrower than the required detour width
will correctly report the detour route infeasible rather than clip it.

## Next slice

Integrate `time_expanded_search` as an alternative reference-generation path
for a branch when `dynabarn`-style prediction indicates a moving obstacle
crosses that branch's corridor within the planning horizon, using the
existing `BranchProposal`/`SamplingMode` machinery so a "wait" or "detour"
space-time route becomes a real sampling mode, not a bolt-on cost term.
Validate against the existing V6.1/V7.1 dynamic scenarios only after that
integration, not before -- re-validating every dynamic result against a
changed pseudopod definition is the bulk of the remaining effort.
