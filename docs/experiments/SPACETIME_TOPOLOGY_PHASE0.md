# Space-time topology: Phase 0 (standalone mechanism) and its first wiring

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

## The integration glue: `spacetime_path_to_proposal`

Phase 0's search returns a bare `(x, y)`-per-time-layer list; the rest of
the pipeline (`SamplingMode`, grouped sampling, MPPI cost) consumes
`proposals.BranchProposal` objects: `[T, dt]`-gridded velocity commands
under real box/rate limits, plus feasibility. `spacetime_path_to_proposal`
is that conversion, so a "wait" or "detour" space-time route can slot in
as an ordinary mode exactly like a static pseudopod branch.

It cannot reuse `proposals.branch_to_control_sequence` for this: that
function only understands geometry (a bare polyline via pure pursuit), not
timing, and its "remove repeated grid points" step would discard a wait
segment (consecutive identical points) as a zero-length segment -- exactly
the information this conversion exists to preserve. Instead it resamples
the path onto the controller's own `(T, dt)` grid by linear interpolation
(holding the final position beyond the path's duration), differentiates
consecutive points into raw velocity commands, and projects them through
the same `grouped_sampling.project_control_sequences` every other proposal
uses.

Two claims mattered enough to verify empirically rather than trust on
paper, both now regression-tested (`tests/test_spacetime.py`,
`SpacetimePathToProposalTests`):

- **The wait survives.** A path holding at the start for 1.0 s resamples to
  *exactly* zero commanded speed for the corresponding 20 controller steps
  (`dt=0.05`) -- not a discarded segment, not spurious drift.
- **Feasibility is judged on what the robot actually does, not what the
  route asked for.** A raw path that swerves away from an obstacle
  immediately looks safe by a comfortable 0.2 m margin. Under a
  deliberately tight turn-rate limit (`w_accel_max=0.3`), the achieved
  rollout can't make that turn in time and keeps heading toward the
  obstacle -- min clearance comes out at -0.247 m, correctly flagged
  infeasible. This is the same failure mode the exact-footprint fix,
  the plant/plan mismatch harness, and the peer-collision fix all guard
  against elsewhere in the sandbox: a plan that looks fine until the real
  dynamics are honored.

Static clearance prefers `RobotModel.exact_clearance` when `env` supports
it (falls back to the sampled `clearance()` otherwise, same convention as
everywhere else this session); dynamic clearance checks the achieved
rollout against `predict_obstacle` at each step's own absolute time,
self-contained rather than routed through `env`'s own prediction machinery
-- full integration with `DynaBarnEnv.predicted_clearance` is next-slice
work, not done here.

## Tests

`tests/test_spacetime.py`, 17 tests: collision-avoidance correctness under
both wait-cost regimes (exhaustive per-layer check, not spot checks); the
wait-vs-detour emergent behavior above; a no-obstacle degenerate control;
infeasibility handling (permanently-occupied goal, occupied start) returns
`(None, inf)` rather than crashing or returning a colliding path; direct
unit tests of `routes_are_distinct` on hand-built trajectories (opposite
sides, same side, too-far-to-matter, and a short/held-position path
correctly padded against a longer one); and the two `spacetime_path_to_
proposal` claims above (wait preserved, feasibility on the achieved
rollout). Full suite: 86/86.

## Explicitly not yet done

Not wired into `LocalFlowField`, `pseudopods.py`, `BranchTracker`, or
`mppi.py`. No multi-obstacle support (`predict_obstacle` is single-obstacle;
extending to several is straightforward -- take the max over per-obstacle
exclusion checks -- but untested). No viscosity weighting (plain Euclidean
step cost, unlike the static flood's viscosity-weighted distance). The
`window` parameter still needs enough lateral room for a detour to be
geometrically possible; a corridor narrower than the required detour width
will correctly report the detour route infeasible rather than clip it.

## The decision layer: wired into `MPPI.build_branch_proposals` (2026-09-06)

`spacetime_modes=False` by default (no prior result changes; verified: with
a genuine predicted crossing present but the flag off, `branch_proposals`
is unchanged from the flag never having existed). When enabled,
`MPPI._spacetime_alternatives(state, branch, base_proposal)` runs after
each branch's ordinary reference is built:

1. Check `env.predicted_moving_obs` (if the env has it) against the
   branch's own nominal-speed continuation along its centreline, out to
   `spacetime_horizon` -- NOT the base proposal's achieved rollout, and
   not capped at the short MPPI horizon `T * dt`. Two real bugs surfaced
   getting this right, both belonging to the same "runs fine, silently
   checks the wrong window" class this week's other fixes guard against:
   - The local goal for the search was first computed using `v_max`, but
     `time_expanded_search`'s own implied speed is `res / dt_layer`
     (coarser). A goal picked as "reachable at `v_max`" was routinely
     *not* reachable by the search's own grid, making both routes
     trivially infeasible for a reason having nothing to do with the
     obstacle.
   - The crossing check first used the base proposal's rollout, which is
     capped at `T * dt` (the MPPI horizon, e.g. 1.5 s in testing) --
     `spacetime_horizon` can be set much longer (6.0 s), but a crossing
     that only matters beyond the short window was invisible no matter
     how far the longer horizon was told to look.
2. If a crossing is found, `two_route_search` runs from the current state
   to a local goal along the branch's centreline (reachable at 90% of the
   search grid's own speed within `spacetime_horizon` -- not the branch's
   full, possibly distant endpoint, which `time_expanded_search` treats as
   a hard arrival requirement rather than partial progress).
3. Each feasible route (`wait`, `detour`) becomes an extra `BranchProposal`
   via `spacetime_path_to_proposal`, with `branch_id` offset by +100/+200
   from the original -- distinct from the -1 fallback and the small
   non-negative ids `pseudopods.py` assigns, and stable across cycles so
   warm-starting and dwell/switch hysteresis apply with no special-casing.

Verified end to end on a synthetic crossing scenario
(`tests/test_spacetime_integration.py`, 5 tests): disabled leaves
`branch_proposals` untouched even with a real crossing present; enabled
with no crossing adds nothing; a genuine crossing yields exactly `[0, 100,
200]` as feasible proposals, picked up by `_sampling_modes` as ordinary
modes (`[-1, 0, 100, 200]`), each carrying a `reference` so the existing
homotopy-consistency cost applies to them with no extra wiring; and a
10-step closed-loop episode with `grouped_sampling=True` runs to
completion with finite commands throughout. One incidental finding along
the way: the two routes' `two_route_search["distinct"]` flag can come out
`False` even when they are procedurally different (one waits, one
doesn't) -- a symmetric scenario let the search nudge both to the same
side of the obstacle. This flag is diagnostic only; `_spacetime_alternatives`
never gates on it, only on each route's own feasibility, so this doesn't
block anything -- just don't read "not distinct" as "one route is
redundant."

Full suite: 96/96.

## Next slice

Still open: the path fallback (not just pseudopod branches) never gets a
space-time alternative; multiple simultaneous moving obstacles collapse to
"the single most-conflicting one" (`nearest_crossing_obstacle`'s documented
limitation); and none of this has been validated against the existing
V6.1/V7.1 dynamic scenarios yet, which is the bulk of the remaining effort
-- re-validating every dynamic result against a controller that can now
propose genuinely new modes, not just re-running them. Do that validation
as its own declared experiment (a small matched panel, `spacetime_modes`
on vs. off, on the existing dynamic/dynamic-blocked scenarios), not folded
into unrelated work.
