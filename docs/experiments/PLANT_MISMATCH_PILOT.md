# Plant/plan mismatch pilot (n=1 per case)

Date: 2026-09-04. `artifacts/results/milestones/v6_ablation_plant_mismatch.csv`,
produced by `python3 -m experiments.v6_ablation --configs slip
--plant-mismatch 0.3`, compared against the self-consistent exact-footprint
re-run (`v6_ablation_exact_footprint.csv`), same worlds/seeds/scenarios.

## What this checks

`simulation.episode(plant_model=...)` and `robot_model.sample_plant_mismatch`
(built earlier the same day) let the executed plant differ from what the
controller assumes. `--plant-mismatch 0.3` draws the plant's `yaw_gain` and
`speed_yaw_loss` independently and uniformly from `[0.7, 1.3] x` the
controller's assumed values, one draw per `(world, seed)`, reused across all
four scenarios for that pair. The controller's own model, and therefore its
plans, are unaffected; only what actually executes changes.

## Result: every one of 24 SLIP cases matched or improved; none got worse

| scenario | self-consistent | single mismatched draw |
|---|---:|---:|
| normal | 4/6 | 6/6 |
| blocked | 5/6 | 6/6 |
| dynamic | 1/6 | 3/6 |
| dynamic_blocked | 0/6 | 3/6 |

Paired case-by-case, 8 of 24 flipped from collision to success; **zero**
flipped the other way. Worst-case clearances that had been −0.001 to
−0.015 m under the self-consistent plant became positive for those flipped
cases.

## Why this is not a robustness claim

The flips do not track the direction of the perturbation: world 49 flips to
success with a draw of `yaw_gain=0.594` (much weaker turning) in one scenario
and `yaw_gain=1.025` (stronger) in another. If mismatch were systematically
helping, the direction should matter; it does not appear to here. The
self-consistent collisions being perturbed were already borderline (clearance
within 1.5 cm of zero) -- exactly the regime where a trajectory sits on a
collision boundary and any change to the dynamics, propagated through
hundreds of replanning cycles, can tip the outcome either way. One random
draw per case cannot distinguish "this perturbation happens to help" from
"this system is chaotically sensitive at the boundary and any perturbation
reshuffles which side of zero a near-miss lands on."

Put differently: **zero regressions out of 24 is itself information** (it
would be unusual by chance alone if the true effect were neutral or harmful),
but it is one sample per case against a two-parameter, correlated random
draw. It is not enough to certify SLIP is robust to mismatch, and it should
not be read as such.

## Conclusion and next step

The plan/plant separation mechanism is validated and should be kept as
standard practice. This single draw is a demonstration of the harness, not a
robustness result, and should not be cited as one. A real claim needs N
independent draws per `(world, seed, scenario)` (5-10 is a reasonable start),
reporting the distribution of outcomes, not a single point. That run is
future work, not required before the green light; if it is run, the honest
framing is "does SLIP degrade gracefully under mismatch," not "does mismatch
help."

This also reinforces a limitation to state plainly in the report: the SLIP
collision counts throughout this sandbox (self-consistent or not) come from
single seeded runs per case. The exact-footprint re-run changed six outcomes;
this pilot shows a further eight can move under a single further
perturbation. Treat any individual SLIP success/collision count as a point
estimate with real variance, not a stable ground truth.
