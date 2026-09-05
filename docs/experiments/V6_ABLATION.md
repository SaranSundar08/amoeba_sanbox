# V6 analytical skid-steer and footprint ablation

V6 is an analytical prototype, not a calibrated SLIP model. It verifies that
planning proposals, MPPI rollouts, episode execution, safety costs, collision
metrics, path-validity checks, and live rendering can share the same robot
model before hardware parameters are available.

## Compared configurations

- `ideal`: the legacy circular differential-drive behavior.
- `slip`: provisional 0.90 by 0.65 m oriented rectangular footprint, yaw gain
  0.82, speed-yaw-loss factor 0.55, turn margin 0.08 m, and speed margin
  0.03 m. Turn/speed margins are hard rollout feasibility allowances.

Both configurations use fixed covariance and V3 weighting. Each ran worlds 4,
48, and 49 with seeds 0 and 1 in normal, blocked, dynamic, and
dynamic-blocked scenarios: 48 runs total. The raw rows are in
`artifacts/results/milestones/v6_ablation.csv`.

| scenario | model | success | collisions | mean successful time | worst clearance | mean stall | mean ESS | mean path error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | ideal | 6/6 | 0 | 20.4 s | 0.085 m | 0.0 s | 89.1 | 0.062 m |
| normal | slip | 5/6 | 1 | 22.1 s | -0.000 m | 1.4 s | 103.9 | 0.042 m |
| blocked | ideal | 6/6 | 0 | 25.8 s | 0.076 m | 0.1 s | 60.0 | 0.322 m |
| blocked | slip | 5/6 | 1 | 30.3 s | -0.019 m | 3.5 s | 63.7 | 0.261 m |
| dynamic | ideal | 5/6 | 1 | 24.4 s | -0.007 m | 0.5 s | 45.9 | 0.205 m |
| dynamic | slip | 4/6 | 2 | 25.2 s | -0.013 m | 1.3 s | 42.4 | 0.117 m |
| dynamic blocked | ideal | 4/6 | 2 | 28.2 s | -0.009 m | 0.6 s | 40.1 | 0.342 m |
| dynamic blocked | slip | 2/6 | 4 | 30.0 s | -0.027 m | 0.7 s | 44.5 | 0.272 m |

## Interpretation

The larger footprint and reduced yaw response expose failures hidden by the
ideal circular model. They improve path-error measurements but reduce success,
especially with moving obstacles. Hard margins do not guarantee continuous
collision avoidance because footprint checking is sampled, A* uses an
orientation-free width proxy, and scripted obstacle motion is not predicted
over the rollout horizon.

The 0.405 m planning proxy is half the assumed width plus the turn margin. A
full rectangular circumcircle is safer but made all three BARN development
worlds unreachable, so it is not a useful stand-in for a heading-aware
footprint planner.

## Re-run under the exact footprint check (2026-09-04)

The table above was produced while the ground-truth collision check sampled
17 fixed points on the SLIP footprint boundary (0.225 m apart on the long
edges). A BARN cylinder (radius 0.075 m) fits between two samples, so the
true rectangle could overlap an obstacle by up to about 0.06 m while every
sample still read positive. `RobotModel.exact_clearance` now computes the
analytic oriented-rectangle distance against every real obstacle and is used
for episode collision/metric reporting, branch feasibility, and the hybrid
path-validity gate (the K x T MPPI rollout cost keeps the fast sampled
lookup on purpose). The same 48-case matrix was re-run with identical
settings; raw rows are in
`artifacts/results/milestones/v6_ablation_exact_footprint.csv`, the original
file is retained, and `experiments/v6_compare_footprint_check.py` pairs them.

| scenario | model | success | collisions | mean successful time | worst clearance | mean stall | mean ESS | mean path error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | ideal | 6/6 | 0 | 20.4 s | 0.091 m | 0.0 s | 89.1 | 0.062 m |
| normal | slip | 4/6 | 2 | 21.3 s | -0.015 m | 1.9 s | 105.1 | 0.042 m |
| blocked | ideal | 6/6 | 0 | 25.8 s | 0.082 m | 0.1 s | 60.0 | 0.322 m |
| blocked | slip | 5/6 | 1 | 32.4 s | -0.001 m | 4.0 s | 74.3 | 0.268 m |
| dynamic | ideal | 5/6 | 1 | 24.4 s | -0.007 m | 0.5 s | 45.9 | 0.205 m |
| dynamic | slip | 1/6 | 5 | 23.3 s | -0.006 m | 0.3 s | 59.9 | 0.191 m |
| dynamic blocked | ideal | 4/6 | 2 | 28.2 s | -0.009 m | 0.6 s | 40.1 | 0.342 m |
| dynamic blocked | slip | 0/6 | 6 | -- | -0.009 m | 0.5 s | 64.6 | 0.197 m |

Ideal-model outcomes, times, paths, switches, and ESS are unchanged; only
their reported clearances rise by 6--8 mm because the exact check has no
grid quantization. The change is confined to the rectangular SLIP footprint,
as predicted. Six of the 24 SLIP cases flipped from success to collision:
normal 5/6 to 4/6, dynamic 4/6 to 1/6, dynamic-blocked 2/6 to 0/6; blocked
stayed at 5/6. Every new collision is shallow (0 to -0.015 m), which is the
signature of grazes the sampled check could not see.

Two caveats. First, this is a matched re-run, not a re-scoring of the old
traces: the exact check also feeds the path-validity gate and branch
feasibility, so controller decisions differ and the flipped cases are
"collide under the corrected model", not necessarily "the old trace was
secretly colliding" (although a world-48, seed-0 spot check showed exactly
that: old check +0.004 m, exact check -0.007 m at the same state). Second,
these runs remain self-consistent: the plant executing the command is the
controller's own model. `simulation.episode(plant_model=...)` together with
`robot_model.sample_plant_mismatch` now allows a genuine plan/plant mismatch
ablation; it has not been run yet.

Cite this table, not the one above, for SLIP rows. The original
interpretation is strengthened rather than reversed: the analytical footprint
exposes moving-obstacle contacts that the circular model hides, and it does
so more than first reported.

A single-draw plant-mismatch pilot on these same SLIP cases
(`PLANT_MISMATCH_PILOT.md`) found 8 of 24 collisions flip to success under
one random perturbation and none flip the other way -- evidence these
collisions sit on a knife-edge (clearances of a centimetre or less), not a
robustness result. Treat the SLIP counts above as point estimates with real
run-to-run variance, not stable ground truth, until a multi-draw study exists.

## Hardware work required

Before using this model for SLIP, measure the full footprint including
protrusions and buffer; record commanded and odometry linear/angular velocity
over a speed/turn grid; identify acceleration, stopping, surface, and load
effects; fit and validate the model on held-out runs; and use Nav2 footprint
collision checking plus dynamic obstacle prediction or Collision Monitor.
