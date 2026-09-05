# Final narrow-navigation benchmark contract

Date: 2026-08-26

## Thesis question

Does a geodesic amoeba that generates pseudopod-derived ancillary control
distributions improve Biased MPPI navigation of the SLIP skid-steer robot in
narrow and locally obstructed environments?

Reciprocal IA-MPPI is outside this primary claim and remains exploratory.

## Frozen controllers

1. `vanilla`: ordinary MPPI with no global-path or amoeba proposal.
2. `path_mppi`: ordinary MPPI with the A* path added to the rollout cost. This
   is a path-tracking baseline, not Biased MPPI.
3. `path_biased`: one A*-derived ancillary control mean, with the full sample
   batch drawn around that mean. This is the single-path Biased MPPI baseline.
4. `amoeba`: the path fallback plus up to three geodesic pseudopod ancillary
   distributions using stable V3 grouped weights.

All controllers share the SLIP model, footprint, cost weights, samples,
horizon, integration step, limits, seeds, environment, and (where applicable)
dynamic prediction. Adaptive covariance and raw mixture-importance correction
are disabled in the final comparison because their earlier ablations did not
establish a reliable advantage. This prevents them from confounding the
pseudopod attribution.

## Development worlds

The default ten-world set was selected from the footprint-feasibility scan:

- critical: 0, 7, 48;
- tight: 1, 5, 20;
- comfortable references: 40, 42, 93, 94.

Unreachable and footprint-collision worlds are excluded from controller success
statistics and retained separately as feasibility evidence.

## Staged execution

1. Three-world, one-seed static pilot: worlds 48, 5, and 42.
2. Ten-world, five-seed static matrix after the pilot passes mechanically.
3. A declared locally blocked-path experiment targeted at the thesis mechanism.
4. A smaller moving-obstacle matrix only after static/blocked results freeze.

Do not tune controller parameters on the final five-seed matrix.

## Three-world static pilot result

The mechanical pilot completed on worlds 48, 5, and 42 with seed 0:

| controller | success | collision | mean successful time | mean clearance |
|---|---:|---:|---:|---:|
| vanilla | 2/3 | 1 | 24.15 s | 0.310 m |
| path MPPI | 3/3 | 0 | 25.32 s | 0.290 m |
| amoeba | 3/3 | 0 | 20.23 s | 0.275 m |

World 48 was discriminating: vanilla collided at -0.019 m clearance, while
path MPPI and amoeba succeeded at 0.115 m and 0.100 m. Parallel execution was
used, so timing columns are outcome diagnostics rather than final isolated
controller timing.

This pilot does **not** yet support the pseudopod-ancillary claim. Amoeba
pseudopods were available in approximately 74% of cycles, but an ancillary
mode was selected in 0% of cycles; the path fallback remained selected. The
amoeba controller can still differ through its gated flow cost/bias, so its
result cannot be attributed to ancillary distributions. Consequently, the
ten-world matrix is paused. The next required experiment is a matched blocked-
path case in which ancillary selection is nonzero and causally relevant.

## Frozen blocked-path pilot result

The blocked-split pilot was rerun with the same final controller definitions,
SLIP footprint, 1024 samples, 56-step horizon, and seed 0:

| controller | result | peer clearance | static clearance | branch selection |
|---|---|---:|---:|---:|
| vanilla | static collision | 0.771 m | approximately 0 m | 0% |
| path MPPI | peer collision | -0.001 m | 0.100 m | 0% |
| amoeba | success, 29.25 s | 0.726 m | 0.158 m | 49.1% |

Amoeba was in ASSIST for 98.3% of cycles, selected ancillary pseudopod modes
in 49.1% of cycles, and switched modes twice. Unlike the ordinary static
pilot, this run activates and attributes the proposed mechanism. It is still
one seed and therefore only authorizes the five-seed blocked-split matrix; it
does not establish a final success-rate claim.

## Five-seed blocked-path result

The frozen five-seed matrix reproduced the pilot without exceptions:

| controller | success | peer collisions | static collisions | mean peer clearance |
|---|---:|---:|---:|---:|
| vanilla | 0/5 | 0 | 5 | 0.767 m |
| path MPPI | 0/5 | 5 | 0 | -0.004 m |
| amoeba | 5/5 | 0 | 0 | 0.710 m |

Amoeba completion time was 29.13 s, minimum static clearance averaged 0.141 m,
ASSIST fraction was 98.3%, pseudopod selection fraction was 49.9%, and mode
switches averaged 2.4. This establishes repeatability across MPPI sampling
seeds for this one deterministic blocked geometry. Parallel execution makes
the recorded timing unsuitable as isolated publication timing.

One attribution control remains before assigning the improvement specifically
to ancillary distributions: `flow_only` uses the identical geodesic field,
path cost/bias, gate, dynamics, and MPPI settings but disables pseudopod
extraction and grouped ancillary proposals. Only its five seeds need to be
run; the completed vanilla/path/amoeba cases must not be repeated.

## Flow-only attribution result

Flow-only succeeded in 2/5 seeds, with one peer collision and two static
collisions. Full Amoeba succeeded in 5/5 with no collisions. Relative to
flow-only, full Amoeba reduced mean stall time from 15.71 s to 3.65 s,
increased mean peer clearance from 0.134 m to 0.710 m, and reduced successful-
run completion time from 36.88 s to 29.13 s. Full Amoeba selected pseudopod
modes in 49.9% of cycles and cost approximately 8.5% more mean controller time
than flow-only in the parallel outcome run.

This supports a practical ancillary contribution beyond the geodesic field,
but five paired seeds give wide statistical uncertainty. The harness therefore
supports `--seed-start`; extend only flow-only and Amoeba through seeds 5--9.
Do not rerun vanilla, path, or seeds 0--4.

## Ten-seed final attribution result

Seeds 5--9 added five new paired flow-only/Amoeba trials. Flow-only timed out
in all five. Amoeba succeeded in four and timed out safely in seed 5. Combining
the non-overlapping files for seeds 0--9 gives:

| method | success | peer collision | static collision | timeout |
|---|---:|---:|---:|---:|
| flow-only | 2/10 | 1 | 2 | 5 |
| full Amoeba | 9/10 | 0 | 0 | 1 |

| metric | flow-only | full Amoeba |
|---|---:|---:|
| mean peer clearance | 0.131 m | 0.703 m |
| mean static clearance | 0.028 m | 0.143 m |
| mean stall time | 19.96 s | 5.02 s |
| mean successful time | 36.88 s | 28.95 s |
| pseudopod selection | 0% | 47.6% |
| mean mode switches | 0 | 4.5 |
| mean controller time | 137.56 ms | 148.68 ms |

The paired outcomes contain seven seeds where Amoeba alone succeeded, zero
where flow-only alone succeeded, two where both succeeded, and one where both
failed. The exact two-sided McNemar test is $p=0.015625$. Wilson 95% intervals
for success are approximately [0.057, 0.510] for flow-only and
[0.596, 0.982] for Amoeba. This supports the claim that pseudopod ancillary
sampling improves this blocked-path task beyond geodesic flow guidance alone.

The limitation is explicit: Amoeba seed 5 timed out with 18.1 s stall time,
19 mode switches, and a lower 36.0% branch-selection fraction. Do not tune on
that seed before the main result is frozen. Parallel outcome timing is not the
final real-time measurement; isolated serial timing remains required.

The blocked-split outcome and attribution experiment is now frozen. Do not run
additional seeds or tune its controller. Proceed to an ordinary narrow-world
generality matrix and retain blocked-split as the primary mechanism test.

## Ten-world static generality result

The completed matrix contains 50 matched trials per controller (ten worlds,
five seeds, SLIP model):

| method | success | collisions | mean successful time | mean clearance | worst clearance |
|---|---:|---:|---:|---:|---:|
| vanilla | 25/50 | 25 | 24.21 s | 0.216 m | -0.025 m |
| path tracking (`path_mppi`) | 39/50 | 11 | 25.36 s | 0.233 m | -0.040 m |
| single-path biased (`path_biased`) | 45/50 | 5 | 19.86 s | 0.251 m | -0.025 m |
| full Amoeba | 50/50 | 0 | 20.87 s | 0.258 m | 0.025 m |

This establishes generality of the complete gated geodesic/path/Amoeba
controller over the declared development worlds. It does not independently
attribute the static improvement to pseudopod sampling: ancillary modes were
available in 70.9% of Amoeba cycles but selected in 0%, ASSIST occupied only
2.0% of cycles, and there were no mode switches. The frozen blocked-split
experiment remains the evidence for the pseudopod mechanism.

The matrix also exposed a terminology gap: `path_mppi` modifies the rollout
cost but does not bias the sampling distribution. The added `path_biased`
configuration supplies the single-path Biased MPPI comparator. It improved
six matched cases over path tracking and never lost a case, but collided in
all five world-7 seeds. Amoeba succeeded in those five matched cases and never
lost a case to `path_biased`. Because all five discordances share one geometry,
they demonstrate a repeatable world-specific weakness rather than five
independent environment-level observations. An exact paired sign/McNemar test
on the 45/5 split gives two-sided p=0.0625 and should not be overstated.
Parallel-run timing remains diagnostic only.

### Re-validated under the exact footprint check (2026-09-05)

This matrix used `simulation.episode()`, which the 2026-09-04
exact-footprint fix (`RobotModel.exact_clearance`) applies to automatically,
unlike `v71_junction_benchmark.py`'s own loop. All 200 cases (vanilla,
path_mppi, path_biased, amoeba) were re-run under the current code
(`artifacts/results/milestones/final_static_generality_revalidated.csv`) and
paired case by case against the frozen file above: **0/200 flips**. Every
success/collision outcome is unchanged; the table's numbers stand as
independently confirmed, not merely assumed, alongside the McNemar
blocked-split result confirmed the same way.

The re-run added `amoeba_nominal_fb` (the accepted horizon-scale-distinctness
candidate from `REFERENCE_DISTINCTNESS.md`) to the same matrix:

| method | success | collisions | mean successful time | mean clearance |
|---|---:|---:|---:|---:|
| `amoeba_nominal_fb` | 50/50 | 0 | 20.9 s | 0.262 m |

**Identical to `amoeba` in every one of the 50 cases** (0/50 differ). This is
expected, not a null result: this matrix's paths stay valid (ASSIST occupies
only 2.0% of cycles, pseudopods are selected in 0%, per the paragraph above),
so `nominal_fb`'s branch-reference speed policy has nothing to act on here.
It says nothing about `nominal_fb`'s effect where branches actually matter
(the blocked-path promotion gate already showed that: 21/24 to 22/24, see
`docs/experiments/REFERENCE_DISTINCTNESS.md`) -- only that it does not
disturb the ordinary-navigation case this matrix measures.

## Isolated serial timing protocol

After freezing outcomes, measure computation separately on worlds 7, 48, and
42, representing two critical geometries and one comfortable reference. Run
the four frozen controllers sequentially with one process, seed 0, 1024
samples, and horizon 56. Report the mean of episode-level controller means,
the mean of episode-level p95 values, and the worst observed callback maximum.
The panel is a reference-Python timing characterization, not a Nav2 real-time
guarantee. No outcome parameters may be tuned from it.

The completed isolated panel produced:

| method | mean update | mean episode p95 | worst update | effective rate |
|---|---:|---:|---:|---:|
| vanilla | 31.45 ms | 35.25 ms | 68.65 ms | 31.6 Hz |
| path tracking | 109.60 ms | 115.98 ms | 179.05 ms | 9.1 Hz |
| single-path biased | 133.80 ms | 166.20 ms | 188.59 ms | 7.5 Hz |
| full Amoeba | 147.97 ms | 194.70 ms | 226.97 ms | 6.8 Hz |

The reference NumPy Amoeba implementation therefore does not meet a 20 Hz
(50 ms) controller budget. Relative to single-path biased MPPI, Amoeba adds
about 14.2 ms mean latency (10.6%); most of the total cost is already present
in path processing and 1024-by-56 rollout evaluation. These figures motivate
asynchronous low-rate geodesic reconstruction and a vectorized C++/GPU rollout
path for deployment. They are not transferred directly to the Nav2 C++ plugin.

## Required metrics

- terminal result, success and collision rate;
- completion time, path length, final distance, and goal progress;
- minimum footprint clearance and stall duration;
- mean, p95, and maximum controller time;
- path tracking error and angular sign flips;
- ancillary availability, active-mode count, selection fraction, fallback
  fraction, ASSIST fraction, mode switches, and ESS.

The episode evaluator and NumPy MPPI both use `RobotModel.clearance()`, so
termination and trajectory scoring share the same oriented footprint model.
The sampled footprint is an acknowledged approximation to a polygon costmap
check. Gazebo/Nav2 validation must use the actual costmap footprint.

## Acceptance and interpretation

The proposed method is supported only if it improves success or recovery in
the blocked/narrow subset without an unacceptable collision, clearance, or
compute-time penalty. Equal performance is a valid negative result. Ordinary
open-world success alone does not establish the value of pseudopods; ancillary
availability and selection telemetry must demonstrate mechanism activation.
