# Amoeba MPPI project status

Last updated: 2026-09-04

This file is the durable handoff for future Codex sessions. Read it together
with `README.md`, `docs/amoeba_mppi_progress.tex`, and the milestone-specific
ablation notes.

## Live visualization backend

PyQtGraph is now the default `--animate` renderer, with Matplotlib retained via
`--renderer matplotlib` and for static PNG/publication output. The PyQtGraph
module is lazily imported, so headless runs have no GUI dependency. An
offscreen equivalence test using two fresh, identically seeded V5 controllers
produced byte-identical traces and identical episode metrics with and without
the renderer callback.

## Independent PyTorch acceleration track

`torch_port/` is a sibling implementation and does not alter or import the
NumPy controller. It covers ideal/slip dynamics, footprint-clearance lookup,
constraint projection, batched rollouts and costs, vanilla updates, and V4
mixture densities. Its standalone runtime also owns a BARN loader, rolling
geodesic body, pseudopods, ancillary proposals, V3-style grouped controller,
episode loop, CLI, and light-mode PyQtGraph observer. The hybrid runtime now
adds independent A*, a protected path proposal, footprint path validity,
NORMAL/ASSIST gating, committed switching, and periodic replanning. Six
deterministic parity tests plus three runtime tests pass; the combined suite
has 54 tests. System Python skips the nine optional torch tests when PyTorch
is absent.

The isolated `amoeba-torch` Conda environment uses Python 3.11 and PyTorch
2.13.0+cu130. CPU validation passes. CUDA cannot be initialized from the
current sandbox because the NVIDIA driver/NVML is not visible; it must be
verified from the host terminal before GPU performance claims are made.

## Implemented

- V0: local geodesic-field-guided MPPI.
- V1: topology-separated pseudopod extraction and branch ID tracking.
- V2: branch-to-control-sequence proposals with dynamic projection.
- V3: fixed grouped sampling, separate mode updates, persistent warm starts,
  and committed mode selection.
- V3.1: fallback allocation, hysteresis, dwell/confirmation, ESS diagnostics,
  and grouped-sampling ablations.
- V3.2: NORMAL/ASSIST gating, Nav2-like path validity, and periodic A*
  replanning.
- V4: stable full-mixture importance correction in latent control space:
  `-S(U)/lambda + log p(U) - log q(U)`.
- V4.1: cycle-coherent ESS fallback to V3.
- V5: per-mode, per-step diagonal covariance from predicted clearance and
  turning demand.
- V6: shared ideal/analytical-skid dynamics, oriented rectangular footprint
  checks, and turn/speed-dependent feasibility margins.
- V7/V7.1: vanilla/path/Amoeba benchmark harnesses, including dynamic-junction
  scenarios.
- V7.2: independent PyTorch controller/runtime and fast PyQtGraph observer.
- V8 Phase 1--3: Nav2 C++ geodesic body, pseudopod extraction, geodesic bias,
  and promise-weighted ancillary control proposals with a vanilla fallback.
- V8 Phase 4: full-horizon ancillary mean rollout, live Nav2 footprint/costmap
  rejection, safe-mode sample reallocation, validated-rollout Path topics, and
  lightweight RViz allocation diagnostics.

## Current Nav2 Phase 4 state

The controller can expose up to three pseudopod reference paths. With a batch
of 2000, 20% (at most 400 rows) is allocated among active pseudopods using a
promise softmax with temperature 0.75; the remaining roughly 1600 rows retain
ordinary Nav2 MPPI sampling. The ancillary means are time-varying `(v, w)`
sequences. The current test configuration enables amoeba bias at strength 0.2,
disables shadow mode, and leaves the separate flow critic disabled.

Visualization is split across `/amoeba_debug` (membrane/field MarkerArray),
`/amoeba/ancillary_path_1` through `_3` (Path messages), and `/trajectories`
(throttled batched trajectory lines). The Path topics must use RViz Path
displays. A stale or duplicate external action client can still crash the
composed Nav2 container by submitting an already-active goal ID; test with one
goal source after setting the initial pose.

Each ancillary mean is now checked using Nav2's live robot footprint at every
predicted pose. Rejected/off-map modes receive zero samples. Safe modes share
the ancillary budget, while rejection of every mode produces a complete
vanilla-MPPI fallback. `/amoeba/ancillary_rollout_1` through `_3` expose only
accepted rollouts.

All 63 unit and integration tests pass (45 original, plus eight added on
2026-09-04 for the exact footprint check and plant/plan separation, and ten
for the homotopy-consistency cost and mode-distinctness metrics).

## V4 empirical check (2026-07-24)

### Prescribed normal-path comparison

World 48, seed 0, radius 0.34, hybrid controller, grouped sampling, automatic
gate, Nav2 path validity, and replanning every 1.0 seconds:

| weighting | result | time | path | min clearance | stall | switches | omega sign flips |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 | success | 19.75 s | 8.74 m | 0.095 m | 0.05 s | 0 | 34 |
| V4 | success | 19.75 s | 8.74 m | 0.095 m | 0.05 s | 0 | 34 |

These identical results are expected but not a meaningful mixture test. The
automatic gate remained in NORMAL and assigned all 1024 samples to the path
mode. With one Gaussian, `p=q`, so V4 reduces exactly to V3.

### Genuine multimodal blocked-path comparison

World 48, seed 0, fake center-lane path, radius 0.34, hybrid controller,
grouped sampling, automatic gate, Nav2 path validity, and no replanning:

| weighting | result | time | path | min clearance | stall | switches | omega sign flips |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 | success | 29.0 s | 10.43 m | 0.100 m | 0.60 s | 6 | 42 |
| V4 | timeout | 60.0 s | 7.56 m | 0.181 m | 17.20 s | 1 | 31 |

V4 remained collision-free but oscillated near the blockage and failed to
complete the route. Corrected branch weights repeatedly had very low ESS,
including values near 1.0. The protected fallback often retained much higher
ESS and dominated the mode evidence, preventing commitment to the detour that
allowed V3 to succeed.

## Current conclusion

V4 is mathematically implemented and its density/weight tests pass, but
enabling it unconditionally is not a controller improvement. The first real
multimodal comparison shows a finite-sample degeneracy regression. Keep V3 as
the active controller baseline.

## V4.1 ESS guard (2026-07-24)

V4.1 was implemented behind `--importance-ess-guard`. It computes raw V4 and
V3 statistics together. If any active mode has corrected
`ESS < max(8, 0.02 N_m)`, the complete cycle uses V3 weights and free energies.
The cycle-level decision is important because V3 and V4 mode evidence must not
be mixed in one mode-selection comparison.

The same blocked world-48 test produced:

| weighting | result | time | path | min clearance | stall | switches | omega sign flips |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4.1 guarded | success | 26.95 s | 10.06 m | 0.110 m | 0.95 s | 6 | 27 |

The guard restored success, but it activated for 498 cycles and 1298 of 1355
mode evaluations (95.8%). This single run therefore shows a successful safety
fallback, not a corrected-weighting performance advantage. Raw V4 should
remain diagnostic-only. V3 remains the controller baseline until a
multi-world, multi-seed V3/V4/V4.1 benchmark says otherwise.

All 32 tests passed after the V4.1 implementation.

## V5 ablation (2026-08-03)

The reproducible 48-run matrix is stored in
`artifacts/results/milestones/v5_ablation.csv`. Fixed and adaptive covariance each ran 24
matched cases: worlds 4, 48, and 49; seeds 0 and 1; normal, blocked, dynamic,
and dynamic-blocked scenarios.

V5 improved worst static clearance and mean ESS in every scenario. Aggregate
success was unchanged: both policies achieved 6/6 normal, 6/6 blocked, 5/6
dynamic, and 4/6 dynamic-blocked. Adaptive covariance was slower and switched
more often in blocked scenarios. It had one collision plus one timeout in the
dynamic-blocked set, versus two collisions for fixed covariance.

V5 is complete as an implemented and ablated research milestone, but remains
experimental. See `experiments/V5_ABLATION.md`.

## V6 analytical prototype (2026-08-03)

V6 routes MPPI rollout, branch proposal rollout, episode execution, collision
metrics, and hybrid path validity through one robot model. The `slip` option
uses a 0.90 by 0.65 m oriented sampled footprint, speed-dependent yaw loss,
and hard turn/speed uncertainty margins. The orientation-free A* proxy uses
half the footprint width plus the turn margin (0.405 m); a full circumcircle
made all three development worlds unreachable.

The matched 48-run ideal/slip matrix is in
`artifacts/results/milestones/v6_ablation.csv`. Success was ideal/slip: normal 6/6 vs
5/6, blocked 6/6 vs 5/6, dynamic 5/6 vs 4/6, and dynamic-blocked 4/6 vs 2/6.
The analytical model improved mean path error, but did not remove sampled or
dynamic contacts and increased completion time. This is useful model plumbing
and a conservative failure-revealing prototype, not evidence of hardware
readiness.

The footprint dimensions, yaw gain 0.82, speed-yaw-loss factor 0.55, and
0.08/0.03 m uncertainty margins are provisional. Before robot use, replace
them using measured footprint geometry, commanded-versus-odometry bags,
acceleration/stopping tests, and surface/load trials. Dynamic deployment also
needs obstacle prediction or Nav2 collision monitoring. Fixed covariance with
V3 weights and the ideal sandbox model remains the comparison baseline. See
`experiments/V6_ABLATION.md`.

## Exact footprint check and plant/plan separation (2026-09-04)

Two evaluation flaws were found by code review and fixed. First, the
ground-truth SLIP collision check sampled 17 boundary points 0.225 m apart on
the long edges; a 0.075 m BARN cylinder fits between them, so the true
rectangle could overlap an obstacle by about 0.06 m while reading clear.
`RobotModel.exact_clearance` (analytic oriented-rectangle distance against
`env.obstacle_centers()`, including live moving obstacles) now backs episode
collision reporting, branch feasibility, and the hybrid path-validity gate.
The K x T rollout cost keeps the fast sampled lookup deliberately. Second,
`simulation.episode` executed commands with the controller's own robot
model, so no plan/plant mismatch ever existed; it now accepts
`plant_model=`, and `robot_model.sample_plant_mismatch` draws jittered
yaw-gain/speed-yaw-loss plants for a real robustness ablation.

The V6 matrix was re-run under the exact check into
`artifacts/results/milestones/v6_ablation_exact_footprint.csv` (original
retained; compare with `experiments/v6_compare_footprint_check.py`). Ideal
rows are unchanged apart from +6--8 mm reported clearance. Six of 24 SLIP
cases flipped success to collision, all shallow (0 to -0.015 m): SLIP success
is now normal 4/6, blocked 5/6, dynamic 1/6, dynamic-blocked 0/6. Cite the
re-run for SLIP numbers; the original SLIP rows are a known lower bound on
collisions. The plant-mismatch ablation has not been run.

Other milestone CSVs used the ideal circular model, whose check was already
point-exact up to grid quantization, so they are not affected by the sampling
gap; the blocked-split and generality matrices used `path_validity_gate`
under the ideal model and are likewise unaffected.

## Space-time topology Phase 0 (2026-09-05)

Started as a background investment (roughly an hour or two a day, alongside
report writing and the Nav2 port), not a green-light item. `spacetime.py`
implements a standalone time-expanded `(x, y, t)` Dijkstra search plus a
synchronized-time distinctness test (the temporal analogue of `homotopy.py`'s
side test), validated on synthetic scenarios only -- not yet wired into
`LocalFlowField`/`pseudopods.py`/`mppi.py`. Found and fixed two real bugs
before they shipped: a deadline-based "two route" construction that silently
degenerates to one dominant strategy (replaced with a wait-cost-variation
construction, verified empirically to produce genuine wait-vs-detour
routes), and a search grid sized only around the start position that
silently clips a distant goal instead of erroring.

Also added `spacetime_path_to_proposal`: converts a raw space-time path
into a `proposals.BranchProposal` (resample onto the controller's `(T, dt)`
grid, differentiate into velocity commands, project through the same
box/rate limits every proposal uses, roll out, check feasibility) -- the
glue needed for a space-time route to slot in as an ordinary mode. Verified
a wait segment survives resampling as exactly zero commanded speed (would
be silently discarded by `proposals.branch_to_control_sequence`, which only
understands geometry, not timing), and that feasibility is judged on the
achieved rollout: a raw path with 0.2 m of paper clearance came out with
-0.247 m once a tight turn-rate limit was honored, correctly flagged
infeasible. Still not wired into `mppi.py`/`pseudopods.py` -- nothing yet
decided when to call this or turned its output into a live `SamplingMode`
-- that decision layer is now wired (2026-09-06):
`MPPI._spacetime_alternatives`, called from `build_branch_proposals`,
detects a predicted crossing against a branch's nominal-speed centreline
continuation (not its short MPPI-horizon rollout -- a second "silently
checks the wrong window" bug found and fixed the same way as the goal-
reachability one above) and adds feasible wait/detour routes as extra
modes (branch_id +100/+200), picked up automatically by `_sampling_modes`.
Off by default, verified to leave `branch_proposals` untouched even with a
real crossing present. 5 new integration tests
(`tests/test_spacetime_integration.py`), 96/96 overall. Not yet validated
against the existing dynamic scenarios, and the path fallback and multiple
simultaneous obstacles remain unhandled. See
`docs/experiments/SPACETIME_TOPOLOGY_PHASE0.md`.

## Homotopy-consistency cost and horizon-scale distinctness (2026-09-04)

A T-MPC-style mode-purity cost (de Groot et al., RA-L; `homotopy.py`,
`--homotopy-w`, off by default, fallback unconstrained) was implemented and
ablated on the blocked path (worlds 4/48/49, seeds 0--1,
`artifacts/results/pilots/homotopy_pilot.csv`). Off and on are identical:
6/6, 25.8 s, 0.082 m worst clearance, 7.5 switches. Sample crossings are
at most 0.07%, executed-update wrong-side mass is zero, and a mode's
optimized mean drifts across its own reference's side in 0.1% of
mode-cycles: the T-MPC collapse mechanism does not occur in the
sampling-based controller, and the cost is inactive. See
`docs/experiments/HOMOTOPY_ABLATION.md`.

The same test on the *references* is the real finding: feasible pseudopod
pairs pass a nearby obstacle on different sides in 0/288 (world 48) and
0/454 (world 49) pairs. The 56-step proposal rollout covers 0.46 m, about
19--20% of the branch centreline, and reference endpoints are 0.08--0.27 m
apart. Pseudopods are distinct at the membrane but not at horizon scale, so
mode groups sample near-identical guidance and their free energies differ
mostly by noise. This explains the blocked-path switch counts and the 0%
branch selection in the static generality matrix.

## Horizon-scale reference distinctness (2026-09-04)

Five branch-reference speed policies were compared on the blocked path
(worlds 4/48/49, seeds 0--1; `experiments/reference_distinctness.py`,
`artifacts/results/pilots/reference_distinctness_pilot*.csv`), varying only
the `reference_*` kwargs of `MPPI`; path proposal, costs, sampling, gating,
and selection unchanged. Half-speed floors (`floor50`, `band20`) never reach
the split (separation 0.000--0.001). Nominal speed reaches it (0.328) but
rejects 42% of references as infeasible, drops guided modes to about one, and
times out twice on world 48. Nominal speed with a per-branch fallback to the
shaped reference when infeasible (`nominal_fb`) gives separation 0.186 with
feasibility 0.94 (baseline 0.89), the same guided-mode count, a free-energy
gap between modes of 2.07 (baseline 0.88), fewer switches (6.3 vs 7.5),
pseudopod selection 0.73 (vs 0.55), and worst clearance 0.100 m (vs 0.082 m)
at 6/6 success. Own drift rises from 0.1% to 2.9%, so the mode-purity cost
is testable only from here. See `docs/experiments/REFERENCE_DISTINCTNESS.md`.

`nominal_fb` is the accepted candidate and remains opt-in; the shaped
reference stays the default and no prior result changes.

**Re-validated 2026-09-05.** The ten-world, five-seed static generality
matrix (vanilla/path_mppi/path_biased/amoeba, the 25/50-39/50-45/50-50/50
headline) was re-run under the exact-footprint fix: 0/200 paired cases
flipped. `amoeba_nominal_fb` was added to the same matrix and is identical to
`amoeba` in all 50 cases -- expected, since ASSIST/branch selection barely
occurs in this matrix, not a claim about the blocked-path cases where
`nominal_fb` actually differs. See
`docs/experiments/FINAL_BENCHMARK_CONTRACT.md` and
`artifacts/results/milestones/final_static_generality_revalidated.csv`.

`nominal_fb` passed the four-scenario promotion gate (worlds 4/48/49, seeds
0-1): 21/24 to 22/24 successes, no scenario regressed, dynamic improved 5/6
to 6/6. One caveat: dynamic_blocked's count held at 4/6 but the failing cases
changed (world 48/seed 0 fixed, world 4/seed 1 newly failed) -- a wash, not a
clean win, in that scenario. `nominal_fb` remains opt-in, not default,
pending validation against the larger generality/attribution matrices.

A single-draw plant-mismatch pilot (`docs/experiments/PLANT_MISMATCH_PILOT.md`)
ran the 24 SLIP exact-footprint cases with the plant's yaw-gain/speed-yaw-loss
independently perturbed +/-30%, unknown to the controller. 8/24 collisions
flipped to success, none flipped the other way -- but the flips don't track
perturbation direction (both weaker and stronger yaw-gain draws fixed world
49), so this reads as knife-edge sensitivity at already-borderline clearances
(within 1.5 cm), not evidence mismatch helps or is tolerated. The harness is
validated and worth keeping; this run is not a robustness claim and should
not be cited as one. A real claim needs N draws per case, averaged -- deferred,
not required before the green light.

Next: validate `nominal_fb` on the blocked-split junction case and the
10-world generality matrix before making it a default; then run `nominal_fb`
x `homotopy_w` on/off as the actual T-MPC test. Space-time (dynamic-obstacle-
aware) pseudopod extraction remains deferred to the post-green-light roadmap.

## Next implementation order

The frozen three-controller static pilot (worlds 48, 5, 42; seed 0; SLIP)
produced vanilla/path/Amoeba success of 2/3, 3/3, and 3/3. However, Amoeba
selected ancillary pseudopods in 0% of cycles even though they were available
about 74% of the time. Do not expand the static matrix yet: first run the
matched blocked-path attribution case required by
`experiments/FINAL_BENCHMARK_CONTRACT.md`.

The frozen five-seed blocked-path matrix then produced vanilla/path/Amoeba
success of 0/5, 0/5, and 5/5. Vanilla had five static collisions, path MPPI had
five peer collisions, and Amoeba had none. Amoeba selected pseudopod modes in
49.9% of cycles, maintained 0.710 m mean peer clearance, and averaged 2.4 mode
switches. Run only the five-seed `flow_only` attribution control next; the
large ordinary-static matrix remains paused.

Flow-only then succeeded in 2/5 with one peer and two static collisions,
versus Amoeba 5/5 with none. Amoeba cut mean stall from 15.71 s to 3.65 s and
raised peer clearance from 0.134 m to 0.710 m while selecting pseudopods in
49.9% of cycles. Extend only flow-only and Amoeba to seeds 5--9 using
`--seed-start 5`; do not repeat completed cases.

The combined ten-seed attribution is now frozen: flow-only succeeded 2/10
(one peer collision, two static collisions, five timeouts), while full Amoeba
succeeded 9/10 with no collisions and one timeout. Seven paired seeds favored
Amoeba and none favored flow-only; exact two-sided McNemar $p=0.015625$.
Amoeba reduced mean stall from 19.96 s to 5.02 s and selected pseudopods in
47.6% of cycles. Seed 5 remains a documented safe timeout with 19 switches.
Do not tune or extend blocked-split further; move to ordinary narrow-world
generality testing and isolated timing.

**Re-validated 2026-09-04.** `v71_junction_benchmark.py` predates today's
exact-footprint fix and has its own episode loop (not `simulation.episode`),
so it was independently checked: its peer-robot collision test used the same
sampled-footprint approximation, now replaced with the exact
`RobotModel.circle_set_clearance` (the obstacle term of `exact_clearance`,
factored out so a caller with non-circular static geometry -- this
environment's walls are rectangles, still sampled and undocumented as exact
-- can still get an exact check against a circular obstacle). Re-running
flow-only and Amoeba at seeds 0--9 under the fix reproduced every one of the
20 outcomes exactly: flow-only 2/10, Amoeba 9/10, identical per-seed results
and clearances (the one collision, flow-only seed 0, moved from -0.004 m to
-0.001 m, still a collision). The McNemar result stands, independently
confirmed. Static-wall collision checking in this script remains the sampled
approximation; an exact oriented-rectangle-vs-rectangle check is deferred, not
required for this confirmation since the peer is the applicable geometry for
the attribution claim. See
`artifacts/results/milestones/final_blocked_attribution_revalidated.csv`.

The ten-world static generality matrix is now complete. Vanilla succeeded
25/50, path-tracking MPPI 39/50, and full Amoeba 50/50; their collision counts
were 25, 11, and zero. Amoeba's mean successful time was 20.87 s and mean
clearance was 0.258 m. However, its pseudopod selection fraction was zero in
this matrix (despite 70.9% availability), so this is evidence for the complete
gated geodesic/path hybrid, while blocked-split remains the causal pseudopod
test. The missing single-A*-ancillary `path_biased` column is now complete:
45/50 successes and five collisions, all on world 7. It improved six matched
cases over path tracking without losing any, while Amoeba improved the five
world-7 cases without losing any. The five Amoeba/path-biased discordances are
clustered in one geometry (two-sided paired p=0.0625), so report this as a
repeatable world-specific advantage rather than broad statistical proof.

The isolated serial Python timing panel is also frozen (worlds 7, 48, and 42;
seed 0; one process). Mean controller latency was 31.45 ms for vanilla,
109.60 ms for path tracking, 133.80 ms for single-path biased MPPI, and
147.97 ms for Amoeba. Amoeba's mean episode p95 was 194.70 ms and its worst
update was 226.97 ms. Thus the reference NumPy implementation is not a 20 Hz
deployment implementation. Amoeba adds 14.2 ms (10.6%) over the closer
single-path-biased baseline; deployment should move geodesic reconstruction to
a low-rate asynchronous worker and retain vectorized C++/GPU rollouts.

The Torch reciprocal interaction-risk ablation is recorded in
`experiments/INTERACTION_RISK_ABLATION.md`. Constant velocity succeeded in
10/10 matched static/dynamic trials, versus 6/10 for worst-case hypotheses and
7/10 for CVaR. A difficult-seed mean screen succeeded in only 1/4 trials.
Risk aggregation alone is therefore closed as a tuning direction. The next
Torch interaction step is observation-based hypothesis probabilities plus a
hysteretic reverse/deadlock-recovery state.

Observation- and ancillary-mode-conditioned probabilities are now implemented
in the Torch reciprocal runner, including probability and entropy telemetry.
Difficult-seed validation and the hysteretic reverse recovery state remain
pending; this is not yet the learned local-goal network. The first difficult-
static screen passed 2/2: mean completion time was 25.93 s, peer clearance
0.327 m, mode switches 8.5, and reverse distance 0.48 m. A matched two-seed
dynamic screen is next before modifying recovery logic.

The dynamic weighted screen passed seed 0 but seed 1 timed out with 72 mode
switches and 4.475 m of reversing even though predicted reverse probability
was below one percent. Reverse has therefore been moved behind a hysteretic
NORMAL/YIELD/RECOVERY_REVERSE/REJOIN state machine. Repeat only dynamic seed 1
before any larger matrix.

The first recovery regression remained a timeout, but switches dropped from
72 to 19, reverse distance from 4.475 m to 1.493 m, and robot 1 reached its
goal without collision. Robot 2 did not finish. Per-robot terminal goal
distance, pose, interaction/recovery state, and assist reason are now recorded
before any further recovery tuning.

The repeat showed robot 2 stopped 2.161 m from goal in `ASSIST/path_blocked`,
not interaction recovery, but had still reversed 1.168 m because negative
velocity sampling was enabled globally. Reverse sampling is now restricted to
the designated yielding robot. Repeat only dynamic seed 1.

That forward-priority regression passed: both robots finished in 49.65 s,
non-yielding reverse distance was zero, and peer clearance was 0.166 m. It
still required five recoveries, 36 mode switches, and reached 0.027 m static
clearance. Validate current-code seed 0 next; do not tune further on seed 1.

1. Add benchmark-grade promise, rejection-reason, and minimum-clearance
   telemetry.
2. Move field reconstruction to an asynchronous snapshot worker so it cannot
   block the controller loop.
3. Run matched vanilla, path-biased, and Amoeba Nav2 benchmarks in scaled BARN
   and dynamic/two-robot junction scenarios.
4. Only then evaluate the optional flow critic and physical-robot calibration.

## Optional learned pseudopod ranking

Phase 0 is implemented independently under `torch_port/learning/`. It compares
the current geometric promise ordering against a finite-horizon counterfactual
surrogate oracle that rolls every feasible pseudopod mean through the Torch
SLIP footprint model. The frozen controllers are unchanged. A small world-48
CPU smoke test produced six multimodal snapshots, 100% agreement, zero regret,
and no avoidable candidate collision; this is only a mechanical test, not
evidence that learning is unnecessary. Run the declared ten-world GPU screen
before implementing a CNN. If it shows no meaningful oracle headroom, stop the
learning extension and retain it as future work.

The declared three-world GPU pilot is complete. Across 97 multimodal snapshots,
the heuristic and surrogate oracle agreed in 70.1%, mean utility regret was
1.87, p95 regret was 18.53, and 9/97 captures contained a heuristic candidate
collision avoided by another pseudopod. Per world, agreement was 65.3% on
world 0, 72.4% on world 7, and 78.9% on world 48. Avoidable-collision captures
were concentrated in world 0 (seven) and world 48 (two), with several adjacent
time samples belonging to the same encounter. Phase 0 therefore shows enough
ranking headroom to justify the full screen, but it does not yet demonstrate
improved closed-loop controller performance or nine independent events.

The full two-seed screen confirms the signal over 588 multimodal snapshots:
69.7% agreement, mean regret 0.92, 25 avoidable-collision captures grouped
into 12 temporal encounter events. Safety headroom appears in worlds 0, 1, 20,
and 48. Phase 1 is now implemented as an offline-only shared CNN ranker. Its
robot-centred input contains occupancy, signed clearance, Amoeba body,
membrane, navigation value, and a candidate pseudopod mask; a small MLP adds
candidate and relative-goal features. Dataset generation, world-level splits,
multi-task ranking/collision/progress training, checkpointing, and held-out
metrics are available under `torch_port/learning/`. No learned output is yet
connected to the controller.

The first 60-epoch offline ranker is complete and is rejected for controller
integration. It improved training top-1 accuracy from the heuristic's 65.5% to
80.3% and validation from 84.5% to 91.4%, but held-out test accuracy fell from
77.2% to 71.5%. Test mean regret improved only from 0.561 to 0.519 and selected
collision fraction remained exactly 10.6%. Per held-out world, the model
slightly improved world 7 (75.9% versus 74.1%, no candidate collisions), but
regressed on safety-critical world 48 (67.7% versus 80.0%, with both selecting
colliding candidates in 20%). The validation worlds contained no selected
candidate collisions, so validation top-1 accuracy was an inadequate model-
selection criterion for safety generalization. No checkpoint output is used by
MPPI. The next learning iteration requires more independent training worlds,
a safety-bearing validation split, and risk-aware model selection; it must not
tune on held-out worlds 7 or 48.

The next ranker iteration now has a resumable full-feasible-BARN collector.
The 134 valid worlds are frozen into 99 training, 18 validation, and 17 new
blind-test worlds; invalid A*-unreachable and footprint-collision worlds remain
excluded. Atomic per-world shards make collection restart-safe. Training reads
the split manifest and omits blind-test evaluation by default; a separate
one-time evaluator is provided for use only after architecture, losses,
checkpoint criterion, and risk scoring are frozen.

The safety-aware full-BARN checkpoint is now frozen before blind evaluation.
On 488 validation snapshots it achieved 79.1% top-1 accuracy versus 75.0% for
the geometric promise, reduced mean regret from 0.195 to 0.067, reduced total
selected-collision captures from 7 to 5, and reduced avoidable selected
collisions from 3 to 1. Risk-adjusted selection uses beta=20 and checkpoint
priority is avoidable collisions, then regret, then top-1 accuracy. The frozen
checkpoint SHA-256 is
`d03c53e21ee0c398d5c21783fb89f9845b4b40d88cc0ed1c6785327d37862416`.
Do not retrain, overwrite, or tune this file before the one-time 17-world blind
evaluation.

The one-time blind evaluation passed without retraining. Across 459 multimodal
snapshots from 17 untouched feasible BARN worlds, the frozen risk-adjusted
ranker achieved 387/459 (84.3%) oracle top-1 agreement versus 358/459 (78.0%)
for geometric promise. Mean utility regret fell from 0.377 to 0.0156. Selected
collision-labelled candidates fell from 16 to 9, and all seven cases where the
heuristic selected a colliding candidate despite a safe alternative were
avoided by the ranker (7 to 0). The result is an offline counterfactual ranking
result, not yet closed-loop MPPI or collision-rate evidence. The dataset hash
is `0fe22e9cde6e6630f3e888ed7cb4222177f54105b1a47a83c39c05fc72456d5c`;
the checkpoint and risk coefficient beta=20 remain frozen. This authorizes a
conservative shadow-mode controller integration, followed by a learned-weight
blend ablation against frozen heuristic Amoeba.

The frozen ranker has now been integrated into the independent PyTorch
controller in observation-only shadow mode. It is evaluated only when the
controller is in `ASSIST` and at least two feasible pseudopods exist. Runtime
telemetry records both proposed branch IDs, learned risk-adjusted scores,
collision probabilities, learned-versus-promise agreement, and inference
latency. A command-parity regression test confirms that enabling shadow mode
does not change the MPPI command for an identical controller seed. No learned
score currently affects sampling, costs, mode selection, or actuation. The
next evidence gate is a development-world shadow latency/agreement panel;
closed-loop learned influence remains unauthorized until that panel passes.
The matched panel is implemented as
`python -m torch_port.learning.shadow_panel --device cuda`. It runs paired
baseline/shadow episodes on development worlds, verifies exact state-trace
parity, records controller and ranker timing, and refuses all 17 blind-test
world IDs.

The next closed-loop development ablation is implemented. The existing Torch
behaviour is correctly identified as equal pseudopod allocation; geometric
promise had selected the telemetry comparator but had not weighted samples.
The new policies are `equal`, `geometric`, `blend`, and `learned`. The blend
uses 25% learned and 75% geometric probability, with a 10% uniform floor and
the existing protected 25% fallback budget. Learned outputs affect allocation
only; they do not bypass feasibility, collision costs, MPPI weighting, or final
mode selection. Run
`python -m torch_port.learning.allocation_panel --device cuda` on development
worlds before any hybrid A*-gated evaluation.

The three-seed allocation panel is complete (four development worlds, 12
episodes per policy). All policies achieved 12/12 successes with zero
collisions. The blend did not retain its one-seed advantage: versus equal
allocation it was 0.363 s slower on average and lost 4.6 mm of mean minimum
clearance. Fully learned allocation was only 0.175 s faster, with 0.25 fewer
mode switches and 1.5 mm less clearance. Paired outcomes were mixed, so these
changes are practically small and do not establish a learned closed-loop
benefit. Keep the learned module as a documented exploratory/negative ablation
and shadow option; use equal allocation for the primary hybrid A* benchmark.

Dynamic visualization exposed a global/local coordination failure: periodic
A* replanning could start from an Amoeba detour and adopt its longer corridor.
Hybrid replanning is now frozen during `ASSIST`. A mode-level rejoin prior also
penalizes pseudopods by mean lateral distance to the protected A* path and by
positive remaining-path deficit relative to the A* proposal; the fallback has
zero prior. Defaults are lateral weight 3.0 and remaining-distance weight 2.0.
A matched dynamic trial is still required before accepting this change into
the final benchmark.

The dynamic handoff now activates prospectively. Upcoming global-path points
are assigned arrival times from path arc length and current speed, then tested
against predicted moving-obstacle centres, physical radii, robot planning
radius, and horizon-growing uncertainty. An intersection enters `ASSIST` with
reason `dynamic_path_blocked`, rather than waiting 20 stationary cycles. A
unit test confirms pre-stall activation. The global A* layer now uses static
map clearance only and applies collision-checked Laplacian smoothing after
shortcutting/densification; world 24 shows a reduced maximum heading jump with
positive footprint clearance. `--raw-astar` retains the piecewise-linear
ablation. Final dynamic trials must be rerun because this changes the global
reference used by both vanilla and hybrid controllers.

Transient blockage now has an explicit temporal response. During predicted
dynamic obstruction, the proposal mixture contains a zero-velocity
`dynamic_wait` mode in addition to safe pseudopods and the path fallback. Once
three consecutive predictions report the corridor clear, assistance exits
without requiring the stationary robot to have made progress; the stall
counter and anchor reset before A* tracking resumes.

Plant/plan mismatch re-run at N=8 independent draws per case (was n=1).
`experiments/v6_ablation.py` gained a `--plant-mismatch-draws` flag and a
`draw` index reseeded per-draw via a 4-part seed sequence. Result overturns
part of the n=1 pilot's headline: aggregate success rates still rise under
+-30% skid-parameter mismatch (as the pilot found), but 2 of the 10
previously-100%-reliable baseline-success cases now show an occasional
failure (`blocked` world 48 seed 1: 7/8 success; world 49 seed 0: 7/8
success) -- the pilot's "every case matched or improved; none got worse"
does not hold at N=8. Separately, 6 of the 14 baseline-collision cases show
zero successes across all 8 draws despite clearances just as close to zero
as the cases that do flip -- a structural weak point independent of plant
mismatch, not a knife-edge one, and a candidate scenario set for the
space-time-topology matched panel. See docs/experiments/PLANT_MISMATCH_N8.md.
96/96 tests still pass.

Method renamed Topology-Guided MPPI (TG-MPPI) per professor feedback;
`docs/amoeba_mathematics_guide.tex` fully re-terminologized (flood region/
body, topological branch, flood boundary, narrow-channel weight). Codebase
identifiers and the ROS/Nav2 side are unchanged; that rename is deliberately
deferred, not forgotten.

Added `spacetime_flow.py`: a full (x, y, t) Dijkstra flood (`SpaceTimeFlood`),
not just the two-route search `spacetime.py` already does. It is
`spacetime.time_expanded_search`'s identical grid/cost function with the
early "stop at goal" exit removed, so every reached cell keeps its true
distance -- `G[i, j, k]`, the direct time-axis analogue of
`env.LocalFlowField.D` -- rather than only the one path that happened to
reach one goal. As a byproduct of checking free space directly rather than
one predicted crossing, it also supports any number of simultaneous moving
obstacles, which `time_expanded_search` does not (see that module's own
"only ever selects ONE obstacle" limitation). 9 new tests
(`tests/test_spacetime_flow.py`, 105/105 overall), including a direct
consistency check against `two_route_search`'s own cost on an identical
scenario. Standalone and NOT wired into `spacetime.py`, `mppi.py`, or
`env.LocalFlowField` -- a full flood is `nx*ny*nk` cells versus the static
flood's `nx*ny`, and unlike the static flood it can't be cached for many
cycles since the obstacle predictions it floods against are only valid
until the next re-prediction. Filed as future work ("generalize the
per-branch wait/detour search into a fully-flooded space-time cost field"),
not attempted before the Sept 13 sandbox freeze. Demo figures:
`make_spacetime_flood_figure.py` (now built directly on the module).

UPDATE 2026-09-08 (same day): wired it in anyway ("Option B"), as a
declared experiment, not a production change. `mppi.py` gained
`spacetime_flow=False` (opt-in) rebuilding a `SpaceTimeFlood` every
`spacetime_flow_reflood_every` cycles (default 1) and adding
`spacetime_flow_w * spacetime_flow_cost(xs)` to both cost-computation
paths. `SpaceTimeFlood.dist_batch` added for vectorized K*T queries.
114/114 tests (9 new: 3 `dist_batch`, 6 integration) confirm zero change
when off. Profiled on real `AmoebaHybrid`+`DynaBarnEnv`: flood build
~10-11ms at the sandbox's existing `spacetime_*` defaults, query <1ms for
a full K=1024,T=56 batch; every-cycle rebuild costs ~+9% on the baseline
controller time, every-5th-cycle ~+1%.

First matched panel (6 worlds x 3 seeds, dynamic scenario, `flow_w=1.0`):
mixed and NOT a clean win -- 17/18 vs 18/18 success (one regression:
world 4 seed 1 turned a success into a timeout), clearance delta mean
~0, 12/18 improved vs 5/18 worsened including two large regressions.
Investigated the world-4/seed-1 case directly: not a hard stall -- the
robot oscillates near one y-band for ~700 steps before self-resolving,
consistent with the added cost term (magnitude ~6-14) being comparable
in scale to the existing flow/track guidance cost, which at MPPI's sharp
`lam=0.3` softmax temperature was enough to occasionally flip which
near-tied sample wins, without being informative enough to reliably pick
the better one. Swept `spacetime_flow_w` on that one case: 0.1 recovers
success, 0.25/0.5 still time out. Re-ran the full 18-pair panel at
`flow_w=0.1`: 18/18 vs 18/18 success (regression gone), clearance delta
mean +0.014 (0.124 -> 0.138), median +0.005, 11/18 improved vs 7/18
worsened. Paired Wilcoxon signed-rank test on the 18 deltas: p=0.316 --
directionally positive, NOT statistically significant at this sample
size. Default `spacetime_flow_w` changed to 0.1 to reflect this.
Results: `artifacts/results/milestones/spacetime_flow_matched_panel.csv`
(w=1.0) and `..._w0.1.csv` (w=0.1).

Honest status: real, working, opt-in feature; directionally promising;
not statistically validated; NOT enabled by default; not part of any
banked generality/benchmark result. A larger-N panel (more seeds) would
be needed before this could be reported as an actual improvement rather
than a plausible one.

UPDATE 2026-09-08 (later same day): ran the matched panel that was
actually next on the pre-detour priority list -- `spacetime_modes` (the
discrete, trigger-based wait/detour mechanism, integrated since the
2026-09-06 session, never before benchmarked against real dynamic
scenarios). Same 18 worlds/seeds as the `spacetime_flow` panel; the
`spacetime_modes=False` arm is identical in every other parameter to that
panel's `flow=False` arm, so those 18 baseline rows were reused rather
than re-run (`experiments/spacetime_modes_matched_panel.py`).

Result: 18/18 success both arms (no collisions, no timeouts either way),
and the mechanism is genuinely active -- 2436 triggers and a substantial
number of `spacetime_modes_added` across the panel, not dormant. But
clearance impact is a wash-to-slightly-negative: 6/18 improved, 7/18
worsened, 5/18 exactly unchanged, mean delta -0.0066, median 0.0000.
Wilcoxon signed-rank p=0.195 -- not significant, and unlike `spacetime_
flow`'s slight positive lean, this one leans (insignificantly) negative.
Two moderate regressions worth a closer look if this is picked up again:
world 42 seed 0 (-0.044) and seed 2 (-0.048). Results:
`artifacts/results/milestones/spacetime_modes_matched_panel.csv`.

Combined honest picture for both space-time mechanisms: mechanically
correct, real, well-tested, firing as designed -- and NEITHER shows a
statistically defensible improvement over the plain dynamic-prediction
baseline at N=18. Both are legitimate future-work items with first-pass
evidence attached, not results to cite as validated. Both mechanisms
remain off by default; no existing benchmark or generality-matrix result
is affected.

## Footprint match and 1:1 Gazebo worlds (2026-09-09)

Two consistency fixes ahead of the Gazebo/Nav2 trials, made so sandbox and
Gazebo outcomes are comparable rather than merely similar.

**Footprint.** The sandbox's SLIP rectangle was the provisional
`0.90 x 0.65 m`; the Nav2 costmap footprint the port actually collides
against is `[[0.42, 0.34], ...]` = `0.84 x 0.68 m`
(`susag_nav2/param/navigation_amoeba*.yaml`). The sandbox now defaults to
`0.84 x 0.68` everywhere the old value lived: `RobotModel` (`robot_model.
py`), `MPPI.__init__` (`mppi.py` -- the site the frozen benchmarks actually
used), `cli.py` flags, `experiments/v6_ablation.py` CONFIGS, and the torch
port (`torch_port/model.py`, `torch_port/scan_slip_worlds.py` -- inside the
`torch_port` git submodule, left uncommitted alongside its other
pre-existing uncommitted work). Planning radius becomes `0.34 + 0.08 = 0.42
m`, identical to the Gazebo tiering's half-width. Every test passes
explicit dimensions, so 114/114 still pass; the torch-port suite is
unchanged (28 run, 20 skipped, 3 errors that are `import torch` failing at
module load -- PyTorch is not in system Python -- pre-existing).

**Banked results are records, not re-run.** All SLIP CSVs dated before
2026-09-09 (`v6_ablation*`, `v6_ablation_plant_mismatch*`, the generality/
blocked-split matrices where `robot_model=slip`) were produced at `0.90 x
0.65`; to reproduce them pass `--footprint-length 0.90 --footprint-width
0.65` (or set `v6_ablation.py`'s CONFIGS back). Their numbers and the
conclusions drawn from them are unchanged; only the default for *future*
runs moved. The feasibility screen was re-run at the new footprint into a
separate file (`slip_world_feasibility_0.84x0.68.csv`, see
`docs/experiments/SLIP_WORLD_FEASIBILITY.md`): nine of the ten frozen
development worlds stay suitable, 0 and 48 improve, and **world 7 flips to
footprint collision (-0.028 m on its A* route)**. Treat world 7 as
feasibility-excluded at 1:1 under the measured footprint in Gazebo and
report it separately; the banked world-7 result stands as recorded.

**Gazebo now runs the 1:1 dataset.** `susag_new_model/launch/gazebo_barn.
launch.py` and `susag_nav2/launch/navigation.launch.py` had `BARN_SCALE =
1.2` hardcoded; both are now `1.0`. At x1.2 all 300 BARN worlds are
"comfortable" for this footprint (`BARN_dataset/scaled_1.2/
susag_suitable_worlds.txt`: SAFE 300, TIGHT 0) and the narrow-environment
claim is untestable there -- and on comfortable worlds the branch
mechanism is never selected (0% in the static matrix). At 1:1: SAFE 114,
TIGHT 186. `BARN_dataset/scaled_1` (the 1:1 output of `scale_barn.py`) was
regenerated with the current script so it carries the sealed lateral-wall
cylinders (`0.080 m`, no lidar leakage through seams) that `scaled_1.2`
already had but the July `scaled_1` did not; obstacle poses verified
numerically identical to the raw dataset (max |diff| 0.0 on worlds 7/48/
93), maps carry the 12 planner-padding rows. `benchmark_barn.py`'s `geo()`
now uses `scaled_<s>` for every scale including 1.0 instead of the raw,
unsealed, unpadded root set.

**Gazebo comparison arms had different footprints.** `benchmark_barn.py`
runs `vanilla`/`biased`/`log`/`lowpass` on `navigation_sim.yaml` (was
`0.82 x 0.66`) and `amoeba` on `navigation_amoeba.yaml` (`0.84 x 0.68`).
Both `navigation_sim.yaml` and `navigation_sim_tight.yaml` are now `0.84 x
0.68` so all arms collide against the same rectangle. `navigation.yaml`
(`0.80 x 0.64`, padding `0.08`; presumably the physical-robot profile) was
deliberately not touched -- reconcile it against the measured chassis
before real-robot runs. The 14 existing rows in `susag_nav2/benchmark/
barn_results.csv` are worlds-42/93 smoke runs made with the raw dataset
and the old vanilla footprint; archive that file before new trials so
rows are not silently mixed (the CSV has no footprint column).

None of the ROS-side files are under version control; the pre-edit
versions of all five (and the July `scaled_1`) are in this session's
scratchpad `ros_backup_2026-09-09/`.

## Two lidars as two costmap sources; AMCL on the front only (2026-09-09)

Diagnosing "gets stuck a bit in highly narrow worlds" in Gazebo turned up
a sensor-wiring problem that predates today. The robot carries a front and
a rear 180-degree lidar. The real robot publishes them as `/front_scan` and
`/rear_scan` (`ydlidar_ros2_driver/launch/dual_launch.py`). The earlier
setup fused them with `ros2_laser_scan_merger` into one 360-degree `/scan`
for AMCL and the costmaps; that produced localization drift, so the merger
was disabled and the sim's front lidar was remapped straight to `/scan`,
leaving the rear lidar unused.

The drift was a wiring bug, not the rear lidar: the merger subscribes to
`/front_scan`, but the sim's front lidar had been remapped to `/scan`, so
the merger never received the front lidar at all -- it fused the rear 180
degrees alone into a "360-degree" scan and republished it on `/scan`,
interleaved with the raw front `/scan` on the same topic. AMCL was fed
two geometrically different scans in alternation. (Its config also has
`range_min: 0.35`, which blanks anything nearer than 35 cm.) The costmap
yamls' second observation source pointed at `/base/scan`, a topic that
exists on neither the sim nor the real robot -- dead everywhere -- and in
`navigation_amoeba_tight.yaml` it was not even listed in
`observation_sources`. With reversing allowed (`vx_min: -0.35`), MPPI could
back into space the front lidar had never observed.

Nav2's costmap obstacle layer accepts any number of observation sources,
each transformed through TF individually; only AMCL is limited to a single
`scan_topic`. So no fusion is needed for either job. Changed, in all four
sim/campaign yamls (`navigation_amoeba_tight`, `navigation_amoeba`,
`navigation_sim`, `navigation_sim_tight`) so every comparison arm is
identical: AMCL `scan_topic: front_scan`; global and local obstacle layers
`observation_sources: front_scan rear_scan` with topics `/front_scan` and
`/rear_scan`. The sim's front lidar is back on `/front_scan`
(`susag_new_model/urdf/susag_updated_model.gazebo`, symlink-installed, live
at next launch), matching the real robot's topic names so one yaml serves
both. RViz shows both scans (`susag_nav.rviz`); `benchmark_barn.py` and
`benchmark_amoeba.py` take their collision proxy from both lidars. Not
touched, still on `/scan`: `navigation.yaml` (real-robot profile) and
`test_config.yaml`, and `slam.yaml` (mapping) -- apply the same change to
the real-robot profile before hardware runs. Pre-edit copies of every file
are in the session scratchpad `ros_backup_2026-09-09/`.

Separately noted for the Gazebo trials, not changed: the progress checker
(`required_movement_radius: 0.5` in `movement_time_allowance: 10.0`) fails
the controller on a slow bottleneck crawl, after which the default BT runs
`recoveries_server`'s Spin / BackUp / Wait -- a Spin with a 0.84 m robot in
a ~0.8 m gap is physically impossible; `inflation_radius: 0.45` with
`cost_scaling_factor: 6.0` puts a 0.399 m half-gap entirely at cost ~200,
which is what makes the crawl slow; and the Gazebo diff-drive plugin's odom
is world-referenced (set `<odometry_source>1</odometry_source>` explicitly),
so a static identity `map->odom` in place of AMCL is the clean localization
A/B and the condition that matches the planned Vicon-based real robot. The
15:35 Gazebo run today was made while the installed launch still loaded the
x1.2 world against the 1:1 map; only runs from 15:38 on are valid.

## RViz shows every TG-MPPI output; rename is presentation-only (2026-09-09)

`susag_nav2/rviz/susag_nav.rviz` (symlink-installed, live) previously had
none of the plugin's outputs -- its "Trajectories" display even pointed at
`/marker`, a topic this plugin never publishes. It now has a "TG-MPPI"
group with everything `optimizer.cpp` / `trajectory_visualizer.cpp`
publish: the `/amoeba_debug` MarkerArray with all eight namespaces
enabled (`amoeba_body`, `amoeba_membrane`, `amoeba_membrane_halo`,
`amoeba_pseudopods`, `flow_dirs`, `amoeba_scan`, `amoeba_mode`,
`amoeba_state`), the three branch reference Paths
(`/amoeba/ancillary_path_1..3`) and three validated-rollout Paths
(`/amoeba/ancillary_rollout_1..3`) in the sandbox's branch colours
(`visualization.BRANCH_COLORS`), the tracked global plan
(`/transformed_global_plan`) in the sandbox's A*-path violet, and the
sample cloud retargeted to `/trajectories`. The stale turtlebot camera
displays now point at the sim camera's real topics (`/image_raw`,
`/susag/depth/depth_camera/points`); "Bumper Hit"
(`/mobile_base/sensors/bumper_pointcloud`) is left as harmless clutter.

Decision (user): the Amoeba -> TG-MPPI rename is **presentation-only**
for now -- RViz display/group names, README and docs prose (done today
for the README's Nav2 section: flood body / flood boundary / topological
branches, with the old term in parentheses where a topic or namespace
still carries it). ROS topics, parameters, marker namespaces, the
`nav2_amoeba_mppi_controller` package and `AmoebaController` class, and
all sandbox identifiers stay as they are until after the Gazebo campaign
and the freeze (plugin: 57 files / 768 occurrences, not under git, C++
rebuild required; sandbox: 342 occurrences plus the repository name).

Why the RViz membrane looks octagonal while the sandbox's looks smooth:
both compute the same thing. `env.py::_geodesic_ball` and
`flow_field.cpp` both run an 8-connected Dijkstra with `res` /
`res*sqrt(2)` edge costs, whose ball in open space is an octagon (the
chamfer metric). The sandbox's `visualization._draw_field` Gaussian-blurs
the body mask (`sigma=1.2`) before contouring at 0.5 -- its own comment
calls this "purely cosmetic, doesn't touch the underlying body/flood" --
while the C++ side draws the raw membrane cells as a `SPHERE_LIST`. The
RViz picture is the honest geometry. Options, none taken: replicate the
blur + marching-squares outline in C++ (cosmetic, post-freeze), or a
16-neighbourhood flood (changes distances, would need re-validation).

**Correction, same day: the `/amoeba_debug` display stalls the controller
while enabled.** Reported as "robot gets stuck when the TG sampling view
is on, moves smoothly when it's off". Cause, from `optimizer.cpp`:
`amoeba_debug: true` makes `publishFlowDebug()` run every control cycle
(call sites in the per-cycle NORMAL/ASSIST paths, not only on reflood);
its only cost gate is `get_subscription_count() == 0`. With no RViz
subscriber it returns immediately. With one, every 50 ms cycle it walks
the whole flow-field grid, pushes ~31,000 SPHERE_LIST points for the body
(one per 0.025 m cell of a 2.5 m disc), copies the membrane again for the
halo, and serializes/publishes ~0.75 MB inside the control loop at 20 Hz,
while RViz rebuilds 31k sphere instances at 20 Hz on the same machine.
The config had no `/amoeba_debug` display before 2026-09-09, so the gate
had always kept this free; adding the display enabled exposed it. The
`/trajectories` sample cloud is not the cause (~4k line points, and it is
built regardless of subscribers). Fix applied: the display now defaults
to off and is labelled as heavy; toggle it on for screenshots. Proper fix
(C++, post-campaign): publish the flood markers only on reflood cycles,
decimate the body (stride 2-3) or use POINTS instead of SPHERE_LIST, drop
the halo copy, and ideally build markers off the control thread.

Also, for the RViz "No map received" warning on the local costmap:
`always_send_full_costmap: true` added to the local costmap in all four
campaign yamls (it was set only on the global one). A rolling costmap
otherwise publishes only updates once its origin stops moving, and a
display that subscribes mid-run can be left waiting for a full grid.
Publishing-only; 200x200 cells at 2 Hz. If the warning persists, check
publisher QoS against the display (`ros2 topic info -v
/local_costmap/costmap`: a Volatile publisher is incompatible with the
display's Transient Local request) and whether the topic is publishing at
all (`ros2 topic hz`). Unverified and worth checking with `ros2 topic
list | grep points`: the depth-camera topic the `stvl_local` layer and
the RViz depth display use (`/susag/depth/depth_camera/points`) may
actually be `/susag/depth_camera/points` given the plugin's namespace and
remap; if so both need the corrected name.

**Confirmed, same day: the depth camera topics were never right.**
gazebo_ros_camera remap keys must be prefixed with the camera name (the
shipped `gazebo_ros_depth_camera_demo.world` says so explicitly); the
`.gazebo` file's `~/image`, `~/depth_image`, `~/points` keys matched
nothing, so all four remaps were silently ignored and the plugin has
always published at its defaults under the `susag` namespace:
`/susag/depth_camera/image_raw`, `/susag/depth_camera/depth/image_raw`,
`/susag/depth_camera/points` (+ camera_info topics). Every consumer
pointed elsewhere: the yamls' `stvl_local` and global `pointcloud` layers
at `/susag/depth/depth_camera/points` (an extra `depth/`), so the
spatio-temporal voxel layers have never received a cloud and the
1280x720 @ 30 Hz camera has been pure simulation load; and RViz's camera
displays at old turtlebot names, then (my retarget) at the yaml's wrong
name -- which is why no camera image was ever visible. Fixed: the dead
remaps replaced by a comment stating the real topics (no functional
change), all four yamls' pointcloud sources and the RViz image/cloud
displays on the real names, a depth-image display added, and the RViz
camera displays set to Best Effort so they match the plugin regardless of
which QoS it publishes with. Not changed: whether to keep the STVL layers
at all for BARN (2-D pillars, nothing above lidar height) or drop the
camera resolution/rate -- a campaign-condition decision; note that from
now on the voxel layers will actually mark, which they never did before.

**rviz2 crashes when a camera display is enabled (2026-09-09).** The
kernel log shows five rviz2 segfaults today. Four -- at 15:37, 15:42,
16:14 and 16:26, including the two launch *shutdowns* when no camera
display was being touched -- fault at the identical instruction in
`librclcpp.so` (`ip ...e3b31`, `error 4`); the fifth (16:43) in `libc`.
The session is x11 on an NVIDIA RTX 4060 (driver 580), so this is not
the Wayland/GPU class of rviz2 crash. It is the ROS-client-library
subscription-teardown class: toggling a display creates or destroys a
subscription, and on a 30 Hz, multi-MB topic (the camera image/cloud
that only began publishing to a subscribed topic after today's topic
fix) the race is hit deterministically. Installed: rviz2 11.2.28,
rclcpp 16.0.19, rmw_fastrtps_cpp 6.2.10 (the default RMW), and
rmw_cyclonedds_cpp 1.3.4 is present. Known related reports:
[ros2/rviz#703](https://github.com/ros2/rviz/issues/703) (rviz2 randomly
crashes with the Nav2 stack), [ros2/rclcpp#2437](https://github.com/ros2/
rclcpp/issues/2437) (segfault in `RMWSubscriptionEvent::
update_data_available()` with Fast DDS), and
[ros2/rviz#574](https://github.com/ros2/rviz/issues/574) (the costmap
footprint Polygon display crashes rviz2; this config has the local one
enabled). Mitigations: the three camera displays now default to off (a
first version of that edit put an unquoted `: ` in the display names and
broke the YAML -- fixed); view images with `ros2 run rqt_image_view
rqt_image_view /susag/depth_camera/image_raw` (installed, separate
process); and, per Nav2's own Humble guidance, run the whole stack on
CycloneDDS (`export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` before every
launch, all terminals and the benchmark alike -- all nodes must share
the RMW). Not applied: switching RMW is a campaign-wide condition the
user should set deliberately; if crashes persist under CycloneDDS,
disable the local-costmap Polygon display next (#574).

## Stock nav2_mppi_controller as the "normal MPPI" comparator (2026-09-09)

`~/robohouse_ws/src/nav2_mppi_controller` is a local checkout/build of the
Nav2 MPPI controller (built at `install/nav2_mppi_controller/lib/
libmppi_controller.so`, not the apt package), forked to add an
`mppi_variant` param (`vanilla | log | lowpass | biased`) selecting
between plain i.i.d. Gaussian sampling and three research variants --
`noise_generator.cpp` labels `vanilla`'s branch "i.i.d. zero-mean Gaussian
perturbations", i.e. genuinely the standard MPPI, not a relabeled custom
behavior. `navigation_sim.yaml`/`navigation_sim_tight.yaml` run `FollowPath`
as this controller with `mppi_variant: "vanilla"`; `navigation_amoeba*.
yaml` run it as `nav2_amoeba_mppi_controller::AmoebaController`. These are
the intended vanilla-vs-TG-MPPI comparator pair, already used that way by
`benchmark_barn.py`'s `CONTROLLERS` dict.

They were not actually parameter-matched. A structural (YAML-parsed, not
grep) diff of `navigation_amoeba_tight.yaml` vs `navigation_sim_tight.yaml`
-- the pair `navigation.launch.py` actually launches by default -- found,
beyond the deliberate `plugin` and `FlowFieldCritic`-in-`critics` (the
"additional parts from the controller we created") differences:
`vx_max` 0.35 vs 0.3, `vx_std` 0.3 vs 0.2, `wz_std` 0.7 vs 0.3,
`PathAlignCritic.cost_weight` 6.0 vs 3.0, `PathAngleCritic.cost_weight`
5.0 vs 2.0, `PreferForwardCritic.cost_weight` 1.0 vs 2.0. `vx_max`/`vx_std`/
`wz_std` in particular would have let vanilla explore a different speed
envelope than TG-MPPI, confounding any comparison. (A naive flat-key grep
across the whole `FollowPath` block is unsafe here -- `cost_weight` and
`consider_footprint` each appear under several differently-named critics,
so a bare-key diff can compare two unrelated critics' values; the fix used
PyYAML and diffed per named critic block.) All six values in
`navigation_sim_tight.yaml` set to match `navigation_amoeba_tight.yaml`;
re-diffed after, zero remaining differences besides `plugin` and the
critics list. Local costmap resolution/size/rates and the progress/goal
checkers were already identical between the two files.

`navigation_amoeba.yaml` vs `navigation_sim.yaml` (the non-tight pair,
used by `benchmark_barn.py` directly rather than the interactive launch)
has the same class of drift -- `vx_max` 0.5 vs 1.0, `wz_std` 0.4 vs 0.6 --
**not yet fixed**; do the same structural-diff-and-match pass before
running `benchmark_barn.py`'s `vanilla` arm for a real result.

To view the stock controller live: same launch, override the params file
(the vanilla controller still needs the front/rear-lidar-fed local
costmap and the matched footprint, both already in this file):

```bash
ros2 launch susag_nav2 navigation.launch.py sim:=true world_idx:=48 \
  nav2_params:=/home/saran/robohouse_ws/src/susag_nav2/param/navigation_sim_tight.yaml
```

The RViz TG-MPPI group's `/amoeba_debug`, `/amoeba/ancillary_*` displays
will show nothing under this controller (it never publishes them) --
expected, not a fault; `/trajectories` (the sample cloud) and
`/transformed_global_plan` are published by both controllers and remain
meaningful for either.

## Ground-truth pose plugin, and a stale-install lesson repeated (2026-09-10)

Reported symptom: laser scans still drift specifically when turning, even
after the front/rear lidar rewiring. Ruled out one candidate directly:
`wheel_separation: 0.74` in the diff_drive plugin (declared twice,
redundantly, but with identical values) exactly matches the real geometry
(URDF wheel joints at y=+-0.37 m) -- not a mismatch.

Two candidates remain, and they need different fixes, so added
`libgazebo_ros_p3d` to `susag_new_model/urdf/susag_updated_model.gazebo`
on `base_link`, `frame_name: world`, zero noise, publishing
`/ground_truth/odom` -- Gazebo's actual simulated rigid-body pose,
independent of whatever the drive plugin assumes about wheel slip.
Verified live: publishes at ~23.6 Hz, first message
`x=-2.2500, y=1.0000` in `frame_id: world` -- an exact match to the
robot's spawn pose (`gazebo_barn.launch.py`'s `x_pose=-2.25, y_pose=1.0`).

**How to use it:** launch normally, send a pure rotation
(`ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist
"{angular: {z: 0.5}}"` for a few seconds, then stop), and compare `/odom`
against `/ground_truth/odom` over that window.
- If they track together: any observed drift is the TF/`use_sim_time`
  clock mismatch flagged earlier this session (`navigation.launch.py`'s
  `sim` arg still defaults to `false`) -- fix is always passing
  `sim:=true`.
- If they diverge specifically during rotation: Gazebo is genuinely
  simulating skid-steer slip. This chassis has no steering DOF (all four
  wheel joints are `continuous`, axis `(0,1,0)` only -- confirmed in the
  URDF) and turns purely by scrubbing the wheels against the ground;
  `libgazebo_ros_diff_drive` publishes `/odom` from ideal no-slip
  kinematics, so a real physics-simulated slip during rotation would not
  match it. This is not a bug to patch -- it is the same phenomenon
  `robot_model.py`'s `yaw_gain`/`speed_yaw_loss` already models for the
  real robot (see V6, `docs/PROJECT_STATUS.md`'s 2026-08-03 entry); the
  remediation is a design choice (accept it as realistic, or switch to a
  kinematic drive plugin such as `libgazebo_ros_planar_move` if a
  slip-free baseline is wanted for controller-only testing), not a
  one-line fix.

**Stale-install lesson, repeated from 2026-09-09's launch-file incident:**
the installed copy of this exact `.gazebo` file was a **plain file, not a
symlink** (dated the previous day, missing this session's own edit)
despite `susag_updated_model_description` having been rebuilt with
`--symlink-install` on 2026-09-09 for an unrelated fix (the Gazebo-world
scale). That rebuild evidently symlinked the file it was explicitly
checked for (the launch file) but left this one a stale copy from before
this session started -- colcon's symlink-install does not necessarily
relink every file in a package on a partial rebuild if it considers the
destination already present. Re-ran
`colcon build --packages-select susag_updated_model_description
--symlink-install`; confirmed the full chain is now `install -> build ->
src` (a real symlink at every hop, not just the endpoints) via
`readlink -f`, and confirmed the resolved content matches source via
`grep`. **Any future edit to a file in this package should have its
install-space symlink verified with `readlink -f`, not assumed** -- a
partial rebuild is not proof every file in the package is live.

## 2026-09-10: nav2_amoeba_mppi_controller cleanup + full amoeba->tgmppi rename

User audit request ("how far and what features from the sandbox are
implemented here") led to two follow-up asks: strip out the controller's
previous-implementation cruft, and rename the whole nomenclature from
"amoeba" to "tgmppi" while already touching the code. Both are done and
build-verified.

**Safety first**: neither `nav2_amoeba_mppi_controller` nor `susag_nav2`
had ever been under version control. Before any edit, `git init` +
baseline commit in both, so the rename/deletion has a real revert path.

**Build-crash lesson (own mistake)**: the very first rebuild attempt ran
unthrottled `colcon build` on this 20-core/15GB-RAM/**zero-swap** machine.
Default parallelism (up to 20 concurrent xtensor/-O3/-ffast-math compiles,
each GB-hungry) exhausted RAM with nothing to page to -- full system
lockup, user had to hard-reset. Every build after that used
`MAKEFLAGS="-j1"` (or `-j2` for the lightweight config-only `susag_nav2`)
+ `colcon --parallel-workers 1`, run backgrounded under a `free -m`
watchdog that kills the build outright if available memory drops toward
~1.5GB. No further incidents; **this machine should never again run a
colcon build without an explicit job cap**.

**Cleanup (dead code removal, zero behavior change)**: both live campaign
yamls ran `amoeba_mode: "flow"` exclusively, so the alternate `"ray"`
polar-gap-scan mode (`computeAmoebaModes()`, `applyAmoebaBias()`,
`buildWrapSequence()`, `amoebaCostAt()`, `publishAmoebaDebug()`, and all
`amoeba_seq_*`/`amoeba_path_*`/`amoeba_bearing_*`/`dbg_ray_*` state) was
live but unreachable -- deleted. `PathAlignLegacyCritic` was registered in
pluginlib but absent from both active `critics:` lists -- deleted
(`.cpp`/`.hpp`, `CMakeLists.txt` line, `critics.xml` entry). Verified with
a capped `colcon build`, clean exit 0, 2min 11s.

**Rename**: package `nav2_amoeba_mppi_controller` -> `nav2_tgmppi_controller`
(dir + include/ subdir moved with `git mv`), namespace `amoeba::` ->
`tgmppi::`, class `AmoebaController` -> `TgMppiController`, all
`amoeba_*` params -> `tgmppi_*`, topics `/amoeba_debug` -> `/tgmppi_debug`
and the three `/amoeba/ancillary_{path,rollout}_N` pairs -> `/tgmppi/...`,
CMake lib targets `amoeba_controller`/`amoeba_critics` ->
`tgmppi_controller`/`tgmppi_critics`. Applied by a scripted 4-rule
case-variant substring replace (`amoeba_mppi`->`tgmppi` first, to collapse
the compound package-name token, then `AMOEBA`/`Amoeba`/`amoeba` ->
`TGMPPI`/`TgMppi`/`tgmppi`) across all 55 source files. The script's first
pass missed one thing: header guards use the *uppercase* `AMOEBA_MPPI`
compound (`NAV2_AMOEBA_MPPI_CONTROLLER__...`), which only the lowercase
collapse rule was written for -- produced `NAV2_TGMPPI_MPPI_CONTROLLER__`
(doubled MPPI) until a follow-up pass fixed it. Verified zero `amoeba`
(case-insensitive) remains in any tracked file, then a capped rebuild with
`-Wall -Wextra -Wpedantic -Werror`: exit 0, 2min 12s, zero warnings.

**Downstream (`susag_nav2`)**: `navigation_amoeba.yaml` ->
`navigation_tgmppi.yaml`, `navigation_amoeba_tight.yaml` ->
`navigation_tgmppi_tight.yaml`, `benchmark_amoeba.py` ->
`benchmark_tgmppi.py` (all `git mv`, history preserved); `plugin:` field,
every `amoeba_*` param key, and RViz topics/marker-namespace toggles
updated to match. Also fixed the now-stale
`navigation_amoeba*.yaml` cross-reference comments left in
`navigation_sim.yaml`/`navigation_sim_tight.yaml` from the earlier
parameter-matching pass, and `benchmark_barn.py`'s controller registry.
Hit the **same stale-symlink trap** documented above: after the file
rename, `install/susag_nav2/share/susag_nav2/param/` still had symlinks
named `navigation_amoeba*.yaml` pointing at now-nonexistent `src/` paths
(a `--symlink-install` rebuild adds new symlinks for new files but does
not remove orphaned old ones) -- found via `find -xtype l` and deleted by
hand.

**Deliberately left alone**: `benchmark/logs/*amoeba*.log` (historical
run records -- renaming them would misrepresent what was actually run
under the old name); the Python sandbox's own "Amoeba" naming
(`amoeba_controllers.py`, `AmoebaHybrid`, etc.) -- separate, already-
published research code, explicitly out of scope for this rename.

**Not started**: the GPU (LibTorch/CUDA) port of the live controller,
discussed and scoped earlier this session but superseded by this cleanup
request. Still the agreed next step if the user wants to continue down
that path -- a genuinely separate, multi-day C++/CUDA undertaking, not
something to start opportunistically at the tail of this session.

## 2026-09-10 (continued): GPU port slice 1 -- NoiseGenerator/rollout, with real numbers

User pushed nav2_tgmppi_controller + susag_nav2/susag_bringup/susag_teleop
(the whole ~/robohouse_ws/src/ tree) to a new home,
https://github.com/SaranSundar08/SLIP -- this is now the canonical remote
for all of Saran's own ROS packages (separate from this amoeba_sandbox
repo, which stays Python-sandbox + docs only). Then: "let's start" on the
GPU port, reordering it ahead of homotopy-consistency cost / report per
an explicit priority decision.

**Toolchain proof (before touching any controller code)**: LibTorch
2.7.1+cu126 (cxx11-ABI) installed to `~/libtorch_cu126/libtorch`, matching
this machine's `cuda-toolkit-12.6` (nvcc at `/usr/local/cuda-12.6/bin/nvcc`,
not on PATH) and driver 580.178.04 (CUDA-13-capable). A tiny standalone
CMake project confirmed a real CUDA matmul running on `cuda:0`
(RTX 4060, auto-detected sm_89) before any production code changed.

**Design: additive knob, not a rewrite-in-place.** Added
`compute_backend: "cpu"|"cuda"` to `OptimizerSettings` (default `"cpu"`,
always available). The existing xtensor CPU path is completely
unchanged and untouched by default; a new `GpuRollout` class
(`tools/gpu_rollout.hpp`/`.cpp`) implements
`Optimizer::updateStateVelocities()` + `Optimizer::integrateStateVelocities()`
(the diff-drive predict + cumulative-kinematics integration -- the K x T
hot loop) in LibTorch, selected at runtime by `compute_backend`. Compiles
to an empty translation unit unless the package is built with
`-DTGMPPI_WITH_CUDA=ON` (new CMake option, OFF by default) --
**no LibTorch dependency exists at all for a normal build.**
`generateNoisedTrajectories()` branches to the GPU path only when
`compute_backend=="cuda"` AND a CUDA device was actually found at
`reset()` time; any mismatch (built without CUDA support, or no GPU at
runtime) logs a warning and silently falls back to `"cpu"` -- the
controller can never fail to start over this.

Deliberately NOT moved to GPU in this slice: noise generation (stays
xtensor/CPU -- cheap, and keeping it CPU-side means the *same* noise
draw feeds both backends, which is what makes the parity test below
possible) and the TG-MPPI flow bias (`applyFlowBias()` -- small, serial,
per-mode logic that reads/writes the same xtensor `state_.cvx/cwz`
critics still consume; nothing to gain from moving it before critics
themselves move).

**Verified, not just written:**
- Default build (`TGMPPI_WITH_CUDA` unset): unchanged, clean, 0 warnings,
  2min11s -- confirms zero regression risk to what's actually driving the
  robot today.
- `-DTGMPPI_WITH_CUDA=ON` build: compiles and links `libtgmppi_controller.so`
  against LibTorch cleanly, 0 compiler warnings, 2min49s (only cosmetic
  upstream LibTorch CMake warnings, e.g. missing NVTX3).
- **Numeric parity**: a standalone test (compiles the *real*
  `gpu_rollout.cpp` directly, not a reimplementation) feeds identical
  random noise into both the xtensor CPU formula and `GpuRollout`, across
  4 (batch, timesteps) shapes including the real K=2000/T=56. Max
  abs-diff on x/y/yaw ~1e-7 (float32 cumsum accumulation-order noise,
  nothing more); vx/wz bit-identical. All 4 cases PASS.
- **Honest timing** at the real shape (K=2000, T=56, 200-iteration
  average, warm CUDA context): CPU(xtensor)=1.980 ms/cycle,
  GPU(libtorch, incl. upload+download)=2.987 ms/cycle -- **0.66x, GPU is
  slower for this slice alone.** Expected and stated up front: a K=2000x56
  tensor is small enough that PCIe transfer + kernel-launch overhead isn't
  amortized by predict+integrate's own compute, which is trivial for AVX2
  to chew through already. The real payoff requires the critics (the
  actual bulk of per-cycle cost) to also run on GPU, so the upload/download
  happens once per cycle instead of being pure overhead on top of an
  otherwise-unmoved CPU pipeline. Next slice: pick one critic (likely
  `obstacles_critic` or `cost_critic`, the two used for every trajectory
  regardless of TG-MPPI mode) and port it the same way (additive class,
  same parity-test discipline) before deciding whether the end-to-end
  number justifies continuing to the rest.

Build command for the CUDA path (not wired into any launch file --
opt-in only):
```
colcon build --packages-select nav2_tgmppi_controller --symlink-install \
  --cmake-args -DTGMPPI_WITH_CUDA=ON \
    -DCMAKE_PREFIX_PATH=$HOME/libtorch_cu126/libtorch \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
```
`$HOME/libtorch_cu126/libtorch/lib` must be on `LD_LIBRARY_PATH` at
launch time too (runtime dlopen dependency of the plugin `.so`).

## 2026-09-10 (continued): GPU port slice 2 -- CostCritic, and an architectural pivot

**First correction before starting**: user asked to port `ObstaclesCritic`.
It's commented out in both live campaign yamls -- `CostCritic` does the
same job (near-identical `costAtPose`/`inCollision` structure) and is
what's actually running. Ported `CostCritic` instead; `ObstaclesCritic`
stays CPU-only and untouched (not in the active critics list either way).

**Design, same additive pattern as slice 1**: added
`const std::string & compute_backend` to `CriticData` (propagated from
`OptimizerSettings::compute_backend`), so any critic can see which
backend is active without a new per-critic param. New `GpuCostCritic`
class (`tools/gpu_cost_critic.hpp`/`.cpp`, same `#ifdef TGMPPI_WITH_CUDA`
empty-TU-by-default pattern as `GpuRollout`) implements
`CostCritic::score()`'s **circular-mode** collision check only
(`consider_footprint:false` -- the active yaml setting); footprint mode
needs per-point polygon rasterization (`footprintCostAtPose`), doesn't
vectorize the same way, and isn't attempted here.  `CostCritic::score()`
checks `data.compute_backend=="cuda" && !consider_footprint_ &&
gpu_critic_.ready()` and only then calls the GPU path; otherwise the
original CPU loop is untouched.

The CPU version's early-break-on-first-collision doesn't need
reproducing on GPU: a colliding trajectory's cost gets unconditionally
overwritten with `collision_cost` regardless of what partial sum
preceded the break, so the GPU version sums every point unconditionally
(a batched costmap gather + elementwise ops) and applies the same
override afterward -- verified equivalent by the parity test, not just
argued.

**Verified:**
- Default build: unchanged (this file's contents are `#ifdef`'d out
  entirely without the flag).
- `-DTGMPPI_WITH_CUDA=ON`: **caught a real bug** on first attempt --
  `-Werror=unused-but-set-variable` on a leftover `cuda_opts_f32` that
  only exists in this code path, so the earlier default-build regression
  check couldn't have caught it. Fixed, rebuilt, clean, 0 warnings
  (only the same two cosmetic upstream LibTorch CMake warnings as
  slice 1), 3min4s. **Lesson reinforced: "the default build passed"
  proves nothing about the CUDA path -- always rebuild both configs
  after touching anything under a TGMPPI_WITH_CUDA guard.**
- Numeric parity: real `gpu_cost_critic.cpp` vs a hand-verified xtensor
  reference, on a synthetic 200x200 costmap (matching the real 5m local
  costmap at 0.025m resolution) with realistic free/inflated/lethal/
  unknown cost distribution. 4 cases (varying `track_unknown_space` and
  `near_goal`), **exact match (0.0 diff)** -- this critic's math is pure
  integer costmap values, no floating-point accumulation-order noise
  like the rollout's trig/cumsum had.

**Honest timing** at the real shape (K=2000, T=56, 200x200 costmap,
200-iteration average): CPU(xtensor)=0.408 ms/cycle,
GPU(libtorch)=0.943 ms/cycle -- **0.43x, GPU slower again.**

**This is now a pattern, not a fluke, and it changes the plan.** Slice 1
(rollout): 0.66x. Slice 2 (CostCritic): 0.43x. Both losses have the same
root cause: each piece does its *own* independent upload -> compute ->
download round trip, and for a K=2000x56 workload the PCIe transfer +
CUDA kernel-launch overhead (each launch ~5-20us, several per op) swamps
compute that AVX2 already does cheaply. Porting a third critic the same
way would just add a third data point confirming the same limitation --
**the round-trip overhead compounds with every additively-ported piece
instead of amortizing.** The actual fix is architectural: stop crossing
the CPU<->GPU boundary per critic. Upload the noised+biased state (+
costmap, path) once at the top of `Optimizer::optimize()`'s cycle, run
the rollout AND every critic's cost chained together on tensors that
stay resident on the GPU the whole time, download the final softmax
result once at the bottom. Presented this finding + the three options
(consolidate now / port more critics first then consolidate / stop here)
to the user explicitly rather than silently continuing the losing
pattern -- **decision: consolidate now.** `GpuRollout`'s and
`GpuCostCritic`'s math get reused as the compute kernels inside the
consolidated pipeline, not thrown away; what changes is who owns the
upload/download (a new persistent-tensor session spanning the whole
cycle, not each piece for itself). Not yet designed in detail or
started -- next actual step.

## 2026-09-10 (continued): GPU port -- consolidation, real result

Refactored `GpuRollout` and `GpuCostCritic` to separate upload/download
from pure device-resident compute: each gained a `computeDevice()`
(rollout) / `computeDevice()`+`uploadCostmap()` (cost critic) method that
takes/returns already-on-GPU tensors, doing zero host<->device traffic
itself. The original `rollout()`/`score()` methods are now thin
upload -> computeDevice -> download wrappers around these, so every
existing parity test (slices 1 and 2) still exercises identical behavior
unchanged -- re-ran both after the refactor, still exact/near-exact
match, confirming no regression from the restructuring itself.

**Design that avoided inventing an unnecessary class**: first drafted a
combo `GpuCycle` orchestrator owning both sub-components, then realized
`CriticData` already threads shared resources (costmap, flow_field) to
critics every cycle -- so instead just added
`const void * gpu_rollout` there (opaque pointer, cast to
`const GpuRollout*` only where `TGMPPI_WITH_CUDA` is defined; deleted the
`GpuCycle` files as unnecessary before they were ever built). `Optimizer`
sets it to `&gpu_rollout_` after running the rollout on GPU (nullptr
otherwise, including on the default build). `CostCritic::score()`'s GPU
branch now checks it: if non-null, calls
`gpu_critic_.computeDevice(rollout->trajX(), rollout->trajY(), ...)`
directly -- reading the rollout's already-resident trajectory tensors,
uploading only its own costmap -- instead of re-uploading trajectories.x/y
itself. This design generalizes: any future GPU-ported critic reads the
same shared `data.gpu_rollout` pointer, so the "one shared upload" benefit
keeps compounding as more critics join, with no combinatorial
orchestrator class needed per new pairing.

**Verified**: both build configs clean (default unaffected; CUDA build
3min3s, only the same two cosmetic upstream LibTorch warnings as before).
A new standalone test chains `computeDevice()` calls exactly as
`CostCritic::score()`'s new branch does, confirms exact parity against a
composed CPU reference (max traj diff 1.19e-7, exact repulsive-cost
match), and runs the real 3-way timing comparison at K=2000/T=56/200x200
costmap:

| | ms/cycle | vs CPU |
|---|---|---|
| A) CPU only (rollout+CostCritic) | 2.016 | 1.00x |
| B) independent GPU (2 round trips, slices 1+2 as built) | 4.421 | 0.46x |
| C) chained GPU (1 round trip, this consolidation) | 3.748 | 0.54x |

**Consolidation works exactly as predicted** -- C is a real, measured
1.18x improvement over B, from removing exactly one redundant upload
(trajectories.x/y, already resident from the rollout). It is **still
behind CPU** with only two pieces sharing the one round trip; the
fixed cost of even a single upload+several-kernel-launches+download
isn't yet amortized by just this much chained compute. The trend is the
right one for the hypothesis in the previous entry: each additional critic
chained onto the same shared `data.gpu_rollout` tensors adds real compute
with ZERO additional round-trip cost, so the ratio should keep improving
as more critics join -- unconfirmed until an actual third critic is
chained, not assumed from two data points. That's the next concrete step
if this is continued: pick one more of the active critics (ConstraintCritic
or GoalCritic look like the next-simplest -- small, dense elementwise
math, no per-point costmap lookup) and chain it onto the same
`data.gpu_rollout` tensors the same way, to see whether 3 pieces crosses
the line CPU still holds at 2.

## 2026-09-10 (continued): GPU port -- 3 chained pieces cross over CPU

Chained `GoalCritic` onto the shared `data.gpu_rollout` tensors the same
way as `CostCritic`: pure elementwise distance-to-goal math over
already-rolled-out trajectories, no costmap, so this critic needs **no
upload of its own at all** when chained -- `goal_x`/`goal_y` are two
scalar kernel arguments, not tensors -- only a `[K]` download of its
result. `GpuGoalCritic` (`tools/gpu_goal_critic.hpp`/`.cpp`, same
`computeDevice()`/standalone-`score()` split as the other two) added;
`GoalCritic::score()` checks `data.gpu_rollout` exactly like
`CostCritic::score()` does.

**Verified**: both configs clean (default unaffected; CUDA build 3min24s,
zero compiler warnings this time -- not even the earlier cosmetic
LibTorch CMake ones). A new 3-critic standalone test (rollout +
CostCritic + GoalCritic, all chained through one upload/download) passes
parity exactly (repulsive-cost diff 0.0, goal-cost diff 5.72e-6) and
measures the real question:

| | ms/cycle | vs CPU |
|---|---|---|
| A) CPU only (3 pieces) | 3.985 | 1.00x |
| B) independent GPU (3 round trips) | 5.238 | 0.76x |
| C) chained GPU (1 round trip) | 3.915 | **1.02x** |

**Crossed over.** Three chained pieces sharing one round trip is
essentially at parity with CPU (slightly ahead, 1.02x) -- confirming the
hypothesis from the previous two entries exactly: consolidation's fixed
cost doesn't grow with each additional critic chained onto the same
resident tensors, but the compute does, so the ratio climbs every critic
added (0.54x at 2 pieces -> 1.02x at 3). This is genuinely the strongest
result all session and the actual proof the redesign was worth doing --
not just directionally right (slice-2-to-consolidation already showed
that) but now demonstrably ahead of CPU with real, verified numbers.

**Not yet true end-to-end**: the live controller runs 9 critics
(`ConstraintCritic, CostCritic, GoalCritic, GoalAngleCritic,
PathAlignCritic, PathFollowCritic, PathAngleCritic, PreferForwardCritic,
FlowFieldCritic`), of which only CostCritic and GoalCritic are chained
onto the GPU path so far -- the other 7 still run CPU-only, meaning
`CriticManager::evalTrajectoriesScores()` today does rollout+2-critics on
GPU (one round trip) THEN 7 more critics on CPU reading the downloaded
trajectories, i.e. the download already had to happen for those 7
regardless. The 1.02x number above is for the 3-piece slice in isolation,
not a measurement of the whole `optimize()` cycle with all 9 critics
active -- that end-to-end number (does moving 2 of 9 critics to GPU
measurably speed up the real cycle, given the other 7's CPU cost and the
still-happening download dominate the total) has NOT been measured and
is the next honest thing to check before claiming a controller-level win,
not just a component-level one.

## 2026-09-10 (continued): GPU port -- measuring the real CPU tail, conclusion, stopping point

Measured the actual question from the previous entry: real (verbatim,
not reimplemented-from-memory) `score()` bodies for the 7 critics NOT yet
on GPU (`ConstraintCritic`, `GoalAngleCritic`, `PathAlignCritic`,
`PathFollowCritic`, `PathAngleCritic`, `PreferForwardCritic`,
`FlowFieldCritic` -- the last using the real `tgmppi::FlowField` class
directly, it has no ROS dependency) plus the shared
`findPathFurthestReachedPoint` precompute (used by 3 of those 7, runs
once/cycle), timed at the real K=2000/T=56 shape against a realistic
200-point local plan segment and 200x200 costmap.

**Result (stable across repeat runs, 4.36-4.40ms both times):**

| | ms/cycle |
|---|---|
| findFurthestPoint (shared precompute) | 0.59 |
| ConstraintCritic | 0.65 |
| GoalAngleCritic | 0.79 |
| PathAlignCritic | 0.54 |
| PathFollowCritic | 0.11 |
| PathAngleCritic | 0.79 |
| PreferForwardCritic | 0.06 |
| FlowFieldCritic | 0.88 |
| **TOTAL** | **~4.4** |

**This overturns the earlier back-of-envelope guess.** The unported
7-critic tail (~4.4ms) is not small relative to the already-GPU-chained
3-piece slice (~3.9ms) -- it's comparable, actually slightly larger.
There is real, substantial remaining opportunity, not diminishing
returns. Breaking down where that 4.4ms actually is:

- **Trivially portable** (pure elementwise/reduction over [K,T], same
  shape as the already-proven `GoalCritic`): `ConstraintCritic`,
  `GoalAngleCritic`, `PathFollowCritic`, `PathAngleCritic`,
  `PreferForwardCritic` -- **2.39ms combined**.
- **Portable with the already-proven `CostCritic` grid-gather pattern**:
  `FlowFieldCritic` (a per-point lookup into `FlowField`'s distance grid,
  structurally the same shape as `CostCritic`'s per-point costmap
  lookup) -- **0.88ms**.
- **Genuinely hard to vectorize**: `PathAlignCritic` (a per-trajectory
  binary-search cursor into cumulative path distance, incrementally
  stateful across the loop -- resists batching without real algorithmic
  redesign) and the shared `findFurthestPoint` precompute (a raw nested
  argmin loop, no vectorized reduction used in the original) --
  **1.13ms combined**.

So ~85% of the remaining CPU cost (3.8 of 4.4ms) follows one of the two
patterns already mechanically proven tonight (`GoalCritic`'s zero-upload
elementwise chain, `CostCritic`'s grid-gather chain) -- not a research
problem, just more of the same work. Given the established trend across
tonight's three data points (0.66x @ 1 piece -> 0.54x @ 2 -> 1.02x @ 3,
each additional chained elementwise critic costing far less on GPU than
its own CPU time once the one round trip is already paid for), chaining
that 3.8ms of mechanically-portable work alongside what's already there
would plausibly push the full cycle meaningfully ahead of CPU, not just
to the parity already shown -- estimated, not measured; the actual
number is only known once it's built. `PathAlignCritic` + the shared
precompute (1.13ms, ~13% of the remaining tail) could reasonably stay
CPU-only indefinitely -- small enough not to matter much either way, and
avoiding a real algorithmic rewrite for a minor piece.

**Conclusion and stopping point for today**: the architecture is proven
correct (numeric parity every step), proven to actually cross over CPU
once enough compute shares one round trip (real, not projected, 1.02x
result), and the remaining work is now well-scoped and estimated rather
than open-ended. This is a genuine, worthwhile continuation -- not
diminishing returns -- but it is realistically another full session's
worth of work (5 mechanical elementwise ports + 1 grid-gather port, each
needing its own parity test and both-build-config verification, same
rigor as tonight) and was explicitly not started tonight per the user's
call to measure, conclude, and stop. **Next-session plan, in priority
order**: (1) `FlowFieldCritic` (biggest single item, proven pattern,
directly reuses the `CostCritic` grid-gather approach against
`FlowField`'s grid instead of the costmap's), (2) the 5 elementwise
critics in any order (`ConstraintCritic`, `GoalAngleCritic`,
`PathFollowCritic`, `PathAngleCritic`, `PreferForwardCritic`), (3) after
all of those are chained, re-measure the REAL end-to-end
`Optimizer::optimize()` cycle (not a component slice) with
`compute_backend:"cuda"` actually wired into a live Nav2 run, which is
the number that finally answers "does the real robot get faster" --
leave `PathAlignCritic` and the shared precompute on CPU unless that
end-to-end number says otherwise.

## 2026-09-11: GPU port -- FlowFieldCritic, a much bigger win than expected

Continued the plan from yesterday's stopping point: `FlowFieldCritic`
first (identified as the biggest single remaining item and a proven-shape
candidate -- a per-point grid gather, structurally identical to
`CostCritic`'s costmap gather, just against `FlowField`'s distance grid
instead). `FlowField::distAt()` turned out even simpler than
`CostCritic`'s `costAtPose()`: it always clamps to the grid rather than
special-casing out-of-bounds, so `GpuFlowFieldCritic::computeDevice()`
needs no in-bounds mask at all. Added one small public accessor to
`FlowField` (`distGrid()`, returning the whole `D_` buffer for bulk
upload -- previously only `cellDist(i,j)` existed, one cell at a time).
One float constant (`kDryPenalty = 2.0f`) had to be duplicated since it's
private to `flow_field.cpp`'s anonymous namespace -- documented as a
sync risk in the new file's docstring, not silently copied.

Same wiring pattern as `CostCritic`/`GoalCritic`: `data.gpu_rollout`
checked first (chains onto the rollout's resident trajectory tensors,
uploading only the flow field's own grid), standalone `score()` as
fallback. Both build configs verified clean, **zero warnings this time**.

**Standalone parity**: exact match (0.0 diff).

**Standalone timing was the surprise**: CPU=1.997ms/cycle,
GPU(standalone)=0.267ms/cycle -- **7.48x**, a much bigger win than
`CostCritic`'s or `GoalCritic`'s standalone numbers ever showed (both of
those *lost* standalone). The reason: `FlowFieldCritic`'s CPU
implementation is a raw, unvectorized double for-loop (112,000 individual
`distAt()` calls, non-SIMD), unlike the xtensor-vectorized elementwise
critics -- so its CPU baseline is much slower to begin with, and the
GPU's batched gather wins even before accounting for the round-trip
savings from chaining.

**4-critic chained result** (rollout + `CostCritic` + `GoalCritic` +
`FlowFieldCritic`, one upload/download, K=2000/T=56/200x200
costmap/150-point path): exact/near-exact parity on all three critics'
outputs (repulsive-cost and flow-cost diffs both 0.0, goal-cost 5.72e-6),
and:

| | ms/cycle | vs CPU |
|---|---|---|
| A) CPU only (4 pieces) | 4.910 | 1.00x |
| B) independent GPU (4 round trips) | 2.073 | 2.37x |
| C) chained GPU (1 round trip) | 1.629 | **3.01x** |

Up from yesterday's 1.02x at 3 pieces to **3.01x at 4** -- a much larger
jump than the previous per-critic increments, driven by `FlowFieldCritic`
alone being worth more than the first three pieces combined. Chaining
still beats independent by 1.27x, consistent with the established
pattern (fixed round-trip cost shared, not per-critic).

**Remaining scope, updated**: of the 5 elementwise critics not yet
ported (`ConstraintCritic`, `GoalAngleCritic`, `PathFollowCritic`,
`PathAngleCritic`, `PreferForwardCritic`, ~2.39ms combined per
yesterday's CPU-only measurement) plus `PathAlignCritic` + the shared
`findFurthestPoint` precompute (~1.13ms, left CPU-only), the single
biggest lever (`FlowFieldCritic`) is now done. The remaining 5 are each
individually cheap and mechanically identical to `GoalCritic` -- real
but smaller incremental wins expected, not another 3x jump. Next actual
step, in order: (1) port the 5 remaining elementwise critics (mechanical,
same pattern each time), (2) measure the real end-to-end
`Optimizer::optimize()` cycle wired into a live Nav2 run -- still not
done, still the number that actually answers "does the robot get
faster," not a component slice.

## 2026-09-11 (continued): GPU port -- all 5 remaining elementwise critics done

Ported the last 5 active critics not yet on GPU: `ConstraintCritic`,
`GoalAngleCritic`, `PathFollowCritic`, `PathAngleCritic`,
`PreferForwardCritic`. All five are single elementwise+reduction passes
over already-resident state/trajectory tensors (same shape as
`GoalCritic`), so bundled into one shared class, `GpuElementwiseCritics`
(one `ready_` flag covers all five -- none holds persistent device
state, matching the other Gpu*Critic classes). Scoped to what the active
yaml actually exercises, same precedent as `CostCritic`'s
`consider_footprint:false`-only: `ConstraintCritic`'s Ackermann branch
(motion model is DiffDrive), `GoalAngleCritic`'s `symmetric_yaw_tolerance`
(unset -> false), and `PathAngleCritic`'s reversing-corrected branch
(`forward_preference` defaults true, never overridden) are all NOT
implemented -- `score()` falls back to CPU whenever a critic's own
config would need one of those, so nothing silently misbehaves if the
yaml changes later.

**Verified**: both build configs clean, zero warnings. Dedicated parity
test (real `GpuElementwiseCritics` class, not reimplemented) -- all 5
critics exact match (0.0 diff) against verbatim CPU references.

**Built a comprehensive test chaining all 8 now-portable pieces**
(rollout + all 8 critics except `PathAlignCritic`) through one
upload/download, and hit a real bug in the *test harness* worth recording
as a lesson: `GpuCostCritic::computeDevice()` uniquely returns the RAW
repulsive sum (weight/traj_len scaling happens once in the caller,
matching `CostCritic::score()`'s own design) while every other critic's
`computeDevice()` already returns the fully-scaled cost internally. The
first version of this test applied CostCritic's weight scaling twice in
one place and not at all in another, producing a large, confusing
mismatch (max diff ~6298) that had nothing to do with production code --
confirmed by isolating each critic's CPU-vs-GPU diff individually (7 of
8 were 0.0 immediately; `CostCritic` was the only one wrong, and only in
the test's own composition logic). Fixed the test, not the production
code, which was correct throughout. **Lesson for next time a critic is
compared/composed manually: check whether its `computeDevice()` returns
a raw or already-scaled value before assuming a uniform contract across
all of them.**

**Full 8-piece parity, after the fix**: exact match (0.0 diff) on every
individual critic and the composed total.

**Timing showed a real, reproducible bimodal split** across repeated
runs of the identical binary -- worth reporting honestly rather than
picking the best number: roughly half the runs land near chained=1.5ms
(~5x vs CPU), the other half near chained=3.45ms (~2.2x vs CPU). CPU
stayed rock-stable throughout (7.45-7.77ms across all 7 runs). GPU
clock was confirmed at P0 (not throttled, 2565MHz) during testing, so
this isn't simple thermal throttling -- more likely CUDA
allocator/caching state varying between process launches. **Honest
range: 2.2x-5.1x vs CPU for the 8-piece chain**, not a single point
estimate. This variance itself is a finding worth carrying into the
real end-to-end test: cycle-to-cycle GPU timing on this hardware isn't
perfectly deterministic, so that measurement should average over many
cycles, not trust a single reading.

**Status**: every critic that follows the two proven patterns
(elementwise/reduction, grid-gather) is now GPU-chained. Only
`PathAlignCritic` (needs real algorithmic redesign for its serial
binary-search cursor) remains CPU-only, by design, per the 2026-09-10
plan. **Next and last step before any real conclusion**: wire
`compute_backend:"cuda"` into an actual live Nav2 run and measure the
real `Optimizer::optimize()` cycle end to end -- every component
measurement so far has been a synthetic slice, not the real thing with
all 9 critics + the softmax control update + real ROS/costmap overhead
included.
