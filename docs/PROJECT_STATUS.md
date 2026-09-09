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
