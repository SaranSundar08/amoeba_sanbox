# Homotopy-consistency (mode-purity) cost ablation

Date: 2026-09-04. Pilot rows: `artifacts/results/pilots/homotopy_pilot.csv`
(`python3 -m experiments.homotopy_ablation --jobs 6`).

## Motivation

De Groot, Ferranti, Gavrila and Alonso-Mora, *Topology-Driven Parallel
Trajectory Optimization in Dynamic Environments* (T-MPC, RA-L), show that
initializing parallel local planners from distinct guidance trajectories is
not enough: without an explicit constraint the optimized trajectories
collapse into one homotopy class (their Fig. 6). They hold each planner to
its guidance trajectory's side of every obstacle with a linear half-plane per
obstacle (their Eq. 8).

Amoeba's analogue of a guidance trajectory is a feasible pseudopod proposal's
rolled mean; its analogue of a local planner is that mode's fixed sample group
and within-mode MPPI update. Between refloods nothing pins the update to the
branch's side of an obstacle except the ordinary collision cost, so the
concern was that a mode's weighted mean could random-walk across a pillar into
a neighbouring mode's corridor.

## Implementation

`homotopy.py` builds one half-plane per obstacle within `homotopy_relevance`
(0.8 m) of a mode's reference rollout, through the obstacle centre and
perpendicular to the reference's closest-approach offset (`beta = 0`, T-MPC's
inactive-at-the-boundary setting). A sample is penalized by
`homotopy_w * sum_t sum_m max(0, A_m x_t - b_m)`, i.e. only when its centre is
on the far side of an obstacle from where the reference passes it, and by how
far. Static obstacles use the closest-approach plane rather than T-MPC's
time-synchronized plane per step, which would penalize samples merely for
being ahead of a slower reference. The unguided fallback is unconstrained by
default (`homotopy_constrain_fallback=False`), the analogue of T-MPC++'s
unguided planner in parallel. All flags default off, so no prior result
changes; `homotopy_monitor=True` reports purity without touching costs.

Purity is reported three ways per ASSIST cycle: the fraction of a guided
mode's samples that cross a plane; the weight mass of the executed update on
the wrong side; and, on the optimized means (`_homotopy_mode_distinctness`,
the direct Fig. 6 analogue), whether each guided mode's update still lies on
its own reference's side (`own_drift`) and whether each guided-mode pair still
passes some nearby obstacle on different sides (`collapsed_pairs`).

## Pilot: blocked path, worlds 4/48/49, seeds 0--1, `homotopy_w = 20`

| config | success | collisions | mean time | worst clearance | switches | sign flips | ESS | near-obstacle crossing fraction | wrong-side mass | own drift | collapsed pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off (monitor) | 6/6 | 0 | 25.8 s | 0.082 m | 7.5 | 12.7 | 60.0 | 0.000 | 0.000 | 0.001 | 0.989 |
| on | 6/6 | 0 | 25.8 s | 0.082 m | 7.5 | 12.5 | 60.1 | 0.000 | 0.000 | 0.000 | 0.984 |

Every outcome, time, and clearance is identical between arms. Per-sample
crossings are essentially never observed (at most 0.07% of samples in cycles
that have planes): a 2.8 s horizon of rate-limited angular noise spreads a
sample laterally by roughly 0.25 m, which rarely reaches past an obstacle
centre (at least 0.4 m away for a collision-free centre). The executed
update's wrong-side mass is zero and a guided mode's optimized mean drifts
across its own reference's side in 0.1% of mode-cycles. The T-MPC collapse
mechanism does not occur here; the cost is inactive in practice.

## The number that matters: pairs are "collapsed" 98--100% of the time

That is not optimizer collapse. Checking the *references* themselves with the
same test (world 48 and 49, blocked, seed 0, K=512): the two feasible
pseudopod proposals pass some nearby obstacle on different sides in **0 of
288** and **0 of 454** pairs. The 56-step proposal rollout covers on average
0.46 m of path, about 19--20% of the branch centreline, and the two
references' endpoints are only 0.27 m (world 48) and 0.08 m (world 49) apart.
Pseudopods are topologically distinct at the membrane, 2.5 m out, but the
MPPI horizon sees only their shared trunk. Within the horizon the mode groups
are sampling around near-identical guidance, so their free energies differ
mostly by sampling noise.

This explains three earlier observations at once: 7.5 mode switches per
blocked episode despite dwell/margin/confirmation hysteresis; branch
selection in 0% of cycles across the static generality matrix although
branches were available 70.9% of the time (`PROJECT_STATUS.md`); and why the
mode-purity cost cannot bite (there is no distinct side to hold a mode to).

## Conclusion

Negative for the cost, positive as a certificate: the existing controller's
mode updates are pure, and this pilot quantifies it. Keep `homotopy_w = 0`
as the baseline; retain `homotopy_monitor` for diagnostics. Do not extend the
matrix.

The actionable finding is horizon-scale distinctness. Candidate remedies, none
yet run: (a) build the branch reference at nominal speed without the
clearance/curvature slowdown so the horizon reaches the split; (b) lengthen
the proposal/rollout horizon during ASSIST; (c) T-MPC-style, plan each
reference to its branch endpoint over the full horizon so references diverge
from the first step. The acceptance metric for any of these is the reference
pair separation fraction above (currently 0.000), measured before rerunning
mode-selection experiments; the homotopy cost only becomes testable after that
number is materially positive.

Follow-up (same day): `REFERENCE_DISTINCTNESS.md` runs option (a) with a
per-branch fallback to the shaped reference when the fast one is infeasible
(`nominal_fb`): separation 0.186, feasibility above baseline, mode free-energy
gap 0.88 to 2.07, switches 7.5 to 6.3. Own drift rises to 2.9%, so the cost
ablation should be repeated under that policy.
