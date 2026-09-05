# Horizon-scale pseudopod reference distinctness

Date: 2026-09-04. Rows: `artifacts/results/pilots/reference_distinctness_pilot.csv`
(shaped, floor50, nominal) and `..._pilot2.csv` (band20, nominal_fb), both
from `python3 -m experiments.reference_distinctness --jobs 6` on the blocked
path, worlds 4/48/49, seeds 0--1, otherwise the V5/V6 hybrid settings.

## Problem

`HOMOTOPY_ABLATION.md` found that feasible pseudopod references pass a nearby
obstacle on different sides in 0 of 742 pairs. The V2 reference
(`proposals.branch_to_control_sequence`) slows to `clearance / 0.45 m` with a
12% floor and by curvature; in BARN, usable clearance is almost always below
0.45 m, so the reference crawls at about 0.16 m/s and a 56-step rollout covers
about 0.5 m: the shared trunk of every branch, never their split. The mode
groups therefore sample near-identical guidance and their free energies differ
mostly by noise.

## What was varied

Only the branch REFERENCE speed policy, through the `reference_*` kwargs of
`MPPI`. The path fallback proposal, costs, sampling, gating, and selection are
unchanged in every arm, so NORMAL behaviour is identical across arms and the
comparison isolates pseudopod guidance.

| policy | speed floor | curvature slowdown | clearance band | infeasible fallback |
|---|---:|---:|---:|---|
| shaped (baseline) | 0.12 | 1.2 | 0.45 m | -- |
| floor50 | 0.50 | 0.6 | 0.45 m | -- |
| band20 | 0.40 | 0.6 | 0.20 m | -- |
| nominal | 1.00 | 0.0 | -- | -- |
| nominal_fb | 1.00 | 0.0 | -- | shaped reference if the nominal one is infeasible |

Acceptance metric, declared before running: reference pair separation
materially above zero **while** feasible-branch fraction and guided-mode
count stay at or above the baseline. Outcomes are reported, not tuned on.

## Result

| policy | success | time | worst clearance | switches | pseudopod selection | ref. separation | ref. length | centreline coverage | feasible branches | guided modes | own drift | free-energy gap | fallback used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shaped | 6/6 | 25.8 s | 0.082 m | 7.5 | 0.55 | 0.000 | 0.53 m | 0.22 | 0.89 | 1.56 | 0.001 | 0.88 | -- |
| floor50 | 6/6 | 25.0 s | 0.099 m | 9.5 | 0.64 | 0.000 | 0.71 m | 0.29 | 0.92 | 1.58 | 0.003 | 1.43 | -- |
| band20 | 6/6 | 24.4 s | 0.096 m | 7.3 | 0.66 | 0.001 | 0.71 m | 0.29 | 0.95 | 1.61 | 0.006 | 1.36 | -- |
| nominal | 4/6 (2 timeouts) | 22.3 s | 0.093 m | 10.5 | 0.45 | 0.328 | 1.36 m | 0.50 | 0.58 | 1.05 | 0.063 | 2.12 | -- |
| **nominal_fb** | 6/6 | 25.1 s | 0.100 m | 6.3 | 0.73 | **0.186** | 0.99 m | 0.40 | **0.94** | 1.57 | 0.029 | **2.07** | 0.32 |

- Half-speed floors (`floor50`, `band20`) reach 0.7 m and never the split:
  separation stays at zero. Speed is not the lever on its own.
- `nominal` confirms the diagnosis (separation 0.33, coverage 0.50) but pays
  in feasibility: 42% of fast references clip a tight gap and are rejected,
  guided modes fall to about one, and world 48 times out in both seeds. It
  trades indistinct modes for missing modes.
- `nominal_fb` keeps the fast reference where it is feasible (68% of
  branches) and the shaped one elsewhere. Separation is 0.186 with feasibility
  above baseline (0.94 vs 0.89) and the same number of guided modes. The
  free-energy gap between modes more than doubles (0.88 to 2.07), mode
  switches fall (7.5 to 6.3), pseudopod selection rises (0.55 to 0.73), and
  worst clearance improves (0.082 to 0.100 m) at equal success.
- Own drift rises from 0.1% to 2.9% under `nominal_fb`: once references are
  distinct, the mode-purity mechanism has something to act on. The homotopy
  cost (`homotopy_w`) is therefore testable only from here.

## Decision

`nominal_fb` is the accepted candidate for horizon-scale distinctness. It is
**opt-in** (`reference_min_speed_ratio=1.0, reference_curvature_slowdown=0.0,
reference_infeasible_fallback=True`); the shaped reference remains the
default so no prior result changes. Before it becomes a default it must pass
the four-scenario matrix (normal, blocked, dynamic, dynamic-blocked; worlds
4/48/49; seeds 0--1) with no success regression, and the blocked-split
junction case. Only after that is the `nominal_fb` x `homotopy_w` on/off
ablation the correct T-MPC test.

Limitations: six blocked episodes on three development worlds; the fallback
decision is per branch per cycle, so a branch's reference speed can alternate
between cycles (the warm start blends 70% of the previous mode mean, which
damps this); the path fallback still uses the shaped reference, so in ASSIST
the branch and path modes now differ in reference speed as well as geometry.
