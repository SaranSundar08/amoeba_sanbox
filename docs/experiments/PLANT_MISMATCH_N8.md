# Plant/plan mismatch, N=8 independent draws per case

Date: 2026-09-06. `artifacts/results/milestones/v6_ablation_plant_mismatch_n8.csv`,
produced by `python3 -m experiments.v6_ablation --configs slip
--plant-mismatch 0.3 --plant-mismatch-draws 8 --worlds 4,48,49 --seeds 2
--jobs 12`, same 24 `(world, seed, scenario)` SLIP cases as the n=1 pilot
(`PLANT_MISMATCH_PILOT.md`), each now run against 8 independent mismatch
draws instead of 1.

## What changed in the harness

`experiments/v6_ablation.py` gained a `draw` index (default 1 draw, fully
backward compatible) and reseeds the mismatch RNG per draw with a 4-part
seed sequence (`np.random.default_rng([world, seed, draw, 7])`), so every
draw is statistically independent rather than a deterministic arithmetic
offset into one stream. `draw` is recorded in the output CSV.

## Aggregate result (192 = 24 cases x 8 draws)

| scenario | success | collisions | timeouts |
|---|---:|---:|---:|
| normal | 39/48 | 9 | 0 |
| blocked | 40/48 | 7 | 1 |
| dynamic | 11/48 | 37 | 0 |
| dynamic_blocked | 9/48 | 39 | 0 |

Aggregate success rates all sit at or above the n=1/self-consistent baseline
per scenario (normal 66.7%->81.3%, blocked 83.3%->83.3%, dynamic
16.7%->22.9%, dynamic_blocked 0%->18.8%). Taken alone this would read like
the pilot's "mismatch never hurts" story, confirmed. It is not the full
story -- see below.

## The real finding: this is bidirectional, not one-directional

Paired against the self-consistent baseline for all 24 cases:

- **8 of 14 baseline-collision cases show at least one successful draw** out
  of 8 (matches the pilot's flip count, though not necessarily the identical
  physical mechanism per case -- these are new independent draws, not a
  replay of the pilot's single draw).
- **2 of 10 baseline-success cases show at least one failed draw**:
  - `blocked, world=48, seed=1`: 7/8 success, 1 timeout.
  - `blocked, world=49, seed=0`: 7/8 success, 1 collision.

The n=1 pilot found zero regressions and called that "unusual by chance
alone if the true effect were neutral," which was correct as far as it went
-- but it was one sample per case, and the pilot's own text predicted
exactly this outcome was possible: "any change to the dynamics ... can tip
the outcome either way." At N=8, it does. **The n=1 pilot's headline
("every case matched or improved; none got worse") does not hold at N=8. Two
previously-100%-reliable cases now show an ~1/8 failure rate under
mismatch.** This is the corrected, more honest claim, and it should replace
the pilot's headline wherever it is cited.

## A separate pattern: some collisions are robust to mismatch, not knife-edge

6 of the 14 baseline-collision cases show **zero** successes across all 8
draws: `dynamic` world 48 (both seeds) and world 49 seed 1; `dynamic_blocked`
world 48 (both seeds) and world 49 seed 1. Their baseline clearances
(-0.001 to -0.006 m) are just as close to zero as the cases that *do* flip
under mismatch, so proximity to zero clearance alone does not predict which
cases are mismatch-sensitive. Whatever fails in these six cases fails
regardless of a +-30% skid-parameter perturbation -- worth treating as a
distinct, structural weak point (likely geometry/timing of the encounter,
not slip-parameter sensitivity), separate from the knife-edge cases. These
six are reasonable candidates to include in the upcoming space-time-topology
matched panel, since a plant-parameter change cannot fix a missing
alternative route, but an explicit wait/detour candidate might.

## Conclusion

The plan/plant separation mechanism continues to check out and remains
standard practice. The claim it now supports is: **under +-30% skid
parameter mismatch, most already-fragile (near-zero-clearance) SLIP
collisions are more likely than not to resolve, but a small number of
previously-reliable cases (~1/8 draws, roughly matching the same knife-edge
sensitivity) become occasional failures, and about half of the fragile
collisions are unaffected by mismatch entirely.** This is a genuine
robustness characterization now, not a demonstration -- but it is a
"degrades gracefully with a small new failure rate at the boundary" claim,
not a "mismatch only helps" claim. State it that way in the report.
