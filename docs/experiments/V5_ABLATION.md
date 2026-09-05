# V5 geometry-adaptive covariance ablation

## Scope

This development ablation isolates covariance while keeping the hybrid
controller, V3 weights, costs, allocation, commitment, paths, seeds, and
moving-obstacle layouts fixed. It compares fixed covariance with V5 per-mode,
per-step diagonal covariance on BARN worlds 4, 48, and 49, seeds 0 and 1.

Scenarios are normal A* navigation, a blocked centre-lane plan, dynamic
obstacles with the A* plan, and dynamic obstacles with the blocked plan. Each
configuration has six runs per scenario and 24 total.

```bash
python3 -m experiments.v5_ablation --worlds 4,48,49 --seeds 2 \
  --scenarios normal,blocked,dynamic,dynamic_blocked \
  --configs fixed,adaptive --jobs 4 \
  --out artifacts/results/milestones/v5_ablation.csv
```

## Aggregate results

| scenario | covariance | success | collisions | time | path | worst clearance | stall | flips | switches | ESS | path error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | fixed | 6/6 | 0 | 20.4 s | 8.98 m | 0.085 m | 0.0 s | 9.5 | 0.0 | 89.1 | 0.062 m |
| normal | adaptive | 6/6 | 0 | 20.4 s | 9.02 m | 0.095 m | 0.0 s | 8.8 | 0.0 | 94.7 | 0.058 m |
| blocked | fixed | 6/6 | 0 | 25.8 s | 9.95 m | 0.076 m | 0.1 s | 12.7 | 7.5 | 60.0 | 0.322 m |
| blocked | adaptive | 6/6 | 0 | 28.3 s | 10.13 m | 0.091 m | 0.7 s | 12.3 | 10.8 | 70.3 | 0.337 m |
| dynamic | fixed | 5/6 | 1 | 24.4 s | 9.86 m | -0.007 m | 0.5 s | 12.0 | 5.5 | 45.9 | 0.205 m |
| dynamic | adaptive | 5/6 | 1 | 23.7 s | 9.44 m | -0.006 m | 0.8 s | 12.3 | 5.0 | 60.4 | 0.132 m |
| dynamic blocked | fixed | 4/6 | 2 | 29.3 s | 10.95 m | -0.007 m | 0.8 s | 10.5 | 10.3 | 40.5 | 0.350 m |
| dynamic blocked | adaptive | 4/6 | 1 | 33.5 s | 10.95 m | -0.011 m | 3.6 s | 13.0 | 15.0 | 71.0 | 0.288 m |

Mean time and path length are calculated over successful runs. Adaptive
dynamic-blocked results contain one collision and one timeout; fixed contains
two collisions.

## Interpretation

V5 is technically complete and behaves as intended statistically. It improves
worst static clearance from 0.085 m to 0.095 m in normal navigation and from
0.076 m to 0.091 m with blocked paths. Mean ESS improves in every scenario,
most strongly in dynamic blocked navigation (40.5 to 71.0).

It is not uniformly superior. Blocked cases are slower, adaptive covariance
increases mode switching in blocked scenarios, and total success is unchanged:
6/6 normal, 6/6 blocked, 5/6 dynamic, and 4/6 dynamic blocked for both
policies. The policies fail on different matched cases, so V5 redistributes
dynamic risk rather than eliminating it.

V5 should remain experimental behind `--adaptive-covariance`. Fixed covariance
with V3 weights remains the deployment baseline until V6 adds the real
footprint, skid-steer behavior, and turn-dependent safety margins.

## Remaining limitations

- Six runs per scenario and one dynamic-obstacle parameterization are not
  sufficient for statistical superiority claims.
- Moving obstacles are scripted circles, not sensed/tracked agents.
- Branch-ID churn still contributes to switching.
- The robot remains a circular ideal differential-drive model.
- The covariance mapping is hand-designed and diagonal, not calibrated from
  SLIP data.
