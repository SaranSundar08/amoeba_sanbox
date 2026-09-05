# Reciprocal interaction-risk ablation

Date: 2026-08-25

## Purpose

This experiment asks how an interaction-aware MPPI controller should combine
several plausible predictions of another robot. It is a diagnostic step toward
local-goal prediction, not a claim that the current hand-written predictor is
the final thesis method.

The reciprocal DynaBARN runner places two slip-model robots in world 93. Their
start and goal poses are exchanged, so they must negotiate a head-on encounter.
The dynamic variant additionally contains two scripted moving obstacles. All
reported runs used 1024 MPPI samples, a 56-step horizon and CUDA.

## Peer hypotheses

For every MPPI sample `k`, the controller evaluates four possible peer futures
`h`:

1. constant velocity;
2. follow the peer's A* path;
3. decelerate and yield;
4. reverse.

If `J_k^(h)` is the peer-interaction cost under hypothesis `h`, the tested
aggregation rules were:

```text
worst:  J_k = max_h J_k^(h)
mean:   J_k = (1/H) sum_h J_k^(h)
CVaR:   J_k = mean of the worst-cost tail
```

CVaR used `alpha=0.5`; with four hypotheses this averages the worst two. The
expected ordering is `mean <= CVaR <= worst`. Constant-velocity prediction was
retained as the matched baseline.

## Five-seed baseline, worst-case and CVaR results

The complete five-seed CSV files are:

- `artifacts/results/interaction/two_robot_prediction_benchmark.csv`
- `artifacts/results/interaction/two_robot_prediction_dynamic.csv`
- `artifacts/results/interaction/two_robot_prediction_cvar.csv`
- `artifacts/results/interaction/two_robot_prediction_dynamic_cvar.csv`

| scenario | method | success | mean successful time | mean minimum peer clearance | mean mode switches | mean reverse distance |
|---|---:|---:|---:|---:|---:|---:|
| static | constant velocity | 5/5 | 25.56 s | 0.127 m | 4.2 | 0.673 m |
| static | worst hypotheses | 3/5 | 29.53 s | 0.680 m | 16.8 | 1.581 m |
| static | CVaR hypotheses | 4/5 | 34.46 s | 0.793 m | 18.8 | 1.442 m |
| dynamic | constant velocity | 5/5 | 25.58 s | 0.182 m | 4.8 | 0.150 m |
| dynamic | worst hypotheses | 3/5 | 42.95 s | 0.693 m | 31.6 | 1.725 m |
| dynamic | CVaR hypotheses | 3/5 | 34.72 s | 0.706 m | 31.2 | 1.862 m |

Worst-case aggregation succeeded in 6/10 trials. CVaR improved this only to
7/10. CVaR reduced successful dynamic completion time relative to worst-case,
but did not meet the required 5/5 static acceptance criterion. Its much larger
peer clearance, reverse distance and switching count indicate conservative
avoidance and oscillation rather than a computational bottleneck. Mean
controller time remained approximately 49--51 ms for all policies.

## Mean-risk screening

Because complete runs are expensive, mean aggregation was screened only on the
previously difficult seeds 0 and 1. The files are:

- `artifacts/results/interaction/two_robot_prediction_mean_screen.csv`
- `artifacts/results/interaction/two_robot_prediction_dynamic_mean_screen.csv`

| scenario | seed 0 | seed 1 |
|---|---|---|
| static | success, 29.1 s | timeout |
| dynamic | timeout | timeout |

The dynamic mean runs switched modes 38 and 69 times and reversed 2.864 m and
3.961 m. Mean therefore did not resolve the deadlock mechanism, and the full
five-seed mean matrix was intentionally not run.

## Interpretation

Changing only the risk aggregator is insufficient:

- all four hand-written intentions are treated as equally plausible;
- a collision penalty is `10000`, so even one collision hypothesis contributes
  about `2500` under a four-way mean and can still dominate ordinary costs;
- reverse is considered even without evidence that the peer will reverse;
- the yielding robot's reverse proposal competes repeatedly with path and
  amoeba proposals, producing excessive mode switching.

The negative result is useful: multi-hypothesis prediction is not automatically
interaction-aware merely because several futures are enumerated. Their
likelihood and the recovery state machine matter.

## Decision and next implementation

Do not spend more benchmark time tuning `worst`, equal-weight `mean`, or CVaR.
Retain all three as ablations. The next controller should use observation-based
hypothesis probabilities:

```text
J_k = J_static,k + J_path,k + sum_h p_h J_peer,k^(h)
```

The probabilities should respond to observed motion: alignment with the peer
path increases the path-following probability, deceleration increases the
yield probability, and reverse receives appreciable probability only after
observed backward motion. Static geometry and scripted-obstacle collision
checks remain hard safety constraints. Reverse should become a hysteretic
deadlock-recovery state rather than an ordinary continuously competing mode.

This is the bridge to learned local-goal probabilities. First validate the
probability-weighted hand-crafted model; only then replace its probability
estimator with a learned predictor and compare both versions.

## Implementation status after the ablation

The first probability-weighted stage is now implemented. Every path or amoeba
ancillary proposal is rolled out as an ego local intention. Its proposed motion
and the observed peer speed change, path alignment, backward motion, predicted
proximity, and right-of-way determine a separate four-way intention
distribution. A 0.7/0.3 previous/current filter smooths each mode's
probabilities. The MPPI samples belonging to that mode use those probabilities
when aggregating peer-hypothesis costs.

The runner accepts `--hypothesis-risk weighted`, now its default, and emits
mean constant-velocity, path, yield, and reverse probabilities plus mean
hypothesis entropy. Unit tests cover probability-weighted collision costs and
the existing worst/mean/CVaR paths. This is the hand-crafted, interpretable
interaction-aware stage. Difficult-seed screening and the hysteretic reverse
recovery state remain pending.

## Probability-weighted static screen

The difficult static seeds 0 and 1 were then run with 1024 samples, a 56-step
horizon, CUDA, and `--hypothesis-risk weighted`. Both succeeded:

| seed | result | time | minimum peer clearance | mode switches | reverse distance |
|---:|---|---:|---:|---:|---:|
| 0 | success | 25.75 s | 0.323 m | 9 | 0.551 m |
| 1 | success | 26.10 s | 0.331 m | 8 | 0.404 m |

Across the two runs, mean probabilities were 0.220 constant velocity, 0.644
path following, 0.134 yielding, and 0.002 reversing; mean entropy was 0.774.
The predictor therefore suppressed unsupported reverse futures while retaining
more than one plausible intention.

This passes the 2/2 difficult-static acceptance screen. Relative to the same
mean-risk screen, average mode switching fell from 23.5 to 8.5, reverse
distance from 2.80 m to 0.48 m, and success increased from 1/2 to 2/2. The
result supports goal-conditioned probability weighting, but two selected seeds
are not a statistical performance claim. The next controlled step is the same
two-seed dynamic screen before changing the reverse state machine.

## Probability-weighted dynamic screen and recovery decision

The matched two-moving-obstacle screen then succeeded in 1/2 difficult seeds.
Seed 0 succeeded in 35.40 s with 0.163 m peer clearance, 12 mode switches, and
0.282 m reverse distance. Seed 1 timed out with 72 switches and 4.475 m reverse
distance, despite its mean reverse-hypothesis probability being only 0.007.
This isolates the remaining problem: the legacy reverse proposal bypassed the
interaction belief and continuously competed with path/amoeba modes.

Reverse is consequently gated by an explicit hysteretic state machine:

```text
NORMAL -> YIELD -> RECOVERY_REVERSE -> REJOIN -> NORMAL
```

Entry requires persistent nearby forward conflict and lack of progress. The
robot must yield before reverse becomes available. Recovery has minimum and
maximum dwell conditions, exits after sufficient displacement or peer
clearance, and is followed by a cooldown. The runner now reports recovery
activation count. The next experiment repeats only dynamic seed 1; the full
matrix remains deferred until that regression passes.

The first recovery regression still timed out at 60 s, so it did not pass the
terminal criterion. It nevertheless changed the mechanism substantially:
mode switches fell from 72 to 19, total reverse distance from 4.475 m to
1.493 m, total path length increased from 16.394 m to 20.109 m, and robot 1
reached its goal without collision. Robot 2 alone remained unfinished. The
minimum peer and static clearances were 0.028 m and 0.041 m, respectively,
which are valid but close. Because the original telemetry omitted robot 2's
terminal goal distance and controller state, no further threshold tuning is
justified from this row alone. Per-robot terminal distance, pose, recovery
count, interaction state, and assist state/reason have now been added.

The diagnostic repeat reproduced the run exactly. Robot 1 finished 0.492 m
from its goal (inside the 0.5 m tolerance) after one recovery activation.
Robot 2 ended 2.161 m from its goal in amoeba `ASSIST`, with reason
`path_blocked`, while its interaction state remained `NORMAL`. It nevertheless
accumulated 1.168 m of reverse motion. This exposed a configuration error:
negative velocity sampling had been enabled for both controllers even though
only robot 1 has deterministic yielding responsibility. Reverse velocity is
now disabled for the non-yielding robot; its interaction avoidance still uses
forward steering and stopping. The next regression again consists only of
dynamic seed 1.

The forward-priority regression then succeeded. Both robots reached their
goals in 49.65 s; the non-yielding robot's reverse distance fell from 1.168 m
to exactly 0, and it finished in `NORMAL` rather than `ASSIST/path_blocked`.
Minimum peer clearance improved from 0.028 m to 0.166 m. This supports the
reverse-scope diagnosis.

The run is not fully tuned: the yielding robot activated recovery five times,
total mode switches rose to 36, completion remained slower than the CV
baseline, and minimum static clearance was only 0.027 m. These are retained as
efficiency and robustness limitations. Do not tune them on seed 1 alone. The
next economical validation is current-code seed 0, followed---only if it
passes---by unseen seeds 2, 3, and 4 rather than rerunning completed cases.
