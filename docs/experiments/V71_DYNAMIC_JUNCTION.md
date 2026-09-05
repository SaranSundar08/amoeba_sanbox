# V7.1 dynamic-junction benchmark

Date: 2026-08-10

## Hypothesis

The experiment targets the thesis claim directly: blob-derived ancillary
controllers should help a path-biased MPPI controller when its single global
path becomes dynamically blocked and another homotopy class remains open.

## Blocked-split scenario

A static A* path is computed through the left side of a central island. After
planning, a scripted robot enters that left branch and remains stopped there.
The right branch remains collision-free. All controllers receive identical
geometry, peer motion, 1024 samples, 56 horizon steps, V6.1 prediction, five
random seeds, and a 40-second limit.

| controller | success | peer collision | wall collision | timeout | mean peer clearance |
|---|---:|---:|---:|---:|---:|
| vanilla MPPI | 5/5 | 0 | 0 | 0 | 1.112 m |
| stale-path MPPI | 1/5 | 0 | 2 | 2 | 0.225 m |
| Amoeba-biased MPPI | 4/5 | 0 | 0 | 1 | 0.717 m |

Vanilla has no path bias and freely discovers either side, so it is an upper
reference here rather than the main scientific comparator. The key comparison
is stale-path versus Amoeba-biased MPPI: ancillary pseudopod modes raised
success from 20% to 80% and removed collisions. Amoeba selected a branch mode
in 62.3% of cycles, confirming that the recovery was not merely dormant blob
infrastructure.

The remaining Amoeba timeout was safe but conservative. Successful Amoeba
runs took 36.3 seconds on average and stalled for roughly 6 seconds; the
failed seed stalled for 17.5 seconds. ASSIST exit and mode commitment therefore
remain the next tuning target.

The preliminary crossing and merge cases were non-discriminating (all methods
succeeded), while the original narrow head-on case was infeasible for every
controller. Those results are retained as benchmark-design evidence, not as
support for the algorithm.
