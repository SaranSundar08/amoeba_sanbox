# Generated artifacts

- `images/demos/`: current figures suitable for reports or presentations.
- `images/experiments/`: diagnostic and milestone figures.
- `images/legacy/`: preserved output from removed ray/flow/commit variants.
- `results/milestones/`: canonical evidence referenced by the thesis notes.
- `results/pilots/`: early or reduced-size checks retained for provenance.
- `results/exploratory/`: non-milestone development comparisons.
- `results/interaction/`: optional reciprocal/IA-MPPI stress-test evidence.

Milestone experiment scripts write to `results/milestones/`. Reciprocal Torch
benchmarks write to `results/interaction/`. Use an explicit `--out` or
`--output` for temporary work rather than placing loose files in `results/`.
`run.py` continues to honor the exact path supplied with `--png`.
