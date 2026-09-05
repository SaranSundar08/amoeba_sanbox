# Repository layout

The sandbox contains one active NumPy reference implementation and one
independent PyTorch port. Version numbers describe research milestones; they
do not correspond to duplicate controller source trees.

| Path | Role |
|---|---|
| root runtime modules | active NumPy controller, topology, simulation and rendering |
| `experiments/` | reproducible versioned benchmark and analysis entry points |
| `tests/` | NumPy reference tests |
| `torch_port/` | independent tensor implementation, tests and runners |
| `docs/` | mathematics, status, experiment contracts and Nav2 notes |
| `artifacts/images/` | curated and legacy figures |
| `artifacts/results/milestones/` | thesis-facing benchmark evidence |
| `artifacts/results/pilots/` | reduced-size preliminary runs |
| `artifacts/results/exploratory/` | older non-milestone comparisons |
| `artifacts/results/interaction/` | reciprocal/IA-MPPI exploratory branch |
| `artifacts/v62/` | ROS bag-derived calibration analysis |

## Git boundary

`torch_port/.git` is intentionally preserved as the separate repository used
for the PyTorch port. The parent sandbox is not currently a Git repository.
Do not remove or relocate that nested metadata during parent-folder cleanup.

## Result policy

- Never delete a CSV that supports a documented decision.
- Put declared thesis matrices in `milestones/`.
- Put quick reduced-sample checks in `pilots/`.
- Put reciprocal interaction work in `interaction/` unless it is formally
  promoted into the thesis scope.
- Supply an explicit output filename for temporary tuning runs.
- Generated Python and LaTeX caches are ignored and may be removed safely.
