# V7 vanilla/path/Amoeba benchmark

Date: 2026-08-06

The matched benchmark used ten BARN worlds, two controller seeds, 1024 MPPI
samples, a 56-step horizon, and a 1200-cycle limit. Dynamic cases used four
matched moving obstacles and enabled the same V6.1 prediction for all three
controllers.

## Configurations

- `vanilla`: goal-distance MPPI without A*, blob, or pseudopods.
- `path_mppi`: A*-path-biased MPPI without pseudopod proposals.
- `amoeba`: final hybrid with blob, pseudopods, grouped sampling, guarded
  importance weighting, adaptive covariance, commitment, and V6.1 prediction.

## Results

| scenario | controller | success | collisions | timeouts | mean clearance | worst clearance |
|---|---:|---:|---:|---:|---:|---:|
| static | vanilla | 15/20 | 5 | 0 | 0.102 m | -0.020 m |
| static | path MPPI | 20/20 | 0 | 0 | 0.133 m | 0.017 m |
| static | Amoeba | 19/20 | 0 | 1 | 0.130 m | 0.013 m |
| dynamic | vanilla | 13/20 | 6 | 1 | 0.056 m | -0.012 m |
| dynamic | path MPPI | 19/20 | 0 | 1 | 0.093 m | 0.010 m |
| dynamic | Amoeba | 18/20 | 0 | 2 | 0.093 m | 0.010 m |

The complete Amoeba system strongly outperformed vanilla and removed all
observed collisions. It did not outperform path-only MPPI on success. Its two
unique regressions were safe stalls: static world 180 seed 0 and dynamic world
160 seed 1. Those cases used branch modes, switched 9--10 times, stalled for
36--40 seconds, and finished only 5.7--6.6 m closer to the goal. This points to
ASSIST exit/commitment behavior, not blob collision geometry.

The natural A* benchmark rarely requires a topological detour, so it mainly
measures the value of path guidance. A separate path-blockage benchmark is
required to establish whether pseudopods beat path-only recovery when the
global plan is genuinely invalid.

The NumPy timing numbers are not a deployment claim. Parallel jobs share CPU
resources, and the synchronous reference implementation rebuilds topology in
the controller process. C++ should compute topology asynchronously and the
PyTorch/C++ hot path should be timed separately.
