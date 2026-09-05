# Slip-footprint BARN feasibility screen

This screen uses the Torch/Nav2 transfer footprint assumptions:

- rectangular base: `0.90 x 0.65 m`;
- A* inflation radius: `0.65 / 2 + 0.08 = 0.405 m`;
- A* grid resolution: `0.10 m`;
- start: `(-2.25, 1.0)`;
- goal: `(-2.25, 10.0)`.

Every A* path is subsequently checked using oriented samples around the full
rectangular footprint. This is a feasibility screen, not a guarantee that the
closed-loop controller will succeed: turning transients and dynamic obstacles
can require additional clearance.

## Summary

| Category | Worlds | Meaning |
|---|---:|---|
| comfortable | 7 | at least 0.15 m footprint clearance |
| tight | 92 | 0.05--0.15 m footprint clearance |
| critical | 35 | less than 0.05 m positive clearance |
| footprint collision | 25 | A* centreline exists but the rectangle overlaps |
| unreachable | 141 | no path after 0.405 m inflation |

## Recommended tiers

Comfortable smoke and baseline worlds:

`93, 42, 40, 61, 67, 75, 94`

The first two have the largest measured bottleneck clearances:

- world 93: `0.725 m`;
- world 42: `0.575 m`.

Useful tight-challenge worlds with at least 0.10 m measured clearance:

`25, 5, 18, 27, 28, 29, 36, 41, 57, 65, 72, 108, 145`

World 48 is classified as **critical** with `0.049 m` oriented-footprint
clearance. It is useful as a near-limit stress test, but should not be the
primary debugging world. World 180 is **unreachable** with this footprint.

The complete per-world measurements are stored in
`artifacts/results/milestones/slip_world_feasibility.csv`.

Regenerate the table with:

```bash
python -m torch_port.scan_slip_worlds
```
