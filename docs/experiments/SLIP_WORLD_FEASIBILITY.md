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

## Re-screen under the Nav2-matched footprint (2026-09-09)

The screen above (and every banked SLIP result before this date) used the
provisional `0.90 x 0.65 m` rectangle with a `0.405 m` A* inflation. On
2026-09-09 the sandbox footprint was changed to match the Nav2 costmap
footprint actually used by the Gazebo/real-robot port
(`susag_nav2/param/navigation_amoeba*.yaml`: `[[0.42, 0.34], ...]` =
`0.84 x 0.68 m`, inflation `0.34 + 0.08 = 0.42 m`). The table above is left
as the record of the scan that selected the frozen development worlds; the
re-screen was written to a **separate** file so the banked CSV is untouched:

```bash
python -m torch_port.scan_slip_worlds --length 0.84 --width 0.68 \
  --output artifacts/results/milestones/slip_world_feasibility_0.84x0.68.csv
```

| Category | 0.90 x 0.65 / 0.405 | 0.84 x 0.68 / 0.42 |
|---|---:|---:|
| comfortable | 7 | 7 |
| tight | 92 | 101 |
| critical | 35 | 18 |
| footprint collision | 25 | 8 |
| unreachable | 141 | 166 |

The wider inflation closes 25 more corridors at the A* stage; the shorter
rectangle makes the oriented check pass more often on the routes that
remain. Net suitable (comfortable + tight): 99 -> 108.

The ten frozen development worlds, side by side:

| world | old category | old fp. clearance | new category | new fp. clearance |
|---:|---|---:|---|---:|
| 0 | critical | 0.009 | tight | 0.055 |
| **7** | critical | 0.037 | **footprint collision** | **-0.028** |
| 48 | critical | 0.049 | tight | 0.073 |
| 1 | tight | 0.070 | tight | 0.081 |
| 5 | tight | 0.125 | tight | 0.110 |
| 20 | tight | 0.081 | tight | 0.081 |
| 40 | comfortable | 0.275 | comfortable | 0.260 |
| 42 | comfortable | 0.575 | comfortable | 0.560 |
| 93 | comfortable | 0.725 | comfortable | 0.710 |
| 94 | comfortable | 0.275 | comfortable | 0.260 |

Nine of ten stay suitable; worlds 0 and 48 improve (critical -> tight)
because the rectangle is shorter. **World 7 flips to footprint collision**:
the 0.42 m inflation routes A* through a different gap (path length 9.674
-> 9.728 m, centreline margin 0.001 -> -0.014 m) and the 0.68 m width then
overlaps an obstacle by 2.8 cm along that route. This is the A*-centreline
proxy, not a closed-loop verdict -- the controllers do not track the
centreline exactly, and the generality matrix's world-7 result
(`path_biased` 0/5, full controller 5/5, at 0.90 x 0.65) stands as recorded.
But under the contract's rule ("unreachable and footprint-collision worlds
are excluded from controller success statistics"), world 7 should be
treated as **feasibility-excluded at 1:1 under the measured footprint** in
the Gazebo campaign, and reported separately as the deliberate near-limit
case rather than counted in the success rate. Expect Nav2's own planner
(NavFn, `inflation_radius: 0.45`, footprint + padding) to refuse or clip it.
