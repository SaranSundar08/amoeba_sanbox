# Proposal attribution ablation

## Question

Does explicit pseudopod topology add value beyond a well-constructed,
constraint-projected global-path proposal?

Four hybrid controllers were compared on BARN worlds 4, 48, and 49, seeds 0
and 1:

- `v0`: original single-Gaussian hybrid.
- `path_only`: one dynamically feasible global-path proposal with all samples.
- `path_branches`: path proposal plus committed pseudopod modes.
- `no_commit`: path plus pseudopods with immediate switching.

All variants use the same trajectory costs within each scenario.

## Normal A* plans

| controller | success | time | path | worst clearance | sign flips | switches | branch selected | path error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 6/6 | 24.5 s | 9.28 m | 0.098 m | 166.7 | 0.0 | 0.00 | 0.102 m |
| path only | 6/6 | 20.1 s | 8.99 m | 0.091 m | 13.5 | 0.0 | 0.00 | 0.068 m |
| path + committed branches | 6/6 | 20.3 s | 8.98 m | 0.095 m | 15.8 | 8.3 | 0.32 | 0.064 m |
| path + uncommitted branches | 6/6 | 20.3 s | 8.99 m | 0.095 m | 19.3 | 53.2 | 0.33 | 0.069 m |

The path proposal, constraint projection, and warm start explain most of the
speed and oscillation improvement. Pseudopods provide a small average path
error improvement and recover some clearance in individual cases, but they do
not establish a large normal-navigation advantage. Commitment is clearly
necessary: removing it causes roughly six times as many mode switches without
improving aggregate performance.

## Blocked centre-lane plans with Nav2-like validity handling

The first blocked-path experiment was unfair to the topology modes:
`AmoebaHybrid.goal_cost()` continued attracting trajectories to occupied path
cells. Actual Nav2 behavior is different:

- `PathAlignCritic` stops scoring when enough local path points are occupied.
- `PathFollowCritic` advances to a valid point beyond an occupied section.

The attribution harness now enables an approximate `path_validity_gate`:
occupied path points cannot be tracking targets, and alignment turns off above
the Nav2 default 0.07 occupancy ratio.

| controller | success | time | path | worst clearance | sign flips | switches | branch selected | path error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 6/6 | 25.2 s | 9.42 m | 0.100 m | 164.8 | 0.0 | 0.00 | 0.365 m |
| path only | 6/6 | 22.1 s | 9.70 m | 0.083 m | 10.3 | 0.0 | 0.00 | 0.418 m |
| path + committed branches | 6/6 | 21.7 s | 9.50 m | 0.086 m | 13.0 | 5.8 | 0.60 | 0.383 m |
| path + uncommitted branches | 6/6 | 21.4 s | 9.51 m | 0.087 m | 11.8 | 70.2 | 0.48 | 0.382 m |

Pseudopods now show a measurable blocked-path contribution over path-only:
shorter/faster recovery, better average path error, and higher clearance in the
world-48 cases. However, aggregate worst clearance remains below V0 and success
is unchanged. This is evidence of useful proposals, not yet evidence of a
superior controller.

## Conclusion

The topology direction remains viable, but the primary normal-navigation gain
comes from the path proposal and constraint handling. The thesis claim must be
supported by targeted topology cases, not ordinary path following.

Before V4:

1. Replace endpoint-based branch identities with persistent corridor/junction
   signatures.
2. Make the sandbox path-validity behavior closer to Nav2 on the local pruned
   plan rather than using a whole-path occupancy approximation.
3. Test blocked paths, dynamic blockage, symmetric homotopy choices, and stale
   plans across more worlds.
4. Preserve commitment; uncommitted switching is not competitive.
5. Treat importance weighting as an offline diagnostic first because minimum
   per-mode ESS still approaches one.

