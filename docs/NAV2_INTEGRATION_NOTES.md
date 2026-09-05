# Nav2 MPPI integration notes

These notes are based on the local sources at:

- `/home/saran/robohouse_ws/src/nav2_mppi_controller`
- `/home/saran/robohouse_ws/src/nav2_amoeba_mppi_controller`

No source files in either package were modified during the Python attribution
work.

## Existing control flow

`MPPIController::computeVelocityCommands()`:

1. Transforms and prunes the global plan through `PathHandler`.
2. Locks the costmap.
3. Calls `Optimizer::evalControl()`.
4. Optionally publishes candidate and optimized trajectories.

`Optimizer::evalControl()`:

1. Prepares pose, measured speed, path tensor, and critic data.
2. Generates noised trajectories.
3. Runs all configured critics.
4. Updates the control sequence.
5. Applies the existing Savitzky-Golay output filter.
6. Returns a command and shifts the sequence when controller period equals
   `model_dt`.

The correct V3/V4 extension seam is therefore between noise generation and
motion-model prediction:

```text
NoiseGenerator / proposal means
→ State.cvx/cwz
→ motion_model_->predict()
→ integrateStateVelocities()
→ existing critics
→ mode-aware updateControlSequence()
```

Path transformation, footprint collision checking, critics, speed limits,
motion prediction, lifecycle behavior, and output filtering should remain
Nav2-owned.

## Required additions

### Proposal snapshot

Add a fixed-capacity, timestamped proposal snapshot containing:

- Stable mode ID.
- `[T]` means for `vx` and `wz`.
- Diagonal standard deviations.
- Prior/allocation weight.
- Feasibility and clearance diagnostics.
- Costmap and transformed-path version/timestamp.

The control loop should atomically read the latest valid immutable snapshot.
It must not build the geodesic body while holding the costmap lock.

### State metadata

Extend the optimizer-side batch storage with:

- Mode index per sample.
- Raw latent controls when V4 is enabled.
- Per-mode sample ranges/counts.
- Per-mode free energy, ESS, and posterior mass.

Use preallocated xtensors/vectors sized during `reset()`.

### Persistent mode bank

Maintain one `ControlSequence` per active stable mode, including the transformed
global-path mode. Shift all active sequences each cycle using Nav2's existing
sequence-shift convention.

### Mode-aware optimizer update

Keep critic evaluation unchanged. Aggregate costs by mode, update every
mode-specific sequence, then select one committed mode. Apply Nav2's existing
constraints and Savitzky-Golay filtering only to the selected output sequence.

## Global planner behavior

Use the transformed and pruned plan from `PathHandler`, not the full global
plan. The path proposal is the defensive/default component.

Do not reproduce the sandbox's unconditional distance-to-path cost. Existing
Nav2 critics already contain important blocked-path behavior:

- `PathAlignCritic` disables alignment when local path occupancy exceeds
  `max_path_occupancy_ratio`.
- `PathFollowCritic` searches forward for a valid path point beyond an occupied
  region.
- `ObstaclesCritic` supports full footprint collision checks.

Pseudopods should complement these behaviors, not replace them.

## Existing fork cautions

The current `nav2_amoeba_mppi_controller` already contains:

- Synchronous flow-field reconstruction in `Optimizer::applyFlowBias()`.
- A flow-field critic.
- A legacy ray/wrap multimodal experiment.
- Direct sample-mean shifts in optimizer code.

Before porting the new branch bank, consolidate or remove competing ray/wrap
and flow-bias paths. Otherwise several proposal mechanisms will modify the same
`State.cvx/cwz` buffers with unclear ownership.

The comments in the existing biased implementation describe Nav2's gamma
control-cost term as an importance correction for shifted samples. That term is
part of the standard MPPI update, but it is not by itself the full
`log p - log q` correction for a Gaussian mixture. V4 should not inherit that
claim without a derivation and density tests.

## Real-time structure

Recommended threads:

```text
costmap/path snapshot
        ↓
topology worker, 2–5 Hz
        ↓ atomic immutable ProposalSnapshot
Nav2 controller, 15–20 Hz
        ↓
grouped MPPI using latest valid snapshot
```

If the snapshot is stale, path-incompatible, or infeasible, use ordinary Nav2
MPPI/path sampling immediately.

