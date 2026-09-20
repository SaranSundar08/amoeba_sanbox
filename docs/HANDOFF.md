# TG-MPPI thesis — session handoff (2026-09-20)

## Who / what
MSc thesis, green light 19 Oct 2026, defense ~Nov. SLIP skid-steer robot, Gazebo
Classic 11 + ROS 2 Humble + Nav2. Comparing TG-MPPI (topology-guided MPPI, my fork)
against stock Nav2 MPPI on dynamic-obstacle scenarios. Sim only so far; real trials
use Vicon + front Tmini lidar.

## HARD CONSTRAINT
The baseline is vanilla `nav2_mppi_controller`. NEVER modify it. All implementation
goes in `nav2_tgmppi_controller`.

## Repos
- `~/robohouse_ws/src` — ROS packages. main @ 87593c0, pushed.
  Key: `nav2_tgmppi_controller/`, `susag_nav2/` (launch, params, benchmark),
  `susag_new_model/` (pkg name is `susag_updated_model_description`).
- `~/robohouse_ws/amoeba_sandbox` — python sandbox + all docs. main @ cd82a7b, **6 commits unpushed**.
  `docs/PROJECT_STATUS.md` = dated engineering log, read the last ~6 entries first.
  `docs/amoeba_mathematics_guide.tex` = the theory. `docs/thesis/` = the IEEE draft.

## Environment — do not break
ROS Humble is PINNED to `http://snapshots.ros.org/humble/2026-08-07/ubuntu`, 579
packages `apt-mark hold`. NEVER `apt upgrade`. Reason: tf2 0.25.23 has a lock-order
inversion (`waitForTransform` takes the Buffer mutex then frame_mutex_, while the TF
listener does the reverse in `setTransform`→`testTransformableRequests`). It deadlocks
RViz in seconds and floods Nav2 logs with `tf_help: Transform data too old`. Symptom
check: `dpkg -l ros-humble-tf2` must read 0.25.22.

Build protocol: `MAKEFLAGS=-j1 colcon build --parallel-workers 1`, always check
`REAL_EXIT=$?` (colcon's own exit code lies). CUDA build needs
`export PATH=/usr/local/cuda/bin:$PATH` — nvcc is NOT on PATH in non-interactive
shells. LibTorch RPATH is baked into the plugin, so no LD_LIBRARY_PATH needed.

Gotchas: `sim:=true` is mandatory for sim launches. `pkill -f <pattern>` matches your
own shell — use `pkill -x` or PIDs.

## Launch
    ros2 launch susag_updated_model_description gazebo_barn.launch.py world_idx:=20
    ros2 launch susag_nav2 navigation.launch.py sim:=true world_idx:=20

## Benchmark
`src/susag_nav2/benchmark/benchmark_dynamic.py`. Conditions: A=stock Nav2 MPPI,
B=TG-MPPI legacy split, Bp=TG-MPPI equal split, D=plain MPPI + prediction critic
(the critical control — B−D isolates what topology adds). 4 cond × 5 worlds × 3 reps
= 60 trials, ~2 min/trial, ~2 h total. Resumes by reusing `--run <name>`.
Results in `~/robohouse_ws/benchmark_results/dynamic/<run>/`.

## Results so far
`thesis_dyn` (60 trials, pre-upgrade, VALID): A 19/45 legs ok, 32/45 with contact;
B 57/59, 10 contacts; Bp 57/59, 9; D 53/57, 9. Prediction gives the whole gain over
stock (p<0.001). B/Bp/D statistically indistinguishable at n=15 — topology did not
separate from prediction.

## CRITICAL — do not write this up wrong
The "96.7% of space-time routes are non-distinct" figure measures the OLD
synchronised-time side test being structurally blind, NOT route redundancy. That test
required both routes near the same obstacle at the same time layer; a pass-before and
a pass-behind route are near it at DIFFERENT layers by construction, so it defaulted to
"not distinct". Replaced on main (commit 764ce21) by T-MPC's winding-number test
(λ = (1/2π)Σ wrapped Δθ, pass threshold 1/(4π)). Do NOT claim "space-time regenerates
duplicate routes".

Also void: the Sept 19 runs `st_winding_check` / `st_winding_3rep` (0/12 legs, 11
recoveries/leg). They ran on the broken tf2 stack — the control on main failed
identically (6480 `tf_help` errors vs 0 in good runs). Not the branch's fault.

## In flight
`thesis_dyn_winding` — the 60-trial re-run with the winding test on the restored
stack. Started 17:16, was at 49/60. Check: `ls <run>/*/*/* -d | wc -l` and `report.md`.

## Next
1. When the re-run finishes: regenerate the distinctness rate, and check whether
   B vs D changes now that distinct modes can actually be detected.
2. Re-run BARN static (`benchmark_barn.py`) — the only prior run is 14 trials on the
   broken stack, worlds 42/93 only.
3. Thesis draft: `docs/thesis/main.tex`, IEEEtran journal, double column, 10–12 pp.
   `\todo{}` = my prose to write, `\pending{}` = awaiting the re-run. `\draftfalse`
   hides both. Needs: `sudo apt install texlive-latex-recommended texlive-publishers
   texlive-science texlive-fonts-recommended latexmk`.
4. Push amoeba_sandbox (6 commits behind origin).

## Open honesty items for the report
- The 9-scenario harness cited in `amoeba_mathematics_guide.tex`
  §sec:spacetime-empirical ("old test true in only 2/9") is NOT in the repo —
  commit it or soften the claim.
- Winding test's own blind spot: a route passing far from an obstacle accumulates
  < 1/(4π) and reads "didn't pass", so such pairs are judged non-distinct.
- Obstacle observation is perfect (Gazebo ground truth, no noise/latency) — T-MPC had
  motion capture, a real lidar tracker would not. `tracking_quality:=good|poor` presets
  exist to test this; not yet run.
