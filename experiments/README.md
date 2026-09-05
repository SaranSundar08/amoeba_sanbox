# Experiment drivers

Run these modules from the sandbox root with `python3 -m experiments.<name>`.
They are retained to reproduce documented milestones; they are not alternate
copies of the controller.

| Module | Purpose | Default result area |
|---|---|---|
| `v31_ablation` | grouped-sampling stability | `artifacts/results/milestones/` |
| `attribution_ablation` | path/topology/commitment attribution | `artifacts/results/milestones/` |
| `v5_ablation` | fixed versus adaptive covariance | `artifacts/results/milestones/` |
| `v6_ablation` | ideal versus analytical SLIP model | `artifacts/results/milestones/` |
| `v61_ablation` | dynamic prediction comparison | `artifacts/results/milestones/` |
| `v7_benchmark` | vanilla/path/Amoeba benchmark | `artifacts/results/milestones/` |
| `v7_visual_compare` | side-by-side path/Amoeba animation | no CSV |
| `final_figures` | thesis figures from frozen results | `artifacts/images/experiments/final/` |
| `v71_junction_benchmark` | blocked-split dynamic junction | `artifacts/results/milestones/` |
| `v71_junction_demo` | visual junction episode | no CSV by default |
| `v62_bag_analysis` | physical/Gazebo ROS bag analysis | explicit output directory |

Pilot and exploratory CSVs are preserved under `artifacts/results/pilots/`
and `artifacts/results/exploratory/`. The reciprocal interaction investigation
is isolated under `artifacts/results/interaction/` because it is optional
future work rather than the main narrow-navigation thesis claim.

Run the discriminating world-7 visual comparison with:

```bash
python3 -m experiments.v7_visual_compare
```

This defaults to the scientifically closer single-path Biased MPPI baseline.
Pass `--baseline path_mppi` to compare against the path-cost controller. Both
panels use the same A* path, seed, SLIP model, sample count, and horizon.

Regenerate the final result figures from the frozen CSV files with:

```bash
python3 -m experiments.final_figures --with-trajectories
```

The trajectory option reruns only the deterministic world-7 illustration and
does not modify or extend the frozen statistical benchmark.
