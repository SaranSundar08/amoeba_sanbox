"""V7 benchmark aggregation contract."""
from experiments.v7_benchmark import CONFIGS, summarize


def test_final_configs_include_single_path_biased_baseline():
    assert CONFIGS == ("vanilla", "path_mppi", "path_biased", "amoeba")


def test_summary_separates_scenario_and_controller():
    base = dict(world=1, seed=0, success=1, result="success", time_s=2.0,
                path_len_m=3.0, min_clear_m=0.1, goal_progress_m=2.5,
                stall_s=0.0, controller_hz=20.0, controller_mean_ms=40.0,
                controller_p95_ms=50.0, controller_max_ms=60.0,
                mean_path_error_m=0.2)
    rows = [dict(base, scenario="static", config="vanilla"),
            dict(base, scenario="static", config="amoeba")]
    summary = summarize(rows)
    assert [(row["scenario"], row["config"]) for row in summary] == [
        ("static", "vanilla"), ("static", "amoeba")]


def test_summary_preserves_controller_latency_statistics():
    base = dict(world=1, success=1, result="success", time_s=2.0,
                path_len_m=3.0, min_clear_m=0.1, goal_progress_m=2.5,
                stall_s=0.0, controller_hz=20.0, mean_path_error_m=0.2,
                scenario="static", config="vanilla")
    rows = [dict(base, seed=0, controller_mean_ms=10.0,
                 controller_p95_ms=15.0, controller_max_ms=20.0),
            dict(base, seed=1, controller_mean_ms=14.0,
                 controller_p95_ms=19.0, controller_max_ms=30.0)]
    summary = summarize(rows)[0]
    assert summary["mean_controller_ms"] == 12.0
    assert summary["mean_controller_p95_ms"] == 17.0
    assert summary["worst_controller_max_ms"] == 30.0
