"""Summary-contract tests for the paired V6.1 ablation."""
from experiments.v61_ablation import paired_deltas, summarize


def _row(config, success, result, clearance, path):
    return {
        "config": config, "world": 1, "seed": 0, "success": success,
        "result": result, "time_s": 2.0, "path_len_m": path,
        "min_clear_m": clearance, "stall_s": 0.1, "controller_hz": 15.0,
        "mean_ess": 10.0, "mode_switches": 1, "mean_path_error_m": 0.2,
    }


def test_summary_and_pair_deltas_keep_direction_explicit():
    rows = [_row("current", 0, "collision", 0.02, 2.0),
            _row("predicted", 1, "success", 0.22, 3.0)]
    summary = summarize(rows)
    assert [item["config"] for item in summary] == ["current", "predicted"]
    delta = paired_deltas(rows)
    assert delta["success_delta"] == 1
    assert delta["collision_delta"] == -1
    assert abs(delta["mean_clearance_delta_m"] - 0.20) < 1e-12
