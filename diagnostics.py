"""Terminal diagnostics for V2/V3 proposal and mode behavior."""


def _print_mode_debug(k, state, info, ctrl):
    """Compact V2/V3 diagnostics suitable for a headless terminal run."""
    selected = info.get("selected_mode")
    selected_text = "none" if selected is None else selected["name"]
    stats = " ".join(
        f"{item['name']}[n={item['count']},F={item['free_energy']:.2f},"
        f"ESS={item['ess']:.1f}"
        f"{',raw=' + format(item['corrected_ess'], '.1f') + ',guard=V3' if item.get('ess_guard_fallback') else ''}"
        f"{',trigger' if item.get('ess_guard_trigger') else ''}]"
        for item in info.get("mode_stats", []))
    proposals = " ".join(
        f"P{p.branch_id}[{'ok' if p.feasible else p.reason},"
        f"clr={p.min_clearance:.2f},prog={p.progress:.2f},"
        f"err={p.mean_tracking_error:.2f}]"
        for p in info.get("branch_proposals", []))
    mode_stds = info.get("mode_stds")
    covariance = "" if mode_stds is None else (
        f" sigma_v={mode_stds[..., 0].min():.3f}.."
        f"{mode_stds[..., 0].max():.3f}"
        f" sigma_w={mode_stds[..., 1].min():.3f}.."
        f"{mode_stds[..., 1].max():.3f}")
    print(
        f"debug k={k:4d} xy=({state[0]:.2f},{state[1]:.2f}) "
        f"u=({ctrl.last_applied[0]:.2f},{ctrl.last_applied[1]:.2f}) "
        f"selected={selected_text} reason={info.get('selection_reason', '-')} "
        f"switches={info.get('mode_switch_count', 0)} "
        f"assist={info.get('assist_state', '-')} "
        f"model={info.get('robot_model', 'ideal')} "
        f"assist_reason={info.get('assist_reason', '-')} "
        f"plan_v={info.get('plan_version', 0)} "
        f"gap={info.get('free_energy_gap', float('nan')):.2f} "
        f"banks={info.get('mode_bank_keys', ())}{covariance} | "
        f"{stats} | {proposals}",
        flush=True)
