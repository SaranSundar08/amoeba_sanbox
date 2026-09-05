"""Multimodal MPPI sampling and density utilities.

All probability densities are evaluated in the latent, pre-projection control
space.  Constraint projection is deterministic but many-to-one, so treating
projected controls as ordinary Gaussian samples would be mathematically wrong.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingMode:
    """One Gaussian proposal component."""

    key: int
    name: str
    mean: np.ndarray
    feasible: bool = True
    std: np.ndarray | None = None
    # (T, 2) rolled position of the proposal that anchors this mode -- its
    # guidance trajectory in T-MPC terms -- used by the homotopy-consistency
    # cost. None for the unguided nominal fallback.
    reference: np.ndarray | None = None


def logsumexp(values, axis=None):
    """Numerically stable ``log(sum(exp(values)))`` using only NumPy."""
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    result = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if axis is None:
        return float(result.squeeze())
    return np.squeeze(result, axis=axis)


def diagonal_gaussian_logpdf(samples, mean, std):
    """Log density of independent Gaussian dimensions.

    ``samples`` may have arbitrary leading batch dimensions. ``mean`` and
    ``std`` must broadcast over the trailing control-sequence dimensions.
    """
    samples = np.asarray(samples, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("Gaussian standard deviations must be finite and > 0")
    delta = (samples - mean) / std
    dimensions = tuple(range(samples.ndim - mean.ndim, samples.ndim))
    return -0.5 * np.sum(
        delta * delta + 2.0 * np.log(std) + np.log(2.0 * np.pi),
        axis=dimensions)


def gaussian_mixture_logpdf(samples, means, std, mixture_weights):
    """Evaluate the complete diagonal Gaussian-mixture density.

    ``std`` may be one shared control schedule or one schedule per component.
    """
    means = np.asarray(means, dtype=float)
    std = np.asarray(std, dtype=float)
    mixture_weights = np.asarray(mixture_weights, dtype=float)
    if means.ndim < 2 or len(means) != len(mixture_weights):
        raise ValueError("one mixture weight is required for every mean")
    if (np.any(~np.isfinite(mixture_weights))
            or np.any(mixture_weights <= 0.0)):
        raise ValueError("mixture weights must be finite and positive")
    mixture_weights = mixture_weights / mixture_weights.sum()
    component_stds = (
        std if std.shape == means.shape
        else np.broadcast_to(std, means.shape))
    component_logs = np.stack([
        diagonal_gaussian_logpdf(samples, mean, component_std)
        for mean, component_std in zip(means, component_stds)
    ], axis=-1)
    return logsumexp(component_logs + np.log(mixture_weights), axis=-1)


def normalized_log_weights(log_weights):
    """Normalize arbitrary log weights and return weights plus log normalizer."""
    log_normalizer = logsumexp(log_weights)
    return np.exp(np.asarray(log_weights) - log_normalizer), log_normalizer


def mixture_importance_weights(raw, costs, target_mean, modes, std,
                               mixture_weights, lam):
    """V4 weights: ``-cost/lambda + log p(raw) - log q(raw)``.

    ``p`` is the diagonal Gaussian around ``target_mean``. ``q`` is the full
    mixture over every active sampling mode.
    """
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lambda must be finite and positive")
    means = np.stack([mode.mean for mode in modes])
    mode_stds = np.stack([
        np.broadcast_to(std, mode.mean.shape)
        if mode.std is None else mode.std
        for mode in modes
    ])
    target_index = next(
        (index for index, mode in enumerate(modes) if mode.key == -1),
        None)
    target_std = (
        np.broadcast_to(std, np.asarray(target_mean).shape)
        if target_index is None else mode_stds[target_index])
    log_p = diagonal_gaussian_logpdf(raw, target_mean, target_std)
    log_q = gaussian_mixture_logpdf(
        raw, means, mode_stds, mixture_weights)
    log_weights = -np.asarray(costs, dtype=float) / lam + log_p - log_q
    weights, log_normalizer = normalized_log_weights(log_weights)
    return weights, log_weights, log_p, log_q, log_normalizer


def importance_mode_statistics(log_weights, labels, modes, lam):
    """Per-mode evidence and conditional weights from corrected V4 weights."""
    stats = []
    conditional = np.zeros(len(log_weights), dtype=float)
    for index, mode in enumerate(modes):
        sample_indices = np.flatnonzero(labels == index)
        local_log = np.asarray(log_weights)[sample_indices]
        local_weights, local_norm = normalized_log_weights(local_log)
        conditional[sample_indices] = local_weights
        free_energy = -lam * (local_norm - np.log(len(sample_indices)))
        stats.append({
            "index": index,
            "key": mode.key,
            "name": mode.name,
            "count": len(sample_indices),
            "free_energy": float(free_energy),
            "min_cost": np.nan,
            "ess": 1.0 / float(np.sum(local_weights ** 2)),
        })
    return stats, conditional


def guard_importance_statistics(corrected_stats, corrected_weights,
                                uncorrected_stats, uncorrected_weights,
                                labels, min_ess=8.0,
                                min_ess_fraction=0.02):
    """Use V3 for the whole cycle if any mode has degenerate V4 weights.

    Mode free energies must be comparable during selection. Mixing a V3 free
    energy for one component with V4 evidence for another is not coherent, so
    one failing component makes the complete cycle fall back. Returned
    statistics always expose the raw corrected values for diagnostics.
    """
    if min_ess < 0.0 or min_ess_fraction < 0.0:
        raise ValueError("ESS guard thresholds must be nonnegative")
    if len(corrected_stats) != len(uncorrected_stats):
        raise ValueError("corrected and uncorrected modes must match")
    thresholds = [
        max(float(min_ess), float(min_ess_fraction) * item["count"])
        for item in corrected_stats
    ]
    triggers = [
        item["ess"] < threshold
        for item, threshold in zip(corrected_stats, thresholds)
    ]
    cycle_fallback = any(triggers)
    guarded_weights = np.asarray(
        uncorrected_weights if cycle_fallback else corrected_weights,
        dtype=float).copy()
    guarded_stats = []
    for corrected, uncorrected, threshold, trigger in zip(
            corrected_stats, uncorrected_stats, thresholds, triggers):
        if (corrected["index"] != uncorrected["index"]
                or corrected["key"] != uncorrected["key"]):
            raise ValueError("corrected and uncorrected modes must match")
        chosen = dict(uncorrected if cycle_fallback else corrected)
        chosen.update({
            "corrected_ess": corrected["ess"],
            "corrected_free_energy": corrected["free_energy"],
            "uncorrected_ess": uncorrected["ess"],
            "uncorrected_free_energy": uncorrected["free_energy"],
            "ess_threshold": threshold,
            "ess_guard_trigger": trigger,
            "ess_guard_fallback": cycle_fallback,
            "weight_source": (
                "v3_cycle_fallback" if cycle_fallback else "v4_corrected"),
        })
        guarded_stats.append(chosen)
    return guarded_stats, guarded_weights, int(cycle_fallback)


def allocate_group_counts(total, n_modes):
    """Deterministic near-equal allocation whose entries sum to ``total``."""
    if total < 1:
        raise ValueError("total must be positive")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    counts = np.full(n_modes, total // n_modes, dtype=int)
    counts[:total % n_modes] += 1
    return counts


def allocate_weighted_counts(total, weights):
    """Deterministic stratified allocation with at least one sample per mode."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) < 1:
        raise ValueError("weights must be a non-empty vector")
    if total < len(weights):
        raise ValueError("total samples must cover every fixed mode")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    if weights.sum() <= 0.0:
        raise ValueError("at least one allocation weight must be positive")
    remaining = total - len(weights)
    quotas = remaining * weights / weights.sum()
    extra = np.floor(quotas).astype(int)
    counts = np.ones(len(weights), dtype=int) + extra
    leftover = total - int(counts.sum())
    order = np.argsort(-(quotas - extra), kind="stable")
    counts[order[:leftover]] += 1
    return counts


def sample_fixed_groups(rng, modes, total, std, allocation_weights=None):
    """Return latent samples, integer labels, and deterministic group counts."""
    if not modes:
        raise ValueError("at least one sampling mode is required")
    if len(modes) > total:
        raise ValueError("total samples must cover every fixed mode")
    counts = (allocate_group_counts(total, len(modes))
              if allocation_weights is None
              else allocate_weighted_counts(total, allocation_weights))
    raw, labels = [], []
    std = np.asarray(std, dtype=float)
    for index, (mode, count) in enumerate(zip(modes, counts)):
        mode_std = std if mode.std is None else np.asarray(mode.std)
        if mode_std.shape not in (std.shape, mode.mean.shape):
            raise ValueError("mode std must match shared or mean shape")
        eps = rng.normal(size=(count,) + mode.mean.shape) * mode_std
        raw.append(mode.mean[None] + eps)
        labels.append(np.full(count, index, dtype=int))
    return np.concatenate(raw), np.concatenate(labels), counts


def project_control_sequences(raw, initial_control, dt, v_max, w_max,
                              v_accel_max, w_accel_max):
    """Deterministically project latent sequences onto box/rate constraints."""
    controls = np.asarray(raw, dtype=float).copy()
    previous = np.broadcast_to(
        np.asarray(initial_control, dtype=float), (len(controls), 2)).copy()
    previous[:, 0] = np.clip(previous[:, 0], 0.0, v_max)
    previous[:, 1] = np.clip(previous[:, 1], -w_max, w_max)
    dv_max, dw_max = v_accel_max * dt, w_accel_max * dt
    for t in range(controls.shape[1]):
        desired = controls[:, t]
        u = previous + np.column_stack((
            np.clip(desired[:, 0] - previous[:, 0], -dv_max, dv_max),
            np.clip(desired[:, 1] - previous[:, 1], -dw_max, dw_max)))
        u[:, 0] = np.clip(u[:, 0], 0.0, v_max)
        u[:, 1] = np.clip(u[:, 1], -w_max, w_max)
        controls[:, t] = u
        previous = u
    return controls


def mode_statistics(costs, labels, modes, lam):
    """Stable per-mode free energies and within-mode vanilla MPPI weights."""
    stats = []
    weights = np.zeros(len(costs), dtype=float)
    for index, mode in enumerate(modes):
        sample_indices = np.flatnonzero(labels == index)
        c = costs[sample_indices]
        cmin = float(c.min())
        likelihood = np.exp(-(c - cmin) / lam)
        local_weights = likelihood / likelihood.sum()
        weights[sample_indices] = local_weights
        free_energy = cmin - lam * np.log(float(likelihood.mean()))
        ess = 1.0 / float(np.sum(local_weights ** 2))
        stats.append({
            "index": index,
            "key": mode.key,
            "name": mode.name,
            "count": len(sample_indices),
            "free_energy": float(free_energy),
            "min_cost": cmin,
            "ess": ess,
        })
    return stats, weights


def select_mode(stats, previous_key=None, switch_margin=0.3):
    """Choose minimum free energy, retaining the old mode within hysteresis."""
    best = min(stats, key=lambda item: item["free_energy"])
    if previous_key is None or best["key"] == previous_key:
        return best
    old = next((item for item in stats if item["key"] == previous_key), None)
    if old is not None and old["free_energy"] <= (
            best["free_energy"] + switch_margin):
        return old
    return best
