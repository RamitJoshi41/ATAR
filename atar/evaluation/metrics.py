"""
Evaluation metrics for ATAR.

All metric functions operate on plain Python lists/dicts so they can be
unit-tested with hand-computed expected values.  No dependency on the env
or LLM — these are pure functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_accuracy(records: list[dict[str, Any]]) -> float:
    """
    Fraction of task records where the final answer was correct.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain ``"correct"`` (bool).

    Returns
    -------
    float  in [0, 1].  Returns 0.0 if *records* is empty.
    """
    if not records:
        return 0.0
    return sum(1 for r in records if r["correct"]) / len(records)


def compute_success_rate(records: list[dict[str, Any]]) -> float:
    """
    Fraction of task records where the episode was successful.

    An episode is successful if it terminated (not truncated) AND the
    final answer was correct.  With the current ATAREnv a final answer only
    exists after a terminal action, so this equals accuracy; it is kept as a
    separate column in case truncation-with-answer semantics ever change.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain ``"success"`` (bool).

    Returns
    -------
    float  in [0, 1].  Returns 0.0 if *records* is empty.
    """
    if not records:
        return 0.0
    return sum(1 for r in records if r["success"]) / len(records)


def compute_efficiency(
    accuracy: float,
    avg_tool_calls: float,
) -> float:
    """
    Efficiency = accuracy / (1 + avg_tool_calls).

    The +1 counts the final answer itself as one unit of work, so an agent
    that answers correctly with no tool calls gets its accuracy (not 0, and
    not infinity), and every extra tool call per episode lowers the score
    for the same accuracy.  See ARCHITECTURE_DECISIONS.md (ADR on eval
    metrics) for why the plain accuracy / avg_tool_calls was replaced.

    Parameters
    ----------
    accuracy : float
        Accuracy in [0, 1].
    avg_tool_calls : float
        Mean tool calls per episode.

    Returns
    -------
    float
    """
    return accuracy / (1.0 + max(avg_tool_calls, 0.0))


def compute_recovery_rate(records: list[dict[str, Any]]) -> float:
    """
    Recovery rate: among episodes with at least one tool failure (a tool
    runtime error OR a tool call rejected by M4 validation), the fraction
    that still ended with a correct answer.

    Only episodes that *had* a tool failure are in the denominator.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain ``"had_tool_failure"`` (bool) and
        ``"correct"`` (bool).

    Returns
    -------
    float  in [0, 1].  Returns 0.0 if no episodes had a tool failure.
    """
    failed = [r for r in records if r.get("had_tool_failure", False)]
    if not failed:
        return 0.0
    return sum(1 for r in failed if r["correct"]) / len(failed)


def compute_mean_latency(records: list[dict[str, Any]]) -> float:
    """
    Mean episode latency in seconds.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain ``"latency_s"`` (float).

    Returns
    -------
    float
    """
    if not records:
        return 0.0
    return sum(r["latency_s"] for r in records) / len(records)


def compute_avg_tool_calls(records: list[dict[str, Any]]) -> float:
    """
    Mean tool calls per episode.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain ``"tool_call_count"`` (int).

    Returns
    -------
    float
    """
    if not records:
        return 0.0
    return sum(r["tool_call_count"] for r in records) / len(records)


def bootstrap_ci(
    values: list[float],
    stat_fn: str = "mean",
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    values : list[float]
        The sample values.
    stat_fn : str
        "mean" or "median".
    n_bootstrap : int
        Number of bootstrap resamples (default 10000).
    ci : float
        Confidence level (default 0.95 → 95% CI).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (point_estimate, lower, upper) : tuple[float, float, float]
    """
    if not values:
        return (0.0, 0.0, 0.0)

    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)

    if stat_fn == "mean":
        fn = np.mean
    elif stat_fn == "median":
        fn = np.median
    else:
        raise ValueError(f"Unknown stat_fn: {stat_fn}")

    point = float(fn(arr))

    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot_stats[i] = fn(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return (point, lower, upper)


def aggregate_metrics(
    records: list[dict[str, Any]],
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """
    Compute all aggregate metrics from a list of per-task records.

    Parameters
    ----------
    records : list[dict]
        Per-task evaluation records.
    bootstrap_seed : int
        RNG seed for bootstrap CI.

    Returns
    -------
    dict with keys: accuracy, accuracy_ci, success_rate, success_rate_ci,
        avg_tool_calls, avg_tool_calls_ci, efficiency, recovery_rate,
        mean_latency, mean_latency_ci, n_tasks.
    """
    acc = compute_accuracy(records)
    sr = compute_success_rate(records)
    avg_tc = compute_avg_tool_calls(records)
    eff = compute_efficiency(acc, avg_tc)
    rr = compute_recovery_rate(records)
    ml = compute_mean_latency(records)

    # Bootstrap CIs
    acc_vals = [float(r["correct"]) for r in records]
    sr_vals = [float(r["success"]) for r in records]
    tc_vals = [float(r["tool_call_count"]) for r in records]
    lat_vals = [r["latency_s"] for r in records]

    acc_ci = bootstrap_ci(acc_vals, seed=bootstrap_seed)
    sr_ci = bootstrap_ci(sr_vals, seed=bootstrap_seed)
    tc_ci = bootstrap_ci(tc_vals, seed=bootstrap_seed)
    lat_ci = bootstrap_ci(lat_vals, seed=bootstrap_seed)

    return {
        "n_tasks": len(records),
        "accuracy": acc,
        "accuracy_ci": [acc_ci[1], acc_ci[2]],
        "success_rate": sr,
        "success_rate_ci": [sr_ci[1], sr_ci[2]],
        "avg_tool_calls": avg_tc,
        "avg_tool_calls_ci": [tc_ci[1], tc_ci[2]],
        "efficiency": eff,
        "recovery_rate": rr,
        "mean_latency": ml,
        "mean_latency_ci": [lat_ci[1], lat_ci[2]],
    }
