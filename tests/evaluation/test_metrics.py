"""
Unit tests for evaluation metrics.

Every expected value is HAND-COMPUTED on small fixed inputs.
No function under test is called to derive its own expected value.
"""

from __future__ import annotations

import pytest

from atar.evaluation.metrics import (
    compute_accuracy,
    compute_success_rate,
    compute_efficiency,
    compute_recovery_rate,
    compute_mean_latency,
    compute_avg_tool_calls,
    bootstrap_ci,
    aggregate_metrics,
)


# ─── compute_accuracy ────────────────────────────────────────────────────

class TestComputeAccuracy:
    def test_all_correct(self) -> None:
        """3 correct out of 3 → 1.0"""
        records = [{"correct": True}, {"correct": True}, {"correct": True}]
        assert compute_accuracy(records) == 1.0

    def test_none_correct(self) -> None:
        """0 correct out of 3 → 0.0"""
        records = [{"correct": False}, {"correct": False}, {"correct": False}]
        assert compute_accuracy(records) == 0.0

    def test_mixed(self) -> None:
        """2 correct out of 5 → 0.4"""
        records = [
            {"correct": True}, {"correct": False}, {"correct": True},
            {"correct": False}, {"correct": False},
        ]
        assert compute_accuracy(records) == pytest.approx(0.4)

    def test_empty(self) -> None:
        """Empty list → 0.0"""
        assert compute_accuracy([]) == 0.0

    def test_single(self) -> None:
        """1 correct out of 1 → 1.0"""
        assert compute_accuracy([{"correct": True}]) == 1.0


# ─── compute_success_rate ─────────────────────────────────────────────────

class TestComputeSuccessRate:
    def test_all_success(self) -> None:
        records = [{"success": True}, {"success": True}]
        assert compute_success_rate(records) == 1.0

    def test_mixed(self) -> None:
        """1 success out of 4 → 0.25"""
        records = [
            {"success": True}, {"success": False},
            {"success": False}, {"success": False},
        ]
        assert compute_success_rate(records) == pytest.approx(0.25)

    def test_empty(self) -> None:
        assert compute_success_rate([]) == 0.0


# ─── compute_efficiency ──────────────────────────────────────────────────

class TestComputeEfficiency:
    # efficiency = accuracy / (1 + avg_tool_calls)

    def test_basic(self) -> None:
        """accuracy=0.8, avg_tool_calls=2.0 → 0.8 / 3"""
        assert compute_efficiency(0.8, 2.0) == pytest.approx(0.8 / 3.0)

    def test_zero_tool_calls(self) -> None:
        """avg_tool_calls=0 → efficiency == accuracy (no-tool agents are not zeroed)"""
        assert compute_efficiency(0.5, 0.0) == pytest.approx(0.5)

    def test_perfect(self) -> None:
        """accuracy=1.0, avg_tool_calls=1.0 → 0.5"""
        assert compute_efficiency(1.0, 1.0) == pytest.approx(0.5)

    def test_high_tool_use(self) -> None:
        """accuracy=0.6, avg_tool_calls=3.0 → 0.15"""
        assert compute_efficiency(0.6, 3.0) == pytest.approx(0.15)

    def test_more_tools_same_accuracy_is_less_efficient(self) -> None:
        assert compute_efficiency(0.6, 1.0) > compute_efficiency(0.6, 2.0)


# ─── compute_recovery_rate ───────────────────────────────────────────────

class TestComputeRecoveryRate:
    def test_no_failures(self) -> None:
        """No episodes had tool failures → 0.0"""
        records = [
            {"had_tool_failure": False, "correct": True},
            {"had_tool_failure": False, "correct": False},
        ]
        assert compute_recovery_rate(records) == 0.0

    def test_all_recovered(self) -> None:
        """2 episodes with tool failure, both correct → 1.0"""
        records = [
            {"had_tool_failure": True, "correct": True},
            {"had_tool_failure": True, "correct": True},
        ]
        assert compute_recovery_rate(records) == 1.0

    def test_partial_recovery(self) -> None:
        """3 episodes with tool failure, 1 correct → 1/3 ≈ 0.3333"""
        records = [
            {"had_tool_failure": True, "correct": True},
            {"had_tool_failure": True, "correct": False},
            {"had_tool_failure": True, "correct": False},
            {"had_tool_failure": False, "correct": True},  # not in denominator
        ]
        # 1 recovered / 3 that failed = 1/3
        assert compute_recovery_rate(records) == pytest.approx(1.0 / 3.0)

    def test_empty(self) -> None:
        assert compute_recovery_rate([]) == 0.0


# ─── compute_mean_latency ────────────────────────────────────────────────

class TestComputeMeanLatency:
    def test_basic(self) -> None:
        """Latencies [1.0, 2.0, 3.0] → mean = 2.0"""
        records = [
            {"latency_s": 1.0}, {"latency_s": 2.0}, {"latency_s": 3.0},
        ]
        assert compute_mean_latency(records) == pytest.approx(2.0)

    def test_single(self) -> None:
        assert compute_mean_latency([{"latency_s": 5.5}]) == pytest.approx(5.5)

    def test_empty(self) -> None:
        assert compute_mean_latency([]) == 0.0


# ─── compute_avg_tool_calls ──────────────────────────────────────────────

class TestComputeAvgToolCalls:
    def test_basic(self) -> None:
        """[2, 4, 0] → mean = 2.0"""
        records = [
            {"tool_call_count": 2}, {"tool_call_count": 4}, {"tool_call_count": 0},
        ]
        assert compute_avg_tool_calls(records) == pytest.approx(2.0)

    def test_empty(self) -> None:
        assert compute_avg_tool_calls([]) == 0.0


# ─── bootstrap_ci ────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_deterministic_with_seed(self) -> None:
        """Same seed → same CI bounds."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = bootstrap_ci(values, seed=99)
        ci2 = bootstrap_ci(values, seed=99)
        assert ci1 == ci2

    def test_point_estimate_is_mean(self) -> None:
        """Point estimate for [1, 2, 3, 4, 5] is 3.0."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        point, lower, upper = bootstrap_ci(values, seed=42)
        assert point == pytest.approx(3.0)

    def test_ci_contains_point(self) -> None:
        """The 95% CI must contain the point estimate."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        point, lower, upper = bootstrap_ci(values, seed=42)
        assert lower <= point <= upper

    def test_empty_returns_zeros(self) -> None:
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)

    def test_single_value(self) -> None:
        """All bootstrap samples from a single value give the same value."""
        point, lower, upper = bootstrap_ci([7.0], seed=42)
        assert point == pytest.approx(7.0)
        assert lower == pytest.approx(7.0)
        assert upper == pytest.approx(7.0)

    def test_all_same_values(self) -> None:
        """If all values are the same, CI is degenerate."""
        point, lower, upper = bootstrap_ci([3.0, 3.0, 3.0, 3.0], seed=42)
        assert point == pytest.approx(3.0)
        assert lower == pytest.approx(3.0)
        assert upper == pytest.approx(3.0)

    def test_binary_values_hand_computed(self) -> None:
        """
        For binary values [1, 1, 0, 0], mean = 0.5.
        CI must be symmetric-ish around 0.5, lower >= 0, upper <= 1.
        """
        values = [1.0, 1.0, 0.0, 0.0]
        point, lower, upper = bootstrap_ci(values, seed=42, n_bootstrap=10_000)
        assert point == pytest.approx(0.5)
        assert 0.0 <= lower <= 0.5
        assert 0.5 <= upper <= 1.0


# ─── aggregate_metrics ───────────────────────────────────────────────────

class TestAggregateMetrics:
    def test_basic_aggregation(self) -> None:
        """
        Hand-computed on a 4-record dataset:
          - accuracy: 2/4 = 0.5
          - success_rate: 2/4 = 0.5
          - avg_tool_calls: (1+2+0+3)/4 = 1.5
          - efficiency: 0.5 / (1 + 1.5) = 0.2
          - recovery_rate: 1 failed with tool failure, 0 recovered → 0.0
          - mean_latency: (1.0+2.0+0.5+3.5)/4 = 1.75
        """
        records = [
            {
                "correct": True, "success": True, "tool_call_count": 1,
                "latency_s": 1.0, "had_tool_failure": False,
            },
            {
                "correct": True, "success": True, "tool_call_count": 2,
                "latency_s": 2.0, "had_tool_failure": False,
            },
            {
                "correct": False, "success": False, "tool_call_count": 0,
                "latency_s": 0.5, "had_tool_failure": True,
            },
            {
                "correct": False, "success": False, "tool_call_count": 3,
                "latency_s": 3.5, "had_tool_failure": False,
            },
        ]
        result = aggregate_metrics(records)

        assert result["n_tasks"] == 4
        assert result["accuracy"] == pytest.approx(0.5)
        assert result["success_rate"] == pytest.approx(0.5)
        assert result["avg_tool_calls"] == pytest.approx(1.5)
        assert result["efficiency"] == pytest.approx(0.2)
        assert result["recovery_rate"] == pytest.approx(0.0)  # failed & not correct
        assert result["mean_latency"] == pytest.approx(1.75)

        # CIs should exist and be lists of 2 floats
        assert len(result["accuracy_ci"]) == 2
        assert len(result["success_rate_ci"]) == 2
