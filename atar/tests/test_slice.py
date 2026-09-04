"""Tests for the ATAR vertical-slice components."""

from __future__ import annotations

import numpy as np
import pytest

from atar.shared.shared_types import (
    STATE_DIM,
    ActionType,
    RewardComponents,
    Task,
    ToolResult,
)
from slice.calculator import execute_calculator, _safe_eval
from slice.dummy_policy import DummyPolicy
from slice.env import SliceEnv
from slice.state_builder import build_state
from slice.tasks import TASKS


# =====================================================================
# Calculator tests
# =====================================================================

class TestCalculator:
    """Tests for the AST-safe calculator."""

    def test_addition(self) -> None:
        r = execute_calculator("2 + 3")
        assert r.success is True
        assert r.output == 5

    def test_subtraction(self) -> None:
        r = execute_calculator("10 - 4")
        assert r.success is True
        assert r.output == 6

    def test_multiplication(self) -> None:
        r = execute_calculator("7 * 8")
        assert r.success is True
        assert r.output == 56

    def test_division(self) -> None:
        r = execute_calculator("144 / 12")
        assert r.success is True
        assert abs(r.output - 12.0) < 1e-9

    def test_power(self) -> None:
        r = execute_calculator("2 ** 10")
        assert r.success is True
        assert r.output == 1024

    def test_complex_expression(self) -> None:
        r = execute_calculator("(100 - 30) * 2")
        assert r.success is True
        assert r.output == 140

    def test_float_expression(self) -> None:
        r = execute_calculator("3.14 * 25")
        assert r.success is True
        assert abs(r.output - 78.5) < 1e-9

    def test_negative_result(self) -> None:
        r = execute_calculator("-5 + 3")
        assert r.success is True
        assert r.output == -2

    def test_returns_toolresult(self) -> None:
        r = execute_calculator("1 + 1")
        assert isinstance(r, ToolResult)
        assert r.error is None
        assert r.latency_ms >= 0.0
        assert r.token_cost == 0

    # --- Rejection cases (fail closed per AGENTS.md rule 8) ---

    def test_rejects_function_call(self) -> None:
        r = execute_calculator("__import__('os').system('ls')")
        assert r.success is False
        assert r.error is not None

    def test_rejects_variable_name(self) -> None:
        r = execute_calculator("x + 1")
        assert r.success is False

    def test_rejects_empty(self) -> None:
        r = execute_calculator("")
        assert r.success is False

    def test_rejects_string_literal(self) -> None:
        r = execute_calculator("'hello'")
        assert r.success is False

    def test_rejects_list_comprehension(self) -> None:
        r = execute_calculator("[x for x in range(10)]")
        assert r.success is False

    def test_division_by_zero(self) -> None:
        r = execute_calculator("1 / 0")
        assert r.success is False
        assert r.error is not None
        assert "division by zero" in r.error.lower()


# =====================================================================
# State builder tests
# =====================================================================

class TestStateBuilder:
    """Tests for state vector construction."""

    def test_shape_and_dtype(self) -> None:
        s = build_state("hello", set(), None, 5, 5, 0, 1000)
        assert s.shape == (STATE_DIM,)
        assert s.dtype == np.float32

    def test_semantic_region_nonzero(self) -> None:
        s = build_state("some query text", set(), None, 5, 5, 0, 1000)
        assert np.count_nonzero(s[0:384]) > 0

    def test_tool_history_flags(self) -> None:
        history = {ActionType.CALCULATOR, ActionType.ANSWER_DIRECTLY}
        s = build_state("q", history, None, 5, 5, 0, 1000)
        assert s[384 + ActionType.CALCULATOR] == 1.0
        assert s[384 + ActionType.ANSWER_DIRECTLY] == 1.0
        assert s[384 + ActionType.SEARCH_WEB] == 0.0

    def test_outcome_encoding(self) -> None:
        tr = ToolResult(success=True, output=42, error=None, latency_ms=1.5, token_cost=0)
        s = build_state("q", set(), tr, 5, 5, 0, 1000)
        assert s[391] == 1.0  # success flag

    def test_budget_features(self) -> None:
        s = build_state("q", set(), None, 3, 5, 200, 1000)
        assert abs(s[455] - 3 / 5) < 1e-6  # turns remaining
        assert abs(s[456] - 200 / 1000) < 1e-6  # token cost

    def test_padding_is_zero(self) -> None:
        s = build_state("q", set(), None, 5, 5, 0, 1000)
        assert np.all(s[457:512] == 0.0)

    def test_deterministic(self) -> None:
        s1 = build_state("test query", set(), None, 5, 5, 0, 1000)
        s2 = build_state("test query", set(), None, 5, 5, 0, 1000)
        np.testing.assert_array_equal(s1, s2)


# =====================================================================
# Tasks tests
# =====================================================================

class TestTasks:
    """Tests for the hand-written task set."""

    def test_ten_tasks(self) -> None:
        assert len(TASKS) == 10

    def test_task_format(self) -> None:
        for t in TASKS:
            assert isinstance(t, Task)
            assert t.id.startswith("arith-")
            assert t.tier == 1
            assert len(t.query) > 0
            assert len(t.ground_truth) > 0
            assert ActionType.CALCULATOR in t.required_tools

    def test_unique_ids(self) -> None:
        ids = [t.id for t in TASKS]
        assert len(set(ids)) == len(ids)


# =====================================================================
# Environment tests
# =====================================================================

class TestSliceEnv:
    """Tests for the Gymnasium environment."""

    def _make_task(self) -> Task:
        return Task(
            id="test-001",
            tier=1,
            query="What is 2 + 3?",
            required_tools=[ActionType.CALCULATOR],
            ground_truth="5",
        )

    def test_reset_returns_correct_shapes(self) -> None:
        env = SliceEnv()
        obs, info = env.reset(task=self._make_task())
        assert obs.shape == (STATE_DIM,)
        assert "task_id" in info

    def test_reset_requires_task(self) -> None:
        env = SliceEnv()
        with pytest.raises(ValueError, match="requires a Task"):
            env.reset()

    def test_step_returns_reward_components(self) -> None:
        env = SliceEnv()
        env.reset(task=self._make_task())
        obs, reward, term, trunc, info = env.step(ActionType.CALCULATOR)
        assert isinstance(reward, RewardComponents)
        assert hasattr(reward, "accuracy")
        assert hasattr(reward, "efficiency")
        assert hasattr(reward, "safety")
        assert hasattr(reward, "bonus")
        assert hasattr(reward, "total")

    def test_calculator_then_terminate_correct(self) -> None:
        env = SliceEnv()
        env.reset(task=self._make_task())
        env.step(ActionType.CALCULATOR)
        _, reward, terminated, _, info = env.step(ActionType.TERMINATE)
        assert terminated is True
        assert reward.accuracy == 1.0
        assert reward.bonus == 0.2

    def test_efficiency_penalty(self) -> None:
        env = SliceEnv()
        env.reset(task=self._make_task())
        _, reward, _, _, _ = env.step(ActionType.CALCULATOR)
        assert reward.efficiency == -0.1

    def test_truncation_at_max_turns(self) -> None:
        env = SliceEnv()
        env.reset(task=self._make_task())
        for _ in range(5):
            _, _, term, trunc, _ = env.step(ActionType.ANSWER_DIRECTLY)
            if term or trunc:
                break
        assert trunc is True

    def test_safety_always_zero(self) -> None:
        env = SliceEnv()
        env.reset(task=self._make_task())
        _, reward, _, _, _ = env.step(ActionType.CALCULATOR)
        assert reward.safety == 0.0


# =====================================================================
# Dummy policy tests
# =====================================================================

class TestDummyPolicy:
    """Tests for the rule-based policy."""

    def test_arithmetic_query_uses_calculator(self) -> None:
        policy = DummyPolicy()
        task = Task(
            id="t1", tier=1, query="What is 2 + 3?",
            required_tools=[ActionType.CALCULATOR], ground_truth="5",
        )
        policy.reset(task)
        assert policy.act() == ActionType.CALCULATOR
        assert policy.act() == ActionType.TERMINATE

    def test_percentage_query_uses_calculator(self) -> None:
        policy = DummyPolicy()
        task = Task(
            id="t2", tier=1, query="Calculate 15% of 200.",
            required_tools=[ActionType.CALCULATOR], ground_truth="30",
        )
        policy.reset(task)
        assert policy.act() == ActionType.CALCULATOR

    def test_non_arithmetic_uses_answer_directly(self) -> None:
        policy = DummyPolicy()
        task = Task(
            id="t3", tier=1, query="What is the capital of France?",
            required_tools=[], ground_truth="Paris",
        )
        policy.reset(task)
        assert policy.act() == ActionType.ANSWER_DIRECTLY
        assert policy.act() == ActionType.TERMINATE

    def test_overrun_returns_terminate(self) -> None:
        policy = DummyPolicy()
        task = Task(
            id="t4", tier=1, query="1+1",
            required_tools=[ActionType.CALCULATOR], ground_truth="2",
        )
        policy.reset(task)
        policy.act()  # CALCULATOR
        policy.act()  # TERMINATE
        assert policy.act() == ActionType.TERMINATE  # beyond plan


# =====================================================================
# Integration test — full episode
# =====================================================================

class TestIntegration:
    """End-to-end episode test."""

    def test_full_episode(self) -> None:
        env = SliceEnv()
        policy = DummyPolicy()
        task = TASKS[0]  # "What is 2 + 3?" → ground truth "5"

        policy.reset(task)
        obs, info = env.reset(task=task)

        done = False
        steps = 0
        reward = None
        while not done:
            action = policy.act()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps == 2  # CALCULATOR → TERMINATE
        assert info["current_answer"] == "5"
        # Final step reward should have accuracy=1.0
        assert reward is not None
        assert reward.accuracy == 1.0
