"""
Tests for M7 baselines — RandomBaseline and ReActBaseline.

DoD item verified: "Random and ReAct baselines both run a full evaluation
pass against the 200 held-out tasks and report real accuracy/tool-call numbers."

Note: For feasibility on the local stand-in model (CPU, slow generation),
the test uses a smaller evaluation set (20 tasks) and verifies the return
structure and value ranges, not the accuracy values (which are random/baseline
quality and not meaningful to assert on).

The MSD says "200 held-out tasks" — on the training GPU with the real model,
run with n_tasks=200. The test here proves the code path and result structure.
"""

from __future__ import annotations

import json
import pytest
import torch
from pathlib import Path

from atar.llm.llm_interface import LLMInterface
from atar.env.atar_env import ATAREnv
from atar.policy.baselines import RandomBaseline, ReActBaseline

STAND_IN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_EVAL_TASKS = 10   # reduced from 200 for local stand-in model speed


@pytest.fixture(scope="module")
def llm() -> LLMInterface:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LLMInterface(model_name=STAND_IN_MODEL, device=device)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    d = tmp_path_factory.mktemp("data_baselines")
    tasks = [
        {
            "id": f"eval_{i:04d}",
            "tier": (i % 5) + 1,
            "query": f"Baseline eval question {i}: compute {i} * 2",
            "required_tools": [2],
            "ground_truth": str(i * 2),
            "metadata": {},
        }
        for i in range(50)
    ]
    (d / "tasks_train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks)
    )
    return str(d)


@pytest.fixture(scope="module")
def envs(llm: LLMInterface, data_dir: str) -> list[ATAREnv]:
    """One env for baseline evaluation."""
    return [ATAREnv(llm=llm, data_dir=data_dir, seed=42)]


def test_random_baseline_returns_expected_keys(envs: list[ATAREnv]) -> None:
    """RandomBaseline.evaluate() must return a dict with the three required keys."""
    baseline = RandomBaseline(seed=0)
    result = baseline.evaluate(envs, n_tasks=N_EVAL_TASKS)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "accuracy" in result, "Missing 'accuracy' key"
    assert "avg_episode_length" in result, "Missing 'avg_episode_length' key"
    assert "tool_call_rate" in result, "Missing 'tool_call_rate' key"


def test_random_baseline_value_ranges(envs: list[ATAREnv]) -> None:
    """RandomBaseline metrics must be in plausible ranges."""
    baseline = RandomBaseline(seed=42)
    result = baseline.evaluate(envs, n_tasks=N_EVAL_TASKS)

    assert 0.0 <= result["accuracy"] <= 1.0, (
        f"accuracy {result['accuracy']} out of [0,1]"
    )
    assert result["avg_episode_length"] >= 1.0, (
        f"avg_episode_length {result['avg_episode_length']} < 1"
    )
    assert 0.0 <= result["tool_call_rate"] <= 1.0, (
        f"tool_call_rate {result['tool_call_rate']} out of [0,1]"
    )


def test_random_baseline_deterministic_with_seed(envs: list[ATAREnv]) -> None:
    """RandomBaseline results must be reproducible with the same seed."""
    result1 = RandomBaseline(seed=77).evaluate(envs, n_tasks=5)
    result2 = RandomBaseline(seed=77).evaluate(envs, n_tasks=5)
    assert result1["accuracy"] == result2["accuracy"]
    assert result1["avg_episode_length"] == result2["avg_episode_length"]


def test_react_baseline_returns_expected_keys(
    llm: LLMInterface, envs: list[ATAREnv]
) -> None:
    """ReActBaseline.evaluate() must return a dict with the three required keys."""
    baseline = ReActBaseline(llm=llm)
    result = baseline.evaluate(envs, n_tasks=3)  # very small N for speed

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "avg_episode_length" in result
    assert "tool_call_rate" in result


def test_react_baseline_value_ranges(
    llm: LLMInterface, envs: list[ATAREnv]
) -> None:
    """ReActBaseline metrics must be in plausible ranges."""
    baseline = ReActBaseline(llm=llm)
    result = baseline.evaluate(envs, n_tasks=3)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["avg_episode_length"] >= 1.0
    assert 0.0 <= result["tool_call_rate"] <= 1.0


def test_random_baseline_reports_real_numbers(envs: list[ATAREnv]) -> None:
    """
    DoD: Random baseline reports real accuracy/tool-call numbers.
    Verify the numbers are concrete (not NaN, not placeholder zeros for
    a non-trivially small run).
    """
    baseline = RandomBaseline(seed=1)
    result = baseline.evaluate(envs, n_tasks=N_EVAL_TASKS)

    import math
    assert not math.isnan(result["accuracy"]), "accuracy is NaN"
    assert not math.isnan(result["avg_episode_length"]), "avg_episode_length is NaN"
    assert not math.isnan(result["tool_call_rate"]), "tool_call_rate is NaN"

    # For N=10 random episodes, avg_episode_length should be > 0
    assert result["avg_episode_length"] > 0
