"""M8 replay records: stepping, captured fields, totals, JSONL round-trip (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from atar.demo.replay import append_records, iter_episode, load_records, run_episode, EpisodeRecord
from atar.env.atar_env import ATAREnv
from atar.shared.shared_types import ActionType
from tests.demo.conftest import ScriptedAgent


def _task(env: ATAREnv, task_id: str):
    return next(t for t in env._tasks if t.id == task_id)


def test_calculator_then_answer_is_recorded_step_by_step(stub_env: ATAREnv) -> None:
    """
    CALCULATOR (valid, succeeds): efficiency -0.5          -> cumulative -0.5
    ANSWER_DIRECTLY "4" (correct, 1 tool): +10 acc +2 bonus -> step +12, cumulative +11.5
    """
    agent = ScriptedAgent([ActionType.CALCULATOR, ActionType.ANSWER_DIRECTLY], with_probs=True)
    rec = run_episode(stub_env, agent, _task(stub_env, "calc-1"), version="step_100",
                      agent_name="atar", model_name="stub", checkpoint_step=100)

    assert rec.action_trace == ["CALCULATOR", "ANSWER_DIRECTLY"]
    s1, s2 = rec.steps
    assert s1.params == {"expression": "2 + 2"}
    assert s1.tool_success is True and s1.tool_output is not None and s1.tool_output.startswith("4")
    assert s1.reward["efficiency"] == pytest.approx(-0.5)
    assert s1.cumulative_reward == pytest.approx(-0.5)
    assert s1.action_probs is not None and s1.action_probs[int(ActionType.CALCULATOR)] == 1.0
    assert s1.value_estimate == pytest.approx(1.5)

    assert s2.reward["accuracy"] == pytest.approx(10.0)
    assert s2.reward["bonus"] == pytest.approx(2.0)
    assert s2.cumulative_reward == pytest.approx(11.5)
    assert s2.final_answer == "4"

    assert rec.correct is True and rec.terminated and not rec.truncated
    assert rec.total_reward == pytest.approx(11.5)
    assert rec.tool_calls == 1
    assert rec.recorded_at


def test_iter_episode_yields_one_step_at_a_time(stub_env: ATAREnv) -> None:
    """Live step-through relies on the generator advancing exactly one env step per next()."""
    task = _task(stub_env, "calc-1")
    record = EpisodeRecord(version="v", agent="atar", checkpoint_step=None, task_id=task.id,
                           tier=task.tier, query=task.query, ground_truth=task.ground_truth,
                           model_name="stub")
    steps = iter_episode(stub_env, ScriptedAgent([ActionType.CALCULATOR, ActionType.ANSWER_DIRECTLY]), task, record)
    first = next(steps)
    assert first.action == "CALCULATOR" and len(record.steps) == 1 and not record.terminated
    next(steps)
    assert record.terminated and record.correct is True
    assert next(steps, None) is None


def test_rejected_params_are_recorded(stub_env: ATAREnv) -> None:
    stub_env._generate_params = lambda conditioned: {}  # type: ignore[method-assign]
    rec = run_episode(stub_env, ScriptedAgent([ActionType.SQL_QUERY, ActionType.TERMINATE]),
                      _task(stub_env, "calc-1"), version="random", agent_name="random", model_name="stub")
    assert rec.steps[0].param_error and rec.steps[0].params is None
    assert rec.steps[0].reward["safety"] == pytest.approx(-10.0)


def test_no_ground_truth_means_ungraded(stub_env: ATAREnv) -> None:
    from atar.shared.shared_types import Task
    task = Task(id="live-1", tier=0, query="Anything?", required_tools=[], ground_truth="")
    rec = run_episode(stub_env, ScriptedAgent([ActionType.ANSWER_DIRECTLY]), task,
                      version="react", agent_name="react", model_name="stub")
    assert rec.correct is None


def test_max_turns_truncates_early(tmp_path: Path) -> None:
    from tests.demo.conftest import make_stub_env
    env = make_stub_env(tmp_path / "d", max_turns=2)
    rec = run_episode(env, ScriptedAgent([ActionType.CALCULATOR] * 5), _task(env, "calc-1"),
                      version="v", agent_name="random", model_name="stub")
    assert len(rec.steps) == 2 and rec.truncated and rec.correct is False


def test_jsonl_round_trip(stub_env: ATAREnv, tmp_path: Path) -> None:
    rec = run_episode(stub_env, ScriptedAgent([ActionType.CALCULATOR, ActionType.ANSWER_DIRECTLY], with_probs=True),
                      _task(stub_env, "calc-1"), version="step_5", agent_name="atar",
                      model_name="stub", checkpoint_step=5)
    path = tmp_path / "r.jsonl"
    append_records(path, [rec])
    append_records(path, [rec])
    loaded = load_records(path)
    assert len(loaded) == 2
    assert loaded[0] == rec


def test_real_atar_agent_probabilities_are_recorded(stub_env: ATAREnv) -> None:
    """
    Regression: the real ATARPolicy returns batched logits for a single state,
    which used to make action_distribution() return a nested list.  Uses the
    real ATARAgent (untrained policy) — the scripted agent above can't catch it.
    """
    import torch
    from atar.evaluation.runner import ATARAgent
    from atar.policy.policy_network import ATARPolicy

    torch.manual_seed(0)
    agent = ATARAgent(ATARPolicy())
    rec = run_episode(stub_env, agent, _task(stub_env, "calc-1"), version="step_0",
                      agent_name="atar", model_name="stub", checkpoint_step=0)
    for step in rec.steps:
        assert step.action_probs is not None
        assert len(step.action_probs) == 7
        assert all(isinstance(p, float) for p in step.action_probs)
        assert sum(step.action_probs) == pytest.approx(1.0, abs=1e-5)
        assert step.action == ActionType(max(range(7), key=step.action_probs.__getitem__)).name
        assert isinstance(step.value_estimate, float)
