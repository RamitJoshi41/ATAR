"""
"No answer = wrong answer" (ADR-22) — fast, no LLM is loaded.

The M5 spec gave 0 accuracy to an episode truncated without an answer and
-5 to a wrong answer.  Five CLARIFYs cost 5 x -0.5 = -2.5, so stalling until
the turn limit was cheaper than answering, and the policy learned to stall
on anything it was unsure about (9 of 20 diagnostic episodes at step 15,360).

Same stubbing as test_clarify_reward.py: skip_internal_encode=True and a
patched _generate_params; the real M4 validator and M1 sandbox still run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atar.env.atar_env import ATAREnv, MAX_TURNS
from atar.shared.shared_types import ActionType


def _env(tmp_path: Path) -> ATAREnv:
    task = {
        "id": "calc", "tier": 1, "query": "What is 2 + 2?",
        "required_tools": [2], "ground_truth": "4", "metadata": {},
    }
    (tmp_path / "tasks_train.jsonl").write_text(json.dumps(task))
    return ATAREnv(llm=object(), data_dir=tmp_path, seed=0, skip_internal_encode=True)  # type: ignore[arg-type]


def _llm_returns(env: ATAREnv, params: dict[str, Any]) -> None:
    env._generate_params = lambda conditioned: dict(params)  # type: ignore[method-assign]


def _episode_return(env: ATAREnv, actions: list[ActionType]) -> tuple[float, dict[str, Any]]:
    env.reset(task=env._tasks[0])
    total, info = 0.0, {}
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total, info


def test_stalling_to_the_turn_limit_is_graded_as_a_wrong_answer(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _llm_returns(env, {"question": "Do you mean 2 + 2?", "clarification_answer": "x"})
    _, info = _episode_return(env, [ActionType.CLARIFY] * MAX_TURNS)
    assert info["reward_components"].accuracy == -5.0
    assert info["correct"] is False


def test_stalling_is_never_cheaper_than_answering_wrong(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _llm_returns(env, {"question": "Do you mean 2 + 2?", "clarification_answer": "x"})
    stall, _ = _episode_return(env, [ActionType.CLARIFY] * MAX_TURNS)

    _llm_returns(env, {"final_answer": "5", "reasoning": "guess"})
    wrong, info = _episode_return(env, [ActionType.TERMINATE])
    assert info["correct"] is False
    assert stall < wrong


def test_rejected_llm_params_are_reported_in_info(tmp_path: Path) -> None:
    """Lets training logs show WHY M4 rejected a tool call (e.g. CALCULATOR)."""
    env = _env(tmp_path)
    env.reset(task=env._tasks[0])
    _llm_returns(env, {"expression": "salary / 4"})
    _, _, _, _, info = env.step(ActionType.CALCULATOR)
    assert info["params_valid"] is False
    assert info["param_error"]
    assert info["rejected_params"] == {"expression": "salary / 4"}

    _llm_returns(env, {"expression": "2 + 2"})
    _, _, _, _, info = env.step(ActionType.CALCULATOR)
    assert info["params_valid"] is True and info["rejected_params"] is None
