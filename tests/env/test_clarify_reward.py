"""
CLARIFY reward regression tests (fast — no LLM is loaded).

M5 spec: "+1 for correctly recognizing and executing a CLARIFY action on an
ambiguous task".  The original env paid +1 for EVERY successful CLARIFY on
ANY task with no tool cost, so the policy could farm it (seen as ~15% of
actions in the 2026-10-14 Kaggle pilot, in a phase with no ambiguous tasks).

Now: +1 once per episode, only on tasks whose solution requires CLARIFY;
an unneeded or repeated CLARIFY costs the normal -0.5 tool-call penalty.

The LLM is stubbed out: skip_internal_encode=True (no encode) and
_generate_params is patched to return valid CLARIFY params; the real M4
validator and M1 sandbox still run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atar.env.atar_env import ATAREnv
from atar.shared.shared_types import ActionType


def _env(tmp_path: Path) -> ATAREnv:
    tasks = [
        {   # ambiguous: tier-4 style, solution starts with CLARIFY
            "id": "amb", "tier": 4, "query": "What is the capital today?",
            "required_tools": [5, 1], "ground_truth": "Paris",
            "metadata": {"clarification_answer": "France"},
        },
        {   # unambiguous arithmetic
            "id": "calc", "tier": 1, "query": "What is 2 + 2?",
            "required_tools": [2], "ground_truth": "4", "metadata": {},
        },
    ]
    (tmp_path / "tasks_train.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    env = ATAREnv(llm=object(), data_dir=tmp_path, seed=0, skip_internal_encode=True)  # type: ignore[arg-type]
    env._generate_params = lambda conditioned: {  # type: ignore[method-assign]
        "question": "Which country do you mean?",
        "clarification_answer": "(replaced by the env)",
    }
    return env


def _task(env: ATAREnv, task_id: str):
    return next(t for t in env._tasks if t.id == task_id)


def test_clarify_on_ambiguous_task_pays_bonus_once(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.reset(task=_task(env, "amb"))

    _, _, _, _, info = env.step(ActionType.CLARIFY)
    rc = info["reward_components"]
    assert info["tool_result"].success
    assert info["params_valid"] is True
    assert rc.safety == pytest.approx(1.0)
    assert rc.efficiency == pytest.approx(0.0)

    _, _, _, _, info = env.step(ActionType.CLARIFY)          # asking again
    rc = info["reward_components"]
    assert rc.safety == pytest.approx(0.0)
    assert rc.efficiency == pytest.approx(-0.5)


def test_clarify_on_unambiguous_task_costs_a_tool_call(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.reset(task=_task(env, "calc"))

    _, reward, _, _, info = env.step(ActionType.CLARIFY)
    rc = info["reward_components"]
    assert rc.safety == pytest.approx(0.0)
    assert rc.efficiency == pytest.approx(-0.5)
    assert reward == pytest.approx(-0.5)


def test_bonus_resets_between_episodes(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.reset(task=_task(env, "amb"))
    env.step(ActionType.CLARIFY)
    env.reset(task=_task(env, "amb"))
    _, _, _, _, info = env.step(ActionType.CLARIFY)
    assert info["reward_components"].safety == pytest.approx(1.0)
