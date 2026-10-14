"""
Shared fixtures for M8 demo tests — no LLM is loaded anywhere here.

``stub_env`` is a real ATAREnv (real M4 validator, real M1 tools, real reward
code) with the two LLM touch points removed: skip_internal_encode=True (no
encode) and _generate_params replaced by fixed, valid params per action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atar.env.atar_env import ATAREnv
from atar.shared.shared_types import ActionType

STUB_PARAMS: dict[str, dict[str, Any]] = {
    "CALCULATOR": {"expression": "2 + 2"},
    "SEARCH_WEB": {"query": "capital of France"},
    "ANSWER_DIRECTLY": {"answer": "4", "confidence": "high"},
    "TERMINATE": {"final_answer": "4", "reasoning": "computed"},
    "CLARIFY": {"question": "Which one?", "clarification_answer": "x"},
}

TASKS = [
    {"id": "calc-1", "tier": 1, "query": "What is 2 + 2?", "required_tools": [2],
     "ground_truth": "4", "metadata": {}},
    {"id": "geo-1", "tier": 2, "query": "Capital of France?", "required_tools": [0],
     "ground_truth": "Paris", "metadata": {}},
]


def make_stub_env(data_dir: Path, **kwargs: Any) -> ATAREnv:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "tasks_train.jsonl").write_text("\n".join(json.dumps(t) for t in TASKS))
    env = ATAREnv(llm=object(), data_dir=data_dir, seed=0, skip_internal_encode=True, **kwargs)  # type: ignore[arg-type]
    env._generate_params = lambda conditioned: dict(STUB_PARAMS.get(conditioned.action.name, {}))  # type: ignore[method-assign]
    return env


@pytest.fixture
def stub_env(tmp_path: Path) -> ATAREnv:
    return make_stub_env(tmp_path / "data")


class ScriptedAgent:
    """Plays a fixed action list; optionally exposes ATAR-style probabilities."""

    def __init__(self, actions: list[ActionType], with_probs: bool = False) -> None:
        self._actions = list(actions)
        self._i = 0
        self._with_probs = with_probs
        if with_probs:
            self.action_distribution = self._dist  # only present when asked

    def reset(self) -> None:
        self._i = 0

    def _dist(self, state: Any) -> tuple[list[float], float]:
        probs = [0.0] * 7
        probs[int(self._actions[self._i])] = 1.0
        return probs, 1.5

    def select_action(self, state: Any) -> int:
        action = self._actions[self._i]
        self._i += 1
        return int(action)
