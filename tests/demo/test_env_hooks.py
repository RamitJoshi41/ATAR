"""ATAREnv hooks added for M8: search backend swap, max_turns, step info."""

from __future__ import annotations

from pathlib import Path

import pytest

from atar.env.atar_env import ATAREnv
from atar.shared.shared_types import ActionType, ToolResult
from tests.demo.conftest import make_stub_env


def test_search_backend_replaces_simulated_search(tmp_path: Path) -> None:
    calls: list[dict] = []

    def backend(params: dict) -> ToolResult:
        calls.append(params)
        return ToolResult(success=True, output="LIVE RESULT", error=None, latency_ms=1.0, token_cost=0)

    env = make_stub_env(tmp_path, search_backend=backend)
    env.reset(task=env._tasks[1])
    _, _, _, _, info = env.step(ActionType.SEARCH_WEB)
    assert calls == [{"query": "capital of France"}]
    assert info["tool_result"].output == "LIVE RESULT"


def test_other_tools_ignore_search_backend(tmp_path: Path) -> None:
    def backend(params: dict) -> ToolResult:
        raise AssertionError("must not be called for CALCULATOR")

    env = make_stub_env(tmp_path, search_backend=backend)
    env.reset(task=env._tasks[0])
    _, _, _, _, info = env.step(ActionType.CALCULATOR)
    assert info["tool_result"].success


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_max_turns_must_shorten_not_extend(tmp_path: Path, bad: int) -> None:
    with pytest.raises(ValueError):
        make_stub_env(tmp_path, max_turns=bad)


def test_step_info_reports_params_and_errors(tmp_path: Path) -> None:
    env: ATAREnv = make_stub_env(tmp_path)
    env.reset(task=env._tasks[0])
    _, _, _, _, info = env.step(ActionType.CALCULATOR)
    assert info["params"] == {"expression": "2 + 2"} and info["param_error"] is None
    assert "cumulative_tokens" in info
    env._generate_params = lambda conditioned: {}  # type: ignore[method-assign]
    _, _, _, _, info = env.step(ActionType.SQL_QUERY)
    assert info["params"] is None and info["param_error"]
