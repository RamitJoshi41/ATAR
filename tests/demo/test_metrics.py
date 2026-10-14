"""Session metrics panel — hand-computed values."""

from __future__ import annotations

import pytest

from atar.demo.metrics import session_metrics
from atar.demo.replay import EpisodeRecord


def _rec(correct: bool | None, tools: int, latency: float, tokens: int) -> EpisodeRecord:
    return EpisodeRecord(version="v", agent="atar", checkpoint_step=1, task_id="t", tier=1, query="q",
                         ground_truth="" if correct is None else "g", model_name="m", correct=correct,
                         tool_calls=tools, latency_s=latency, token_cost=tokens)


def test_empty() -> None:
    m = session_metrics([])
    assert m.episodes == 0 and m.accuracy is None


def test_hand_computed() -> None:
    """3 episodes: correct, wrong, ungraded -> accuracy 1/2 over 2 graded; tools (2+0+1)/3; latency (1+2+3)/3."""
    m = session_metrics([_rec(True, 2, 1.0, 100), _rec(False, 0, 2.0, 50), _rec(None, 1, 3.0, 0)])
    assert m.episodes == 3 and m.graded == 2
    assert m.accuracy == pytest.approx(0.5)
    assert m.avg_tool_calls == pytest.approx(1.0)
    assert m.mean_latency_s == pytest.approx(2.0)
    assert m.total_tokens == 150


def test_all_ungraded_accuracy_is_none() -> None:
    assert session_metrics([_rec(None, 0, 1.0, 0)]).accuracy is None
