"""
M8 — the demo's bottom "live metrics" panel: running totals over the
episodes viewed or run in this browser session, per agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from atar.demo.replay import EpisodeRecord


@dataclass
class SessionMetrics:
    episodes: int
    graded: int                 # episodes with a ground truth to grade against
    accuracy: float | None      # over graded episodes; None if none graded
    avg_tool_calls: float
    mean_latency_s: float
    total_tokens: int


def session_metrics(records: list[EpisodeRecord]) -> SessionMetrics:
    """Aggregate metrics over ``records`` (one agent's episodes)."""
    if not records:
        return SessionMetrics(0, 0, None, 0.0, 0.0, 0)
    graded = [r for r in records if r.correct is not None]
    accuracy = (
        sum(1 for r in graded if r.correct) / len(graded) if graded else None
    )
    return SessionMetrics(
        episodes=len(records),
        graded=len(graded),
        accuracy=accuracy,
        avg_tool_calls=sum(r.tool_calls for r in records) / len(records),
        mean_latency_s=sum(r.latency_s for r in records) / len(records),
        total_tokens=sum(r.token_cost for r in records),
    )
