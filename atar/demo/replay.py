"""
M8 — episode records shared by recorded replays and live runs.

``iter_episode`` steps any agent (ATAR, ReAct, zero-shot, random) through
one ATAREnv episode and yields a ``StepRecord`` per step, so the Streamlit
app can render a step the moment it happens (live mode / step-through).
``run_episode`` drains it into an ``EpisodeRecord``.  Recorded replays
(scripts/record_replays.py, run on the training GPU) are the same records
serialised to JSONL, so recorded and live episodes render identically.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from atar.env.answer_matching import is_answer_correct
from atar.env.atar_env import ATAREnv
from atar.shared.shared_types import ActionType, Task

TOOL_OUTPUT_CHARS = 600  # cap stored tool output so replay files stay small


@dataclass
class StepRecord:
    """One env step, as the demo shows it."""

    turn: int
    action: str
    action_probs: list[float] | None   # ATAR only: softmax over the 7 actions
    value_estimate: float | None       # ATAR only: critic's V(s)
    thought: str | None                # ReAct only: its Thought before acting
    params: dict[str, Any] | None      # LLM-generated params that passed M4
    param_error: str | None            # M4's rejection reason, if rejected
    tool_success: bool | None
    tool_output: str | None
    tool_error: str | None
    reward: dict[str, float]           # accuracy, efficiency, safety, bonus, total
    cumulative_reward: float
    final_answer: str | None
    latency_s: float


@dataclass
class EpisodeRecord:
    """A whole episode: which policy version ran which task, and how it went."""

    version: str                 # e.g. "step_6144", "react", "zeroshot", "random"
    agent: str                   # "atar" | "react" | "zeroshot" | "random"
    checkpoint_step: int | None  # training step for ATAR versions
    task_id: str
    tier: int
    query: str
    ground_truth: str            # "" for a free-text live query
    model_name: str
    steps: list[StepRecord] = field(default_factory=list)
    final_answer: str | None = None
    correct: bool | None = None  # None when there is no ground truth
    terminated: bool = False
    truncated: bool = False
    total_reward: float = 0.0
    tool_calls: int = 0
    token_cost: int = 0
    latency_s: float = 0.0
    live_tools: bool = False     # True if SEARCH_WEB used the real web
    recorded_at: str = ""

    @property
    def action_trace(self) -> list[str]:
        return [s.action for s in self.steps]


def _truncate(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= TOOL_OUTPUT_CHARS else text[:TOOL_OUTPUT_CHARS] + " …"


def iter_episode(
    env: ATAREnv,
    agent: Any,
    task: Task,
    record: EpisodeRecord,
) -> Iterator[StepRecord]:
    """
    Run ``agent`` on ``task`` in ``env`` one step at a time.

    Each yielded StepRecord is also appended to ``record.steps``; episode
    totals on ``record`` are filled in as the episode ends.  ``agent`` needs
    ``select_action(state) -> int``; optional hooks are used when present:
    ``reset()`` / ``reset(query)`` (zero-shot / ReAct), ``action_distribution``
    (ATAR), ``last_thought`` and ``observe(action, info)`` (ReAct).
    """
    if hasattr(agent, "reset"):
        try:
            agent.reset(task.query)
        except TypeError:
            agent.reset()

    state, _ = env.reset(task=task)
    t_start = time.monotonic()
    cumulative = 0.0
    done = False
    while not done:
        t0 = time.monotonic()
        probs: list[float] | None = None
        value: float | None = None
        if hasattr(agent, "action_distribution"):
            probs, value = agent.action_distribution(np.asarray(state))
        action_int = int(agent.select_action(state))
        thought = getattr(agent, "last_thought", None) or None

        state, reward, terminated, truncated, info = env.step(action_int)
        done = bool(terminated or truncated)
        if hasattr(agent, "observe"):
            agent.observe(action_int, info)

        rc = info["reward_components"]
        cumulative += float(reward)
        tool_result = info.get("tool_result")
        step = StepRecord(
            turn=int(info.get("turn", len(record.steps) + 1)),
            action=ActionType(action_int).name,
            action_probs=probs,
            value_estimate=value,
            thought=thought,
            params=info.get("params"),
            param_error=info.get("param_error"),
            tool_success=None if tool_result is None else bool(tool_result.success),
            tool_output=None if tool_result is None else _truncate(tool_result.output),
            tool_error=None if tool_result is None else _truncate(tool_result.error),
            reward={
                "accuracy": rc.accuracy, "efficiency": rc.efficiency,
                "safety": rc.safety, "bonus": rc.bonus, "total": float(reward),
            },
            cumulative_reward=cumulative,
            final_answer=info.get("final_answer"),
            latency_s=time.monotonic() - t0,
        )
        record.steps.append(step)

        if done:
            record.final_answer = info.get("final_answer")
            record.correct = (
                is_answer_correct(record.final_answer, task.ground_truth)
                if task.ground_truth else None
            )
            record.terminated = bool(terminated)
            record.truncated = bool(truncated)
            record.total_reward = cumulative
            record.tool_calls = int(info.get("tool_call_count", 0))
            record.token_cost = int(info.get("cumulative_tokens", 0))
            record.latency_s = time.monotonic() - t_start
            record.recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        yield step


def run_episode(
    env: ATAREnv,
    agent: Any,
    task: Task,
    *,
    version: str,
    agent_name: str,
    model_name: str,
    checkpoint_step: int | None = None,
    live_tools: bool = False,
) -> EpisodeRecord:
    """Run a whole episode and return its record."""
    record = EpisodeRecord(
        version=version, agent=agent_name, checkpoint_step=checkpoint_step,
        task_id=task.id, tier=task.tier, query=task.query,
        ground_truth=task.ground_truth, model_name=model_name,
        live_tools=live_tools,
    )
    for _ in iter_episode(env, agent, task, record):
        pass
    return record


# ─── JSONL I/O ───────────────────────────────────────────────────────────

def record_to_dict(record: EpisodeRecord) -> dict[str, Any]:
    return asdict(record)


def record_from_dict(raw: dict[str, Any]) -> EpisodeRecord:
    raw = dict(raw)
    steps = [StepRecord(**s) for s in raw.pop("steps", [])]
    return EpisodeRecord(**raw, steps=steps)


def append_records(path: Path, records: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(record_to_dict(r)) + "\n")


def load_records(path: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(record_from_dict(json.loads(line)))
    return records
