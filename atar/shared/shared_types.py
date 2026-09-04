"""
Shared types for ATAR. Locked contract — see interfaces.md for rationale.
Do not modify without human architect approval (see AGENTS.md rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ActionType(IntEnum):
    ANSWER_DIRECTLY = 0
    SEARCH_WEB = 1
    CALCULATOR = 2
    SQL_QUERY = 3
    PYTHON_EXEC = 4
    CLARIFY = 5
    TERMINATE = 6


# --- State vector layout constants (see interfaces.md section 2) ---
STATE_DIM = 512
SEMANTIC_RANGE = (0, 384)
TOOL_HISTORY_RANGE = (384, 391)
OUTCOME_EMBED_RANGE = (391, 455)
BUDGET_RANGE = (455, 457)
PADDING_RANGE = (457, 512)


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str | None
    latency_ms: float
    token_cost: int


@dataclass
class Task:
    id: str
    tier: int  # 1-5
    query: str
    required_tools: list[ActionType]
    ground_truth: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RewardComponents:
    accuracy: float
    efficiency: float
    safety: float
    bonus: float

    @property
    def total(self) -> float:
        return self.accuracy + self.efficiency + self.safety + self.bonus


@dataclass
class RegistryEntry:
    action: ActionType
    system_suffix: str
    output_schema: dict
    few_shot_examples: list[str] = field(default_factory=list)
    constraint_rules: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)


@dataclass
class ConditionedPrompt:
    action: ActionType
    system_message: str
    json_schema: dict


@dataclass
class EpisodeStep:
    """One (s, a, r, s') tuple for the rollout buffer, plus PPO bookkeeping."""

    state: Any  # np.ndarray, shape (STATE_DIM,)
    action: ActionType
    reward: RewardComponents
    next_state: Any
    done: bool
    log_prob: float
    value_estimate: float
