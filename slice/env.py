"""
Minimal Gymnasium-style environment for the ATAR vertical slice.

Action space: {ANSWER_DIRECTLY=0, CALCULATOR=2, TERMINATE=6}
Observation space: Box(-1, 1, shape=(512,), dtype=float32)

Reward components per interfaces.md:
  - accuracy:   +1.0 if final answer matches ground truth, else 0.0
  - efficiency: -0.1 per step (encourages shorter episodes)
  - safety:     0.0 (no safety-critical tool in this slice)
  - bonus:      +0.2 if the correct tool was used at least once
"""

from __future__ import annotations

import re
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atar.shared.shared_types import (
    STATE_DIM,
    ActionType,
    RewardComponents,
    Task,
    ToolResult,
)
from slice.calculator import execute_calculator
from slice.state_builder import build_state

# Actions available in this slice.
SLICE_ACTIONS: list[ActionType] = [
    ActionType.ANSWER_DIRECTLY,  # 0
    ActionType.CALCULATOR,       # 2
    ActionType.TERMINATE,        # 6
]

MAX_TURNS = 5


class SliceEnv(gym.Env):
    """Tiny ATAR environment for the vertical-slice demo."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        # Gymnasium spaces (for compatibility, not strictly needed here).
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        # Full 7-action space so indices match ActionType values.
        self.action_space = spaces.Discrete(7)

        self._task: Task | None = None
        self._tool_history: set[ActionType] = set()
        self._last_result: ToolResult | None = None
        self._current_answer: str = ""
        self._step_count: int = 0
        self._cumulative_token_cost: int = 0
        self._trajectory: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        task: Task | None = None,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode.

        Args:
            task: The Task to solve.  Required for this slice.
        """
        super().reset(seed=seed, options=options)
        if task is None:
            raise ValueError("SliceEnv.reset() requires a Task.")
        self._task = task
        self._tool_history = set()
        self._last_result = None
        self._current_answer = ""
        self._step_count = 0
        self._cumulative_token_cost = 0
        self._trajectory = []

        obs = self._build_obs()
        info: dict[str, Any] = {"task_id": task.id}
        return obs, info

    def step(  # type: ignore[override]
        self, action: int | ActionType
    ) -> tuple[np.ndarray, RewardComponents, bool, bool, dict[str, Any]]:
        """Execute one step.

        Returns:
            (obs, reward, terminated, truncated, info)
            Matches the M5 signature in interfaces.md — reward is a
            RewardComponents dataclass, not a bare float.
        """
        action = ActionType(action)
        self._step_count += 1
        self._tool_history.add(action)

        step_record: dict[str, Any] = {
            "step": self._step_count,
            "action": action.name,
        }

        terminated = False
        truncated = False

        # --- Execute the action ---
        if action == ActionType.CALCULATOR:
            expr = self._extract_expression(self._task.query)  # type: ignore[union-attr]
            result = execute_calculator(expr)
            self._last_result = result
            self._cumulative_token_cost += result.token_cost
            if result.success:
                self._current_answer = str(result.output)
            step_record["tool_result"] = {
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }

        elif action == ActionType.ANSWER_DIRECTLY:
            # Dummy "LLM" answer — just echo the query.
            self._current_answer = self._task.query  # type: ignore[union-attr]
            step_record["answer"] = self._current_answer

        elif action == ActionType.TERMINATE:
            terminated = True
            step_record["final_answer"] = self._current_answer

        else:
            # Action not available in this slice — treated as no-op.
            step_record["note"] = f"Action {action.name} unavailable in slice."

        # --- Truncate if budget exhausted ---
        if self._step_count >= MAX_TURNS and not terminated:
            truncated = True

        # --- Compute reward (only meaningful at episode end) ---
        reward = self._compute_reward(terminated or truncated)
        step_record["reward"] = {
            "accuracy": reward.accuracy,
            "efficiency": reward.efficiency,
            "safety": reward.safety,
            "bonus": reward.bonus,
            "total": reward.total,
        }

        self._trajectory.append(step_record)

        obs = self._build_obs()
        info: dict[str, Any] = {
            "task_id": self._task.id,  # type: ignore[union-attr]
            "step": self._step_count,
            "current_answer": self._current_answer,
            "trajectory": list(self._trajectory),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        return build_state(
            query=self._task.query if self._task else "",
            tool_history=self._tool_history,
            last_tool_result=self._last_result,
            turns_remaining=MAX_TURNS - self._step_count,
            max_turns=MAX_TURNS,
            cumulative_token_cost=self._cumulative_token_cost,
            max_token_budget=1000,
        )

    def _compute_reward(self, episode_done: bool) -> RewardComponents:
        """Compute reward components.

        - accuracy: awarded only at episode end, 1.0 if answer matches
          ground truth, else 0.0.
        - efficiency: -0.1 per step.
        - safety: always 0.0 (calculator is safe).
        - bonus: +0.2 if the correct tool was used.
        """
        efficiency = -0.1  # per-step cost

        if not episode_done:
            return RewardComponents(
                accuracy=0.0,
                efficiency=efficiency,
                safety=0.0,
                bonus=0.0,
            )

        # --- Episode-end rewards ---
        gt = self._task.ground_truth.strip() if self._task else ""
        answer = self._current_answer.strip()

        # Numeric comparison with tolerance.
        accuracy = 0.0
        try:
            if abs(float(answer) - float(gt)) < 1e-6:
                accuracy = 1.0
        except (ValueError, TypeError):
            if answer == gt:
                accuracy = 1.0

        # Bonus for using the right tool.
        bonus = 0.0
        if self._task and ActionType.CALCULATOR in self._tool_history:
            if ActionType.CALCULATOR in self._task.required_tools:
                bonus = 0.2

        return RewardComponents(
            accuracy=accuracy,
            efficiency=efficiency,
            safety=0.0,
            bonus=bonus,
        )

    @staticmethod
    def _extract_expression(query: str) -> str:
        """Best-effort extraction of an arithmetic expression from a natural
        language query.  Handles the 10 hand-written tasks, not general NLP.
        """
        # Handle "X + Y% of Z" pattern (e.g., "50 + 25% of 50")
        # Must be checked BEFORE plain "Y% of Z" to avoid partial match.
        pct_add_match = re.search(
            r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
            query,
        )
        if pct_add_match:
            base = pct_add_match.group(1)
            pct = pct_add_match.group(2)
            of_val = pct_add_match.group(3)
            return f"{base} + {pct} / 100 * {of_val}"

        # Handle percentage patterns like "15% of 200"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", query)
        if pct_match:
            return f"{pct_match.group(1)} / 100 * {pct_match.group(2)}"

        # Handle "spend X% ... Y" pattern
        spend_match = re.search(
            r"(\d+(?:\.\d+)?)\s*and\s*spend\s*(\d+(?:\.\d+)?)\s*%",
            query,
        )
        if spend_match:
            amount = spend_match.group(1)
            pct = spend_match.group(2)
            return f"{pct} / 100 * {amount}"

        # Handle word-based division like "Divide 999 by 3"
        div_match = re.search(
            r"[Dd]ivide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)",
            query,
        )
        if div_match:
            return f"{div_match.group(1)} / {div_match.group(2)}"

        # Fall back: extract arithmetic expression characters.
        expr = re.sub(r"[^0-9+\-*/().%\s\*]", "", query)
        expr = expr.strip()
        if not expr:
            expr = "0"
        return expr
