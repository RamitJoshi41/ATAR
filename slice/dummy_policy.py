"""
Dummy policy — hardcoded rules, no learning.

Strategy:
  1. If the query contains a digit and '%', use CALCULATOR.
  2. If the query contains only arithmetic-looking tokens, use CALCULATOR.
  3. Otherwise, ANSWER_DIRECTLY.
  4. Then TERMINATE.

This cycles through at most 2 actions per episode.
"""

from __future__ import annotations

import re

from atar.shared.shared_types import ActionType, Task


class DummyPolicy:
    """Deterministic rule-based policy for the vertical slice."""

    def __init__(self) -> None:
        self._step: int = 0
        self._plan: list[ActionType] = []

    def reset(self, task: Task) -> None:
        """Plan the action sequence for a new episode."""
        self._step = 0
        query = task.query

        # Decide whether to use the calculator.
        has_number = bool(re.search(r"\d", query))
        has_percent = "%" in query
        has_arith = bool(re.search(r"[+\-*/]", query))
        arith_keywords = {"calculate", "compute", "divide", "multiply", "add", "subtract"}
        has_keyword = bool(arith_keywords & set(query.lower().split()))

        if has_number and (has_percent or has_arith or has_keyword):
            self._plan = [ActionType.CALCULATOR, ActionType.TERMINATE]
        else:
            self._plan = [ActionType.ANSWER_DIRECTLY, ActionType.TERMINATE]

    def act(self) -> ActionType:
        """Return the next action in the plan."""
        if self._step < len(self._plan):
            action = self._plan[self._step]
        else:
            action = ActionType.TERMINATE
        self._step += 1
        return action
