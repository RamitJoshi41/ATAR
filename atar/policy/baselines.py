"""
M7 Baselines — Random and ReAct for comparison against the RL policy.

RandomBaseline: uniform random action selection at each step.
ReActBaseline:  uses M3 generate() directly with a ReAct prompting pattern
                (Yao et al. 2023) — no policy network, no RL.

Neither baseline is batched.  The MSD explicitly says ReAct can remain
single-example; the Random baseline has no LLM calls at all.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from atar.env.atar_env import ATAREnv
from atar.llm.llm_interface import LLMInterface
from atar.shared.shared_types import ActionType

logger = logging.getLogger(__name__)

# ReAct system prompt — instructs the model to follow Thought/Action/Observation
# interleaving (Yao et al. 2023, https://arxiv.org/abs/2210.03629).
_REACT_SYSTEM_PROMPT = """You are a ReAct agent that solves tasks by alternating
between Thought, Action, and Observation steps.

Available actions (choose exactly one per step):
  0 - ANSWER_DIRECTLY  (answer from internal knowledge, no tool)
  1 - SEARCH_WEB       (retrieve information from web/Wikipedia)
  2 - CALCULATOR       (evaluate arithmetic expression)
  3 - SQL_QUERY        (run SELECT query on database)
  4 - PYTHON_EXEC      (execute restricted Python code)
  5 - CLARIFY          (ask a clarifying question)
  6 - TERMINATE        (end the episode with current answer)

Your output must be valid JSON with key "action" (int 0-6)."""

_REACT_ACTION_SCHEMA = {
    "title": "ReActAction",
    "type": "object",
    "properties": {
        "action": {"type": "integer", "minimum": 0, "maximum": 6},
    },
    "required": ["action"],
}


class RandomBaseline:
    """
    Baseline: uniform random action selection each step.

    Parameters
    ----------
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def evaluate(
        self,
        envs: list[ATAREnv],
        n_tasks: int,
    ) -> dict[str, Any]:
        """
        Run n_tasks episodes with uniform random action selection.

        Parameters
        ----------
        envs : list[ATAREnv]
            Pool of pre-constructed environments to use.
        n_tasks : int
            Number of episodes to evaluate.

        Returns
        -------
        dict with keys:
            accuracy          — fraction of episodes where TERMINATE was chosen
                                (proxy for "gave an answer").
            avg_episode_length — mean number of steps per episode.
            tool_call_rate    — fraction of steps that were not ANSWER_DIRECTLY
                                or TERMINATE.
        """
        env = envs[0]  # use first env
        n_actions = len(ActionType)
        total_lengths: list[int] = []
        tool_calls = 0
        total_steps = 0
        n_terminated = 0

        for _ in range(n_tasks):
            env.reset()
            done = False
            ep_length = 0
            while not done:
                action_int = int(self._rng.integers(0, n_actions))
                _, _, terminated, truncated, _ = env.step(action_int)
                done = terminated or truncated
                ep_length += 1
                total_steps += 1
                if action_int not in (ActionType.ANSWER_DIRECTLY, ActionType.TERMINATE):
                    tool_calls += 1
                if terminated and action_int == ActionType.TERMINATE:
                    n_terminated += 1
            total_lengths.append(ep_length)

        return {
            "accuracy": n_terminated / n_tasks,
            "avg_episode_length": float(np.mean(total_lengths)),
            "tool_call_rate": tool_calls / max(total_steps, 1),
        }


class ReActBaseline:
    """
    Baseline: ReAct-style prompted reasoning+acting loop using M3 generate()
    directly, with no policy network.

    Follows the ReAct prompting pattern (Yao et al. 2023):
    each step appends a Thought/Action prompt and the env's tool result
    as an Observation before the next call.

    Not batched — MSD explicitly allows this baseline to be single-example.

    Parameters
    ----------
    llm : LLMInterface
        The shared LLM instance.
    """

    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    def evaluate(
        self,
        envs: list[ATAREnv],
        n_tasks: int,
    ) -> dict[str, Any]:
        """
        Run n_tasks episodes using the ReAct prompt loop.

        Parameters
        ----------
        envs : list[ATAREnv]
            Pool of environments; uses envs[0].
        n_tasks : int
            Number of episodes.

        Returns
        -------
        dict with keys: accuracy, avg_episode_length, tool_call_rate.
        """
        env = envs[0]
        total_lengths: list[int] = []
        tool_calls = 0
        total_steps = 0
        n_terminated = 0

        for _ in range(n_tasks):
            _, reset_info = env.reset()
            query = reset_info.get("query", "")
            done = False
            ep_length = 0

            # Build a conversation context for the ReAct loop.
            messages: list[dict] = [
                {"role": "system", "content": _REACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {query}\n\nBegin."},
            ]

            while not done:
                # Generate next action via constrained JSON generation.
                try:
                    result = self._llm.generate(
                        messages=messages,
                        json_schema=_REACT_ACTION_SCHEMA,
                        max_new_tokens=64,
                    )
                    action_int = int(result.get("action", ActionType.TERMINATE))
                    action_int = max(0, min(action_int, len(ActionType) - 1))
                except Exception as exc:
                    logger.warning("ReAct generate() failed: %s; using TERMINATE", exc)
                    action_int = ActionType.TERMINATE

                _, _, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated
                ep_length += 1
                total_steps += 1

                if action_int not in (ActionType.ANSWER_DIRECTLY, ActionType.TERMINATE):
                    tool_calls += 1
                if terminated and action_int == ActionType.TERMINATE:
                    n_terminated += 1

                # Append the tool result as an Observation turn.
                tool_result = info.get("tool_result")
                if tool_result is not None:
                    observation = (
                        f"Observation: success={tool_result.success}, "
                        f"output={str(tool_result.output)[:200]}"
                    )
                    messages.append({"role": "assistant", "content": observation})

            total_lengths.append(ep_length)

        return {
            "accuracy": n_terminated / n_tasks,
            "avg_episode_length": float(np.mean(total_lengths)),
            "tool_call_rate": tool_calls / max(total_steps, 1),
        }
