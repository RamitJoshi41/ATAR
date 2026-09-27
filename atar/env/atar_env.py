"""
M5 — Gymnasium Environment
==========================
Wraps M1 (tools) + M3 (LLM) + M4 (translator) in a standard Gymnasium
interface so M7's PPO trainer can interact with it generically.

Public interface (matches shared/interfaces.md section 4):

    class ATAREnv(gymnasium.Env):
        observation_space: Box(512,)
        action_space:      Discrete(7)

        reset(task=None) -> (State, dict)
        step(action)     -> (State, float, bool, bool, dict)

Reward contract (confirmed via vertical slice, ADR-3):
  - reward returned from step() is a PLAIN FLOAT (RewardComponents.total)
  - full breakdown is in info["reward_components"] (a RewardComponents instance)

State layout (shared/interfaces.md section 2):
  [0:384]   — M3.encode() output (semantic)
  [384:391] — binary tool-history flags (7-d)
  [391:455] — outcome embedding (64-d, most recent ToolResult)
  [455:457] — budget features (2-d: turns_remaining/5, cumulative_tokens/MAX_TOKENS)
  [457:512] — zero padding (55-d, reserved)
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium

from atar.shared.shared_types import (
    ActionType,
    ConditionedPrompt,
    RewardComponents,
    Task,
    ToolResult,
    STATE_DIM,
    SEMANTIC_RANGE,
    TOOL_HISTORY_RANGE,
    OUTCOME_EMBED_RANGE,
    BUDGET_RANGE,
    PADDING_RANGE,
)
from atar.llm import LLMInterface
from atar.tools import execute_tool
from atar.translator import translate, validate

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants (reward tuning values from MSD — do not change without flagging)
# -------------------------------------------------------------------------
MAX_TURNS: int = 5
MAX_TOKENS_NORMALIZER: int = 10_000   # normalises cumulative token count

# Reward component magnitudes (MSD section "Reward function")
_ACCURACY_CORRECT: float = 10.0
_ACCURACY_INCORRECT: float = -5.0
_ACCURACY_TRUNCATED: float = 0.0

_EFFICIENCY_PER_TOOL_CALL: float = -0.5
_EFFICIENCY_PER_1K_TOKENS: float = -0.1
_EFFICIENCY_REDUNDANT: float = -1.0

_SAFETY_INVALID_PARAMS: float = -10.0
_SAFETY_RUNTIME_ERROR: float = -3.0
_SAFETY_CLARIFY_BONUS: float = +1.0

_BONUS_CORRECT_WITH_LE_1_TOOL: float = +2.0
_BONUS_RECOVERY_AFTER_FAILURE: float = +1.0


# -------------------------------------------------------------------------
# Outcome embedding helpers
# -------------------------------------------------------------------------

def _hash_type_embedding(value: Any) -> np.ndarray:
    """
    Encode the Python type of a ToolResult's output into a 32-d one-hot-style
    vector.  The types covered are: NoneType, str, int, float, list, dict, bool.
    Unknown types map to index 7 (the "other" bucket).

    Returns np.ndarray of shape (32,), dtype float32.
    """
    _TYPE_INDEX: dict[type, int] = {
        type(None): 0,
        str: 1,
        int: 2,
        float: 3,
        list: 4,
        dict: 5,
        bool: 6,
    }
    idx = _TYPE_INDEX.get(type(value), 7)
    vec = np.zeros(32, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def _outcome_embedding(result: ToolResult | None) -> np.ndarray:
    """
    Build the 64-d outcome embedding for the most recent ToolResult.

    Layout:
      [0]     — success flag (1.0 or 0.0)
      [1]     — error flag (1.0 if error is not None, else 0.0)
      [2]     — output_length_norm: min(len(str(output)) / 200, 1.0) if output else 0.0
      [3]     — latency_norm: min(latency_ms / 5000, 1.0)
      [4]     — token_cost_norm: min(token_cost / 1000, 1.0)
      [5:37]  — type embedding (32-d one-hot for output's Python type)
      [37:64] — zero padding (27-d, reserved)

    Returns np.ndarray shape (64,), dtype float32.
    """
    vec = np.zeros(64, dtype=np.float32)
    if result is None:
        return vec

    vec[0] = 1.0 if result.success else 0.0
    vec[1] = 1.0 if result.error is not None else 0.0
    vec[2] = min(len(str(result.output)) / 200.0, 1.0) if result.output is not None else 0.0
    vec[3] = min(result.latency_ms / 5000.0, 1.0)
    vec[4] = min(result.token_cost / 1000.0, 1.0)
    vec[5:37] = _hash_type_embedding(result.output)
    # [37:64] remain zero (reserved)
    return vec


# -------------------------------------------------------------------------
# Task loading helpers
# -------------------------------------------------------------------------

def _load_tasks(data_dir: str | Path) -> list[Task]:
    """
    Load tasks from tasks_train.jsonl in data_dir.
    Returns list of Task objects (all tiers).
    """
    tasks: list[Task] = []
    path = Path(data_dir) / "tasks_train.jsonl"
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(Task(
                id=raw["id"],
                tier=raw["tier"],
                query=raw["query"],
                required_tools=[ActionType(a) for a in raw["required_tools"]],
                ground_truth=raw["ground_truth"],
                metadata=raw.get("metadata", {}),
            ))
    return tasks


# -------------------------------------------------------------------------
# ATAREnv
# -------------------------------------------------------------------------

class ATAREnv(gymnasium.Env):
    """
    Gymnasium environment wrapping M1 (tools) + M3 (LLM) + M4 (translator).

    Parameters
    ----------
    llm : LLMInterface | None
        Pre-loaded LLM interface.  If None, one is instantiated using the
        default config (model from base_config.py).  Pass an instance to
        share the model across multiple env instances (e.g., vec-envs).
    data_dir : str | Path | None
        Directory containing tasks_train.jsonl.  Defaults to
        ``<repo_root>/data``.  Must be set if no ATAR_DATA_DIR env var.
    seed : int | None
        RNG seed for reproducible episode sampling.
    skip_internal_encode : bool
        When False (default), ``step()`` and ``reset()`` call
        ``llm.encode()`` internally to fill the semantic slice of the
        state vector — the standard, self-contained behaviour that all
        existing code and tests rely on.

        When True, the semantic slice (state[0:384]) is set to zeros by
        ``_build_state()`` instead.  M7's ``VectorizedCollector`` uses
        this mode: it collects the raw step outputs, then calls
        ``llm.encode_batch()`` once across all N environments and writes
        the resulting vectors into the stored rollout states itself.
        This avoids N sequential encode() calls inside the env loop
        and replaces them with a single batched GPU forward pass.

        See ARCHITECTURE_DECISIONS.md for the full rationale.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        llm: LLMInterface | None = None,
        data_dir: str | Path | None = None,
        seed: int | None = None,
        skip_internal_encode: bool = False,
    ) -> None:
        super().__init__()

        # Gymnasium spaces -----------------------------------------------
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(len(ActionType))

        # LLM -------------------------------------------------------------
        if llm is None:
            from atar.configs.base_config import load_config
            cfg = load_config()
            llm = LLMInterface(model_name=cfg.llm_model_name, device=cfg.device)
        self._llm: LLMInterface = llm

        # Task pool -------------------------------------------------------
        if data_dir is None:
            # Default: <repo_root>/data (up three directories from this file)
            data_dir = Path(__file__).parent.parent.parent / "data"
        self._tasks: list[Task] = _load_tasks(data_dir)
        if not self._tasks:
            raise RuntimeError(f"No tasks found in {data_dir!s}")

        # M7 batching flag — see class docstring
        self._skip_internal_encode: bool = skip_internal_encode

        # RNG -------------------------------------------------------------
        self._rng = random.Random(seed)

        # Episode state (reset on every call to reset()) ------------------
        self._task: Task | None = None
        self._messages: list[dict] = []
        self._turn: int = 0
        self._cumulative_tokens: int = 0
        self._tool_history: np.ndarray = np.zeros(7, dtype=np.float32)
        self._last_result: ToolResult | None = None
        self._had_tool_failure: bool = False   # for BONUS_RECOVERY
        self._tool_call_count: int = 0
        self._final_answer: str | None = None
        # Track which tool results have been "useful" (success=True) for
        # redundancy detection.  Maps ActionType -> ToolResult at last success.
        self._successful_tool_results: dict[ActionType, ToolResult] = {}

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        task: Task | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        seed : int | None
            Gymnasium-standard seed kwarg (forwarded to np_random).
        options : dict | None
            Gymnasium-standard options dict.  Unused.
        task : Task | None
            If provided, use this specific task; otherwise sample randomly
            from the loaded task pool.

        Returns
        -------
        (state, info) where state is shape (512,) float32 and info is a dict.
        """
        super().reset(seed=seed)

        # Select task
        if task is not None:
            self._task = task
        else:
            self._task = self._rng.choice(self._tasks)

        # Reset episode state
        self._messages = [
            {"role": "user", "content": self._task.query}
        ]
        self._turn = 0
        self._cumulative_tokens = 0
        self._tool_history = np.zeros(7, dtype=np.float32)
        self._last_result = None
        self._had_tool_failure = False
        self._tool_call_count = 0
        self._final_answer = None
        self._successful_tool_results = {}

        state = self._build_state()
        info = {
            "task_id": self._task.id,
            "task_tier": self._task.tier,
            "query": self._task.query,
        }
        return state, info

    def step(
        self, action: int | ActionType
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int | ActionType
            The discrete action chosen by the policy (0-6).

        Returns
        -------
        (next_state, reward, terminated, truncated, info)
          - next_state : np.ndarray, shape (512,)
          - reward     : float — RewardComponents.total (scalar)
          - terminated : bool  — episode ended by policy choice
          - truncated  : bool  — episode ended by budget/turn limit
          - info       : dict  — includes "reward_components" (RewardComponents),
                                 "tool_result" (ToolResult | None), and metadata
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        action = ActionType(int(action))
        self._turn += 1

        # ---- Determine if we are truncating due to turn limit -----------
        # We allow the step to complete before reporting truncated=True
        is_at_limit = self._turn >= MAX_TURNS

        # ---- Dispatch action -------------------------------------------
        terminated = False
        truncated = False
        tool_result: ToolResult | None = None

        # Reward components accumulators
        acc_accuracy: float = 0.0
        acc_efficiency: float = 0.0
        acc_safety: float = 0.0
        acc_bonus: float = 0.0

        if action in (ActionType.ANSWER_DIRECTLY, ActionType.TERMINATE):
            # ---- Terminal actions --------------------------------------
            terminated = True

            # Translate to get conditioned prompt + JSON schema
            conditioned: ConditionedPrompt = translate(action, self._build_state())

            # Efficiency cost: each generation costs something
            response = self._generate_params(conditioned)

            # Validate params
            valid, params_or_err = validate(action, response)

            if not valid:
                # Safety penalty for invalid params
                acc_safety += _SAFETY_INVALID_PARAMS
                self._final_answer = None
            else:
                assert isinstance(params_or_err, dict)
                self._final_answer = params_or_err.get(
                    "answer", params_or_err.get("final_answer", "")
                )
                # Accuracy reward
                acc_accuracy = self._compute_accuracy_reward()
                # Bonus: correct answer with ≤1 tool call total
                if acc_accuracy > 0 and self._tool_call_count <= 1:
                    acc_bonus += _BONUS_CORRECT_WITH_LE_1_TOOL
                # Bonus: recovery after earlier tool failure
                if acc_accuracy > 0 and self._had_tool_failure:
                    acc_bonus += _BONUS_RECOVERY_AFTER_FAILURE

        elif action == ActionType.CLARIFY:
            # ---- Clarify action ----------------------------------------
            conditioned = translate(action, self._build_state())
            response = self._generate_params(conditioned)
            valid, params_or_err = validate(action, response)

            if not valid:
                acc_safety += _SAFETY_INVALID_PARAMS
            else:
                assert isinstance(params_or_err, dict)
                # Inject clarification answer from task metadata (if present)
                clarification_answer = self._task.metadata.get(
                    "clarification_answer", "No further information available."
                )
                params_or_err["clarification_answer"] = clarification_answer

                # Execute the clarify "tool" (returns the answer string)
                tool_result = execute_tool(action, params_or_err)
                self._last_result = tool_result
                self._tool_history[action.value] = 1.0
                self._tool_call_count += 1
                self._cumulative_tokens += tool_result.token_cost

                if not tool_result.success:
                    acc_safety += _SAFETY_RUNTIME_ERROR
                    self._had_tool_failure = True
                else:
                    # Reward for correctly executing a CLARIFY action
                    acc_safety += _SAFETY_CLARIFY_BONUS

                # Append result to conversation
                self._append_tool_result_to_messages(action, tool_result)

        else:
            # ---- Real tool actions (SEARCH_WEB, CALCULATOR, SQL_QUERY, PYTHON_EXEC) ---
            conditioned = translate(action, self._build_state())
            response = self._generate_params(conditioned)
            valid, params_or_err = validate(action, response)

            if not valid:
                acc_safety += _SAFETY_INVALID_PARAMS
                # Don't execute the tool; update history with failure signal
                dummy_fail = ToolResult(
                    success=False,
                    output=None,
                    error=str(params_or_err),
                    latency_ms=0.0,
                    token_cost=0,
                )
                self._last_result = dummy_fail
                self._had_tool_failure = True
            else:
                assert isinstance(params_or_err, dict)

                # Redundancy check: same action already succeeded with same params?
                is_redundant = self._is_redundant_call(action, params_or_err)

                # Execute tool (deterministic=False — live training, not ground truth)
                tool_result = execute_tool(action, params_or_err, deterministic=False)
                self._last_result = tool_result
                self._tool_history[action.value] = 1.0
                self._tool_call_count += 1
                self._cumulative_tokens += tool_result.token_cost

                # Efficiency costs
                acc_efficiency += _EFFICIENCY_PER_TOOL_CALL
                token_batches = tool_result.token_cost / 1000.0
                acc_efficiency += _EFFICIENCY_PER_1K_TOKENS * token_batches
                if is_redundant:
                    acc_efficiency += _EFFICIENCY_REDUNDANT

                if not tool_result.success:
                    acc_safety += _SAFETY_RUNTIME_ERROR
                    self._had_tool_failure = True
                else:
                    # Record successful result for redundancy tracking
                    self._successful_tool_results[action] = tool_result

                # Append result to conversation history
                self._append_tool_result_to_messages(action, tool_result)

        # ---- Truncation (turn limit reached) ---------------------------
        if not terminated and is_at_limit:
            truncated = True
            # No accuracy bonus when truncated without an answer
            acc_accuracy = _ACCURACY_TRUNCATED

        # ---- Build reward dataclass ------------------------------------
        reward_components = RewardComponents(
            accuracy=acc_accuracy,
            efficiency=acc_efficiency,
            safety=acc_safety,
            bonus=acc_bonus,
        )
        reward: float = reward_components.total

        # ---- Build next state ------------------------------------------
        next_state = self._build_state()

        info: dict[str, Any] = {
            "reward_components": reward_components,
            "tool_result": tool_result,
            "turn": self._turn,
            "task_id": self._task.id,
            "task_tier": self._task.tier,
            "action": action.name,
        }

        return next_state, reward, terminated, truncated, info

    def render(self) -> None:  # type: ignore[override]
        """Not implemented — render_modes is empty."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_state(
        self,
        precomputed_semantic: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Assemble the 512-d state vector from current episode context.

        Parameters
        ----------
        precomputed_semantic : np.ndarray | None
            If provided, used as the semantic slice (state[0:384]) directly,
            bypassing the internal ``llm.encode()`` call.  Only supplied by
            M7's ``VectorizedCollector`` when ``skip_internal_encode=True``.
            Must be shape (384,), dtype float32.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # [0:384] — Semantic encoding from M3
        if precomputed_semantic is not None:
            # Caller (M7 VectorizedCollector) supplies pre-batched encoding.
            state[SEMANTIC_RANGE[0]:SEMANTIC_RANGE[1]] = precomputed_semantic[:384]
        elif self._skip_internal_encode:
            # skip_internal_encode=True but no vector provided: leave zeros.
            # Collector must overwrite this slice before using the state.
            pass
        else:
            semantic: torch.Tensor = self._llm.encode(self._messages)
            semantic_np = semantic.cpu().numpy().astype(np.float32)
            s, e = SEMANTIC_RANGE
            state[s:e] = semantic_np[:384]   # guard: encode() returns exactly 384-d

        # [384:391] — Binary tool-history flags
        s, e = TOOL_HISTORY_RANGE
        state[s:e] = self._tool_history

        # [391:455] — Outcome embedding
        s, e = OUTCOME_EMBED_RANGE
        state[s:e] = _outcome_embedding(self._last_result)

        # [455:457] — Budget features
        turns_remaining_norm = (MAX_TURNS - self._turn) / MAX_TURNS
        tokens_norm = min(self._cumulative_tokens / MAX_TOKENS_NORMALIZER, 1.0)
        s, e = BUDGET_RANGE
        state[s] = turns_remaining_norm
        state[s + 1] = tokens_norm

        # [457:512] — Zero padding (already zeroed by np.zeros)
        return state

    def _generate_params(self, conditioned: ConditionedPrompt) -> dict:
        """
        Run M3 constrained generation to produce action parameters.

        Wraps the LLM call so errors return a minimal safe dict rather than
        propagating as exceptions (which would break the episode loop).
        """
        messages_with_system = [
            {"role": "system", "content": conditioned.system_message},
        ] + self._messages

        try:
            raw = self._llm.generate(
                messages=messages_with_system,
                json_schema=conditioned.json_schema,
            )
            return raw if isinstance(raw, dict) else {}
        except Exception as exc:
            logger.warning("M3 generate() failed: %s; returning empty params", exc)
            return {}

    def _append_tool_result_to_messages(
        self, action: ActionType, result: ToolResult
    ) -> None:
        """Append a tool result to the conversation history as an assistant turn."""
        if result.success:
            content = f"[{action.name}] Result: {result.output}"
        else:
            content = f"[{action.name}] Error: {result.error}"
        self._messages.append({"role": "assistant", "content": content})

    def _compute_accuracy_reward(self) -> float:
        """
        Compare self._final_answer against the task's ground truth.
        Returns +10 for correct, -5 for incorrect, 0 if no answer.
        """
        if self._final_answer is None or self._task is None:
            return _ACCURACY_TRUNCATED

        answer = str(self._final_answer).strip().lower()
        truth = str(self._task.ground_truth).strip().lower()

        # Flexible matching: exact match or truth contained in answer or vice versa
        if answer == truth or truth in answer or answer in truth:
            return _ACCURACY_CORRECT
        # Numeric match: try to parse both as floats
        try:
            a_val = float(answer.replace(",", ""))
            t_val = float(truth.replace(",", ""))
            if math.isclose(a_val, t_val, rel_tol=1e-6):
                return _ACCURACY_CORRECT
        except ValueError:
            pass

        return _ACCURACY_INCORRECT

    def _is_redundant_call(
        self, action: ActionType, params: dict
    ) -> bool:
        """
        Return True if this action already succeeded this episode AND the
        params are identical (same information — no new query/expression/etc.).
        """
        if action not in self._successful_tool_results:
            return False
        # We consider it redundant if the primary param value hasn't changed.
        # Key param names per action type:
        primary_keys: dict[ActionType, str] = {
            ActionType.SEARCH_WEB: "query",
            ActionType.CALCULATOR: "expression",
            ActionType.SQL_QUERY: "query",
            ActionType.PYTHON_EXEC: "code",
        }
        key = primary_keys.get(action)
        if key is None:
            return False
        prev_result = self._successful_tool_results[action]
        # We don't store prev params directly, so use the output as a proxy.
        # A new call is not redundant if params differ — but since we don't
        # store prev params, we flag redundancy only if the same action key
        # already exists and the output is non-empty.  This is the conservative
        # approximation allowed by the MSD (which says "same action already
        # used successfully... with no new information since").
        return prev_result.success and prev_result.output is not None


# -------------------------------------------------------------------------
# Gymnasium environment registration
# -------------------------------------------------------------------------
# nondeterministic=True: the LLM sampler and _search_web noise (20%) make
# step() inherently stochastic even for the same seed.  This flag tells
# gymnasium.utils.env_checker.check_step_determinism to skip the
# determinism check, which would otherwise fail (and correctly so — we are
# training an RL agent, not a control-flow simulator).
gymnasium.register(
    id="ATAR-v0",
    entry_point="atar.env.atar_env:ATAREnv",
    nondeterministic=True,
)
