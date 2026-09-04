"""
Build a State vector (512-d numpy array) matching the layout in
shared/interfaces.md section 2.

For this vertical slice the semantic region [0:384] gets a simple
bag-of-characters hash projected into 384 dims — just enough to be non-zero
and deterministic.  Real M3 will replace this with LLM embeddings.
"""

from __future__ import annotations

import hashlib

import numpy as np

from atar.shared.shared_types import (
    STATE_DIM,
    SEMANTIC_RANGE,
    TOOL_HISTORY_RANGE,
    OUTCOME_EMBED_RANGE,
    BUDGET_RANGE,
    ActionType,
    ToolResult,
)


def _pseudo_embed(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic pseudo-embedding: SHA-512 hash expanded to `dim` floats
    in [-1, 1].  Not meaningful semantically — placeholder for LLM projection.
    """
    digest = hashlib.sha512(text.encode("utf-8")).digest()
    # Expand by repeating the hash bytes until we have enough.
    raw = digest * ((dim // len(digest)) + 1)
    arr = np.frombuffer(raw[:dim], dtype=np.uint8).astype(np.float32)
    # Normalize to [-1, 1]
    return (arr / 127.5) - 1.0


def build_state(
    query: str,
    tool_history: set[ActionType],
    last_tool_result: ToolResult | None,
    turns_remaining: int,
    max_turns: int,
    cumulative_token_cost: int,
    max_token_budget: int,
) -> np.ndarray:
    """Construct a 512-d state vector.

    Returns:
        np.ndarray of shape (512,) and dtype float32.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)

    # Region 1: semantic content [0:384]
    sem_start, sem_end = SEMANTIC_RANGE
    state[sem_start:sem_end] = _pseudo_embed(query, sem_end - sem_start)

    # Region 2: tool-history flags [384:391]
    th_start, th_end = TOOL_HISTORY_RANGE
    for action in tool_history:
        idx = th_start + int(action)
        if idx < th_end:
            state[idx] = 1.0

    # Region 3: outcome embedding [391:455]
    oe_start, oe_end = OUTCOME_EMBED_RANGE
    if last_tool_result is not None:
        # Simple encoding: first float = success flag, second = latency (ms, clipped)
        state[oe_start] = 1.0 if last_tool_result.success else -1.0
        state[oe_start + 1] = min(last_tool_result.latency_ms / 1000.0, 1.0)
        # Rest of outcome region stays zero for now.

    # Region 4: budget features [455:457]
    b_start, _ = BUDGET_RANGE
    state[b_start] = turns_remaining / max(max_turns, 1)
    state[b_start + 1] = cumulative_token_cost / max(max_token_budget, 1)

    # Region 5: padding [457:512] — zero-filled, untouched.

    return state
