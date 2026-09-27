"""
Tests for M7 RolloutBuffer.

No GPU or LLM required — pure tensor/data structure tests.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from atar.shared.shared_types import ActionType, EpisodeStep, RewardComponents, STATE_DIM
from atar.policy.rollout_buffer import RolloutBuffer
from atar.policy.types import RolloutBatch


def _make_step(
    action: ActionType = ActionType.ANSWER_DIRECTLY,
    reward: float = 1.0,
    done: bool = False,
    log_prob: float = -1.5,
    value: float = 0.5,
) -> EpisodeStep:
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[0] = float(action)   # just distinguish steps
    return EpisodeStep(
        state=state,
        action=action,
        reward=RewardComponents(accuracy=reward, efficiency=0.0, safety=0.0, bonus=0.0),
        next_state=np.zeros(STATE_DIM, dtype=np.float32),
        done=done,
        log_prob=log_prob,
        value_estimate=value,
    )


def test_buffer_add_and_len() -> None:
    """Buffer length increments correctly after add() calls."""
    buf = RolloutBuffer(capacity=10)
    assert len(buf) == 0
    buf.add(_make_step())
    assert len(buf) == 1
    buf.add(_make_step())
    assert len(buf) == 2


def test_buffer_is_full() -> None:
    """is_full() triggers at capacity; add() beyond capacity raises."""
    buf = RolloutBuffer(capacity=3)
    for _ in range(3):
        buf.add(_make_step())
    assert buf.is_full()
    with pytest.raises(RuntimeError, match="full"):
        buf.add(_make_step())


def test_buffer_clear() -> None:
    """clear() resets buffer to empty."""
    buf = RolloutBuffer(capacity=5)
    for _ in range(5):
        buf.add(_make_step())
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_full()


def test_to_batch_shapes() -> None:
    """to_batch() produces tensors of the correct shape for T transitions."""
    T = 6
    buf = RolloutBuffer(capacity=T)
    for i in range(T):
        buf.add(_make_step(
            action=ActionType(i % len(ActionType)),
            reward=float(i),
            done=(i == T - 1),
        ))
    batch = buf.to_batch()

    assert isinstance(batch, RolloutBatch)
    assert batch.states.shape == (T, STATE_DIM)
    assert batch.actions.shape == (T,)
    assert batch.rewards.shape == (T,)
    assert batch.values.shape == (T,)
    assert batch.log_probs.shape == (T,)
    assert batch.advantages.shape == (T,)
    assert batch.returns.shape == (T,)
    assert batch.dones.shape == (T,)


def test_to_batch_dtypes() -> None:
    """Tensor dtypes must match documented types."""
    T = 4
    buf = RolloutBuffer(capacity=T)
    for _ in range(T):
        buf.add(_make_step())
    batch = buf.to_batch()

    assert batch.states.dtype == torch.float32
    assert batch.actions.dtype == torch.int64
    assert batch.rewards.dtype == torch.float32
    assert batch.values.dtype == torch.float32
    assert batch.log_probs.dtype == torch.float32
    assert batch.advantages.dtype == torch.float32
    assert batch.returns.dtype == torch.float32
    assert batch.dones.dtype == torch.bool


def test_to_batch_reward_values() -> None:
    """Rewards in batch must match the total from RewardComponents."""
    buf = RolloutBuffer(capacity=2)
    buf.add(_make_step(reward=3.0))
    buf.add(_make_step(reward=7.0))
    batch = buf.to_batch()
    assert float(batch.rewards[0]) == pytest.approx(3.0)
    assert float(batch.rewards[1]) == pytest.approx(7.0)


def test_to_batch_done_flags() -> None:
    """Done flags must correctly reflect step.done."""
    buf = RolloutBuffer(capacity=3)
    buf.add(_make_step(done=False))
    buf.add(_make_step(done=True))
    buf.add(_make_step(done=False))
    batch = buf.to_batch()
    assert batch.dones[0].item() is False
    assert batch.dones[1].item() is True
    assert batch.dones[2].item() is False


def test_advantages_returns_initialised_zero() -> None:
    """Advantages and returns start as zeros from to_batch(); GAE fills them."""
    buf = RolloutBuffer(capacity=4)
    for _ in range(4):
        buf.add(_make_step())
    batch = buf.to_batch()
    assert torch.all(batch.advantages == 0.0)
    assert torch.all(batch.returns == 0.0)


def test_to_batch_empty_raises() -> None:
    """Converting an empty buffer raises RuntimeError."""
    buf = RolloutBuffer(capacity=10)
    with pytest.raises(RuntimeError, match="empty"):
        buf.to_batch()


def test_minibatches_count_and_shapes() -> None:
    """
    minibatches() splits a RolloutBatch into correct-sized chunks.
    Last batch may be smaller; all shapes must be consistent.
    """
    T = 10
    buf = RolloutBuffer(capacity=T)
    for _ in range(T):
        buf.add(_make_step())
    batch = buf.to_batch()

    mbs = buf.minibatches(batch, minibatch_size=4, shuffle=False)
    # T=10, minibatch_size=4 → [4, 4, 2]
    assert len(mbs) == 3
    assert mbs[0].states.shape[0] == 4
    assert mbs[1].states.shape[0] == 4
    assert mbs[2].states.shape[0] == 2


def test_minibatches_total_coverage() -> None:
    """All T transitions must appear exactly once across minibatches."""
    T = 9
    buf = RolloutBuffer(capacity=T)
    for i in range(T):
        buf.add(_make_step(reward=float(i)))
    batch = buf.to_batch()

    mbs = buf.minibatches(batch, minibatch_size=3, shuffle=True)
    all_rewards = torch.cat([mb.rewards for mb in mbs]).sort().values
    expected = torch.arange(T, dtype=torch.float32).sort().values
    assert torch.allclose(all_rewards, expected)
