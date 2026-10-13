"""
Regression tests for the 2026-10-13 M7 fixes (fast — no LLM is loaded):

  1. GAE is computed per env, not across the round-robin-interleaved buffer.
  2. Each env's cut-off trajectory bootstraps from V(current state).
  3. returns = values + RAW advantages (not normalised ones).
  4. The PPO update re-applies the projection with gradients (co-training),
     and leaves stored states untouched when the projection is frozen.
  5. compute_ppo_loss stays finite on a size-1 minibatch.

Expected numbers are hand-computed in the docstrings.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from atar.policy.policy_network import ATARPolicy
from atar.policy.ppo_stubs import compute_ppo_loss
from atar.policy.rollout_buffer import RolloutBuffer
from atar.policy.trainer import PPOTrainer
from atar.policy.types import RolloutBatch
from atar.shared.shared_types import (
    ActionType,
    EpisodeStep,
    RewardComponents,
    SEMANTIC_RANGE,
    STATE_DIM,
)


def _batch(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    env_ids: list[int] | None,
    bootstrap: list[float] | None,
) -> RolloutBatch:
    T = len(rewards)
    return RolloutBatch(
        states=torch.zeros(T, STATE_DIM),
        actions=torch.zeros(T, dtype=torch.int64),
        rewards=torch.tensor(rewards),
        values=torch.tensor(values),
        log_probs=torch.zeros(T),
        advantages=torch.zeros(T),
        returns=torch.zeros(T),
        dones=torch.tensor(dones),
        env_ids=torch.tensor(env_ids) if env_ids is not None else None,
        bootstrap_values=torch.tensor(bootstrap) if bootstrap is not None else None,
    )


def test_gae_is_computed_per_env_on_interleaved_buffer() -> None:
    """
    Two envs, buffer filled round-robin: [e0t0, e1t0, e0t1, e1t1].
    gamma = 0.5, lam = 0.5.

    env0: r=[0, 10], V=[1, 2], done=[F, T]          (episode ends)
      t1: delta = 10 - 2 = 8                      -> A = 8
      t0: delta = 0 + 0.5*2 - 1 = 0               -> A = 0 + 0.25*8 = 2
    env1: r=[-1, 0], V=[0, 4], done=[F, F], V(s_now)=6   (cut mid-episode)
      t1: delta = 0 + 0.5*6 - 4 = -1             -> A = -1
      t0: delta = -1 + 0.5*4 - 0 = 1             -> A = 1 + 0.25*(-1) = 0.75

    Flat advantages: [2, 0.75, 8, -1]
    Returns = V + A: [1+2, 0+0.75, 2+8, 4-1] = [3, 0.75, 10, 3]
    """
    batch = _batch(
        rewards=[0.0, -1.0, 10.0, 0.0],
        values=[1.0, 0.0, 2.0, 4.0],
        dones=[False, False, True, False],
        env_ids=[0, 1, 0, 1],
        bootstrap=[99.0, 6.0],   # env0's 99 must be ignored: its last step is terminal
    )
    PPOTrainer._fill_gae(SimpleNamespace(_gamma=0.5, _lam=0.5), batch)  # type: ignore[arg-type]

    assert batch.advantages.tolist() == pytest.approx([2.0, 0.75, 8.0, -1.0])
    assert batch.returns.tolist() == pytest.approx([3.0, 0.75, 10.0, 3.0])


def test_returns_use_raw_not_normalised_advantages() -> None:
    """
    Single 1-step terminal episode: r=10, V=0 -> A = 10, return = 10.
    A normalised advantage would be 0 (single element) or ~1, never 10.
    """
    batch = _batch([10.0], [0.0], [True], env_ids=[0], bootstrap=[0.0])
    PPOTrainer._fill_gae(SimpleNamespace(_gamma=0.95, _lam=0.95), batch)  # type: ignore[arg-type]
    assert batch.returns.item() == pytest.approx(10.0)
    assert batch.advantages.item() == pytest.approx(10.0)


def test_gae_without_env_ids_treats_buffer_as_one_trajectory() -> None:
    """Back-compat for hand-built batches: r=[1, 1], V=[0, 0, (0)], done=[F, T], gamma=lam=1 -> A=[2, 1]."""
    batch = _batch([1.0, 1.0], [0.0, 0.0], [False, True], env_ids=None, bootstrap=None)
    PPOTrainer._fill_gae(SimpleNamespace(_gamma=1.0, _lam=1.0), batch)  # type: ignore[arg-type]
    assert batch.advantages.tolist() == pytest.approx([2.0, 1.0])


class _FakeLLM:
    """Just the projection half of LLMInterface (hidden size 8)."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.projection = nn.Linear(8, 384)

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.projection(pooled.to(self.projection.weight.dtype))


def _projected_minibatch(llm: _FakeLLM, B: int = 4) -> RolloutBatch:
    pooled = torch.randn(B, 8)
    states = torch.randn(B, STATE_DIM)
    s, e = SEMANTIC_RANGE
    with torch.no_grad():
        states[:, s:e] = llm.project(pooled)
    return RolloutBatch(
        states=states,
        actions=torch.randint(0, len(ActionType), (B,)),
        rewards=torch.zeros(B),
        values=torch.zeros(B),
        log_probs=torch.full((B,), -1.9),
        advantages=torch.randn(B),
        returns=torch.randn(B),
        dones=torch.zeros(B, dtype=torch.bool),
        pooled=pooled,
    )


def test_ppo_update_sends_gradients_into_projection() -> None:
    llm = _FakeLLM()
    policy = ATARPolicy()
    mb = _projected_minibatch(llm)
    fake_trainer = SimpleNamespace(_llm=llm, _freeze_projection=False, _device="cpu")

    rebuilt = PPOTrainer._with_projected_states(fake_trainer, mb)  # type: ignore[arg-type]

    # Same projection weights -> identical states (so the PPO ratio starts at 1)...
    assert torch.allclose(rebuilt.states, mb.states, atol=1e-6)
    # ...but now differentiable w.r.t. the projection.
    loss = compute_ppo_loss(rebuilt, policy, 0.2, 0.5, 0.01).total_loss
    loss.backward()
    grad = llm.projection.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "projection received no gradient"


def test_frozen_projection_uses_stored_states() -> None:
    llm = _FakeLLM()
    mb = _projected_minibatch(llm)
    fake_trainer = SimpleNamespace(_llm=llm, _freeze_projection=True, _device="cpu")
    assert PPOTrainer._with_projected_states(fake_trainer, mb) is mb  # type: ignore[arg-type]


def test_ppo_loss_finite_on_single_sample_minibatch() -> None:
    """std() of one element is NaN; the loss must still be finite."""
    llm = _FakeLLM()
    mb = _projected_minibatch(llm, B=1)
    out = compute_ppo_loss(mb, ATARPolicy(), 0.2, 0.5, 0.01)
    assert torch.isfinite(out.total_loss)


def test_rollout_buffer_carries_pooled_and_env_ids_into_minibatches() -> None:
    buf = RolloutBuffer(capacity=4)
    for k in range(4):
        buf.add(
            EpisodeStep(
                state=torch.zeros(STATE_DIM).numpy(),
                action=ActionType.CALCULATOR,
                reward=RewardComponents(0.0, 0.0, 0.0, 0.0),
                next_state=torch.zeros(STATE_DIM).numpy(),
                done=False,
                log_prob=0.0,
                value_estimate=0.0,
            ),
            pooled=torch.full((8,), float(k)),
            env_id=k % 2,
        )
    buf.bootstrap_values = torch.tensor([1.0, 2.0])
    batch = buf.to_batch()

    assert batch.env_ids.tolist() == [0, 1, 0, 1]
    assert batch.pooled[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert batch.bootstrap_values.tolist() == [1.0, 2.0]

    for mb in buf.minibatches(batch, minibatch_size=2, shuffle=True):
        # pooled row k was filled with value k, and env_id = k % 2
        assert (mb.pooled[:, 0].long() % 2).tolist() == mb.env_ids.tolist()
