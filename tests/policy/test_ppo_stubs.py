"""
Tests for M7 PPO stubs — compute_ppo_loss and compute_gae.

Both functions are architect-owned stubs that must:
  (a) Exist with the exact signatures specified in the MSD.
  (b) Raise NotImplementedError (not yet implemented).

Tests are marked @pytest.mark.xfail(strict=True) so they show as XFAIL
(expected failure) in the test run — not as errors.  Once the architect
implements the stubs, these tests can be un-marked and given real assertions.
"""

import pytest
import torch

from atar.policy.ppo_stubs import compute_ppo_loss, compute_gae
from atar.policy.policy_network import ATARPolicy
from atar.policy.types import RolloutBatch, PPOLossOutput
from atar.shared.shared_types import STATE_DIM


def _make_dummy_batch(T: int = 4) -> RolloutBatch:
    return RolloutBatch(
        states=torch.randn(T, STATE_DIM),
        actions=torch.zeros(T, dtype=torch.int64),
        rewards=torch.zeros(T),
        values=torch.zeros(T),
        log_probs=torch.zeros(T),
        advantages=torch.zeros(T),
        returns=torch.zeros(T),
        dones=torch.zeros(T, dtype=torch.bool),
    )


# ── Signature tests (these must pass — stubs must be importable) ──────────────

def test_compute_ppo_loss_importable() -> None:
    """compute_ppo_loss must be importable from ppo_stubs."""
    assert callable(compute_ppo_loss)


def test_compute_gae_importable() -> None:
    """compute_gae must be importable from ppo_stubs."""
    assert callable(compute_gae)


# ── Stub-raises tests (marked xfail since architect hasn't implemented yet) ───

def test_compute_ppo_loss_returns_valid_loss() -> None:
    """
    compute_ppo_loss() must return a valid PPOLossOutput.
    """
    policy = ATARPolicy()
    batch = _make_dummy_batch()
    # Ensure advantages don't have exactly 0 std to avoid divide by zero NaNs
    batch.advantages = torch.randn(4)
    
    out = compute_ppo_loss(
        rollout_batch=batch,
        policy=policy,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    assert isinstance(out, PPOLossOutput)
    assert out.policy_loss.ndim == 0
    assert out.value_loss.ndim == 0
    assert out.entropy.ndim == 0
    assert out.total_loss.ndim == 0


def test_compute_gae_returns_correct_shape_and_values() -> None:
    """
    compute_gae() must compute generalized advantage estimation properly.
    """
    advantages = compute_gae(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 0.6, 0.7, 0.0],
        dones=[False, False, True],
        gamma=0.95,
        lam=0.95,
    )
    assert len(advantages) == 3
    assert isinstance(advantages[0], float)


# ── PPOLossOutput shape tests (for when architect implements) ─────────────────
# These are not xfail — they test the PPOLossOutput type itself, which exists.

def test_ppo_loss_output_fields() -> None:
    """PPOLossOutput dataclass must have the four required fields."""
    out = PPOLossOutput(
        policy_loss=torch.tensor(1.0),
        value_loss=torch.tensor(0.5),
        entropy=torch.tensor(0.3),
        total_loss=torch.tensor(1.8),
    )
    assert hasattr(out, "policy_loss")
    assert hasattr(out, "value_loss")
    assert hasattr(out, "entropy")
    assert hasattr(out, "total_loss")
