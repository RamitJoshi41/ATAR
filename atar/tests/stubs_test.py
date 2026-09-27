"""
Hand-verified unit tests for compute_gae and compute_ppo_loss.

These tests exist to catch implementation bugs (sign errors, off-by-ones)
against numbers worked out by hand, BEFORE spending real GPU time on
Kaggle discovering something is subtly wrong. Read the comments — the
expected values aren't magic, they're derived step by step below.
"""
import torch
import pytest
from atar.policy.ppo_stubs import compute_gae, compute_ppo_loss
from atar.policy.types import RolloutBatch, PPOLossOutput

import torch.nn as nn

class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)  # gives it real parameters to report a device

    def forward(self, states):
        batch_size = states.shape[0]
        logits = torch.zeros(batch_size, 7)
        values = torch.zeros(batch_size, 1)
        return logits, values

policy = FakePolicy()

def test_compute_gae_matches_hand_calculation():
    """
    3-step trajectory, hand-computed backward through the GAE recursion.

    rewards = [1.0, 0.0, 2.0]
    values  = [0.5, 0.5, 0.5, 0.0]   <- note: 4 entries (3 steps + 1 bootstrap V(s_T))
    dones   = [False, False, True]
    gamma = lam = 0.95

    Recursion: delta_t = r_t + gamma*V(s_{t+1})*(1-done_t) - V(s_t)
               A_t = delta_t + gamma*lam*(1-done_t)*A_{t+1}

    t=2 (last step, done=True):
        delta = 2.0 + 0.95*0.0*(0) - 0.5 = 1.5
        A_2 = 1.5 + 0.95*0.95*(0)*0 = 1.5          <- done zeroes out the carry term

    t=1:
        delta = 0.0 + 0.95*0.5*(1) - 0.5 = -0.025
        A_1 = -0.025 + 0.95*0.95*1*1.5 = -0.025 + 1.35375 = 1.32875

    t=0:
        delta = 1.0 + 0.95*0.5*(1) - 0.5 = 0.975
        A_0 = 0.975 + 0.95*0.95*1*1.32875 = 0.975 + 1.19929... = 2.17429...
    """
    rewards = [1.0, 0.0, 2.0]
    values = [0.5, 0.5, 0.5, 0.0]
    dones = [False, False, True]

    advantages = compute_gae(rewards, values, dones, gamma=0.95, lam=0.95)

    expected = [2.17429, 1.32875, 1.5]
    assert len(advantages) == 3
    for i, (a, e) in enumerate(zip(advantages, expected)):
        assert abs(a - e) < 1e-3, f"step {i}: got {a}, expected {e}"


def test_compute_gae_single_step_episode():
    """Sanity check on the simplest possible case: one step, done immediately.
    A_0 should just be the TD error with no bootstrap and no carry term."""
    rewards = [3.0]
    values = [1.0, 0.0]  # bootstrap value irrelevant since done=True zeroes it
    dones = [True]

    advantages = compute_gae(rewards, values, dones, gamma=0.95, lam=0.95)
    # delta = 3.0 + 0.95*0*(0) - 1.0 = 2.0 ; A_0 = 2.0 (no carry, first/last step)
    assert abs(advantages[0] - 2.0) < 1e-6


def test_compute_ppo_loss_ratio_one_reduces_to_negative_mean_advantage():
    """
    When new_log_probs == old_log_probs exactly, ratio = exp(0) = 1.0 for
    every sample. With ratio=1, both the clipped and unclipped surrogate
    terms are numerically identical to the (normalized) advantages
    themselves — clip(1.0, 1-eps, 1+eps) = 1.0 regardless of eps, since 1.0
    is always inside that range. So min(surr1, surr2) = advantages, and
    policy_loss = -advantages.mean().

    After normalization, a 2-element vector's mean is always ~0 by
    construction — so policy_loss should land at ~0.0 here. This isolates
    whether the clipping logic is wired correctly, independent of the
    harder-to-verify value loss and entropy terms.
    """
    torch.manual_seed(0)

    # Fake a policy whose forward() we control directly rather than a real
    # ATARPolicy — we only need it to return known logits/values.
 

    policy = FakePolicy()

    states = torch.zeros(2, 512)
    actions = torch.tensor([0, 1])
    # Uniform logits over 7 actions -> log_prob = log(1/7) for any action.
    # log_probs set to the SAME value -> ratio = exp(0) = 1.0 exactly.
    uniform_log_prob = torch.log(torch.tensor(1.0 / 7))
    log_probs = torch.full((2,), uniform_log_prob.item())
    advantages = torch.tensor([1.0, -1.0])  # will be normalized inside the function
    returns = torch.tensor([0.0, 0.0])      # values are also 0 -> value_loss = 0

    batch = RolloutBatch(
        states=states,
        actions=actions,
        rewards=torch.zeros(2),
        values=torch.zeros(2),
        log_probs=log_probs,
        advantages=advantages,
        returns=returns,
        dones=torch.zeros(2, dtype=torch.bool),
    )

    result = compute_ppo_loss(
        rollout_batch=batch,
        policy=policy,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    assert abs(result.policy_loss.item()) < 1e-5, (
        f"expected policy_loss ~0 with ratio=1 and normalized advantages, "
        f"got {result.policy_loss.item()}"
    )
    assert abs(result.value_loss.item()) < 1e-5, "values=returns=0, value_loss should be 0"