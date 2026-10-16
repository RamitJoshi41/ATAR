"""
CLARIFY-once action mask (ADR-23) — fast, no LLM.

The 2026-10-16 diagnostic (greedy replays of checkpoint_step_15360) showed
the policy repeating CLARIFY until the turn limit on 9 of 20 tasks; the env
answers a repeated CLARIFY with the same text, so it can never help.  The
policy now reads the CLARIFY tool-history flag from its own input state and
gives CLARIFY probability 0 once it has been used.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from atar.policy.policy_network import ATARPolicy, action_mask
from atar.policy.ppo_stubs import compute_ppo_loss
from atar.policy.types import RolloutBatch
from atar.shared.shared_types import TOOL_HISTORY_RANGE, ActionType

CLARIFY = ActionType.CLARIFY.value
FLAG = TOOL_HISTORY_RANGE[0] + CLARIFY


def _states(n: int, clarify_used: bool) -> torch.Tensor:
    s = torch.randn(n, 512)
    s[:, TOOL_HISTORY_RANGE[0]:TOOL_HISTORY_RANGE[1]] = 0.0
    if clarify_used:
        s[:, FLAG] = 1.0
    return s


def test_mask_blocks_only_clarify_and_only_after_it_was_used() -> None:
    fresh, used = _states(4, False), _states(4, True)
    assert action_mask(fresh).all()
    m = action_mask(used)
    assert not m[:, CLARIFY].any()
    assert m[:, [a.value for a in ActionType if a != ActionType.CLARIFY]].all()


def test_used_clarify_has_zero_probability_and_is_never_sampled() -> None:
    torch.manual_seed(0)
    policy = ATARPolicy()
    with torch.no_grad():
        policy.policy_head.bias[CLARIFY] = 50.0     # a policy that "wants" CLARIFY
        logits, _ = policy(_states(8, True))
    dist = Categorical(logits=logits)
    assert torch.all(dist.probs[:, CLARIFY] == 0.0)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(8))
    assert not (dist.sample((500,)) == CLARIFY).any()
    assert torch.isfinite(dist.entropy()).all()

    with torch.no_grad():
        logits_fresh, _ = policy(_states(8, False))
    assert (Categorical(logits=logits_fresh).probs[:, CLARIFY] > 0.99).all()


def test_ppo_loss_and_gradients_stay_finite_with_masked_states() -> None:
    """The PPO update recomputes log-probs from the stored states, so it
    applies the same mask the collector sampled under."""
    torch.manual_seed(0)
    policy = ATARPolicy()
    states = torch.cat([_states(6, True), _states(6, False)])
    with torch.no_grad():
        logits, values = policy(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
    assert not (actions[:6] == CLARIFY).any()
    batch = RolloutBatch(
        states=states, actions=actions, rewards=torch.zeros(12),
        values=values.squeeze(-1), log_probs=log_probs,
        advantages=torch.randn(12), returns=torch.randn(12),
        dones=torch.zeros(12, dtype=torch.bool),
    )
    out = compute_ppo_loss(batch, policy, 0.2, 0.5, 0.01)
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
