"""
M7 PPO loss and GAE (originally architect stubs; file name kept so imports
don't move).

PPOTrainer calls compute_gae() once per env trajectory (_fill_gae) and
compute_ppo_loss() once per minibatch (_run_ppo_update).

References:
- PPO: Schulman et al. 2017, https://arxiv.org/abs/1707.06347
- GAE: Schulman et al. 2016, https://arxiv.org/abs/1506.02438
"""
import torch
from atar.policy.types import RolloutBatch, PPOLossOutput
from atar.policy.policy_network import ATARPolicy


def compute_ppo_loss(
    rollout_batch: RolloutBatch,
    policy: ATARPolicy,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
) -> PPOLossOutput:
    """
    Clipped-surrogate PPO loss for one minibatch:
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy.

    Advantages are normalised per minibatch; returns are not (they are the
    value-head targets).
    """
    device = next(policy.parameters()).device
    states = rollout_batch.states.to(device)
    actions = rollout_batch.actions.to(device)
    old_log_probs = rollout_batch.log_probs.to(device)   # <-- field is log_probs, not old_log_probs
    advantages = rollout_batch.advantages.to(device)
    returns = rollout_batch.returns.to(device)

    logits, values = policy(states)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)
    # std() of a single element is NaN (a size-1 final minibatch would turn
    # every weight into NaN) — only normalise when it's defined.
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = ((values.squeeze(-1) - returns) ** 2).mean()
    entropy = dist.entropy().mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return PPOLossOutput(policy_loss, value_loss, entropy, total_loss)


def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                 gamma: float, lam: float) -> list[float]:
    """
    GAE over ONE env's time-ordered trajectory.  ``values`` has
    len(rewards) + 1 entries; the last bootstraps a trajectory cut
    mid-episode.  Never pass transitions interleaved from several envs —
    see PPOTrainer._fill_gae.
    """
    advantages = [0.0] * len(rewards)
    last_advantage = 0.0
    # values needs one extra entry: V(s_{T}) for bootstrapping the last step
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
        advantages[t] = last_advantage
    return advantages