"""
M7 PPO loss and GAE stubs — architect implements these personally.

These functions define the exact signatures that PPOTrainer calls.  They
raise NotImplementedError until the human architect fills in the bodies.
Do NOT implement these — see AGENTS.md and the M7 MSD for the split of
responsibility.

References for the architect:
- PPO: Schulman et al. 2017, https://arxiv.org/abs/1707.06347
- GAE: Schulman et al. 2016, https://arxiv.org/abs/1506.02438
"""
import torch
from atar.policy.types import RolloutBatch, PPOLossOutput
from atar.policy.policy_network import ATARPolicy


def compute_ppo_loss(rollout_batch, policy, clip_range, value_coef, entropy_coef):
    logits, values = policy(rollout_batch.states)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(rollout_batch.actions)

    ratio = torch.exp(new_log_probs - rollout_batch.log_probs)
    advantages = rollout_batch.advantages
    # normalize advantages — do this, it stabilizes training significantly
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = ((values.squeeze(-1) - rollout_batch.returns) ** 2).mean()
    entropy = dist.entropy().mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return PPOLossOutput(policy_loss, value_loss, entropy, total_loss)

def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                 gamma: float, lam: float) -> list[float]:
    advantages = [0.0] * len(rewards)
    last_advantage = 0.0
    # values needs one extra entry: V(s_{T}) for bootstrapping the last step
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
        advantages[t] = last_advantage
    return advantages