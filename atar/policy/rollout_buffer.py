"""
M7 Rollout Buffer — stores EpisodeStep records from N parallel environments.

Each PPO collection cycle fills the buffer to `capacity` total transitions
(across all N environments combined), then the buffer is converted to a
RolloutBatch tensor for the PPO update.

Episodes contribute independently — terminated episodes reset and keep
filling the buffer with fresh transitions from the new episode.  There is
no synchronisation across environment instances.
"""

from __future__ import annotations

import torch

from atar.shared.shared_types import EpisodeStep, SEMANTIC_RANGE
from atar.policy.types import RolloutBatch


class RolloutBuffer:
    """
    Fixed-capacity transition store for one PPO rollout.

    Parameters
    ----------
    capacity : int
        Total number of transitions to store before the buffer is considered
        full (MSD default: rollout_steps = 2048).
    state_dim : int
        Dimensionality of the state vector (must be 512).
    """

    def __init__(self, capacity: int = 2048, state_dim: int = 512) -> None:
        self._capacity = capacity
        self._state_dim = state_dim
        self._steps: list[EpisodeStep] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, step: EpisodeStep) -> None:
        """
        Append one transition to the buffer.

        Parameters
        ----------
        step : EpisodeStep
            The transition produced by VectorizedCollector for one env step.
        """
        if len(self._steps) >= self._capacity:
            raise RuntimeError(
                f"RolloutBuffer is full ({self._capacity} steps). "
                "Call clear() before adding more transitions."
            )
        self._steps.append(step)

    def is_full(self) -> bool:
        """True when the buffer has reached its capacity."""
        return len(self._steps) >= self._capacity

    def __len__(self) -> int:
        return len(self._steps)

    def to_batch(self) -> RolloutBatch:
        """
        Convert stored EpisodeStep records to a RolloutBatch of tensors.

        Advantages and returns are initialised to zeros here; the caller
        (PPOTrainer) must call compute_gae() to fill them before the update.

        Returns
        -------
        RolloutBatch
            All fields as tensors of shape (T,) or (T, state_dim).
        """
        if not self._steps:
            raise RuntimeError("Cannot convert an empty RolloutBuffer to a batch.")

        T = len(self._steps)

        states = torch.zeros(T, self._state_dim, dtype=torch.float32)
        actions = torch.zeros(T, dtype=torch.int64)
        rewards = torch.zeros(T, dtype=torch.float32)
        values = torch.zeros(T, dtype=torch.float32)
        log_probs = torch.zeros(T, dtype=torch.float32)
        dones = torch.zeros(T, dtype=torch.bool)

        for i, step in enumerate(self._steps):
            states[i] = torch.as_tensor(step.state, dtype=torch.float32)
            actions[i] = int(step.action)
            rewards[i] = float(step.reward.total)
            values[i] = float(step.value_estimate)
            log_probs[i] = float(step.log_prob)
            dones[i] = bool(step.done)

        return RolloutBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            advantages=torch.zeros(T, dtype=torch.float32),   # filled by compute_gae
            returns=torch.zeros(T, dtype=torch.float32),       # filled by compute_gae
            dones=dones,
        )

    def minibatches(
        self,
        batch: RolloutBatch,
        minibatch_size: int = 64,
        shuffle: bool = True,
    ) -> list[RolloutBatch]:
        """
        Split a RolloutBatch into shuffled minibatches for the PPO update.

        Parameters
        ----------
        batch : RolloutBatch
            Full rollout batch (T transitions).
        minibatch_size : int
            Target size for each minibatch (last may be smaller).
        shuffle : bool
            Whether to shuffle indices before splitting.

        Returns
        -------
        list[RolloutBatch]
            List of RolloutBatch slices.
        """
        T = batch.states.shape[0]
        indices = torch.randperm(T) if shuffle else torch.arange(T)
        minibatches: list[RolloutBatch] = []

        for start in range(0, T, minibatch_size):
            idx = indices[start:start + minibatch_size]
            minibatches.append(
                RolloutBatch(
                    states=batch.states[idx],
                    actions=batch.actions[idx],
                    rewards=batch.rewards[idx],
                    values=batch.values[idx],
                    log_probs=batch.log_probs[idx],
                    advantages=batch.advantages[idx],
                    returns=batch.returns[idx],
                    dones=batch.dones[idx],
                )
            )
        return minibatches

    def clear(self) -> None:
        """Reset the buffer for the next rollout."""
        self._steps.clear()
