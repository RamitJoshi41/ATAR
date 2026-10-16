"""
M7-local types for the RL Policy & Training module.

These are distinct from shared_types.py (which is the locked contract between
all modules).  These types are internal to M7 and not exposed to other modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch


class CurriculumPhase(IntEnum):
    """Three-phase curriculum as specified in the M7 MSD."""
    PHASE_1 = 1   # tiers 1-2 only   (0-30% of total training steps)
    PHASE_2 = 2   # tiers 1-4        (30-70%)
    PHASE_3 = 3   # all 5 tiers      (70-100%)


@dataclass
class RolloutBatch:
    """
    All tensors needed for a single PPO update, produced by RolloutBuffer.to_batch().

    Shapes assume T total collected transitions across all N environments.
    """
    states: torch.Tensor       # (T, 512) float32
    actions: torch.Tensor      # (T,)     int64
    rewards: torch.Tensor      # (T,)     float32
    values: torch.Tensor       # (T,)     float32  — from policy value head
    log_probs: torch.Tensor    # (T,)     float32  — log π(a|s) at collection time
    advantages: torch.Tensor   # (T,)     float32  — filled by compute_gae()
    returns: torch.Tensor      # (T,)     float32  — filled by compute_gae()
    dones: torch.Tensor        # (T,)     bool
    # Mean-pooled frozen-LLM hidden states (pre-projection).  When present,
    # the PPO update re-applies M3's projection to these WITH gradients so
    # the projection is co-trained (states[:, 0:384] holds the value it had
    # at collection time).  None in unit tests that build batches by hand.
    pooled: torch.Tensor | None = None          # (T, hidden_size) float32
    # Which of the N parallel envs produced each transition.  The buffer is
    # filled round-robin across envs, so GAE must be computed per env_id —
    # never across the flat buffer.  None = treat as one trajectory.
    env_ids: torch.Tensor | None = None         # (T,) int64
    # V(s) of each env's current (post-rollout) state, used to bootstrap
    # GAE when an env's last stored transition is not terminal.
    bootstrap_values: torch.Tensor | None = None  # (N,) float32


@dataclass
class PPOLossOutput:
    """
    Four loss terms returned by compute_ppo_loss() (architect-implemented stub).
    All are scalar tensors with gradients attached.
    """
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    total_loss: torch.Tensor


# Identifies the training problem (reward function, action mask, task set).
# Bump it whenever a change makes old checkpoints a different problem;
# PPOTrainer refuses to resume across versions.
#   2026-10-16: no answer = wrong answer (ADR-22), CLARIFY once per episode
#               (ADR-23), fixed tier-3 "capital length" ground truths.
TRAINING_SETUP_VERSION: str = "2026-10-16"


@dataclass
class CheckpointState:
    """
    Everything required to resume a training session exactly.

    Saved to disk as a .pt file.  Restored by PPOTrainer when --resume-from
    is passed to scripts/train.py.
    """
    step: int
    curriculum_phase: int        # int so it serialises cleanly; cast to CurriculumPhase on load
    policy_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    # M3 projection weights (co-trained with the policy).  None for
    # checkpoints written before the projection was co-trained.
    projection_state_dict: dict[str, Any] | None = None
    # W&B run id so a resumed session (e.g. a new Kaggle notebook) logs to
    # the same run instead of starting a new one.
    wandb_run_id: str | None = None
    # TRAINING_SETUP_VERSION the checkpoint was trained under.  None for
    # checkpoints written before the field existed (2026-10-16).
    training_setup_version: str | None = None
