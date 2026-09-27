"""
M7-local types for the RL Policy & Training module.

These are distinct from shared_types.py (which is the locked contract between
all modules).  These types are internal to M7 and not exposed to other modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
