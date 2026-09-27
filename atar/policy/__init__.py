"""
M7 — RL Policy & Training module.

Public exports:
    ATARPolicy          — shared-backbone actor-critic network
    RolloutBuffer       — transition store for PPO updates
    VectorizedCollector — N-env batched rollout collection
    CurriculumManager   — three-phase curriculum phase management
    PPOTrainer          — PPO training loop scaffolding
    RandomBaseline      — uniform random action baseline
    ReActBaseline       — ReAct-prompted LLM baseline
    compute_ppo_loss    — PPO loss stub (architect implements)
    compute_gae         — GAE stub (architect implements)
    CurriculumPhase     — phase enum (PHASE_1/2/3)
    RolloutBatch        — tensor container for one PPO update
    PPOLossOutput       — four loss terms from compute_ppo_loss
    CheckpointState     — checkpoint serialisation container
"""

from atar.policy.policy_network import ATARPolicy
from atar.policy.rollout_buffer import RolloutBuffer
from atar.policy.collector import VectorizedCollector
from atar.policy.curriculum import CurriculumManager
from atar.policy.trainer import PPOTrainer
from atar.policy.baselines import RandomBaseline, ReActBaseline
from atar.policy.ppo_stubs import compute_ppo_loss, compute_gae
from atar.policy.types import (
    CurriculumPhase,
    RolloutBatch,
    PPOLossOutput,
    CheckpointState,
)

__all__ = [
    "ATARPolicy",
    "RolloutBuffer",
    "VectorizedCollector",
    "CurriculumManager",
    "PPOTrainer",
    "RandomBaseline",
    "ReActBaseline",
    "compute_ppo_loss",
    "compute_gae",
    "CurriculumPhase",
    "RolloutBatch",
    "PPOLossOutput",
    "CheckpointState",
]
