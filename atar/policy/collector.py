"""
M7 Vectorized Collector — batched rollout collection across N ATAREnv instances.

Architecture overview
---------------------
N ATAREnv instances share a single LLMInterface object (one set of model weights).
Each collection cycle advances all N environments by one step, using:

  1. A SINGLE encode_batch() call across all N current message lists →
     one GPU forward pass producing (N, 384) semantic vectors.
  2. A SINGLE batched policy.forward() call → (N, 7) logits + (N, 1) values.
  3. N sequential env.step() calls (I/O bound, not GPU bound).

ATAREnv is constructed with skip_internal_encode=True, which prevents each
env.step() from calling llm.encode() internally.  The collector takes ownership
of the semantic encoding and writes the correct batched vectors into the stored
EpisodeStep states.  This trades N sequential encode() calls (one inside each
env.step()) for a single encode_batch() call — the primary GPU throughput gain.

Episode management
------------------
Episodes that terminate or are truncated reset independently and continue
contributing to the shared rollout buffer.  No synchronisation is needed or
performed across the N instances.

See ARCHITECTURE_DECISIONS.md for the full design rationale and the trade-offs
considered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from atar.llm.llm_interface import LLMInterface
from atar.env.atar_env import ATAREnv
from atar.policy.policy_network import ATARPolicy
from atar.policy.rollout_buffer import RolloutBuffer
from atar.policy.types import CurriculumPhase
from atar.shared.shared_types import (
    ActionType,
    EpisodeStep,
    RewardComponents,
    SEMANTIC_RANGE,
)

logger = logging.getLogger(__name__)


class VectorizedCollector:
    """
    Manages N parallel ATAREnv instances and collects rollouts via batched
    encode() and policy forward passes.

    Parameters
    ----------
    llm : LLMInterface
        Shared LLM instance (frozen base model + trainable projection).
    policy : ATARPolicy
        Policy network — called in eval mode during collection (no grad).
    n_envs : int
        Number of parallel environment instances.  Configurable; 8-16
        is the MSD-recommended starting point.
    rollout_steps : int
        Total transitions to collect per call to collect().
        MSD default: 2048.
    data_dir : str | Path | None
        Path to the data directory for ATAREnv.
    seed : int | None
        RNG seed; each env receives a different seed derived from this.
    allowed_tiers : list[int] | None
        If set, restricts which task tiers are sampled.  Updated by
        PPOTrainer via set_allowed_tiers() on phase transitions.
    """

    def __init__(
        self,
        llm: LLMInterface,
        policy: ATARPolicy,
        n_envs: int = 8,
        rollout_steps: int = 2048,
        data_dir: str | Path | None = None,
        seed: int | None = None,
        allowed_tiers: list[int] | None = None,
    ) -> None:
        self._llm = llm
        self._policy = policy
        self._n_envs = n_envs
        self._rollout_steps = rollout_steps
        self._allowed_tiers = allowed_tiers  # None = all tiers

        # Determine device from the policy (or default to cpu).
        try:
            self._device = next(policy.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        # Create N envs with skip_internal_encode=True.
        # Each env gets a different seed so episode sampling is independent.
        logger.info("Creating %d ATAREnv instances (skip_internal_encode=True)", n_envs)
        self._envs: list[ATAREnv] = [
            ATAREnv(
                llm=llm,
                data_dir=data_dir,
                seed=(seed + i) if seed is not None else None,
                skip_internal_encode=True,
            )
            for i in range(n_envs)
        ]

        # Reset all envs and capture initial states.
        # _messages[i] mirrors envs[i]._messages — we keep a live reference.
        self._obs: list[np.ndarray] = []
        self._episode_info: list[dict[str, Any]] = []
        for env in self._envs:
            obs, info = env.reset()
            self._obs.append(obs)
            self._episode_info.append(info)

        # Per-episode accumulators (for logging)
        self._episode_rewards: list[float] = [0.0] * n_envs
        self._episode_lengths: list[int] = [0] * n_envs
        self._completed_episodes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_allowed_tiers(self, tiers: list[int]) -> None:
        """Update the allowed task tiers on all envs (called on phase transition)."""
        self._allowed_tiers = tiers
        # ATAREnv doesn't filter by tier during reset() — the collector
        # enforces curriculum by re-rolling reset() until a valid tier is drawn.
        # This is a no-op on the env objects themselves; the filtering happens
        # in _curriculum_reset() below.

    def collect(self, n_steps: int | None = None) -> tuple[RolloutBuffer, dict[str, Any]]:
        """
        Collect rollout data across all N environments.

        Each cycle:
          1. encode_batch() → (N, 384) — single GPU pass.
          2. policy.forward() on (N, 512) states — single GPU pass.
          3. Loop over N envs calling step() with the sampled action.
          4. Terminated/truncated envs reset independently.

        Parameters
        ----------
        n_steps : int | None
            Total transitions to collect.  Defaults to self._rollout_steps.

        Returns
        -------
        buffer : RolloutBuffer
            Filled buffer (n_steps transitions, advantages/returns = 0).
        stats : dict
            Collection statistics for W&B logging.
        """
        steps = n_steps if n_steps is not None else self._rollout_steps
        buffer = RolloutBuffer(capacity=steps)

        self._completed_episodes.clear()
        total_steps_collected = 0

        while total_steps_collected < steps:
            # ── Step 1: Batch-encode all N current message lists ──────────
            batch_msgs = [env._messages for env in self._envs]   # N message lists
            # encode_batch() → (N, 384) float32 on CPU
            semantic_vecs: torch.Tensor = self._llm.encode_batch(batch_msgs)

            # ── Step 2: Build full (N, 512) state vectors ─────────────────
            states_np = np.stack(self._obs, axis=0)   # (N, 512) from last obs
            states = torch.as_tensor(states_np, dtype=torch.float32)
            # Overwrite semantic slice with the freshly-computed batch encoding.
            s, e = SEMANTIC_RANGE
            states[:, s:e] = semantic_vecs             # (N, 384) → slice of (N, 512)

            # ── Step 3: Policy forward pass ───────────────────────────────
            self._policy.eval()
            with torch.no_grad():
                states_dev = states.to(self._device)
                logits, values = self._policy(states_dev)   # (N, 7), (N, 1)
            dist = Categorical(logits=logits)
            actions = dist.sample()                         # (N,)
            log_probs = dist.log_prob(actions)              # (N,)

            # ── Step 4: Step all N environments ───────────────────────────
            for i, env in enumerate(self._envs):
                if total_steps_collected >= steps:
                    break

                action_int = int(actions[i].item())
                action = ActionType(action_int)

                next_obs, reward_float, terminated, truncated, info = env.step(action)

                # Reconstruct a RewardComponents from the scalar (M5 step() returns
                # the total as a plain float; the full breakdown is in info).
                reward_components: RewardComponents = info.get(
                    "reward_components",
                    RewardComponents(
                        accuracy=reward_float,
                        efficiency=0.0,
                        safety=0.0,
                        bonus=0.0,
                    ),
                )

                done = bool(terminated or truncated)

                # Build the EpisodeStep — state has the batched semantic vector.
                ep_step = EpisodeStep(
                    state=states[i].cpu().numpy(),       # (512,) with batched semantic
                    action=action,
                    reward=reward_components,
                    next_state=next_obs,                 # from env (semantic=zeros, ok for buffer)
                    done=done,
                    log_prob=float(log_probs[i].item()),
                    value_estimate=float(values[i].item()),
                )
                buffer.add(ep_step)
                total_steps_collected += 1

                # Per-episode tracking
                self._episode_rewards[i] += reward_float
                self._episode_lengths[i] += 1

                if done:
                    self._completed_episodes.append({
                        "reward": self._episode_rewards[i],
                        "length": self._episode_lengths[i],
                        "tier": info.get("task_tier", -1),
                    })
                    self._episode_rewards[i] = 0.0
                    self._episode_lengths[i] = 0
                    # Reset — apply curriculum tier filtering if active
                    next_obs, reset_info = self._curriculum_reset(env)
                    self._episode_info[i] = reset_info

                self._obs[i] = next_obs

        stats = self._compute_stats()
        return buffer, stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _curriculum_reset(self, env: ATAREnv) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset env, strictly sampling only from allowed tiers.
        """
        if self._allowed_tiers is None:
            return env.reset()

        allowed_tasks = [t for t in env._tasks if t.tier in self._allowed_tiers]
        if not allowed_tasks:
            raise RuntimeError(f"No tasks available for allowed tiers {self._allowed_tiers}")
        
        task = env._rng.choice(allowed_tasks)
        return env.reset(task=task)

    def _compute_stats(self) -> dict[str, Any]:
        """Aggregate collection statistics for W&B logging."""
        if not self._completed_episodes:
            return {
                "n_episodes_completed": 0,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
            }
        rewards = [ep["reward"] for ep in self._completed_episodes]
        lengths = [ep["length"] for ep in self._completed_episodes]
        return {
            "n_episodes_completed": len(self._completed_episodes),
            "mean_episode_reward": float(np.mean(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
        }
