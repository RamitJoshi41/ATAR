"""
M7 Vectorized Collector — batched rollout collection across N ATAREnv instances.

Architecture overview
---------------------
N ATAREnv instances share a single LLMInterface object (one set of model weights).
Each collection cycle advances all N environments by one step, using:

  1. A SINGLE encode_pooled_batch() call across all N current message lists →
     one GPU forward pass producing (N, hidden) pooled features, projected
     to (N, 384) by M3's projection (pooled features are stored so the
     PPO update can co-train the projection).
  2. A SINGLE batched policy.forward() call → (N, 7) logits + (N, 1) values.
  3. N sequential env.step() calls (I/O bound, not GPU bound).

ATAREnv is constructed with skip_internal_encode=True, which prevents each
env.step() from calling llm.encode() internally.  The collector takes ownership
of the semantic encoding and writes the correct batched vectors into the stored
EpisodeStep states.  This trades N sequential encode() calls (one inside each
env.step()) for a single batched encode call — the primary GPU throughput gain.

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
            obs, info = self._curriculum_reset(env)
            self._obs.append(obs)
            self._episode_info.append(info)

        # Per-episode accumulators (for logging)
        self._episode_rewards: list[float] = [0.0] * n_envs
        self._episode_lengths: list[int] = [0] * n_envs
        self._completed_episodes: list[dict[str, Any]] = []
        self._action_counts: np.ndarray = np.zeros(len(ActionType), dtype=np.int64)
        self._action_reward_sum: np.ndarray = np.zeros(len(ActionType), dtype=np.float64)
        self._action_invalid: np.ndarray = np.zeros(len(ActionType), dtype=np.int64)
        # (tier, action) step counts: shows whether e.g. CLARIFY is used on the
        # ambiguous tier-4 tasks or everywhere.
        self._tier_action_counts: dict[int, np.ndarray] = {}

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

    def reset_all(self) -> None:
        """
        Abandon every in-flight episode and start fresh ones from the
        currently allowed tiers.  Called after restoring a checkpoint so a
        resumed run samples from the correct curriculum phase immediately.
        """
        for i, env in enumerate(self._envs):
            obs, info = self._curriculum_reset(env)
            self._obs[i] = obs
            self._episode_info[i] = info
            self._episode_rewards[i] = 0.0
            self._episode_lengths[i] = 0

    def collect(self, n_steps: int | None = None) -> tuple[RolloutBuffer, dict[str, Any]]:
        """
        Collect rollout data across all N environments.

        Each cycle:
          1. encode_pooled_batch() + project() → (N, 384) — single GPU pass.
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
        self._action_counts[:] = 0
        self._action_reward_sum[:] = 0.0
        self._action_invalid[:] = 0
        self._tier_action_counts = {}
        total_steps_collected = 0

        while total_steps_collected < steps:
            # ── Steps 1-3: batch-encode all N envs, build states, run policy ──
            # pooled: (N, hidden) pre-projection features, stored per
            # transition so the PPO update can co-train the projection.
            pooled, states, logits, values = self._encode_and_evaluate()
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
                buffer.add(ep_step, pooled=pooled[i], env_id=i)
                total_steps_collected += 1
                self._action_counts[action_int] += 1
                self._action_reward_sum[action_int] += reward_float
                if not info.get("params_valid", True):
                    self._action_invalid[action_int] += 1
                tier = int(info.get("task_tier", -1))
                self._tier_action_counts.setdefault(
                    tier, np.zeros(len(ActionType), dtype=np.int64)
                )[action_int] += 1

                # Per-episode tracking
                self._episode_rewards[i] += reward_float
                self._episode_lengths[i] += 1

                if done:
                    self._completed_episodes.append({
                        "reward": self._episode_rewards[i],
                        "length": self._episode_lengths[i],
                        "tier": info.get("task_tier", -1),
                        "correct": bool(info.get("correct", False)),
                    })
                    self._episode_rewards[i] = 0.0
                    self._episode_lengths[i] = 0
                    # Reset — apply curriculum tier filtering if active
                    next_obs, reset_info = self._curriculum_reset(env)
                    self._episode_info[i] = reset_info

                self._obs[i] = next_obs

        # Bootstrap values for GAE: V(s) of the state each env is in now.
        # Only used for envs whose last stored transition was not terminal.
        _, _, _, last_values = self._encode_and_evaluate()
        buffer.bootstrap_values = last_values.squeeze(-1).detach().cpu().to(torch.float32)

        stats = self._compute_stats()
        return buffer, stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_and_evaluate(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One batched encode + one batched policy forward over all N envs.

        Returns
        -------
        pooled : (N, hidden_size) float32 CPU — pre-projection LLM features.
        states : (N, 512) float32 CPU — full states (semantic slice filled).
        logits : (N, 7) on the policy device.
        values : (N, 1) on the policy device.
        """
        batch_msgs = [env._messages for env in self._envs]   # N message lists
        pooled = self._llm.encode_pooled_batch(batch_msgs)    # one GPU pass
        with torch.no_grad():
            semantic_vecs = self._llm.project(pooled).cpu()   # (N, 384)

        states = torch.as_tensor(np.stack(self._obs, axis=0), dtype=torch.float32)
        s, e = SEMANTIC_RANGE
        states[:, s:e] = semantic_vecs

        self._policy.eval()
        with torch.no_grad():
            logits, values = self._policy(states.to(self._device))   # (N, 7), (N, 1)
        return pooled, states, logits, values

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
        stats: dict[str, Any] = {
            "n_episodes_completed": len(self._completed_episodes),
            "mean_episode_reward": float(np.mean(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
            "success_rate": float(np.mean([ep["correct"] for ep in self._completed_episodes])),
        }
        # Per-tier success — shows whether the curriculum is being learned.
        for tier in sorted({ep["tier"] for ep in self._completed_episodes}):
            tier_eps = [ep for ep in self._completed_episodes if ep["tier"] == tier]
            stats[f"success_rate/tier_{tier}"] = float(
                np.mean([ep["correct"] for ep in tier_eps])
            )
        # Action distribution — the first thing to check for policy collapse
        # (e.g. always ANSWER_DIRECTLY).
        total_actions = max(int(self._action_counts.sum()), 1)
        for action in ActionType:
            n = int(self._action_counts[action.value])
            stats[f"action_frac/{action.name}"] = float(n / total_actions)
            if n > 0:
                # Why an action is (un)popular: what it earns per step, and how
                # often the LLM's parameters for it are rejected by M4.
                stats[f"step_reward/{action.name}"] = float(
                    self._action_reward_sum[action.value] / n
                )
                stats[f"invalid_param_rate/{action.name}"] = float(
                    self._action_invalid[action.value] / n
                )
        for tier, counts in sorted(self._tier_action_counts.items()):
            n_tier = max(int(counts.sum()), 1)
            for action in ActionType:
                stats[f"action_frac_by_tier/tier_{tier}/{action.name}"] = float(
                    counts[action.value] / n_tier
                )
        return stats
