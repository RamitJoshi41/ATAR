"""
M7 PPO Trainer — training loop scaffolding.

Implements:
  - Rollout collection via VectorizedCollector (batched encode/generate).
  - Optimizer with ATARPolicy + M3 projection parameters co-trained.
  - Cosine LR scheduler (3e-4 → 1e-5).
  - Entropy coefficient linear decay (0.01 → 0.001).
  - Curriculum phase management + checkpoint before phase transitions.
  - Checkpointing (save/load) for resume support.
  - Weights & Biases logging.
  - Ablation config flags: no_curriculum, freeze_projection.

The actual PPO update (compute_ppo_loss / compute_gae) is left as stubs
for the architect to implement — see ppo_stubs.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import wandb

from atar.llm.llm_interface import LLMInterface
from atar.policy.policy_network import ATARPolicy
from atar.policy.rollout_buffer import RolloutBuffer
from atar.policy.collector import VectorizedCollector
from atar.policy.curriculum import CurriculumManager
from atar.policy.ppo_stubs import compute_ppo_loss, compute_gae
from atar.policy.types import CheckpointState, CurriculumPhase, RolloutBatch

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO training scaffolding for ATAR.

    Parameters
    ----------
    policy : ATARPolicy
        The policy network to train.
    llm : LLMInterface
        Shared LLM (frozen base model + trainable projection).
    data_dir : str | Path
        Data directory for ATAREnv task loading.
    n_envs : int
        Number of parallel environment instances (default 8).
    rollout_steps : int
        Transitions collected per PPO update cycle (default 2048).
    total_steps : int
        Total training steps across all updates (default 100_000).
    batch_size : int
        Minibatch size for PPO update (default 64).
    epochs_per_update : int
        Number of passes over the rollout buffer per update (default 10).
    lr : float
        Initial learning rate (default 3e-4; cosine decays to 1e-5).
    lr_min : float
        Cosine decay target (default 1e-5).
    gamma : float
        Discount factor (default 0.95).
    lam : float
        GAE lambda (default 0.95).
    clip_range : float
        PPO clip parameter (default 0.2).
    entropy_coef_start : float
        Entropy coefficient at step 0 (default 0.01).
    entropy_coef_end : float
        Entropy coefficient at total_steps (default 0.001).
    value_coef : float
        Value function coefficient (default 0.5).
    max_grad_norm : float
        Gradient clipping norm (default 0.5).
    checkpoint_dir : str | Path
        Directory to write checkpoint files.
    checkpoint_interval : int
        Save a checkpoint every N optimizer updates (default 50 ≈ 500 steps
        with rollout_steps=2048 — adjusted so wall-clock frequency is sensible).
    device : str
        PyTorch device string (default "cpu").
    wandb_project : str
        W&B project name.
    wandb_mode : str
        W&B mode ("online", "offline", "disabled").
    seed : int | None
        Global RNG seed.
    no_curriculum : bool
        Ablation: skip curriculum, train on all tiers from step 0.
    freeze_projection : bool
        Ablation: freeze M3's projection layer (don't co-train with policy).
    stop_after_updates : int | None
        Stop training cleanly after this many updates in this invocation.
    """

    def __init__(
        self,
        policy: ATARPolicy,
        llm: LLMInterface,
        data_dir: str | Path,
        n_envs: int = 8,
        rollout_steps: int = 2048,
        total_steps: int = 100_000,
        batch_size: int = 64,
        epochs_per_update: int = 10,
        lr: float = 3e-4,
        lr_min: float = 1e-5,
        gamma: float = 0.95,
        lam: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef_start: float = 0.01,
        entropy_coef_end: float = 0.001,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        checkpoint_dir: str | Path = "./checkpoints",
        checkpoint_interval: int = 50,
        device: str = "cpu",
        wandb_project: str = "atar",
        wandb_mode: Literal["disabled", "offline", "online", "shared"] | None = "online",
        seed: int | None = None,
        no_curriculum: bool = False,
        freeze_projection: bool = False,
        stop_after_updates: int | None = None,
    ) -> None:
        self._policy = policy.to(device)
        self._llm = llm
        self._device = device
        self._total_steps = total_steps
        self._rollout_steps = rollout_steps
        self._batch_size = batch_size
        self._epochs_per_update = epochs_per_update
        self._gamma = gamma
        self._lam = lam
        self._clip_range = clip_range
        self._entropy_coef_start = entropy_coef_start
        self._entropy_coef_end = entropy_coef_end
        self._value_coef = value_coef
        self._max_grad_norm = max_grad_norm
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_interval = checkpoint_interval
        self._no_curriculum = no_curriculum
        self._freeze_projection = freeze_projection
        self._stop_after_updates = stop_after_updates

        # Optimizer parameter groups (MSD requirement: both policy + projection).
        param_groups: list[dict] = [{"params": list(policy.parameters())}]
        if not freeze_projection:
            param_groups.append({"params": list(llm.projection.parameters())})
        else:
            logger.info("Ablation: freeze_projection=True — M3 projection not co-trained")

        self._optimizer = torch.optim.Adam(param_groups, lr=lr)

        # Total update count = ceil(total_steps / rollout_steps)
        total_updates = max(1, total_steps // rollout_steps)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=total_updates, eta_min=lr_min
        )

        # Curriculum
        if no_curriculum:
            logger.info("Ablation: no_curriculum=True — training on all tiers from step 0")
            self._curriculum: CurriculumManager | None = None
        else:
            self._curriculum = CurriculumManager(total_steps)

        # Collector
        initial_tiers: list[int] | None = (
            None if (no_curriculum or self._curriculum is None)
            else self._curriculum.allowed_tiers(CurriculumPhase.PHASE_1)
        )
        self._collector = VectorizedCollector(
            llm=llm,
            policy=policy,
            n_envs=n_envs,
            rollout_steps=rollout_steps,
            data_dir=data_dir,
            seed=seed,
            allowed_tiers=initial_tiers,
        )

        # Training state
        self._global_step = 0
        self._update_count = 0
        self._updates_this_invocation = 0
        self._current_phase = CurriculumPhase.PHASE_1

        # W&B
        hparams = {
            "n_envs": n_envs,
            "rollout_steps": rollout_steps,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "epochs_per_update": epochs_per_update,
            "lr": lr,
            "gamma": gamma,
            "lam": lam,
            "clip_range": clip_range,
            "entropy_coef_start": entropy_coef_start,
            "entropy_coef_end": entropy_coef_end,
            "value_coef": value_coef,
            "max_grad_norm": max_grad_norm,
            "no_curriculum": no_curriculum,
            "freeze_projection": freeze_projection,
        }
        self._run = wandb.init(
            project=wandb_project,
            config=hparams,
            mode=wandb_mode,
            resume="allow",
        )

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(self, max_steps: int | None = None) -> None:
        """
        Run the PPO training loop.

        Stops when global_step >= total_steps (or max_steps if provided).
        The PPO update itself raises NotImplementedError until the architect
        implements compute_ppo_loss() and compute_gae() in ppo_stubs.py.

        Parameters
        ----------
        max_steps : int | None
            Override the total_steps passed at construction.  Used by tests
            to run short training sessions.
        """
        stop_at = max_steps if max_steps is not None else self._total_steps
        logger.info("Starting PPO training loop. Stop at step %d.", stop_at)

        while self._global_step < stop_at:
            # ── Curriculum phase management ───────────────────────────────
            if self._curriculum is not None:
                new_phase = self._curriculum.get_phase(self._global_step)
                if new_phase != self._current_phase:
                    logger.info(
                        "Curriculum transition: Phase %d → Phase %d at step %d. "
                        "Saving pre-transition checkpoint.",
                        self._current_phase, new_phase, self._global_step
                    )
                    self.save_checkpoint(tag=f"pre_phase_{int(new_phase)}")
                    self._current_phase = new_phase
                    allowed = self._curriculum.allowed_tiers(new_phase)
                    self._collector.set_allowed_tiers(allowed)
                    logger.info("Allowed tiers now: %s", allowed)

            # ── Rollout collection ─────────────────────────────────────────
            remaining = stop_at - self._global_step
            n_steps = min(self._rollout_steps, remaining)
            buffer, collection_stats = self._collector.collect(n_steps=n_steps)
            self._global_step += len(buffer)

            # ── GAE (architect stub) ───────────────────────────────────────
            # compute_gae raises NotImplementedError until architect implements it.
            # When implemented, it fills batch.advantages and batch.returns.
            batch = buffer.to_batch()
            # NOTE: The following block is wrapped in try/except so the training
            # scaffold can be tested end-to-end without the architect stubs.
            try:
                self._fill_gae(batch)
                self._run_ppo_update(batch)
            except NotImplementedError:
                logger.debug(
                    "compute_gae/compute_ppo_loss not yet implemented (architect stub). "
                    "Skipping PPO update at step %d.", self._global_step
                )

            # ── Logging ───────────────────────────────────────────────────
            entropy_coef = self._current_entropy_coef()
            log_dict: dict[str, Any] = {
                "global_step": self._global_step,
                "curriculum_phase": int(self._current_phase),
                "entropy_coef": entropy_coef,
                **collection_stats,
            }
            self._run.log(log_dict, step=self._global_step)

            # ── Periodic checkpointing ─────────────────────────────────────
            self._update_count += 1
            self._updates_this_invocation += 1
            
            # Save checkpoint if it's the periodic interval
            if self._update_count % self._checkpoint_interval == 0:
                self.save_checkpoint(tag=f"step_{self._global_step}")
                
            # Check if we should stop early after N updates
            if self._stop_after_updates is not None and self._updates_this_invocation >= self._stop_after_updates:
                logger.info(
                    "Reached --stop-after-updates limit (%d). Saving checkpoint and exiting cleanly.", 
                    self._stop_after_updates
                )
                if self._update_count % self._checkpoint_interval != 0:
                    # Save a checkpoint with the same naming as normal interval checkpoints
                    self.save_checkpoint(tag=f"step_{self._global_step}")
                self._run.summary["final_step"] = self._global_step
                self._run.finish()
                return

        # Final checkpoint
        self.save_checkpoint(tag="final")
        logger.info("Training complete at step %d.", self._global_step)
        self._run.summary["final_step"] = self._global_step
        self._run.finish()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> Path:
        """
        Save a full training checkpoint.

        Saved file contains: policy weights, optimizer state, scheduler state,
        global_step, and current curriculum phase.

        Parameters
        ----------
        tag : str
            Filename tag (e.g. "step_2048", "pre_phase_2", "final").

        Returns
        -------
        Path to the saved .pt file.
        """
        ckpt = CheckpointState(
            step=self._global_step,
            curriculum_phase=int(self._current_phase),
            policy_state_dict=self._policy.state_dict(),
            optimizer_state_dict=self._optimizer.state_dict(),
            scheduler_state_dict=self._scheduler.state_dict(),
        )
        path = self._checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt.__dict__, path)
        logger.info("Checkpoint saved → %s", path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> CheckpointState:
        """
        Load a checkpoint from disk.

        Parameters
        ----------
        path : str | Path
            Path to the .pt file written by save_checkpoint().

        Returns
        -------
        CheckpointState
        """
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return CheckpointState(**raw)

    def restore_from_checkpoint(self, path: str | Path) -> None:
        """
        Restore trainer state from a checkpoint file.

        Called when --resume-from is passed to scripts/train.py.
        Restores: policy weights, optimizer state, scheduler state,
        global_step, current curriculum phase.

        Parameters
        ----------
        path : str | Path
            Path to the checkpoint .pt file.
        """
        ckpt = self.load_checkpoint(path)
        self._policy.load_state_dict(ckpt.policy_state_dict)
        self._optimizer.load_state_dict(ckpt.optimizer_state_dict)
        if ckpt.scheduler_state_dict is not None:
            self._scheduler.load_state_dict(ckpt.scheduler_state_dict)
        self._global_step = ckpt.step
        self._current_phase = CurriculumPhase(ckpt.curriculum_phase)
        logger.info(
            "Resumed from %s at step %d, curriculum phase %d.",
            path, self._global_step, self._current_phase
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_entropy_coef(self) -> float:
        """Linear decay of entropy coefficient from start to end over total_steps."""
        progress = min(self._global_step / max(self._total_steps, 1), 1.0)
        return self._entropy_coef_start + progress * (
            self._entropy_coef_end - self._entropy_coef_start
        )

    def _fill_gae(self, batch: RolloutBatch) -> None:
        """
        Call compute_gae() for each episode segment in the batch to fill
        batch.advantages and batch.returns.

        This is a scaffold wrapper — the actual GAE is in ppo_stubs.py
        (architect implements).
        """
        T = batch.rewards.shape[0]
        rewards = batch.rewards.tolist()
        values = batch.values.tolist() + [0.0]   # bootstrap with 0 at end
        dones = batch.dones.tolist()

        advantages = compute_gae(rewards, values, dones, self._gamma, self._lam)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        # Normalise advantages for training stability.
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        batch.advantages = adv_tensor
        batch.returns = batch.values + adv_tensor   # V(s) + A(s,a)

    def _run_ppo_update(self, batch: RolloutBatch) -> None:
        """
        Run epochs_per_update passes of the PPO update over minibatches.

        Calls compute_ppo_loss() (architect stub) for each minibatch.
        """
        buffer_obj = RolloutBuffer(capacity=len(batch.states))
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(self._epochs_per_update):
            minibatches = RolloutBuffer(capacity=1).minibatches(
                batch, minibatch_size=self._batch_size
            )
            for mb in minibatches:
                self._optimizer.zero_grad()
                entropy_coef = self._current_entropy_coef()
                loss_out = compute_ppo_loss(
                    mb, self._policy, self._clip_range,
                    self._value_coef, entropy_coef
                )
                loss_out.total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self._max_grad_norm
                )
                self._optimizer.step()
                total_policy_loss += loss_out.policy_loss.item()
                total_value_loss += loss_out.value_loss.item()
                total_entropy += loss_out.entropy.item()
                n_updates += 1

        self._scheduler.step()

        if n_updates > 0:
            self._run.log({
                "policy_loss": total_policy_loss / n_updates,
                "value_loss": total_value_loss / n_updates,
                "entropy": total_entropy / n_updates,
                "lr": self._scheduler.get_last_lr()[0],
            }, step=self._global_step)
