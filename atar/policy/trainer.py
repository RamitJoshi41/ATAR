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

The PPO maths lives in ppo_stubs.py (compute_ppo_loss / compute_gae).
This file handles everything around it: per-env GAE, re-applying the
projection with gradients, minibatching, clipping, scheduling, checkpoints.
"""

from __future__ import annotations

import dataclasses
import json
import logging
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
from atar.shared.shared_types import SEMANTIC_RANGE

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
    wandb_run_id : str | None
        Resume logging into this existing W&B run (multi-session training).
        None starts a new run.
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
        wandb_run_id: str | None = None,
    ) -> None:
        self._policy = policy.to(device)
        self._llm = llm
        self._device = device
        self._total_steps = total_steps
        self._rollout_steps = rollout_steps
        self._lr_min = lr_min
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
        # Gradients reach the projection because the PPO update re-applies it
        # to the stored pooled LLM features (see _with_projected_states).
        param_groups: list[dict] = [{"params": list(policy.parameters())}]
        if not freeze_projection:
            param_groups.append({"params": list(llm.projection.parameters())})
        else:
            llm.projection.requires_grad_(False)
            logger.info("Ablation: freeze_projection=True — M3 projection not co-trained")
        self._trainable_params: list[nn.Parameter] = [
            p for group in param_groups for p in group["params"]
        ]

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
            id=wandb_run_id,
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

            # ── GAE + PPO update ───────────────────────────────────────────
            batch = buffer.to_batch()
            self._fill_gae(batch)
            update_stats = self._run_ppo_update(batch)

            # ── Logging ───────────────────────────────────────────────────
            entropy_coef = self._current_entropy_coef()
            log_dict: dict[str, Any] = {
                "global_step": self._global_step,
                "curriculum_phase": int(self._current_phase),
                "entropy_coef": entropy_coef,
                **collection_stats,
                **update_stats,
            }
            self._run.log(log_dict, step=self._global_step)
            # Plain-file copy of every update's metrics, next to the
            # checkpoints, so the full history survives without W&B access.
            with open(self._checkpoint_dir / "metrics.jsonl", "a") as fh:
                fh.write(json.dumps(log_dict) + "\n")

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

        # Final checkpoint ("final" for the planned run; an extension keeps
        # that file and writes final_<step> instead).
        self.save_checkpoint(
            tag="final" if stop_at == self._total_steps else f"final_{stop_at}"
        )
        logger.info("Training complete at step %d.", self._global_step)
        self._run.summary["final_step"] = self._global_step
        self._run.finish()

    # ------------------------------------------------------------------
    # Extension beyond the planned run
    # ------------------------------------------------------------------

    def start_extension(self, extend_to: int, lr: float) -> None:
        """
        Prepare to keep training a FINISHED run up to ``extend_to`` steps.

        Call after restore_from_checkpoint(), then train(max_steps=extend_to).
        The trainer must be built with the ORIGINAL total_steps, so the
        curriculum boundaries are unchanged: every extra step is past the
        Phase-3 boundary and samples all 5 tiers.  The entropy coefficient
        stays at its end value.  The learning rate restarts at ``lr`` and
        cosine-decays to lr_min over the extra updates.

        Resuming in the middle of an extension (step > total_steps) keeps the
        extension's scheduler state that restore_from_checkpoint() loaded, so
        repeating the same command in every session is safe.
        """
        if extend_to <= self._total_steps:
            raise ValueError(
                f"extend_to ({extend_to}) must exceed the planned total_steps "
                f"({self._total_steps})"
            )
        if self._global_step < self._total_steps:
            raise RuntimeError(
                f"The planned run is not finished (step {self._global_step} of "
                f"{self._total_steps}); resume it normally before extending."
            )
        if self._global_step > self._total_steps:
            logger.info(
                "Resuming an extension at step %d (scheduler state restored).",
                self._global_step,
            )
            return

        extra_updates = max(1, -(-(extend_to - self._total_steps) // self._rollout_steps))
        for group in self._optimizer.param_groups:
            group["lr"] = lr
            group["initial_lr"] = lr
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=extra_updates, eta_min=self._lr_min
        )
        logger.info(
            "Extending from step %d to %d: %d more updates, LR %.1e -> %.1e, all tiers.",
            self._global_step, extend_to, extra_updates, lr, self._lr_min,
        )

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
            projection_state_dict=self._llm.projection.state_dict(),
            wandb_run_id=getattr(self._run, "id", None),
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
        Restores: policy weights, M3 projection weights, optimizer state,
        scheduler state, global_step, current curriculum phase — and points
        the collector at that phase's tiers (restarting in-flight episodes).

        Parameters
        ----------
        path : str | Path
            Path to the checkpoint .pt file.
        """
        ckpt = self.load_checkpoint(path)
        self._policy.load_state_dict(ckpt.policy_state_dict)
        if ckpt.projection_state_dict is not None:
            self._llm.projection.load_state_dict(ckpt.projection_state_dict)
        else:
            logger.warning(
                "Checkpoint %s has no projection weights (written before "
                "projection co-training). Keeping the seeded initial projection.",
                path,
            )
        self._optimizer.load_state_dict(ckpt.optimizer_state_dict)
        if ckpt.scheduler_state_dict is not None:
            self._scheduler.load_state_dict(ckpt.scheduler_state_dict)
        self._global_step = ckpt.step
        self._current_phase = CurriculumPhase(ckpt.curriculum_phase)
        # The collector was built with Phase-1 tiers; without this a run
        # resumed in Phase 2/3 keeps sampling tiers 1-2 until the next
        # phase boundary (or forever, if resumed in Phase 3).
        if self._curriculum is not None:
            self._collector.set_allowed_tiers(
                self._curriculum.allowed_tiers(self._current_phase)
            )
        self._collector.reset_all()
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
        Fill batch.advantages and batch.returns by calling compute_gae()
        separately for each env's trajectory.

        The collector fills the buffer round-robin across N envs
        (env0, env1, ..., envN-1, env0, ...), so consecutive buffer entries
        belong to DIFFERENT episodes.  GAE must therefore run per env_id; on
        the flat buffer, V(s_{t+1}) would come from another env's episode
        and terminal rewards would never be credited to the actions that led
        to them.

        Each env's last transition bootstraps from V(current state) when it
        is not terminal (the rollout cut the episode mid-way), else 0.

        Returns are V(s) + raw advantages — the value-head target.  Advantage
        normalisation happens per minibatch inside compute_ppo_loss, AFTER
        returns are formed (normalising first would put the value targets
        on the wrong scale).
        """
        T = batch.rewards.shape[0]
        env_ids = (
            batch.env_ids if batch.env_ids is not None
            else torch.zeros(T, dtype=torch.int64)
        )
        advantages = torch.zeros(T, dtype=torch.float32)

        for env_id in torch.unique(env_ids).tolist():
            idx = (env_ids == env_id).nonzero(as_tuple=True)[0]   # time-ordered
            dones = batch.dones[idx].tolist()
            last_value = 0.0
            if not dones[-1] and batch.bootstrap_values is not None:
                last_value = float(batch.bootstrap_values[env_id])
            env_adv = compute_gae(
                batch.rewards[idx].tolist(),
                batch.values[idx].tolist() + [last_value],
                dones,
                self._gamma,
                self._lam,
            )
            advantages[idx] = torch.tensor(env_adv, dtype=torch.float32)

        batch.advantages = advantages
        batch.returns = batch.values + advantages   # V(s) + A(s,a), unnormalised

    def _with_projected_states(self, mb: RolloutBatch) -> RolloutBatch:
        """
        Re-apply M3's projection to the stored pooled LLM features so the
        semantic slice of the state carries gradients into the projection.

        At the first minibatch of an update the result equals the stored
        state exactly (same projection weights), so PPO's probability ratio
        starts at 1 as it should.  If pooled features are unavailable or the
        projection is frozen (ablation), the stored states are used as-is.
        """
        if mb.pooled is None or self._freeze_projection:
            return mb
        semantic = self._llm.project(mb.pooled).to(self._device)   # (B, 384), grad
        s, e = SEMANTIC_RANGE
        states = torch.cat(
            [
                mb.states[:, :s].to(self._device),
                semantic,
                mb.states[:, e:].to(self._device),
            ],
            dim=1,
        )
        return dataclasses.replace(mb, states=states)

    def _run_ppo_update(self, batch: RolloutBatch) -> dict[str, float]:
        """
        Run epochs_per_update passes of the PPO update over minibatches.

        Calls compute_ppo_loss() (architect stub) for each minibatch.
        """
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
                    self._with_projected_states(mb), self._policy,
                    self._clip_range, self._value_coef, entropy_coef
                )
                loss_out.total_loss.backward()
                # Clip policy AND projection gradients together.
                nn.utils.clip_grad_norm_(
                    self._trainable_params, self._max_grad_norm
                )
                self._optimizer.step()
                total_policy_loss += loss_out.policy_loss.item()
                total_value_loss += loss_out.value_loss.item()
                total_entropy += loss_out.entropy.item()
                n_updates += 1

        self._scheduler.step()

        if n_updates == 0:
            return {}
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "lr": self._scheduler.get_last_lr()[0],
        }
