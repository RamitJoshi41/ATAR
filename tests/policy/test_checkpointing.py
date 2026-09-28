"""
Tests for M7 checkpointing and --resume-from functionality.

DoD items verified:
  - "Checkpointing saves/restores a policy such that its outputs are
     identical before and after a save/load cycle."
  - "--resume-from tested explicitly: run a short training session,
     interrupt it, resume from the saved checkpoint, and confirm curriculum
     phase + step count match what they would have been without interruption."

All tests use the tiny stand-in model on CPU.
W&B is run in offline mode (mode="offline") — no API key required.
The "appears in dashboard" part of the W&B DoD requires the architect to
verify on the training GPU with a real API key.
"""

from __future__ import annotations

import json
import os
import pytest
import torch
import numpy as np
from pathlib import Path

from atar.llm.llm_interface import LLMInterface
from atar.policy.policy_network import ATARPolicy
from atar.policy.trainer import PPOTrainer
from atar.policy.types import CheckpointState, CurriculumPhase

STAND_IN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def llm() -> LLMInterface:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LLMInterface(model_name=STAND_IN_MODEL, device=device)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("data_ckpt")
    tasks = [
        {
            "id": f"t{i:04d}",
            "tier": (i % 2) + 1,
            "query": f"Checkpoint test question {i}",
            "required_tools": [0],
            "ground_truth": "42",
            "metadata": {},
        }
        for i in range(20)
    ]
    (d / "tasks_train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks)
    )
    return d


def test_checkpoint_save_and_load_identity(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    DoD: Policy outputs must be identical (within float32 precision) before
    and after a save/load checkpoint cycle.
    """
    policy = ATARPolicy()
    state = torch.randn(1, 512)

    with torch.no_grad():
        logits_before, values_before = policy(state)

    trainer = PPOTrainer(
        policy=policy,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=20,
        checkpoint_dir=str(tmp_path / "ckpts"),
        wandb_mode="offline",
    )
    ckpt_path = trainer.save_checkpoint(tag="identity_test")
    assert ckpt_path.exists(), "Checkpoint file was not created"

    # Load checkpoint and restore into a fresh policy
    ckpt = PPOTrainer.load_checkpoint(ckpt_path)
    policy2 = ATARPolicy()
    policy2.load_state_dict(ckpt.policy_state_dict)

    with torch.no_grad():
        logits_after, values_after = policy2(state)

    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Logits differ after checkpoint save/load"
    )
    assert torch.allclose(values_before, values_after, atol=1e-6), (
        "Values differ after checkpoint save/load"
    )


def test_checkpoint_contains_required_fields(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """Checkpoint file must contain all fields required for full resumption."""
    policy = ATARPolicy()
    trainer = PPOTrainer(
        policy=policy,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=20,
        checkpoint_dir=str(tmp_path / "ckpts2"),
        wandb_mode="offline",
    )
    ckpt_path = trainer.save_checkpoint(tag="fields_test")
    ckpt = PPOTrainer.load_checkpoint(ckpt_path)

    assert isinstance(ckpt, CheckpointState)
    assert isinstance(ckpt.step, int)
    assert isinstance(ckpt.curriculum_phase, int)
    assert isinstance(ckpt.policy_state_dict, dict)
    assert isinstance(ckpt.optimizer_state_dict, dict)
    # scheduler_state_dict may be None or dict
    assert ckpt.scheduler_state_dict is None or isinstance(ckpt.scheduler_state_dict, dict)


def test_resume_restores_step_and_phase(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    DoD: --resume-from must restore step count and curriculum phase exactly.

    Procedure:
      1. Create a trainer and simulate being partway through training by
         manually setting _global_step and _current_phase.
      2. Save a checkpoint.
      3. Create a fresh trainer and restore from that checkpoint.
      4. Assert global_step and current_phase match the saved values.
    """
    policy = ATARPolicy()
    ckpt_dir = tmp_path / "ckpts_resume"

    # ── First session: advance state, save checkpoint ─────────────────────
    trainer1 = PPOTrainer(
        policy=policy,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=50,
        checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
    )
    # Simulate being partway through training
    trainer1._global_step = 22
    trainer1._current_phase = CurriculumPhase.PHASE_2
    ckpt_path = trainer1.save_checkpoint(tag="resume_test")

    # ── Second session: fresh trainer, restore from checkpoint ────────────
    policy2 = ATARPolicy()
    trainer2 = PPOTrainer(
        policy=policy2,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=50,
        checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
    )
    trainer2.restore_from_checkpoint(ckpt_path)

    assert trainer2._global_step == 22, (
        f"Restored step {trainer2._global_step} != saved step 22"
    )
    assert trainer2._current_phase == CurriculumPhase.PHASE_2, (
        f"Restored phase {trainer2._current_phase} != saved phase PHASE_2"
    )


def test_wandb_offline_run_created(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    W&B logging DoD (offline portion): wandb.init() must succeed in offline
    mode and the run object must support log() and have a summary attribute.

    NOTE: The "appears in dashboard" part of the W&B DoD requires the
    architect to verify on the training GPU with a real API key and
    mode="online". This test only confirms the offline path is wired.
    """
    import wandb

    policy = ATARPolicy()
    trainer = PPOTrainer(
        policy=policy,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=20,
        checkpoint_dir=str(tmp_path / "ckpts_wandb"),
        wandb_mode="offline",
    )

    # The run must be a wandb.sdk.wandb_run.Run (or a disabled/offline variant)
    assert trainer._run is not None, "W&B run object is None"

    # Should be able to log without error
    trainer._run.log({"test_metric": 42.0}, step=0)

    # Summary must exist and be settable
    trainer._run.summary["test_key"] = "verified"
    assert trainer._run.summary.get("test_key") == "verified"

    trainer._run.finish()


def test_stop_after_updates_and_resume(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    DoD: test --stop-after-updates 1.
    Run a trainer with stop_after_updates=1. Confirm exactly one update ran,
    a checkpoint exists, and resuming from it continues the step count.
    """
    policy = ATARPolicy()
    ckpt_dir = tmp_path / "ckpts_stop"
    
    trainer1 = PPOTrainer(
        policy=policy,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=50,
        checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
        stop_after_updates=1,
    )
    
    trainer1.train()
    
    assert trainer1._updates_this_invocation == 1, "Should have stopped after exactly 1 update"
    
    # After 1 update (with rollout_steps=5), global_step should be 5
    expected_step_after_1_update = 5
    assert trainer1._global_step == expected_step_after_1_update
    
    ckpt_path = ckpt_dir / f"checkpoint_step_{expected_step_after_1_update}.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} was not created"
    
    # Now resume
    policy2 = ATARPolicy()
    trainer2 = PPOTrainer(
        policy=policy2,
        llm=llm,
        data_dir=str(data_dir),
        n_envs=2,
        rollout_steps=5,
        total_steps=50,
        checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
        stop_after_updates=1,
    )
    trainer2.restore_from_checkpoint(ckpt_path)
    assert trainer2._global_step == expected_step_after_1_update, "Step count should continue"
    
    # Train again, should do 1 more update and stop
    trainer2.train()
    assert trainer2._updates_this_invocation == 1, "Second invocation should stop after 1 update"
    expected_step_after_2_updates = 10
    assert trainer2._global_step == expected_step_after_2_updates, "Step count should continue instead of restarting"
