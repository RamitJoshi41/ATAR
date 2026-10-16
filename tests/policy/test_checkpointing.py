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
import pytest
import torch
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


def test_checkpoint_round_trips_projection_and_update_changes_it(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    Projection co-training end-to-end (stand-in model): one real PPO update
    must change the projection weights, the checkpoint must contain them, and
    restoring must load them back into the LLM's projection.
    """
    before = llm.projection.weight.detach().clone()
    ckpt_dir = tmp_path / "ckpts_proj"
    trainer = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=8, total_steps=50, batch_size=4, epochs_per_update=2,
        checkpoint_dir=str(ckpt_dir), wandb_mode="offline", stop_after_updates=1,
    )
    trainer.train()
    after = llm.projection.weight.detach().clone()
    assert not torch.equal(before, after), "PPO update did not change the projection"

    ckpt = PPOTrainer.load_checkpoint(ckpt_dir / "checkpoint_step_8.pt")
    assert ckpt.projection_state_dict is not None
    assert torch.equal(ckpt.projection_state_dict["weight"].cpu(), after.cpu())
    # Adam state exists for the projection params too (was 0 entries before the fix).
    assert len(ckpt.optimizer_state_dict["state"]) == len(list(ATARPolicy().parameters())) + 2

    # Scramble, then restore -> weights come back.
    with torch.no_grad():
        llm.projection.weight.add_(1.0)
    trainer2 = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=8, total_steps=50, checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
    )
    trainer2.restore_from_checkpoint(ckpt_dir / "checkpoint_step_8.pt")
    assert torch.equal(llm.projection.weight.detach().cpu(), after.cpu())


def test_resume_applies_curriculum_tiers_immediately(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    Regression: a run resumed inside Phase 2/3 used to keep sampling Phase-1
    tiers [1, 2], because no phase *transition* happens after the restore.
    """
    ckpt_dir = tmp_path / "ckpts_tiers"
    trainer1 = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=5, total_steps=50, checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
    )
    trainer1._global_step = 40                       # 80% -> Phase 3
    trainer1._current_phase = CurriculumPhase.PHASE_3
    ckpt_path = trainer1.save_checkpoint(tag="phase3")

    trainer2 = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=5, total_steps=50, checkpoint_dir=str(ckpt_dir),
        wandb_mode="offline",
    )
    assert trainer2._collector._allowed_tiers == [1, 2]   # fresh trainer: Phase 1
    trainer2.restore_from_checkpoint(ckpt_path)
    assert trainer2._collector._allowed_tiers == [1, 2, 3, 4, 5]


def test_checkpoint_stores_wandb_run_id(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """The run id is saved so a new Kaggle session can continue the same W&B run."""
    trainer = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=5, total_steps=20, checkpoint_dir=str(tmp_path / "ck"),
        wandb_mode="offline",
    )
    ckpt = PPOTrainer.load_checkpoint(trainer.save_checkpoint(tag="wb"))
    assert ckpt.wandb_run_id == trainer._run.id
    trainer._run.finish()


def test_extension_continues_finished_run_with_all_tiers_and_fresh_lr(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    --extend-to: a finished 10-step plan (2 updates of 5) is extended to 20
    steps.  Extra steps run in Phase 3 (all tiers), the LR restarts at the
    extension LR and decays to lr_min over the 2 extra updates, and the
    planned run's checkpoint_final.pt is not overwritten.
    """
    ckpt_dir = tmp_path / "ckpts_ext"
    common = dict(
        llm=llm, data_dir=str(data_dir), n_envs=2, rollout_steps=5,
        total_steps=10, batch_size=5, epochs_per_update=1,
        checkpoint_dir=str(ckpt_dir), checkpoint_interval=1, wandb_mode="offline",
    )
    PPOTrainer(policy=ATARPolicy(), **common).train()
    final_path = ckpt_dir / "checkpoint_final.pt"
    final_bytes = final_path.read_bytes()

    trainer = PPOTrainer(policy=ATARPolicy(), **common)
    trainer.restore_from_checkpoint(final_path)
    trainer.start_extension(20, lr=1e-4)
    assert trainer._optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    trainer.train(max_steps=20)

    assert trainer._global_step == 20
    assert trainer._current_phase == CurriculumPhase.PHASE_3
    assert trainer._collector._allowed_tiers == [1, 2, 3, 4, 5]
    assert trainer._optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)  # lr_min
    assert (ckpt_dir / "checkpoint_final_20.pt").exists()
    assert final_path.read_bytes() == final_bytes, "extension overwrote the planned run's final checkpoint"


def test_extension_resume_mid_way_keeps_its_schedule(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """Repeating the same --extend-to command in a new session must not restart the LR."""
    ckpt_dir = tmp_path / "ckpts_ext_mid"
    common = dict(
        llm=llm, data_dir=str(data_dir), n_envs=2, rollout_steps=5,
        total_steps=10, batch_size=5, epochs_per_update=1,
        checkpoint_dir=str(ckpt_dir), checkpoint_interval=1, wandb_mode="offline",
    )
    PPOTrainer(policy=ATARPolicy(), **common).train()

    session1 = PPOTrainer(policy=ATARPolicy(), stop_after_updates=1, **common)
    session1.restore_from_checkpoint(ckpt_dir / "checkpoint_final.pt")
    session1.start_extension(30, lr=1e-4)
    session1.train(max_steps=30)                        # one extension update, then stop
    lr_after_one = session1._optimizer.param_groups[0]["lr"]
    assert session1._global_step == 15
    assert lr_after_one < 1e-4                          # decayed one step

    session2 = PPOTrainer(policy=ATARPolicy(), **common)
    session2.restore_from_checkpoint(ckpt_dir / "checkpoint_step_15.pt")
    session2.start_extension(30, lr=1e-4)               # same command again
    assert session2._optimizer.param_groups[0]["lr"] == pytest.approx(lr_after_one)
    assert session2._scheduler.T_max == 4               # 20 extra steps / 5 per update


def test_extension_refuses_unfinished_run(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    trainer = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=5, total_steps=50, checkpoint_dir=str(tmp_path / "ck"),
        wandb_mode="offline",
    )
    trainer._global_step = 20
    with pytest.raises(RuntimeError, match="not finished"):
        trainer.start_extension(100, lr=1e-4)
    trainer._global_step = 50
    with pytest.raises(ValueError):
        trainer.start_extension(40, lr=1e-4)


def test_metrics_jsonl_has_one_record_per_update(
    llm: LLMInterface, data_dir: Path, tmp_path: Path
) -> None:
    """
    Every update appends its full metrics (collection stats + PPO losses) to
    checkpoint_dir/metrics.jsonl, so the training history can be shared
    without W&B access.
    """
    import json

    ckpt_dir = tmp_path / "ckpts_metrics"
    trainer = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=8, total_steps=50, batch_size=4, epochs_per_update=1,
        checkpoint_dir=str(ckpt_dir), wandb_mode="offline", stop_after_updates=2,
    )
    trainer.train()
    records = [json.loads(line) for line in (ckpt_dir / "metrics.jsonl").read_text().splitlines()]

    assert [r["global_step"] for r in records] == [8, 16]
    for r in records:
        assert {"value_loss", "policy_loss", "entropy", "lr", "curriculum_phase"} <= r.keys()
        if r["n_episodes_completed"] > 0:
            assert any(k.startswith("action_frac/") for k in r)
            assert any(k.startswith("invalid_param_rate/") for k in r)


def test_wandb_failure_never_costs_training_progress(
    llm: LLMInterface, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A W&B outage (e.g. its background process dying on Kaggle) must not stop
    training: the update still lands in metrics.jsonl and its checkpoint is
    still saved, and the run finishes cleanly.
    """
    import json

    ckpt_dir = tmp_path / "ckpts_wandb_down"
    trainer = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=8, total_steps=50, batch_size=4, epochs_per_update=1,
        checkpoint_dir=str(ckpt_dir), checkpoint_interval=1,   # as on Kaggle
        wandb_mode="offline", stop_after_updates=2,
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("wandb backend process has shut down")

    monkeypatch.setattr(trainer._run, "log", boom)
    monkeypatch.setattr(trainer._run, "finish", boom)
    trainer.train()

    assert trainer._global_step == 16
    assert (ckpt_dir / "checkpoint_step_8.pt").exists()
    assert (ckpt_dir / "checkpoint_step_16.pt").exists()
    lines = (ckpt_dir / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["global_step"] for line in lines] == [8, 16]


def test_wandb_init_failure_falls_back_to_disabled(
    llm: LLMInterface, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wandb

    real_init = wandb.init

    def flaky_init(*args: object, **kwargs: object):  # noqa: ANN202
        if kwargs.get("mode") != "disabled":
            raise RuntimeError("could not reach W&B")
        return real_init(*args, **kwargs)

    monkeypatch.setattr(wandb, "init", flaky_init)
    trainer = PPOTrainer(
        policy=ATARPolicy(), llm=llm, data_dir=str(data_dir), n_envs=2,
        rollout_steps=8, total_steps=50, batch_size=4, epochs_per_update=1,
        checkpoint_dir=str(tmp_path / "ck"), wandb_mode="online", stop_after_updates=1,
    )
    trainer.train()
    assert trainer._global_step == 8
    assert (tmp_path / "ck" / "checkpoint_step_8.pt").exists()
