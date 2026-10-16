"""
Tests for M7 VectorizedCollector — batched rollout collection.

All tests use the tiny stand-in model (Qwen/Qwen2.5-0.5B-Instruct) on CPU.
They prove the collection code path, not training quality.

DoD item verified: "Rollout collection loop runs against a real ATAREnv
for at least 100 steps without error, storing correctly-typed EpisodeStep
entries."
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from atar.llm.llm_interface import LLMInterface
from atar.policy.policy_network import ATARPolicy
from atar.policy.collector import VectorizedCollector
from atar.policy.rollout_buffer import RolloutBuffer
from atar.shared.shared_types import (
    EpisodeStep, ActionType, STATE_DIM, SEMANTIC_RANGE
)

STAND_IN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def llm() -> LLMInterface:
    """Stand-in model, loaded once for this test module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LLMInterface(model_name=STAND_IN_MODEL, device=device)


@pytest.fixture(scope="module")
def policy() -> ATARPolicy:
    return ATARPolicy()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a minimal tasks_train.jsonl for the collector to load."""
    import json
    d = tmp_path_factory.mktemp("data")
    tasks = []
    for i in range(30):
        tasks.append({
            "id": f"t{i:04d}",
            "tier": (i % 5) + 1,
            "query": f"Test question {i}: What is {i} + {i}?",
            "required_tools": [2],   # CALCULATOR
            "ground_truth": str(i * 2),
            "metadata": {},
        })
    (d / "tasks_train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks)
    )
    return str(d)


def test_collector_100_steps_smoke(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """
    DoD: collect() runs against a real ATAREnv for at least 100 steps
    without error and stores correctly-typed EpisodeStep entries.

    Uses n_envs=2 and n_steps=100 to keep runtime manageable on the
    stand-in model (avoids loading 8 envs on a CPU-only dev machine).
    """
    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=2,
        rollout_steps=100,
        data_dir=data_dir,
        seed=42,
    )
    buffer, stats = collector.collect(n_steps=100)

    assert isinstance(buffer, RolloutBuffer)
    assert len(buffer) == 100, f"Expected 100 steps, got {len(buffer)}"
    assert buffer.is_full()


def test_collector_episode_steps_typed(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """
    Every EpisodeStep in the buffer must be correctly typed:
      - state shape (512,) float32
      - action is a valid ActionType
      - log_prob is finite float
      - value_estimate is finite float
    """
    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=2,
        rollout_steps=20,
        data_dir=data_dir,
        seed=7,
    )
    buffer, _ = collector.collect(n_steps=20)

    batch = buffer.to_batch()
    assert batch.states.shape == (20, STATE_DIM)

    for i, step in enumerate(buffer._steps):
        assert isinstance(step, EpisodeStep), f"Step {i} is not EpisodeStep"
        assert np.array(step.state).shape == (STATE_DIM,), (
            f"Step {i} state shape wrong: {np.array(step.state).shape}"
        )
        assert step.action in list(ActionType), (
            f"Step {i} action {step.action} not a valid ActionType"
        )
        assert np.isfinite(step.log_prob), f"Step {i} log_prob not finite"
        assert np.isfinite(step.value_estimate), f"Step {i} value_estimate not finite"


def test_collector_semantic_slice_nonzero(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """
    The semantic slice (state[0:384]) of stored steps must be non-zero,
    confirming that encode_batch() output was correctly written into the
    stored state (not left as zeros from skip_internal_encode=True).
    """
    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=2,
        rollout_steps=5,
        data_dir=data_dir,
        seed=99,
    )
    buffer, _ = collector.collect(n_steps=5)
    batch = buffer.to_batch()

    s, e = SEMANTIC_RANGE
    semantic_slices = batch.states[:, s:e]  # (5, 384)

    # At least some elements must be non-zero (encode_batch produced real output)
    assert semantic_slices.abs().sum() > 0, (
        "Semantic slice is all zeros — encode_batch() output was not written "
        "into stored states."
    )


def test_collector_encode_batch_called_not_loop(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm the batched encode path (encode_pooled_batch) is used during
    collection, never the single-example encode().
    """
    calls = {"batch": 0, "single": 0}
    original_pooled = llm.encode_pooled_batch
    original_encode = llm.encode

    def counting_pooled(batch_messages):
        calls["batch"] += 1
        return original_pooled(batch_messages)

    def counting_encode(messages):
        calls["single"] += 1
        return original_encode(messages)

    monkeypatch.setattr(llm, "encode_pooled_batch", counting_pooled)
    monkeypatch.setattr(llm, "encode", counting_encode)

    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=2,
        rollout_steps=6,
        data_dir=data_dir,
        seed=11,
    )
    collector.collect(n_steps=6)

    assert calls["batch"] >= 1, "encode_pooled_batch() was never called"
    assert calls["single"] == 0, "collector fell back to per-env encode()"


def test_collector_batch_has_pooled_env_ids_and_bootstrap(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """
    The rollout batch must carry what the PPO update needs for projection
    co-training (pooled) and per-env GAE (env_ids, bootstrap_values), and the
    stored semantic slice must equal project(pooled) — the invariant that
    makes the first PPO ratio exactly 1.
    """
    n_envs = 3
    collector = VectorizedCollector(
        llm=llm, policy=policy, n_envs=n_envs, rollout_steps=7,
        data_dir=data_dir, seed=5,
    )
    buffer, _ = collector.collect(n_steps=7)
    batch = buffer.to_batch()

    assert batch.pooled is not None
    assert batch.pooled.shape == (7, llm.projection.in_features)
    # Round-robin fill: env ids 0,1,2,0,1,2,0
    assert batch.env_ids is not None
    assert batch.env_ids.tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert batch.bootstrap_values is not None
    assert batch.bootstrap_values.shape == (n_envs,)

    with torch.no_grad():
        reprojected = llm.project(batch.pooled).cpu()
    s, e = SEMANTIC_RANGE
    assert torch.allclose(batch.states[:, s:e], reprojected, atol=1e-5)


def test_collector_stats_returned(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """collect() must return a stats dict with the required keys."""
    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=2,
        rollout_steps=10,
        data_dir=data_dir,
        seed=0,
    )
    _, stats = collector.collect(n_steps=10)
    assert "n_episodes_completed" in stats
    assert "mean_episode_reward" in stats
    assert "mean_episode_length" in stats
    if stats["n_episodes_completed"] > 0:
        assert 0.0 <= stats["success_rate"] <= 1.0
        action_fracs = [v for k, v in stats.items() if k.startswith("action_frac/")]
        assert len(action_fracs) == len(ActionType)
        assert sum(action_fracs) == pytest.approx(1.0)
        # Per-tier action shares: each tier's 7 fractions sum to 1.
        tiers = {k.split("/")[1] for k in stats if k.startswith("action_frac_by_tier/")}
        assert tiers
        for t in tiers:
            fracs = [v for k, v in stats.items() if k.startswith(f"action_frac_by_tier/{t}/")]
            assert len(fracs) == len(ActionType) and sum(fracs) == pytest.approx(1.0)

def test_curriculum_reset_exact_filtering(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """
    Regression test: _curriculum_reset must never fall back to an unallowed tier
    or log a failure warning. It must strictly filter env._tasks.
    """
    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=1,
        rollout_steps=10,
        data_dir=data_dir,
        seed=123,
    )
    collector.set_allowed_tiers([1])
    env = collector._envs[0]
    
    # Skew the task distribution to make rejection sampling fail
    # 100 Tier 5 tasks, 1 Tier 1 task
    from atar.shared.shared_types import Task
    env._tasks = [
        Task(id="t1", tier=1, query="q", required_tools=[], ground_truth="1")
    ] + [
        Task(id=f"t5_{i}", tier=5, query="q", required_tools=[], ground_truth="5")
        for i in range(100)
    ]
    
    # Under old code, 20 random samples from 101 tasks (1% chance of success) 
    # would fail ~81% of the time, falling back to Tier 5 and returning info["task_tier"] == 5.
    # Under new code, it explicitly filters and picks the Tier 1 task, never failing.
    
    # Test 10 times to ensure it never falls back
    for _ in range(10):
        obs, info = collector._curriculum_reset(env)
        assert info["task_tier"] == 1

def test_curriculum_reset_raises_runtime_error_if_no_tasks(
    llm: LLMInterface, policy: ATARPolicy, data_dir: str
) -> None:
    """
    Verify that _curriculum_reset raises a RuntimeError if the allowed_tiers
    filter results in zero valid tasks, rather than silently falling back
    or crashing obscurely downstream.
    """
    collector = VectorizedCollector(
        llm=llm,
        policy=policy,
        n_envs=1,
        rollout_steps=10,
        data_dir=data_dir,
        seed=44,
    )
    
    # data_dir fixture creates tasks up to tier 5
    # Requesting tier 9 (which has 0 tasks)
    collector.set_allowed_tiers([9])
    env = collector._envs[0]
    
    import pytest
    with pytest.raises(RuntimeError, match="No tasks available for allowed tiers"):
        collector._curriculum_reset(env)
