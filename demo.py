#!/usr/bin/env python3
"""
ATAR vertical-slice demo — runs one full episode end-to-end and prints
the trajectory, state shapes, and reward breakdown.

Usage:
    python demo.py            # runs first task
    python demo.py --all      # runs all 10 tasks
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from atar.shared.shared_types import STATE_DIM, ActionType, RewardComponents, ToolResult
from slice.calculator import execute_calculator
from slice.dummy_policy import DummyPolicy
from slice.env import SliceEnv
from slice.tasks import TASKS


def run_episode(env: SliceEnv, policy: DummyPolicy, task_idx: int) -> dict:
    """Run a single episode and return a summary dict."""
    task = TASKS[task_idx]
    policy.reset(task)
    obs, info = env.reset(task=task)

    print(f"\n{'='*60}")
    print(f"  Task {task.id}: {task.query}")
    print(f"  Ground truth: {task.ground_truth}")
    print(f"{'='*60}")
    print(f"  Initial state shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Semantic region [0:384] non-zero: {np.count_nonzero(obs[:384])}")
    print(f"  Tool history  [384:391]: {obs[384:391]}")
    print(f"  Outcome embed [391:455]: max={obs[391:455].max():.4f}")
    print(f"  Budget        [455:457]: {obs[455:457]}")
    print(f"  Padding       [457:512]: all-zero={np.all(obs[457:512] == 0)}")

    done = False
    total_reward = RewardComponents(accuracy=0.0, efficiency=0.0, safety=0.0, bonus=0.0)

    while not done:
        action = policy.act()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Accumulate reward components.
        total_reward = RewardComponents(
            accuracy=total_reward.accuracy + reward.accuracy,
            efficiency=total_reward.efficiency + reward.efficiency,
            safety=total_reward.safety + reward.safety,
            bonus=total_reward.bonus + reward.bonus,
        )

        print(f"\n  Step {info['step']}: action={action.name}")
        step_data = info["trajectory"][-1]
        if "tool_result" in step_data:
            tr = step_data["tool_result"]
            print(f"    Tool result: success={tr['success']}, output={tr['output']}")
        if "final_answer" in step_data:
            print(f"    Final answer: {step_data['final_answer']}")
        print(
            f"    Reward: acc={reward.accuracy:.2f} eff={reward.efficiency:.2f} "
            f"saf={reward.safety:.2f} bon={reward.bonus:.2f} total={reward.total:.2f}"
        )

    print(f"\n  Episode done — total reward = {total_reward.total:.2f}")
    print(f"    accuracy={total_reward.accuracy:.2f}  efficiency={total_reward.efficiency:.2f}  "
          f"safety={total_reward.safety:.2f}  bonus={total_reward.bonus:.2f}")

    correct = False
    try:
        correct = abs(float(info["current_answer"]) - float(task.ground_truth)) < 1e-6
    except (ValueError, TypeError):
        correct = info["current_answer"].strip() == task.ground_truth.strip()

    print(f"  Correct: {'✓' if correct else '✗'}")

    return {
        "task_id": task.id,
        "correct": correct,
        "total_reward": total_reward.total,
        "steps": info["step"],
    }


def print_type_report() -> None:
    """Print exact shapes and types for comparison with interfaces.md."""
    print("\n" + "=" * 60)
    print("  TYPE / SHAPE REPORT")
    print("=" * 60)

    # State
    dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
    print(f"\n  State:")
    print(f"    type:   np.ndarray")
    print(f"    shape:  {dummy_state.shape}")
    print(f"    dtype:  {dummy_state.dtype}")
    print(f"    layout: [0:384]=semantic, [384:391]=tool_history, "
          f"[391:455]=outcome, [455:457]=budget, [457:512]=padding")

    # ToolResult
    tr = execute_calculator("1+1")
    print(f"\n  ToolResult:")
    print(f"    type:       {type(tr).__name__} (dataclass)")
    print(f"    fields:")
    print(f"      success:    {type(tr.success).__name__}  (value: {tr.success})")
    print(f"      output:     {type(tr.output).__name__}   (value: {tr.output})")
    print(f"      error:      {type(tr.error).__name__}  (value: {tr.error})")
    print(f"      latency_ms: {type(tr.latency_ms).__name__} (value: {tr.latency_ms:.4f})")
    print(f"      token_cost: {type(tr.token_cost).__name__}  (value: {tr.token_cost})")

    # RewardComponents
    rc = RewardComponents(accuracy=1.0, efficiency=-0.2, safety=0.0, bonus=0.2)
    print(f"\n  RewardComponents:")
    print(f"    type:       {type(rc).__name__} (dataclass)")
    print(f"    fields:")
    print(f"      accuracy:   {type(rc.accuracy).__name__}")
    print(f"      efficiency: {type(rc.efficiency).__name__}")
    print(f"      safety:     {type(rc.safety).__name__}")
    print(f"      bonus:      {type(rc.bonus).__name__}")
    print(f"      total:      {type(rc.total).__name__} (property = {rc.total})")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATAR vertical-slice demo")
    parser.add_argument("--all", action="store_true", help="Run all 10 tasks")
    args = parser.parse_args()

    env = SliceEnv()
    policy = DummyPolicy()

    if args.all:
        results = []
        for i in range(len(TASKS)):
            results.append(run_episode(env, policy, i))
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {sum(r['correct'] for r in results)}/{len(results)} correct")
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print(f"  Average total reward: {avg_reward:.2f}")
    else:
        run_episode(env, policy, 0)

    print_type_report()


if __name__ == "__main__":
    main()
