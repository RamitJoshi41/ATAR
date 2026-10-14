#!/usr/bin/env python3
"""
scripts/record_replays.py — record demo episodes for the M8 Streamlit app.

Runs every given ATAR checkpoint (a "version" of the policy) and the
baselines on the same small set of held-out demo tasks, and appends one
EpisodeRecord per (version, task) to a JSONL file: every step's action,
action probabilities, generated tool params, tool result and reward.

Run it on the training GPU with the SAME model the checkpoints were trained
on.  Resumable: (version, task) pairs already in --out are skipped, so it is
safe to re-run after a Kaggle session cut-off.

Example (a shell; in a notebook cell build the --checkpoints list in Python
first, since {...} inside a ! command is Python interpolation):
    python scripts/record_replays.py \\
        --checkpoints /kaggle/input/run/checkpoints/checkpoint_step_{0,6144,12288,18432}.pt \\
                      /kaggle/input/run/checkpoints/checkpoint_final.pt \\
        --baselines react,zeroshot,random --tasks-per-tier 3 \\
        --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \\
        --out /kaggle/working/replays.jsonl

For the untrained starting point ("step_0"), first create a checkpoint with
scripts/make_untrained_checkpoint.py and pass it like any other.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="record_replays.py",
        description="Record demo episodes per policy version (M8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints", nargs="*", default=[], help="ATAR checkpoint .pt files (one version each).")
    p.add_argument("--baselines", type=str, default="react,zeroshot,random",
                   help="Comma-separated baselines to record (react, zeroshot, random); empty for none.")
    p.add_argument("--tasks-per-tier", type=int, default=3, help="Demo tasks drawn from each tier of the test set.")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="./replays.jsonl")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def pick_demo_tasks(tasks: list, per_tier: int, seed: int) -> list:
    """The same tasks for every version: ``per_tier`` per tier, chosen by seed."""
    rng = random.Random(seed)
    chosen = []
    for tier in sorted({t.tier for t in tasks}):
        tier_tasks = sorted((t for t in tasks if t.tier == tier), key=lambda t: t.id)
        chosen.extend(rng.sample(tier_tasks, min(per_tier, len(tier_tasks))))
    return chosen


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("record_replays")

    # Deferred imports keep --help fast.
    from atar.demo.replay import append_records, load_records, run_episode
    from atar.env.atar_env import ATAREnv
    from atar.evaluation.runner import (
        CheckpointMismatchError, RandomAgent, ReActAgent, ZeroShotAgent,
        load_atar_agent, load_tasks_from_jsonl,
    )
    from atar.llm.llm_interface import LLMInterface

    data_dir = Path(args.data_dir)
    demo_tasks = pick_demo_tasks(
        load_tasks_from_jsonl(data_dir / "tasks_test.jsonl"), args.tasks_per_tier, args.seed
    )
    logger.info("Demo tasks: %d (%d per tier) from the held-out test set.", len(demo_tasks), args.tasks_per_tier)

    out = Path(args.out)
    done: set[tuple[str, str]] = set()
    if out.exists():
        done = {(r.version, r.task_id) for r in load_records(out)}
        logger.info("Resuming: %d episodes already recorded in %s.", len(done), out)

    llm = LLMInterface(model_name=args.model_name, device=args.device)
    initial_projection = {k: v.clone() for k, v in llm.projection.state_dict().items()}

    def record(version: str, agent_name: str, agent: object, step: int | None) -> None:
        env = ATAREnv(llm=llm, data_dir=data_dir, seed=args.seed)
        for task in demo_tasks:
            if (version, task.id) in done:
                continue
            rec = run_episode(env, agent, task, version=version, agent_name=agent_name,
                              model_name=args.model_name, checkpoint_step=step)
            append_records(out, [rec])
            done.add((version, task.id))
            logger.info("[%s] %s tier %d: %s -> %s (reward %.1f)", version, task.id[:8], task.tier,
                        " > ".join(rec.action_trace), "correct" if rec.correct else "wrong", rec.total_reward)

    # ── ATAR versions ──────────────────────────────────────────────────────
    for ckpt_path in args.checkpoints:
        try:
            agent, ckpt = load_atar_agent(ckpt_path, llm, args.device)
        except CheckpointMismatchError as exc:
            logger.error("%s", exc)
            return 1
        record(f"step_{ckpt.step}", "atar", agent, int(ckpt.step))

    # ── Baselines (don't use the projection; reset it for tidiness) ─────────
    llm.projection.load_state_dict(initial_projection)
    makers = {
        "react": lambda: ReActAgent(llm=llm),
        "zeroshot": ZeroShotAgent,
        "random": lambda: RandomAgent(seed=args.seed),
    }
    for name in [b.strip() for b in args.baselines.split(",") if b.strip()]:
        if name not in makers:
            logger.error("Unknown baseline %r (choose from %s)", name, ", ".join(makers))
            return 1
        record(name, name, makers[name](), None)

    logger.info("Done: %d episodes in %s", len(done), out)
    print(json.dumps({"episodes": len(done), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
