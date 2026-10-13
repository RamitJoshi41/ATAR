#!/usr/bin/env python3
"""
scripts/evaluate.py — Standalone evaluation CLI for ATAR.

Evaluates one or more agents against the held-out test task set and
produces structured output for analysis and M6 (prompt optimizer).

Usage examples:
    # Full evaluation with a trained checkpoint:
    python scripts/evaluate.py \\
        --checkpoint ./checkpoints/checkpoint_final.pt \\
        --agents atar,random,zeroshot,react \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --out-dir ./eval_results \\
        --device cuda

    # Smoke test (5 tasks, local 0.5B stand-in, CPU):
    python scripts/evaluate.py \\
        --agents random,zeroshot \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --max-tasks 5 \\
        --out-dir ./eval_smoke \\
        --device cpu

    # Resume an interrupted evaluation:
    python scripts/evaluate.py \\
        --checkpoint ./checkpoints/checkpoint_final.pt \\
        --agents atar \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --out-dir ./eval_results \\
        --device cuda
        # (already-completed tasks in per_task_atar.jsonl are skipped)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description="ATAR evaluation harness — standalone CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a trained policy checkpoint (.pt file). "
            "Required when 'atar' is in --agents."
        ),
    )
    p.add_argument(
        "--agents",
        type=str,
        default="atar,random,zeroshot,react",
        help="Comma-separated list of agents to evaluate: atar,random,zeroshot,react.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./eval_results",
        help="Output directory for results.",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="If set, only evaluate this many tasks per agent (smoke-test flag).",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HuggingFace model name. "
            "Use Qwen/Qwen2.5-0.5B-Instruct locally; "
            "Qwen/Qwen2.5-7B-Instruct on the eval GPU."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing tasks_train.jsonl and tasks_test.jsonl.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("evaluate")

    agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]
    logger.info("ATAR Evaluation Harness")
    logger.info("Agents:     %s", agent_names)
    logger.info("Model:      %s", args.model_name)
    logger.info("Data dir:   %s", args.data_dir)
    logger.info("Device:     %s", args.device)
    logger.info("Max tasks:  %s", args.max_tasks)
    logger.info("Out dir:    %s", args.out_dir)

    # ── Validate args ─────────────────────────────────────────────────────
    if "atar" in agent_names and args.checkpoint is None:
        logger.error("--checkpoint is required when 'atar' is in --agents.")
        return 1

    data_dir = Path(args.data_dir)
    test_path = data_dir / "tasks_test.jsonl"
    train_path = data_dir / "tasks_train.jsonl"

    if not test_path.exists():
        logger.error("Test data not found: %s", test_path)
        return 1
    if not train_path.exists():
        logger.error("Train data not found: %s", train_path)
        return 1

    # ── Deferred imports (keep --help fast) ────────────────────────────────
    from atar.evaluation.runner import (
        EvalRunner,
        ATARAgent,
        RandomAgent,
        ZeroShotAgent,
        ReActAgent,
        load_tasks_from_jsonl,
        check_disjoint,
        verify_stratification,
    )
    from atar.llm.llm_interface import LLMInterface
    from atar.policy.policy_network import ATARPolicy
    from atar.policy.trainer import PPOTrainer

    # ── Load test tasks ───────────────────────────────────────────────────
    test_tasks = load_tasks_from_jsonl(test_path)
    logger.info("Loaded %d test tasks.", len(test_tasks))

    # Verify every tier is present and tier shares match the training set
    train_tasks = load_tasks_from_jsonl(train_path)
    tier_counts = verify_stratification(test_tasks, reference_tasks=train_tasks)
    logger.info("Tier distribution: %s", tier_counts)

    # Verify disjointness with training set
    check_disjoint(test_tasks, train_path)

    # ── Load LLM ──────────────────────────────────────────────────────────
    logger.info("Loading LLM: %s ...", args.model_name)
    llm = LLMInterface(model_name=args.model_name, device=args.device)
    logger.info("LLM loaded.")

    # ── Build agents ──────────────────────────────────────────────────────
    agents: dict[str, object] = {}
    # Identity of each agent's run — resuming into an out-dir produced by a
    # different checkpoint/model/seed is refused (see EvalRunner).
    base_fp = {"model_name": args.model_name, "seed": args.seed, "grader": "answer_matching_v2"}
    fingerprints: dict[str, dict] = {}

    for name in agent_names:
        fingerprints[name] = dict(base_fp)
        if name == "atar":
            logger.info("Loading ATAR checkpoint: %s", args.checkpoint)
            ckpt = PPOTrainer.load_checkpoint(args.checkpoint)
            policy = ATARPolicy()
            policy.load_state_dict(ckpt.policy_state_dict)
            policy = policy.to(args.device)

            # The policy was trained on states built by THIS projection, so
            # it must be restored too.  A hidden-size mismatch means the
            # checkpoint was trained on a different LLM (e.g. 7B vs 0.5B).
            if ckpt.projection_state_dict is not None:
                ckpt_in = ckpt.projection_state_dict["weight"].shape[1]
                if ckpt_in != llm.projection.in_features:
                    logger.error(
                        "Checkpoint projection expects hidden size %d but %s has %d. "
                        "Evaluate with the same --model-name used for training.",
                        ckpt_in, args.model_name, llm.projection.in_features,
                    )
                    return 1
                llm.projection.load_state_dict(ckpt.projection_state_dict)
                logger.info("Loaded co-trained projection weights from checkpoint.")
            else:
                logger.warning(
                    "Checkpoint has no projection weights (pre-co-training format). "
                    "Using the seeded initial projection — results are only "
                    "meaningful if --model-name matches the model used in training."
                )

            agents["atar"] = ATARAgent(policy=policy, device=args.device)
            fingerprints["atar"]["checkpoint_sha256"] = _sha256(args.checkpoint)
            logger.info(
                "ATAR policy loaded (step=%d, phase=%d).",
                ckpt.step, ckpt.curriculum_phase,
            )
        elif name == "random":
            agents["random"] = RandomAgent(seed=args.seed)
        elif name == "zeroshot":
            agents["zeroshot"] = ZeroShotAgent()
        elif name == "react":
            agents["react"] = ReActAgent(llm=llm)
        else:
            logger.warning("Unknown agent '%s' — skipping.", name)

    if not agents:
        logger.error("No valid agents specified.")
        return 1

    # ── Run evaluation ────────────────────────────────────────────────────
    runner = EvalRunner(
        tasks=test_tasks,
        llm=llm,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        fingerprints=fingerprints,
    )

    results = runner.evaluate_all(
        agents=agents,
        max_tasks=args.max_tasks,
        bootstrap_seed=args.seed,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    for agent_name, agent_results in results.items():
        overall = agent_results["overall"]
        logger.info(
            "[%s] accuracy=%.4f  success_rate=%.4f  avg_tool_calls=%.2f  "
            "efficiency=%.4f  recovery_rate=%.4f  mean_latency=%.2fs",
            agent_name,
            overall["accuracy"],
            overall["success_rate"],
            overall["avg_tool_calls"],
            overall["efficiency"],
            overall["recovery_rate"],
            overall["mean_latency"],
        )

    logger.info("Output directory: %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
