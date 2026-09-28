#!/usr/bin/env python3
"""
scripts/train.py — Standalone PPO training CLI for ATAR.

This script is designed to be copied to and executed inside a Kaggle notebook,
RunPod instance, or other remote GPU environment.  It must NOT assume anything
about the local Antigravity dev environment — it only depends on the 'atar'
package being installed (or the repo root being in PYTHONPATH).

Usage examples:
    # Initial training run:
    python scripts/train.py \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --data-dir ./data \\
        --n-envs 8 \\
        --total-steps 100000 \\
        --checkpoint-dir ./checkpoints \\
        --wandb-project atar \\
        --device cuda

    # Resume from checkpoint:
    python scripts/train.py \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --data-dir ./data \\
        --n-envs 8 \\
        --total-steps 100000 \\
        --checkpoint-dir ./checkpoints \\
        --resume-from ./checkpoints/checkpoint_step_50000.pt \\
        --device cuda

    # Local smoke test (tiny stand-in model, offline W&B):
    python scripts/train.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --data-dir ./data \\
        --n-envs 2 \\
        --total-steps 10 \\
        --rollout-steps 5 \\
        --checkpoint-dir /tmp/atar_test_ckpts \\
        --wandb-mode offline \\
        --device cpu

M3 note: set --model-name to Qwen/Qwen2.5-0.5B-Instruct for local testing
on a 4GB GPU; use Qwen/Qwen2.5-7B-Instruct on the remote training GPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="ATAR PPO training — standalone CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model / environment ───────────────────────────────────────────────
    p.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HuggingFace model name or local path. "
            "Use Qwen/Qwen2.5-0.5B-Instruct locally; "
            "Qwen/Qwen2.5-7B-Instruct on the training GPU."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing tasks_train.jsonl and tasks_eval.jsonl.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )

    # ── Parallelism ───────────────────────────────────────────────────────
    p.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help=(
            "Number of parallel ATAREnv instances.  8-16 is the MSD "
            "recommended range.  Lower for local testing."
        ),
    )

    # ── PPO hyperparameters ───────────────────────────────────────────────
    p.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
        help="Total environment steps across all training.",
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="Transitions collected per PPO update cycle.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO update.",
    )
    p.add_argument(
        "--epochs-per-update",
        type=int,
        default=10,
        help="Passes over the rollout buffer per update.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate (cosine decays to --lr-min).",
    )
    p.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Cosine LR decay target.",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor.",
    )
    p.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="GAE lambda.",
    )
    p.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip parameter epsilon.",
    )
    p.add_argument(
        "--entropy-coef-start",
        type=float,
        default=0.01,
        help="Entropy coefficient at step 0.",
    )
    p.add_argument(
        "--entropy-coef-end",
        type=float,
        default=0.001,
        help="Entropy coefficient at total_steps (linear decay).",
    )
    p.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="Value function loss coefficient.",
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm.",
    )

    # ── Checkpointing ─────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to write checkpoint .pt files.",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save a checkpoint every N optimizer updates.",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint .pt file to resume from. "
            "Restores weights, optimizer state, step count, and "
            "curriculum phase — training continues as if uninterrupted."
        ),
    )
    p.add_argument(
        "--stop-after-updates",
        type=int,
        default=None,
        help="Checkpoint and exit cleanly after N updates in this invocation.",
    )

    # ── Weights & Biases ──────────────────────────────────────────────────
    p.add_argument(
        "--wandb-project",
        type=str,
        default="atar",
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help=(
            "W&B logging mode. Use 'offline' for local testing; "
            "'online' requires WANDB_API_KEY to be set."
        ),
    )

    # ── Ablations ─────────────────────────────────────────────────────────
    p.add_argument(
        "--no-curriculum",
        action="store_true",
        default=False,
        help=(
            "Ablation: disable curriculum — train on all 5 tiers from step 0. "
            "For comparison against the curriculumed run."
        ),
    )
    p.add_argument(
        "--freeze-projection",
        action="store_true",
        default=False,
        help=(
            "Ablation: freeze M3's projection layer (don't co-train with policy). "
            "For comparison against the co-adapted projection baseline."
        ),
    )

    # ── Logging ───────────────────────────────────────────────────────────
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
    logger = logging.getLogger("train")

    logger.info("ATAR PPO Training")
    logger.info("Model:      %s", args.model_name)
    logger.info("Data dir:   %s", args.data_dir)
    logger.info("Device:     %s", args.device)
    logger.info("N envs:     %d", args.n_envs)
    logger.info("Total steps: %d", args.total_steps)
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)

    # ── Imports (deferred to keep --help fast) ───────────────────────────
    import torch
    from atar.llm.llm_interface import LLMInterface
    from atar.policy import ATARPolicy, PPOTrainer

    # ── Load model ───────────────────────────────────────────────────────
    logger.info("Loading LLM: %s ...", args.model_name)
    llm = LLMInterface(model_name=args.model_name, device=args.device)
    logger.info("LLM loaded.")

    # ── Build policy ─────────────────────────────────────────────────────
    policy = ATARPolicy()

    # ── Build trainer ─────────────────────────────────────────────────────
    trainer = PPOTrainer(
        policy=policy,
        llm=llm,
        data_dir=args.data_dir,
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        epochs_per_update=args.epochs_per_update,
        lr=args.lr,
        lr_min=args.lr_min,
        gamma=args.gamma,
        lam=args.lam,
        clip_range=args.clip_range,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        seed=args.seed,
        no_curriculum=args.no_curriculum,
        freeze_projection=args.freeze_projection,
        stop_after_updates=args.stop_after_updates,
    )

    # ── Resume from checkpoint if requested ───────────────────────────────
    if args.resume_from is not None:
        logger.info("Restoring from checkpoint: %s", args.resume_from)
        trainer.restore_from_checkpoint(args.resume_from)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
