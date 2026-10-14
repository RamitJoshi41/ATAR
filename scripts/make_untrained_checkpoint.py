#!/usr/bin/env python3
"""
scripts/make_untrained_checkpoint.py — a step-0 checkpoint for the learning curve.

Training saves its first checkpoint after the first update, so there is no
file for the untrained starting point.  This writes one: a freshly
initialised ATARPolicy (seeded) plus the projection's deterministic initial
weights for the given model's hidden size (the exact weights every training
run starts from — see llm_interface.init_projection).  evaluate.py and
record_replays.py load it like any other checkpoint.

No model download needed:
    python scripts/make_untrained_checkpoint.py --hidden-size 3584 --out checkpoint_step_0.pt
(3584 = Qwen2.5-7B-Instruct's hidden size; 896 for the 0.5B stand-in.)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden-size", type=int, required=True, help="LLM hidden size (3584 for Qwen2.5-7B).")
    p.add_argument("--seed", type=int, default=42, help="Seed for the policy's random initialisation.")
    p.add_argument("--out", type=str, default="checkpoint_step_0.pt")
    args = p.parse_args()

    import torch
    from atar.llm.llm_interface import init_projection
    from atar.policy.policy_network import ATARPolicy
    from atar.policy.types import CheckpointState

    torch.manual_seed(args.seed)
    ckpt = CheckpointState(
        step=0,
        curriculum_phase=1,
        policy_state_dict=ATARPolicy().state_dict(),
        optimizer_state_dict={},
        scheduler_state_dict=None,
        projection_state_dict=init_projection(args.hidden_size).state_dict(),
    )
    torch.save(ckpt.__dict__, args.out)
    print(f"Wrote untrained step-0 checkpoint (hidden size {args.hidden_size}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
