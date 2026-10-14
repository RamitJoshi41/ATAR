#!/usr/bin/env python3
"""
scripts/build_demo_bundle.py — assemble the M8 demo's data folder.

Combines:
  - evaluate.py output dirs (one per checkpoint, plus baselines) → curve.json
  - record_replays.py JSONL files                                  → replays.jsonl
  - a manifest.json describing where the data came from

Runs anywhere (no GPU, no model).  Example:
    python scripts/build_demo_bundle.py \\
        --eval-dirs eval_step6144 eval_step12288 eval_final eval_baselines \\
        --replays replays.jsonl \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --out demo_data

--sample marks the bundle as SAMPLE data (e.g. stand-in model, toy
checkpoints); the app then shows a banner saying the numbers are not real
results.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_demo_bundle.py",
        description="Assemble the M8 demo data folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-dirs", nargs="*", default=[], help="scripts/evaluate.py --out-dir folders.")
    p.add_argument("--replays", nargs="*", default=[], help="scripts/record_replays.py --out files.")
    p.add_argument("--model-name", type=str, required=True, help="Model the checkpoints were trained/evaluated on.")
    p.add_argument("--out", type=str, default="./demo_data")
    p.add_argument("--sample", action="store_true", help="Mark the bundle as sample (not real) data.")
    p.add_argument("--note", type=str, default="", help="Free-text note shown in the app.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    from atar.demo.curve import build_learning_curve
    from atar.demo.replay import load_records, record_to_dict

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    curve = build_learning_curve([Path(d) for d in args.eval_dirs])
    (out / "curve.json").write_text(json.dumps(curve, indent=2))

    seen: set[tuple[str, str]] = set()
    n_replays = 0
    with open(out / "replays.jsonl", "w", encoding="utf-8") as fh:
        for path in args.replays:
            for rec in load_records(Path(path)):
                key = (rec.version, rec.task_id)
                if key in seen:
                    continue
                seen.add(key)
                fh.write(json.dumps(record_to_dict(rec)) + "\n")
                n_replays += 1

    manifest = {
        "model_name": args.model_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_sample": bool(args.sample),
        "note": args.note,
        "eval_dirs": [str(d) for d in args.eval_dirs],
        "n_checkpoints_on_curve": len(curve["atar"]),
        "baselines": sorted(curve["baselines"]),
        "n_replays": n_replays,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
