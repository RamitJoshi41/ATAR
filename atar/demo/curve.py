"""
M8 — the learning curve: held-out evaluation results per checkpoint.

Input: evaluation output directories written by scripts/evaluate.py, one per
checkpoint (each holds results.json, per_task_<agent>.jsonl and, for ATAR,
run_meta.json with the checkpoint's training step).  Baselines (react,
zeroshot, random) may live in any of those directories or in their own.

Output (curve.json in the demo bundle):
{
  "atar": [ {"version", "step", "phase", "overall", "per_tier", "action_mix"}, ... ]  # sorted by step
  "baselines": { "react": {"overall", "per_tier", "action_mix"}, ... }
}
Everything here is computed from held-out evaluation, never training reward.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASELINES = ("react", "zeroshot", "random")


def action_mix_by_tier(per_task_path: Path) -> dict[str, dict[str, float]]:
    """Fraction of each action among all steps, per tier ("tier_1", ...) and "all"."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with open(per_task_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for action in rec.get("action_trace", []):
                counts[f"tier_{rec['tier']}"][action] += 1
                counts["all"][action] += 1
    return {
        tier: {a: n / sum(c.values()) for a, n in sorted(c.items())}
        for tier, c in sorted(counts.items())
    }


def _entry(eval_dir: Path, agent: str, results: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "overall": results[agent]["overall"],
        "per_tier": results[agent].get("per_tier", {}),
        "eval_dir": str(eval_dir),
    }
    per_task = eval_dir / f"per_task_{agent}.jsonl"
    entry["action_mix"] = action_mix_by_tier(per_task) if per_task.exists() else {}
    return entry


def build_learning_curve(eval_dirs: list[Path]) -> dict[str, Any]:
    """Collect ATAR checkpoints (by training step) and baselines from eval dirs."""
    atar: dict[int, dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}

    for eval_dir in eval_dirs:
        results_path = eval_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} not found — run scripts/evaluate.py first")
        results = json.loads(results_path.read_text())

        if "atar" in results:
            meta_path = eval_dir / "run_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"{meta_path} missing: can't tell which training step {eval_dir} "
                    "evaluated (re-run it with the current scripts/evaluate.py)"
                )
            meta = json.loads(meta_path.read_text())
            step = int(meta["checkpoint_step"])
            if step in atar:
                raise ValueError(f"Two eval dirs for training step {step}: {atar[step]['eval_dir']} and {eval_dir}")
            entry = _entry(eval_dir, "atar", results)
            entry.update(version=f"step_{step}", step=step, phase=meta.get("curriculum_phase"))
            atar[step] = entry

        for name in BASELINES:
            if name not in results:
                continue
            candidate = _entry(eval_dir, name, results)
            current = baselines.get(name)
            # Keep the largest evaluation of each baseline.
            if current is None or candidate["overall"]["n_tasks"] > current["overall"]["n_tasks"]:
                baselines[name] = candidate

    return {"atar": [atar[s] for s in sorted(atar)], "baselines": baselines}


# ─── Chart rows (pure; the app turns these into Altair charts) ───────────

METRIC_LABELS = {
    "accuracy": "Success rate",
    "avg_tool_calls": "Average tool calls per task",
    "efficiency": "Efficiency (accuracy / (1 + tool calls))",
}


def _metrics_for(entry: dict[str, Any], tier: str) -> dict[str, Any] | None:
    return entry["overall"] if tier == "all" else entry.get("per_tier", {}).get(tier)


def curve_rows(curve: dict[str, Any], metric: str = "accuracy", tier: str = "all") -> list[dict[str, Any]]:
    """One row per ATAR checkpoint: step, value, and the 95% CI when it exists."""
    rows = []
    for entry in curve.get("atar", []):
        m = _metrics_for(entry, tier)
        if m is None or metric not in m:
            continue
        ci = m.get(f"{metric}_ci")
        rows.append({
            "step": entry["step"], "version": entry["version"], "value": m[metric],
            "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
            "n_tasks": m.get("n_tasks"),
        })
    return rows


def baseline_rows(curve: dict[str, Any], metric: str = "accuracy", tier: str = "all") -> list[dict[str, Any]]:
    """One row per baseline: its value for the metric (drawn as a flat reference line)."""
    rows = []
    for name, entry in sorted(curve.get("baselines", {}).items()):
        m = _metrics_for(entry, tier)
        if m is not None and metric in m:
            rows.append({"baseline": name, "value": m[metric]})
    return rows


def action_mix_rows(curve: dict[str, Any], tier: str = "all") -> list[dict[str, Any]]:
    """Long-format rows (version, step, action, fraction) for a stacked bar per version."""
    rows = []
    for entry in curve.get("atar", []):
        for action, frac in entry.get("action_mix", {}).get(tier, {}).items():
            rows.append({"version": entry["version"], "step": entry["step"], "action": action, "fraction": frac})
    for name, entry in sorted(curve.get("baselines", {}).items()):
        for action, frac in entry.get("action_mix", {}).get(tier, {}).items():
            rows.append({"version": name, "step": None, "action": action, "fraction": frac})
    return rows
