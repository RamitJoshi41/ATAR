"""
Re-grading saved evaluation answers with the ADR-27 rules (reporting only).

Shared by scripts/regrade.py (the report) and atar.demo.curve (the demo's
"re-graded" view).  The original verdicts stay what evaluate.py stored; this
only adds a second, corrected verdict per answer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from atar.env.answer_matching import is_answer_correct, is_answer_correct_regraded
from atar.evaluation.metrics import bootstrap_ci, compute_efficiency


def is_approx_numeric_task(query: str) -> bool:
    """Boiling-point tasks: their ground truths are rounded constants (generator SUBSTANCES)."""
    return "boiling point" in query.lower()


def regrade_record(record: dict[str, Any]) -> tuple[bool, bool]:
    """(original verdict, re-graded verdict) for one per_task.jsonl record."""
    answer = record.get("final_answer")
    answer_str = None if answer is None else str(answer)
    truth = str(record["ground_truth"])
    original = is_answer_correct(answer_str, truth)
    regraded = is_answer_correct_regraded(
        answer_str, truth, approx_numeric=is_approx_numeric_task(record["query"])
    )
    return original, regraded


def _summary(rows: list[tuple[bool, int]]) -> dict[str, Any]:
    point, lo, hi = bootstrap_ci([float(ok) for ok, _ in rows])
    tools = sum(t for _, t in rows) / len(rows)
    return {"n_tasks": len(rows), "accuracy": point, "accuracy_ci": [lo, hi],
            "efficiency": compute_efficiency(point, tools)}


def regraded_metrics(per_task_path: Path) -> dict[str, Any] | None:
    """
    Re-graded accuracy (with 95% bootstrap CI) and efficiency, overall and per
    tier ("tier_1", ...), from one agent's per_task_<agent>.jsonl.  None if the
    file lacks the fields needed to re-grade (query, ground_truth, final_answer).
    """
    rows: list[dict[str, Any]] = []
    with open(per_task_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows or any(k not in rows[0] for k in ("query", "ground_truth", "final_answer")):
        return None
    by_tier: dict[str, list[tuple[bool, int]]] = defaultdict(list)
    overall: list[tuple[bool, int]] = []
    for rec in rows:
        item = (regrade_record(rec)[1], int(rec.get("tool_call_count", 0)))
        overall.append(item)
        by_tier[f"tier_{rec['tier']}"].append(item)
    return {"overall": _summary(overall),
            "per_tier": {t: _summary(r) for t, r in sorted(by_tier.items())}}
