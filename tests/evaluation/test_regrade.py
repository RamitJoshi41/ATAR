"""Tests for scripts/regrade.py — offline re-scoring of saved results (ADR-27)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_regrade() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "regrade.py"
    spec = importlib.util.spec_from_file_location("regrade", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regrade = _load_regrade()


def _rec(task_id: str, agent: str, query: str, truth: str, answer: str | None, correct: bool,
         tier: int = 2) -> dict[str, Any]:
    return {"task_id": task_id, "tier": tier, "query": query, "ground_truth": truth,
            "agent": agent, "final_answer": answer, "correct": correct}


def test_approx_numeric_only_for_boiling_points() -> None:
    assert regrade.is_approx_numeric_task("Query the boiling point of ethanol in Celsius")
    assert not regrade.is_approx_numeric_task("Compute the number of hours in 252 days")


def test_regrade_record_applies_both_rules() -> None:
    bp = _rec("t1", "atar", "What is the boiling point of benzene in Celsius", "80", "80.1°C", False)
    hours = _rec("t2", "atar", "Compute the number of hours in 2 days", "48", "49", False)
    accent = _rec("t3", "atar", "Find the capital city of Colombia", "Bogota", "Bogotá", False)
    none = _rec("t4", "atar", "Find the capital city of Peru", "Lima", None, False)
    assert regrade.regrade_record(bp) == (False, True)
    assert regrade.regrade_record(hours) == (False, False)
    assert regrade.regrade_record(accent) == (False, True)
    assert regrade.regrade_record(none) == (False, False)


def test_mcnemar_exact_p() -> None:
    assert regrade.mcnemar_exact_p(0, 0) == 1.0
    assert regrade.mcnemar_exact_p(5, 5) == 1.0
    assert regrade.mcnemar_exact_p(31, 10) == pytest.approx(0.0015, abs=1e-4)


def test_build_report_counts_and_pairs() -> None:
    q = "Find the capital city of Colombia"
    groups = {
        "atar@100": [_rec("a", "atar", q, "Bogota", "Bogotá", False),
                     _rec("b", "atar", q, "Bogota", "Bogota", True)],
        "react": [_rec("a", "react", q, "Bogota", "Lima", False),
                  _rec("b", "react", q, "Bogota", "Bogota", True)],
    }
    report = regrade.build_report(groups, ("atar@100", "react"))
    assert "| atar@100 | 2 |" in report
    assert "| 1 | 0 |" in report  # atar@100 gained 1, lost 0
    assert "| re-graded | 1 | 0 |" in report
    assert "Bogotá" in report


def test_build_report_stops_when_stored_verdicts_disagree() -> None:
    groups = {"react": [_rec("a", "react", "Find the capital city of Peru", "Lima", "Lima", False)]}
    with pytest.raises(SystemExit, match="not produced by this grader"):
        regrade.build_report(groups, None)


def test_load_eval_dir_labels_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "run_meta.json").write_text(json.dumps({"checkpoint_step": 24576}))
    (tmp_path / "per_task.jsonl").write_text(
        json.dumps(_rec("a", "atar", "q", "1", "1", True)) + "\n"
    )
    assert list(regrade.load_eval_dir(tmp_path)) == ["atar@24576"]
