"""Learning curve assembly from evaluate.py output dirs (hand-made fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from atar.demo.curve import action_mix_by_tier, action_mix_rows, baseline_rows, build_learning_curve, curve_rows
from tests.demo.helpers import make_eval_dir, metrics


def test_checkpoints_sorted_by_step_and_baselines_collected(tmp_path: Path) -> None:
    dirs = [
        make_eval_dir(tmp_path, "b", {"atar": {"overall": metrics(0.7, 200)}}, step=2048),
        make_eval_dir(tmp_path, "a", {"atar": {"overall": metrics(0.3, 200)}}, step=0),
        make_eval_dir(tmp_path, "base", {"react": {"overall": metrics(0.5, 200)},
                                         "random": {"overall": metrics(0.1, 200)}}),
    ]
    curve = build_learning_curve(dirs)
    assert [e["step"] for e in curve["atar"]] == [0, 2048]
    assert [e["version"] for e in curve["atar"]] == ["step_0", "step_2048"]
    assert sorted(curve["baselines"]) == ["random", "react"]


def test_largest_baseline_evaluation_wins(tmp_path: Path) -> None:
    dirs = [
        make_eval_dir(tmp_path, "small", {"react": {"overall": metrics(0.9, 20)}}),
        make_eval_dir(tmp_path, "full", {"react": {"overall": metrics(0.5, 200)}}),
    ]
    assert build_learning_curve(dirs)["baselines"]["react"]["overall"]["n_tasks"] == 200


def test_missing_run_meta_or_results_raise(tmp_path: Path) -> None:
    no_meta = make_eval_dir(tmp_path, "x", {"atar": {"overall": metrics(0.5, 10)}})
    with pytest.raises(FileNotFoundError, match="run_meta"):
        build_learning_curve([no_meta])
    with pytest.raises(FileNotFoundError, match="results.json"):
        build_learning_curve([tmp_path / "nothing-here"])


def test_duplicate_step_raises(tmp_path: Path) -> None:
    dirs = [make_eval_dir(tmp_path, n, {"atar": {"overall": metrics(0.5, 10)}}, step=64) for n in ("a", "b")]
    with pytest.raises(ValueError, match="64"):
        build_learning_curve(dirs)


def test_action_mix_hand_computed(tmp_path: Path) -> None:
    """Tier 1: CALC, ANSWER, CALC -> CALC 2/3; tier 2: ANSWER -> 1.0; all: CALC 2/4, ANSWER 2/4."""
    d = make_eval_dir(tmp_path, "m", {}, traces={"atar": [
        (1, ["CALCULATOR", "ANSWER_DIRECTLY"]), (1, ["CALCULATOR"]), (2, ["ANSWER_DIRECTLY"])]})
    mix = action_mix_by_tier(d / "per_task_atar.jsonl")
    assert mix["tier_1"]["CALCULATOR"] == pytest.approx(2 / 3)
    assert mix["tier_2"] == {"ANSWER_DIRECTLY": 1.0}
    assert mix["all"] == {"ANSWER_DIRECTLY": 0.5, "CALCULATOR": 0.5}


def test_chart_rows(tmp_path: Path) -> None:
    dirs = [
        make_eval_dir(tmp_path, "a", {"atar": {"overall": metrics(0.3, 200), "per_tier": {"tier_1": metrics(0.4, 50)}}},
                      step=0, traces={"atar": [(1, ["CALCULATOR"])]}),
        make_eval_dir(tmp_path, "base", {"react": {"overall": metrics(0.5, 200)}}),
    ]
    curve = build_learning_curve(dirs)
    rows = curve_rows(curve, "accuracy", "all")
    assert rows == [{"step": 0, "version": "step_0", "value": 0.3, "ci_low": pytest.approx(0.2),
                     "ci_high": pytest.approx(0.4), "n_tasks": 200}]
    assert curve_rows(curve, "accuracy", "tier_1")[0]["value"] == 0.4
    assert curve_rows(curve, "accuracy", "tier_5") == []
    assert baseline_rows(curve, "accuracy") == [{"baseline": "react", "value": 0.5}]
    assert action_mix_rows(curve, "all") == [{"version": "step_0", "step": 0, "action": "CALCULATOR", "fraction": 1.0}]


def _write_answers(d: Path, agent: str, rows: list[tuple[int, str, str, str]]) -> None:
    """rows: (tier, query, ground_truth, final_answer) — one tool call each."""
    import json

    with open(d / f"per_task_{agent}.jsonl", "w") as fh:
        for tier, query, truth, answer in rows:
            fh.write(json.dumps({"tier": tier, "query": query, "ground_truth": truth, "final_answer": answer,
                                 "tool_call_count": 1, "action_trace": ["TERMINATE"]}) + "\n")


def test_regraded_view(tmp_path: Path) -> None:
    from atar.demo.curve import has_regraded

    d = make_eval_dir(tmp_path, "a", {"atar": {"overall": metrics(0.25, 4),
                                               "per_tier": {"tier_2": metrics(0.25, 4)}}}, step=100)
    _write_answers(d, "atar", [
        (2, "Find the capital city of Brazil", "Brasilia", "Brasília"),          # accent: gained
        (2, "What is the boiling point of benzene in Celsius", "80", "80.1°C"),  # ±1 °C: gained
        (2, "Compute the number of hours in 2 days", "48", "49"),                # exact: still wrong
        (2, "Find the capital city of Peru", "Lima", "Lima"),                     # already right
    ])
    curve = build_learning_curve([d])
    assert has_regraded(curve)
    assert curve_rows(curve, "accuracy", "all")[0]["value"] == 0.25                 # original untouched
    regraded = curve_rows(curve, "accuracy", "all", grader="regraded")[0]
    assert regraded["value"] == pytest.approx(0.75)
    assert curve_rows(curve, "accuracy", "tier_2", grader="regraded")[0]["value"] == pytest.approx(0.75)
    # tool calls don't depend on the grader; efficiency follows the re-graded accuracy
    assert curve_rows(curve, "avg_tool_calls", "all", grader="regraded")[0]["value"] == 1.0
    assert curve_rows(curve, "efficiency", "all", grader="regraded")[0]["value"] == pytest.approx(0.75 / 2)


def test_no_regraded_view_without_answers(tmp_path: Path) -> None:
    from atar.demo.curve import has_regraded

    d = make_eval_dir(tmp_path, "a", {"atar": {"overall": metrics(0.3, 2)}}, step=0,
                      traces={"atar": [(1, ["CALCULATOR"])]})   # old-style file: no answers saved
    curve = build_learning_curve([d])
    assert not has_regraded(curve)
    assert curve_rows(curve, "accuracy", "all", grader="regraded") == []
    assert curve_rows(curve, "accuracy", "all")[0]["value"] == 0.3
