"""Fabricated eval dirs and bundles for demo tests. Clearly fake numbers — test fixtures only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atar.demo.replay import EpisodeRecord, StepRecord, append_records


def metrics(acc: float, n: int, tools: float = 1.0) -> dict[str, Any]:
    return {"n_tasks": n, "accuracy": acc, "accuracy_ci": [acc - 0.1, acc + 0.1],
            "success_rate": acc, "avg_tool_calls": tools, "efficiency": acc / (1 + tools),
            "recovery_rate": 0.0, "mean_latency": 1.0}


def make_eval_dir(root: Path, name: str, results: dict[str, Any], step: int | None = None,
                  traces: dict[str, list[tuple[int, list[str]]]] | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(results))
    if step is not None:
        (d / "run_meta.json").write_text(json.dumps({"checkpoint_step": step, "curriculum_phase": 1}))
    for agent, rows in (traces or {}).items():
        with open(d / f"per_task_{agent}.jsonl", "w") as fh:
            for tier, trace in rows:
                fh.write(json.dumps({"tier": tier, "action_trace": trace}) + "\n")
    return d


def make_bundle(root: Path, sample: bool = True) -> Path:
    from atar.demo.curve import build_learning_curve
    evals = [
        make_eval_dir(root / "ev", "s0", {"atar": {"overall": metrics(0.2, 10), "per_tier": {"tier_1": metrics(0.2, 5)}}},
                      step=0, traces={"atar": [(1, ["SQL_QUERY", "TERMINATE"])]}),
        make_eval_dir(root / "ev", "s100", {"atar": {"overall": metrics(0.6, 10), "per_tier": {"tier_1": metrics(0.6, 5)}}},
                      step=100, traces={"atar": [(1, ["CALCULATOR", "ANSWER_DIRECTLY"])]}),
        make_eval_dir(root / "ev", "base", {"react": {"overall": metrics(0.4, 10), "per_tier": {}}}),
    ]
    out = root / "bundle"
    out.mkdir()
    (out / "curve.json").write_text(json.dumps(build_learning_curve(evals)))
    (out / "manifest.json").write_text(json.dumps({"model_name": "stub", "is_sample": sample, "note": "test"}))
    recs = []
    for version, agent, step, actions, correct in [
        ("step_0", "atar", 0, ["SQL_QUERY", "TERMINATE"], False),
        ("step_100", "atar", 100, ["CALCULATOR", "ANSWER_DIRECTLY"], True),
        ("react", "react", None, ["ANSWER_DIRECTLY"], False),
    ]:
        steps = [StepRecord(turn=i + 1, action=a,
                            action_probs=[1 / 7] * 7 if agent == "atar" else None,
                            value_estimate=0.5 if agent == "atar" else None,
                            thought="I should compute it." if agent == "react" else None,
                            params={"expression": "2 + 2"} if a == "CALCULATOR" else None,
                            param_error="bad sql" if a == "SQL_QUERY" else None,
                            tool_success=True if a == "CALCULATOR" else None,
                            tool_output="4.0" if a == "CALCULATOR" else None, tool_error=None,
                            reward={"accuracy": 0.0, "efficiency": -0.5, "safety": 0.0, "bonus": 0.0, "total": -0.5},
                            cumulative_reward=-0.5 * (i + 1),
                            final_answer="4" if i == len(actions) - 1 else None, latency_s=0.1)
                 for i, a in enumerate(actions)]
        recs.append(EpisodeRecord(version=version, agent=agent, checkpoint_step=step, task_id="calc-1",
                                  tier=1, query="What is 2 + 2?", ground_truth="4", model_name="stub",
                                  steps=steps, final_answer="4", correct=correct, terminated=True,
                                  total_reward=-1.0, tool_calls=1, latency_s=0.5))
    append_records(out / "replays.jsonl", recs)
    return out
