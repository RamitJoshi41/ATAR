#!/usr/bin/env python3
"""
scripts/regrade.py — re-score saved evaluation results with the ADR-27 grader.

Reads per_task.jsonl from evaluate.py output dirs and checks every saved final
answer twice: with the original grader (it must reproduce the stored verdicts,
otherwise the script stops) and with is_answer_correct_regraded (accents
ignored; ±1.0 on boiling-point tasks, whose ground truths are whole degrees).
Writes a markdown report with both sets of numbers.  No GPU, no model.

    python scripts/regrade.py \\
        --eval-dirs results/run2/eval_step6144 ... results/run2/eval_baselines \\
        --paired atar@24576 react --out results/run2/regrade.md
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atar.env.answer_matching import is_answer_correct, is_answer_correct_regraded  # noqa: E402
from atar.evaluation.metrics import bootstrap_ci  # noqa: E402


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


def load_eval_dir(eval_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Records grouped by label: 'atar@<step>' for checkpoints, else the agent name."""
    step = None
    meta_path = eval_dir / "run_meta.json"
    if meta_path.exists():
        step = json.loads(meta_path.read_text()).get("checkpoint_step")
    groups: dict[str, list[dict[str, Any]]] = {}
    for line in (eval_dir / "per_task.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        label = f"atar@{step}" if rec["agent"] == "atar" and step is not None else rec["agent"]
        groups.setdefault(label, []).append(rec)
    return groups


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def _fmt_ci(values: list[float]) -> str:
    point, lo, hi = bootstrap_ci(values)
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def build_report(groups: dict[str, list[dict[str, Any]]], paired: tuple[str, str] | None) -> str:
    verdicts: dict[str, list[tuple[dict[str, Any], bool, bool]]] = {}
    for label, records in groups.items():
        rows = []
        for rec in records:
            original, regraded = regrade_record(rec)
            if original != bool(rec["correct"]):
                raise SystemExit(
                    f"{label} task {rec['task_id']}: original grader gives {original}, "
                    f"stored verdict is {rec['correct']} — results were not produced by this grader."
                )
            rows.append((rec, original, regraded))
        verdicts[label] = rows

    lines = [
        "# Run 2 — original vs re-graded accuracy (ADR-27)",
        "",
        "Same saved answers, two graders. **Original** = the grader run 2 was trained and",
        "evaluated with (reproduces every stored verdict). **Re-graded** = accents ignored,",
        "and boiling-point answers within ±1 °C accepted (ground truths are whole degrees).",
        "95% bootstrap CIs (10,000 resamples, seed 42).",
        "",
        "| agent | n | original | re-graded | gained | lost |",
        "|---|---|---|---|---|---|",
    ]
    for label, rows in verdicts.items():
        orig = [float(o) for _, o, _ in rows]
        regr = [float(r) for _, _, r in rows]
        gained = sum(1 for _, o, r in rows if r and not o)
        lost = sum(1 for _, o, r in rows if o and not r)
        lines.append(
            f"| {label} | {len(rows)} | {_fmt_ci(orig)} | {_fmt_ci(regr)} | {gained} | {lost} |"
        )

    tiers = sorted({rec["tier"] for rows in verdicts.values() for rec, _, _ in rows})
    lines += ["", "## Per tier (original → re-graded)", ""]
    lines.append("| agent | " + " | ".join(f"tier {t}" for t in tiers) + " |")
    lines.append("|---|" + "---|" * len(tiers))
    for label, rows in verdicts.items():
        cells = []
        for t in tiers:
            tier_rows = [(o, r) for rec, o, r in rows if rec["tier"] == t]
            n = len(tier_rows)
            cells.append(
                f"{sum(o for o, _ in tier_rows) / n:.2f} → {sum(r for _, r in tier_rows) / n:.2f}"
                if n else "—"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    if paired is not None:
        a, b = paired
        by_task_b = {rec["task_id"]: (o, r) for rec, o, r in verdicts[b]}
        lines += ["", f"## Paired: {a} vs {b} (same tasks)", ""]
        lines += [f"| grader | only {a} right | only {b} right | exact McNemar p |", "|---|---|---|---|"]
        for idx, name in ((0, "original"), (1, "re-graded")):
            only_a = only_b = 0
            for rec, o, r in verdicts[a]:
                va = (o, r)[idx]
                vb = by_task_b[rec["task_id"]][idx]
                only_a += int(va and not vb)
                only_b += int(vb and not va)
            p = mcnemar_exact_p(only_a, only_b)
            lines.append(f"| {name} | {only_a} | {only_b} | {'< 0.0001' if p < 1e-4 else f'{p:.4f}'} |")

    lines += ["", "## Every changed verdict", "", "| agent | tier | answer | ground truth | now |", "|---|---|---|---|---|"]
    for label, rows in verdicts.items():
        for rec, o, r in rows:
            if o != r:
                answer = str(rec.get("final_answer")).replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {label} | {rec['tier']} | {answer[:80]} | {rec['ground_truth']} | "
                    f"{'correct' if r else 'wrong'} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--paired", nargs=2, metavar=("AGENT_A", "AGENT_B"), default=None,
                        help="Two labels to compare task by task, e.g. atar@24576 react.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, Any]]] = {}
    for d in args.eval_dirs:
        for label, records in load_eval_dir(d).items():
            if label in groups:
                raise SystemExit(f"label {label} appears in more than one eval dir")
            groups[label] = records
    report = build_report(groups, tuple(args.paired) if args.paired else None)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
