"""
Tests for evaluation metrics — hand-computed expected values.

See tests/evaluation/test_metrics.py for the metric function unit tests.
This file tests the runner-level functions: disjointness check,
stratification, and answer correctness — against the REAL task generator.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from atar.evaluation.runner import (
    EvalRunner,
    check_answer_correct,
    check_disjoint,
    verify_stratification,
    load_tasks_from_jsonl,
)
from atar.shared.shared_types import Task


# ─── check_answer_correct ────────────────────────────────────────────────

class TestCheckAnswerCorrect:
    def test_exact_match(self) -> None:
        assert check_answer_correct("Paris", "Paris") is True

    def test_case_insensitive(self) -> None:
        assert check_answer_correct("paris", "Paris") is True

    def test_truth_in_answer(self) -> None:
        assert check_answer_correct("The answer is Paris.", "Paris") is True

    def test_answer_in_truth(self) -> None:
        assert check_answer_correct("42", "The answer is 42") is True

    def test_numeric_match(self) -> None:
        assert check_answer_correct("24679.0", "24679") is True

    def test_numeric_with_commas(self) -> None:
        assert check_answer_correct("1,000", "1000") is True

    def test_wrong_answer(self) -> None:
        assert check_answer_correct("London", "Paris") is False

    def test_none_answer(self) -> None:
        assert check_answer_correct(None, "Paris") is False

    def test_whitespace_handling(self) -> None:
        assert check_answer_correct("  Paris  ", "Paris") is True

    # Regressions: the old grader accepted substrings in both directions.
    def test_digit_substring_is_not_correct(self) -> None:
        assert check_answer_correct("6", "61559.0") is False
        assert check_answer_correct("There are 12 projects", "2") is False

    def test_word_substring_is_not_correct(self) -> None:
        assert check_answer_correct("I do not know", "no") is False

    def test_equation_answer_is_correct(self) -> None:
        assert check_answer_correct("2660 + 58899 = 61559", "61559.0") is True

    def test_non_string_answer(self) -> None:
        assert check_answer_correct(42, "42") is True


# ─── Disjointness check against real task generator ───────────────────────

class TestDisjointness:
    """Test disjointness against the REAL generated data files."""

    def test_real_data_disjoint(self) -> None:
        """
        Verify that the actual tasks_test.jsonl and tasks_train.jsonl
        in data/ are disjoint.  This uses the real task generator output,
        not mocks.
        """
        data_dir = Path(__file__).parent.parent.parent / "data"
        test_path = data_dir / "tasks_test.jsonl"
        train_path = data_dir / "tasks_train.jsonl"

        if not test_path.exists() or not train_path.exists():
            pytest.skip("Data files not generated yet — run generate_dataset() first.")

        test_tasks = load_tasks_from_jsonl(test_path)
        # This should NOT raise
        check_disjoint(test_tasks, train_path)

    def test_overlap_detected(self, tmp_path: Path) -> None:
        """Verify that overlapping queries cause a RuntimeError."""
        import json
        from atar.shared.shared_types import ActionType

        # Create two JSONL files with overlapping queries
        task = {
            "id": "overlap-id",
            "tier": 1,
            "query": "What is 1 + 1?",
            "required_tools": [ActionType.CALCULATOR.value],
            "ground_truth": "2",
            "metadata": {},
        }
        train_path = tmp_path / "tasks_train.jsonl"
        test_path = tmp_path / "tasks_test.jsonl"

        train_path.write_text(json.dumps(task) + "\n")
        test_path.write_text(json.dumps(task) + "\n")

        test_tasks = load_tasks_from_jsonl(test_path)

        with pytest.raises(RuntimeError, match="overlap"):
            check_disjoint(test_tasks, train_path)


# ─── Stratification against real task generator ───────────────────────────

class TestStratification:
    """Test stratification against the REAL generated data files."""

    def test_real_data_stratification(self) -> None:
        """
        Verify the actual tasks_test.jsonl has the expected tier distribution:
        {1: 50, 2: 38, 3: 50, 4: 37, 5: 25} = 200 total.
        """
        data_dir = Path(__file__).parent.parent.parent / "data"
        test_path = data_dir / "tasks_test.jsonl"

        if not test_path.exists():
            pytest.skip("Test data not generated yet.")

        test_tasks = load_tasks_from_jsonl(test_path)
        tier_counts = verify_stratification(test_tasks)

        assert len(test_tasks) == 200
        assert tier_counts[1] == 50
        assert tier_counts[2] == 38
        assert tier_counts[3] == 50
        assert tier_counts[4] == 37
        assert tier_counts[5] == 25

    def test_real_data_matches_train_tier_shares(self) -> None:
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not (data_dir / "tasks_test.jsonl").exists():
            pytest.skip("Test data not generated yet.")
        test_tasks = load_tasks_from_jsonl(data_dir / "tasks_test.jsonl")
        train_tasks = load_tasks_from_jsonl(data_dir / "tasks_train.jsonl")
        verify_stratification(test_tasks, reference_tasks=train_tasks)  # no raise

    def test_missing_tier_raises(self) -> None:
        tasks = [_task(f"t{i}", tier) for i, tier in enumerate([1, 2, 3, 4])]
        with pytest.raises(RuntimeError, match="tier"):
            verify_stratification(tasks)

    def test_skewed_shares_raise(self) -> None:
        """Test: 60% tier 1 vs reference 20% -> outside 5% tolerance."""
        test = [_task(f"a{i}", t) for i, t in enumerate([1, 1, 1, 2, 3, 4, 5, 1, 1, 1])]
        ref = [_task(f"b{i}", t) for i, t in enumerate([1, 2, 3, 4, 5] * 2)]
        with pytest.raises(RuntimeError, match="tier 1"):
            verify_stratification(test, reference_tasks=ref)


def _task(task_id: str, tier: int) -> Task:
    return Task(id=task_id, tier=tier, query=f"q {task_id}", required_tools=[], ground_truth="1")


# ─── Resume safety (fingerprints) ──────────────────────────────────────────

class TestResumeFingerprint:
    """
    Resuming must never mix per-task results from different checkpoints or
    models.  Uses EvalRunner._check_or_write_fingerprint directly — no LLM.
    """

    def _runner(self, out_dir: Path, fp: dict) -> EvalRunner:
        return EvalRunner(
            tasks=[], llm=None, data_dir=out_dir, out_dir=out_dir,  # type: ignore[arg-type]
            fingerprints={"atar": fp},
        )

    def test_fresh_dir_writes_fingerprint(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path, {"checkpoint_sha256": "aaa"})
        runner._check_or_write_fingerprint("atar", tmp_path / "per_task_atar.jsonl")
        assert (tmp_path / "per_task_atar.meta.json").exists()

    def test_same_fingerprint_resumes(self, tmp_path: Path) -> None:
        per_task = tmp_path / "per_task_atar.jsonl"
        self._runner(tmp_path, {"checkpoint_sha256": "aaa"})._check_or_write_fingerprint("atar", per_task)
        per_task.write_text('{"task_id": "x"}\n')
        # No raise
        self._runner(tmp_path, {"checkpoint_sha256": "aaa"})._check_or_write_fingerprint("atar", per_task)

    def test_different_checkpoint_refuses_resume(self, tmp_path: Path) -> None:
        per_task = tmp_path / "per_task_atar.jsonl"
        self._runner(tmp_path, {"checkpoint_sha256": "aaa"})._check_or_write_fingerprint("atar", per_task)
        per_task.write_text('{"task_id": "x"}\n')
        with pytest.raises(RuntimeError, match="Refusing to resume"):
            self._runner(tmp_path, {"checkpoint_sha256": "bbb"})._check_or_write_fingerprint("atar", per_task)

    def test_results_without_fingerprint_refuse_resume(self, tmp_path: Path) -> None:
        """Legacy out-dirs (pre-fingerprint) can't be verified -> fail closed."""
        per_task = tmp_path / "per_task_atar.jsonl"
        per_task.write_text('{"task_id": "x"}\n')
        with pytest.raises(RuntimeError, match="no per_task_atar.meta.json"):
            self._runner(tmp_path, {"checkpoint_sha256": "aaa"})._check_or_write_fingerprint("atar", per_task)


# ─── --max-tasks subset ─────────────────────────────────────────────────────

def test_stratified_subset_covers_every_tier() -> None:
    """tasks_test.jsonl is sorted by tier; a smoke subset must not be tier 1 only."""
    from atar.evaluation.runner import stratified_subset
    tasks = [_task(f"t{tier}_{i}", tier) for tier in range(1, 6) for i in range(4)]
    picked = stratified_subset(tasks, 7)
    assert [t.tier for t in picked] == [1, 2, 3, 4, 5, 1, 2]
    assert [t.id for t in picked[:2]] == ["t1_0", "t2_0"]          # file order within a tier
    assert len(stratified_subset(tasks, 100)) == 20                 # never more than exist


def test_real_test_set_smoke_subset_is_stratified() -> None:
    from atar.evaluation.runner import stratified_subset
    data_dir = Path(__file__).parent.parent.parent / "data"
    if not (data_dir / "tasks_test.jsonl").exists():
        pytest.skip("Test data not generated yet.")
    picked = stratified_subset(load_tasks_from_jsonl(data_dir / "tasks_test.jsonl"), 20)
    assert sorted({t.tier for t in picked}) == [1, 2, 3, 4, 5]
