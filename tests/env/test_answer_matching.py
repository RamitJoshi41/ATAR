"""
Tests for atar.env.answer_matching.is_answer_correct — the single grader used
by ATAREnv's accuracy reward and by atar.evaluation.

Several cases are regressions found in the 2026-10-13 review, where the old
bidirectional-substring grader was wrong in both directions.
"""

from __future__ import annotations

import pytest

from atar.env.answer_matching import is_answer_correct, is_answer_correct_regraded, is_numeric_truth

RL_TRUTH = (
    "Reinforcement learning is an area of machine learning concerned with how "
    "intelligent agents ought to take actions in an environment."
)
PARIS_TRUTH = "The capital of France is Paris, located in Europe."
LINUX_TRUTH = (
    "Linux is a family of open-source Unix-like operating systems based on the "
    "Linux kernel."
)


# ─── numeric ground truth ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "answer, truth",
    [
        ("61559", "61559.0"),
        ("2660 + 58899 = 61559", "61559.0"),       # was a false negative
        ("The answer is 95,000", "95000"),         # was a false negative
        ("$95,000.00", "95000"),
        ("206.15", "206.14999999999998"),          # float artefact in data
        ("There are 2 projects.", "2"),
        ("-3", "-3"),
    ],
)
def test_numeric_correct(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is True


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("6", "61559.0"),                          # was a false positive (substring)
        ("There are 12 projects", "2"),            # was a false positive
        ("5", "50"),                               # was a false positive
        ("81559", "61559.0"),
        ("206.1", "206.14999999999998"),           # outside 0.005 tolerance
        ("Not specified in the given information.", "95000"),
        ("2 + 2 = 5", "4"),                        # last number is the answer
    ],
)
def test_numeric_incorrect(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is False


# ─── short text ground truth ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "answer, truth",
    [
        ("Berlin", "Berlin"),
        ("The capital is Berlin.", "Berlin"),
        ("  berlin  ", "Berlin"),
        ("Washington, D.C.", "Washington D.C."),
        ("Washington DC", "Washington D.C."),
        ("It is in the Engineering department", "Engineering"),
    ],
)
def test_short_text_correct(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is True


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("I do not know", "no"),                   # was a false positive
        ("no", "Norway"),                          # was a false positive
        ("London", "Berlin"),
        ("Washington", "Washington D.C."),         # partial multi-word truth
        ("Sales", "Engineering"),
    ],
)
def test_short_text_incorrect(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is False


# ─── long reference-sentence ground truth (search abstracts) ─────────────

@pytest.mark.parametrize(
    "answer, truth",
    [
        ("Paris", PARIS_TRUTH),
        ("The capital of France is Paris.", PARIS_TRUTH),
        ("an area of machine learning", RL_TRUTH),
        (RL_TRUTH, RL_TRUTH),
        ("Linux is a family of Unix-like operating systems", LINUX_TRUTH),
    ],
)
def test_long_text_correct(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is True


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("Linux", RL_TRUTH),
        ("space exploration", RL_TRUTH),
        ("The capital of France is Lyon.", PARIS_TRUTH),
        ("Berlin", PARIS_TRUTH),
    ],
)
def test_long_text_incorrect(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is False


# ─── fail-closed edge cases ──────────────────────────────────────────────

@pytest.mark.parametrize("answer", [None, "", "   ", "the of a"])
def test_empty_or_contentless_answers_are_incorrect(answer: str | None) -> None:
    assert is_answer_correct(answer, "Berlin") is False
    assert is_answer_correct(answer, "42") is False
    assert is_answer_correct(answer, PARIS_TRUTH) is False


def test_is_numeric_truth() -> None:
    assert is_numeric_truth("61559.0")
    assert is_numeric_truth("95,000")
    assert is_numeric_truth("-3")
    assert not is_numeric_truth("Washington D.C.")
    assert not is_numeric_truth("Route 66")


# ─── re-grading rules (ADR-27, W6) — reporting only ─────────────────────

@pytest.mark.parametrize(
    "answer, truth",
    [
        ("Brasília", "Brasilia"),
        ("The capital city of Brazil is Brasília.", "Brasilia"),
        ("Bogotá", "Bogota"),
    ],
)
def test_regraded_ignores_accents(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is False  # the training grader stays strict
    assert is_answer_correct_regraded(answer, truth) is True


@pytest.mark.parametrize(
    "answer, truth",
    [("78.3°C", "78"), ("356.73°C", "356"), ("357.0", "356"), ("64.7°C", "65"), ("118.1 °C", "118")],
)
def test_regraded_approx_numeric_within_one(answer: str, truth: str) -> None:
    assert is_answer_correct(answer, truth) is False
    assert is_answer_correct_regraded(answer, truth) is False  # tolerance only when asked for
    assert is_answer_correct_regraded(answer, truth, approx_numeric=True) is True


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("-64.7°C", "65"),                                  # wrong sign
        ("approximately 337.4°C (639.32°F)", "337"),        # last number is Fahrenheit
        ("358.5", "356"),                                   # more than 1 away
    ],
)
def test_regraded_approx_numeric_still_rejects(answer: str, truth: str) -> None:
    assert is_answer_correct_regraded(answer, truth, approx_numeric=True) is False


def test_regraded_exact_numeric_has_no_extra_tolerance() -> None:
    # "hours in 1589 days" is exact arithmetic: off by one stays wrong
    assert is_answer_correct_regraded("38137", "38136") is False
    assert is_answer_correct_regraded("38136.0", "38136") is True


def test_regraded_none_and_empty() -> None:
    assert is_answer_correct_regraded(None, "Paris") is False
    assert is_answer_correct_regraded("", "Paris") is False
