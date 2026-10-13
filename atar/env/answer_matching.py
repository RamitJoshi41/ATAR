"""
Answer matching — the single source of truth for "is this final answer correct?"

Used by ATAREnv (accuracy reward) and by atar.evaluation (accuracy metric) so
that the number the policy is trained on and the number it is evaluated on
can never drift apart.

Rules
-----
Numeric ground truth (e.g. "61559.0", "95000", "206.14999999999998"):
    Extract every number from the answer (commas allowed as thousands
    separators) and compare the LAST one to the ground truth with a small
    tolerance (rel 1e-6, abs 0.005 so answers rounded to cents still match).
    "2660 + 58899 = 61559" -> 61559 -> correct.
    "There are 12 projects" vs "2" -> 12 -> incorrect.

Short text ground truth (<= 3 content words, e.g. "Berlin", "Washington D.C."):
    Every content word of the ground truth must appear as a whole word in
    the answer.  "The capital is Berlin." -> correct; "no" never matches
    "Norway", and "I do not know" never matches "no".

Long text ground truth (a reference sentence, e.g. a search abstract):
    Correct if the answer covers >= 60% of the truth's content words, OR the
    answer is a non-trivial fragment of the truth: it has at least one
    content word and every content word in it appears in the truth.
    "Paris" vs "The capital of France is Paris, located in Europe." -> correct.
    "Linux" vs a sentence about reinforcement learning -> incorrect.

Empty / None answers are always incorrect.
"""

from __future__ import annotations

import math
import re

_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_FULL_NUMBER_RE = re.compile(r"^\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
        "on", "at", "to", "for", "with", "by", "and", "or", "it", "its", "this",
        "that", "as", "from", "how", "what", "which", "who", "answer",
    }
)

_SHORT_TRUTH_MAX_WORDS = 3
_LONG_TRUTH_RECALL = 0.6


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def _content_words(text: str) -> list[str]:
    """Lower-cased word tokens minus stopwords, with a naive plural strip."""
    words: list[str] = []
    for w in _WORD_RE.findall(text.lower()):
        w = w.replace(".", "")  # "d.c." == "dc"
        if not w or w in _STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return words


def is_numeric_truth(ground_truth: str) -> bool:
    """True if the ground truth is a single number (commas allowed)."""
    return bool(_FULL_NUMBER_RE.match(ground_truth))


def is_answer_correct(answer: str | None, ground_truth: str) -> bool:
    """
    Decide whether ``answer`` matches ``ground_truth``.  See module docstring
    for the exact rules.  Fails closed: anything unparseable is incorrect.
    """
    if answer is None:
        return False
    answer = answer.strip()
    truth = ground_truth.strip()
    if not answer or not truth:
        return False

    if is_numeric_truth(truth):
        numbers = _NUMBER_RE.findall(answer)
        if not numbers:
            return False
        try:
            a_val = _parse_number(numbers[-1])
            t_val = _parse_number(truth)
        except ValueError:
            return False
        return math.isclose(a_val, t_val, rel_tol=1e-6, abs_tol=0.005)

    truth_words = _content_words(truth)
    answer_words = _content_words(answer)
    if not truth_words or not answer_words:
        return False

    answer_set = set(answer_words)
    truth_set = set(truth_words)

    if len(truth_set) <= _SHORT_TRUTH_MAX_WORDS:
        return truth_set <= answer_set

    recall = len(truth_set & answer_set) / len(truth_set)
    if recall >= _LONG_TRUTH_RECALL:
        return True
    return answer_set <= truth_set
