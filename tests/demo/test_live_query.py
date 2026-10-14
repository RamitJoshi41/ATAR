"""Free-text query validation for live mode (a malformed query must not crash the app)."""

from __future__ import annotations

import pytest

from atar.demo.live import MAX_QUERY_CHARS, QueryError, clean_query, make_live_task


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 42, ["list"], "x" * (MAX_QUERY_CHARS + 1), "\x00\x07\x1b"])
def test_rejected(bad: object) -> None:
    with pytest.raises(QueryError):
        clean_query(bad)


def test_control_characters_stripped_and_text_kept() -> None:
    assert clean_query("  What is\x00 2 + 2?\x07 ") == "What is 2 + 2?"
    assert clean_query("Line one\nline two") == "Line one\nline two"


def test_make_live_task() -> None:
    t = make_live_task("  Capital of France? ", " Paris ")
    assert t.query == "Capital of France?" and t.ground_truth == "Paris"
    assert t.tier == 0 and t.id.startswith("live-")
    assert make_live_task("q").id != t.id
