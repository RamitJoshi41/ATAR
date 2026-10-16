"""DemoBundle loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from atar.demo.bundle import DemoBundle
from tests.demo.helpers import make_bundle


def test_bundle_views(tmp_path: Path) -> None:
    b = DemoBundle(make_bundle(tmp_path))
    assert b.is_sample
    assert b.atar_versions() == [("step_0", 0), ("step_100", 100)]
    assert b.baseline_versions() == ["react"]
    assert [t.task_id for t in b.tasks()] == ["calc-1"]
    ep = b.episode("step_100", "calc-1")
    assert ep is not None and ep.correct is True
    assert b.episode("step_999", "calc-1") is None
    assert [e["step"] for e in b.curve["atar"]] == [0, 100]


def test_not_a_bundle(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        DemoBundle(tmp_path)
