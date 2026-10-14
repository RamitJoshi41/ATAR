"""
Streamlit app smoke tests via streamlit.testing.AppTest (headless, no browser,
no LLM).  They cover what can be tested without a GPU: the app renders every
tab from a bundle without exceptions, step-through pauses between actions,
and a missing bundle produces an error message instead of a crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.demo.helpers import make_bundle

APP = str(Path(__file__).resolve().parents[2] / "atar" / "demo" / "app.py")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("ATAR_DEMO_DATA", str(make_bundle(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning]
    parts += [s.value for s in at.subheader] + [e.value for e in at.error] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def test_renders_all_tabs_with_sample_banner(app: AppTest) -> None:
    text = _text(app)
    assert "SAMPLE DATA" in text
    assert "held-out tasks, by training step" in text            # progress tab
    assert "ATAR @ step 100" in text and "ReAct (prompted LLM)" in text  # replay tab defaults
    assert "Turn 2 · ANSWER_DIRECTLY" in text


def test_step_through_pauses_between_actions(app: AppTest) -> None:
    app.toggle[0].set_value(True).run()                          # replay "Step through"
    assert not app.exception
    assert "Turn 1 · CALCULATOR" in _text(app)
    assert "Turn 2 · ANSWER_DIRECTLY" not in _text(app)
    next(b for b in app.button if "Next" in b.label).click().run()
    assert "Turn 2 · ANSWER_DIRECTLY" in _text(app)


def test_selecting_untrained_version_shows_its_rejected_call(app: AppTest) -> None:
    version_box = next(s for s in app.selectbox if s.label.startswith("…with ATAR version"))
    version_box.set_value("step_0").run()
    assert not app.exception
    assert "Parameters rejected by the validator: bad sql" in _text(app)


def test_missing_bundle_is_an_error_message_not_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATAR_DEMO_DATA", str(tmp_path / "nope"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert "No demo data" in _text(at)


def test_live_tab_without_model_shows_load_form(app: AppTest) -> None:
    assert any(b.label == "Load model and checkpoint" for b in app.button)
