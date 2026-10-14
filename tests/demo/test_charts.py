"""
Chart specs (Altair → Vega-Lite dicts).  These pin the fixes found by looking
at the rendered app on 2026-10-14: fixed 0–100% axis for rates, baselines
identified by dash pattern + legend (no colliding in-chart labels), every
action label shown on the probability chart, one bar row per version.
"""

from __future__ import annotations

import json
from pathlib import Path

from atar.demo.charts import ACTIONS, action_mix_chart, action_probs_chart, learning_curve_chart
from tests.demo.helpers import make_bundle


def _curve(tmp_path: Path) -> dict:
    return json.loads((make_bundle(tmp_path) / "curve.json").read_text())


def _layers(spec: dict) -> list[dict]:
    return spec["layer"]


def test_rate_metrics_use_a_fixed_zero_to_one_axis(tmp_path: Path) -> None:
    spec = learning_curve_chart(_curve(tmp_path), "accuracy", "all").to_dict()
    band = _layers(spec)[0]
    assert band["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert band["encoding"]["y"]["axis"]["format"] == "%"


def test_tool_calls_axis_is_not_forced_to_fractions(tmp_path: Path) -> None:
    spec = learning_curve_chart(_curve(tmp_path), "avg_tool_calls", "all").to_dict()
    assert "domain" not in _layers(spec)[0]["encoding"]["y"].get("scale", {})


def test_baselines_are_dashed_rules_with_a_legend_not_text(tmp_path: Path) -> None:
    layers = _layers(learning_curve_chart(_curve(tmp_path), "accuracy", "all").to_dict())

    def mark_type(layer: dict) -> str:
        return layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]

    assert "text" not in [mark_type(layer) for layer in layers]
    rule = next(layer for layer in layers if mark_type(layer) == "rule")
    assert rule["encoding"]["strokeDash"]["legend"]["title"] == "Baselines"


def test_no_curve_means_no_chart(tmp_path: Path) -> None:
    assert learning_curve_chart({"atar": [], "baselines": {}}, "accuracy", "all") is None


def test_probability_chart_never_skips_action_labels() -> None:
    spec = action_probs_chart([1 / 7] * 7, "CALCULATOR").to_dict()
    assert spec["encoding"]["y"]["axis"]["labelOverlap"] is False
    assert spec["encoding"]["y"]["sort"] == ACTIONS


def test_action_mix_one_row_per_version_in_step_order(tmp_path: Path) -> None:
    spec = action_mix_chart(_curve(tmp_path), "all").to_dict()
    assert spec["encoding"]["y"]["sort"] == ["ATAR @ step 0", "ATAR @ step 100"]
    assert spec["encoding"]["x"]["stack"] == "normalize"
    assert spec["height"] == {"step": 30}
    assert spec["encoding"]["color"]["scale"]["domain"] == ACTIONS
