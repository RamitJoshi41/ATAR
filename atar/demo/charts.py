"""
M8 — Altair chart builders (pure: data in, chart spec out; no Streamlit).

Colour follows the dataviz reference palette: ATAR's line and the chosen
action use one accent; baselines and unchosen actions are neutral grey; the
7 actions keep fixed categorical colours (by action id) in every chart.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from atar.demo.curve import METRIC_LABELS, action_mix_rows, baseline_rows, curve_rows
from atar.shared.shared_types import ActionType

ACTIONS = [a.name for a in ActionType]
ACTION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
ACCENT = "#2a78d6"
MUTED = "#8a8880"
BASELINE_LABELS = {
    "react": "ReAct (prompted LLM)",
    "zeroshot": "Zero-shot (answer directly)",
    "random": "Random actions",
}
BASELINE_DASHES = {"react": [6, 3], "zeroshot": [2, 3], "random": [10, 4, 2, 4]}
FRACTION_METRICS = {"accuracy", "efficiency"}  # bounded in [0, 1]


def version_label(version: str) -> str:
    if version.startswith("step_"):
        return f"ATAR @ step {int(version[5:]):,}"
    return BASELINE_LABELS.get(version, version)


def learning_curve_chart(curve: dict[str, Any], metric: str, tier: str) -> alt.LayerChart | None:
    rows = curve_rows(curve, metric, tier)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    label = METRIC_LABELS.get(metric, metric)
    if metric in FRACTION_METRICS:
        # Fixed 0–100%: a narrow auto-range would make small changes look large.
        y_scale = alt.Scale(domain=[0, 1])
        y_axis = alt.Axis(format="%", tickCount=5)
    else:
        y_scale = alt.Scale(zero=True)
        y_axis = alt.Axis(tickCount=5)
    x = alt.X("step:Q", title="Training steps", axis=alt.Axis(format=",d", tickCount=8, grid=False))

    band = alt.Chart(df).mark_area(opacity=0.15, color=ACCENT).encode(
        x=x, y=alt.Y("ci_low:Q", title=label, scale=y_scale, axis=y_axis), y2="ci_high:Q",
    )
    line = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=2).encode(x=x, y="value:Q")
    points = alt.Chart(df).mark_circle(color=ACCENT, size=90, opacity=1).encode(
        x=x, y="value:Q",
        tooltip=[
            alt.Tooltip("version:N", title="Checkpoint"),
            alt.Tooltip("step:Q", title="Step", format=",d"),
            alt.Tooltip("value:Q", title=label, format=".3f"),
            alt.Tooltip("ci_low:Q", title="95% CI low", format=".3f"),
            alt.Tooltip("ci_high:Q", title="95% CI high", format=".3f"),
            alt.Tooltip("n_tasks:Q", title="Held-out tasks"),
        ],
    )
    layers: list[alt.Chart] = [band, line, points]
    base = baseline_rows(curve, metric, tier)
    if base:
        bdf = pd.DataFrame(base)
        bdf["label"] = bdf["baseline"].map(lambda b: BASELINE_LABELS.get(b, b))
        names = list(bdf["label"])
        dashes = [BASELINE_DASHES.get(b, [4, 4]) for b in bdf["baseline"]]
        # Identity by dash pattern + legend (no in-chart text that collides
        # when baselines share a value); colour stays neutral.
        layers.append(alt.Chart(bdf).mark_rule(color=MUTED, strokeWidth=1.5).encode(
            y="value:Q",
            strokeDash=alt.StrokeDash("label:N", scale=alt.Scale(domain=names, range=dashes),
                                      legend=alt.Legend(title="Baselines", orient="bottom")),
            tooltip=[alt.Tooltip("label:N", title="Baseline"), alt.Tooltip("value:Q", title=label, format=".3f")],
        ))
    return alt.layer(*layers).properties(height=340)


def action_mix_chart(curve: dict[str, Any], tier: str) -> alt.Chart | None:
    rows = action_mix_rows(curve, tier)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["label"] = df["version"].map(version_label)
    df["action_order"] = df["action"].map(ACTIONS.index)
    order = list(dict.fromkeys(df.sort_values("step", na_position="last")["label"]))
    return alt.Chart(df).mark_bar(stroke="white", strokeWidth=1).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("sum(fraction):Q", stack="normalize", title="Share of actions", axis=alt.Axis(format="%")),
        color=alt.Color("action:N", scale=alt.Scale(domain=ACTIONS, range=ACTION_COLORS),
                        legend=alt.Legend(title="Action", orient="bottom", columns=4)),
        order=alt.Order("action_order:Q"),
        tooltip=[alt.Tooltip("label:N", title="Version"), alt.Tooltip("action:N", title="Action"),
                 alt.Tooltip("sum(fraction):Q", title="Share", format=".1%")],
    ).properties(height=alt.Step(30))


def action_probs_chart(probs: list[float], chosen: str) -> alt.Chart:
    df = pd.DataFrame({"action": ACTIONS, "probability": probs})
    df["chosen"] = df["action"] == chosen
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format="%")),
        # All 7 labels, always: a skipped label would hide which action a bar is.
        y=alt.Y("action:N", sort=ACTIONS, title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.condition("datum.chosen", alt.value(ACCENT), alt.value(MUTED)),
        tooltip=[alt.Tooltip("action:N"), alt.Tooltip("probability:Q", format=".1%")],
    ).properties(height=alt.Step(20))

