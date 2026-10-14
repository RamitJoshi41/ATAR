"""
ATAR demo (M8) — Streamlit app.

    streamlit run atar/demo/app.py

Tabs:
  Training progress — held-out learning curve per checkpoint vs. baselines,
                      action mix per version, cost per version.
  Replay            — recorded episodes: ReAct (or any version) vs. an ATAR
                      version on the same task, step-through, action
                      probabilities; plus the same task across all versions.
  Live              — run ATAR and ReAct on a typed query (needs a GPU host
                      that can load the training model), optional live
                      Wikipedia search.

Data comes from a bundle folder (scripts/build_demo_bundle.py): the
ATAR_DEMO_DATA env var, else ./demo_data, else ./demo_data_sample.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from atar.demo.bundle import DemoBundle
from atar.demo.charts import (
    action_mix_chart,
    action_probs_chart,
    learning_curve_chart,
    version_label,
)
from atar.demo.curve import METRIC_LABELS
from atar.demo.metrics import session_metrics
from atar.demo.replay import EpisodeRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Data loading ────────────────────────────────────────────────────────

def default_bundle_path() -> str:
    env = os.environ.get("ATAR_DEMO_DATA")
    if env:
        return env
    for name in ("demo_data", "demo_data_sample"):
        if (REPO_ROOT / name / "manifest.json").exists():
            return str(REPO_ROOT / name)
    return str(REPO_ROOT / "demo_data")


@st.cache_resource(show_spinner=False)
def load_bundle(path: str, mtime: float) -> DemoBundle:  # mtime busts the cache on rebuild
    return DemoBundle(path)


# ─── Episode rendering ───────────────────────────────────────────────────

def render_step(step: Any, is_last: bool) -> None:
    with st.container(border=True):
        st.markdown(f"**Turn {step.turn} · {step.action}**")
        if step.thought:
            st.markdown(f"_Thought:_ {step.thought}")
        if step.action_probs:
            st.altair_chart(action_probs_chart(step.action_probs, step.action), width="stretch")
            if step.value_estimate is not None:
                st.caption(f"Critic's estimate of future reward here: {step.value_estimate:+.2f}")
        if step.param_error:
            st.error(f"Parameters rejected by the validator: {step.param_error}")
        elif step.params and step.action not in ("ANSWER_DIRECTLY", "TERMINATE"):
            st.code(json.dumps(step.params, indent=2), language="json")
        if step.tool_success is True:
            st.success(step.tool_output or "(empty result)")
        elif step.tool_success is False:
            st.warning(f"Tool failed: {step.tool_error}")
        if is_last and step.final_answer:
            st.markdown(f"**Final answer:** {step.final_answer}")
        parts = [f"{k} {v:+.1f}" for k, v in step.reward.items() if k != "total" and abs(v) > 1e-9]
        st.caption(
            f"Reward {step.reward['total']:+.1f}" + (f" ({', '.join(parts)})" if parts else "")
            + f" · cumulative {step.cumulative_reward:+.1f}"
        )


def outcome_label(rec: EpisodeRecord) -> str:
    if rec.truncated:
        return "No answer (ran out of turns)"
    if rec.correct is None:
        return "Not graded"
    return "Correct" if rec.correct else "Wrong"


def render_outcome(rec: EpisodeRecord) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Outcome", outcome_label(rec))
    c2.metric("Total reward", f"{rec.total_reward:+.1f}")
    c3.metric("Tool calls", rec.tool_calls)
    st.caption(f"Answer: {rec.final_answer or '—'}"
               + (f" · Ground truth: {rec.ground_truth}" if rec.ground_truth else " · no ground truth to grade against"))


def render_episode(rec: EpisodeRecord, upto: int | None = None, finished: bool = True) -> None:
    st.subheader(version_label(rec.version))
    if rec.live_tools:
        st.caption("Live tools: SEARCH_WEB used real Wikipedia, not the simulated search the policy trained on.")
    shown = rec.steps if upto is None else rec.steps[:upto]
    for i, step in enumerate(shown):
        render_step(step, is_last=finished and i == len(rec.steps) - 1)
    if finished and (upto is None or upto >= len(rec.steps)):
        render_outcome(rec)


def render_session_metrics(groups: dict[str, list[EpisodeRecord]]) -> None:
    st.divider()
    st.markdown("**This session** (episodes viewed or run so far)")
    cols = st.columns(max(1, len(groups)))
    for col, (label, records) in zip(cols, groups.items()):
        m = session_metrics(records)
        with col:
            st.markdown(f"_{label}_")
            a, b, c, d = st.columns(4)
            a.metric("Accuracy", "—" if m.accuracy is None else f"{m.accuracy:.0%}",
                     help=f"{m.graded} graded of {m.episodes} episodes")
            b.metric("Avg tool calls", f"{m.avg_tool_calls:.1f}")
            c.metric("Avg latency", f"{m.mean_latency_s:.1f}s")
            d.metric("Tokens", f"{m.total_tokens:,}")


def remember(rec: EpisodeRecord) -> None:
    st.session_state.setdefault("seen", {})[(rec.version, rec.task_id)] = rec


def seen_for(version: str) -> list[EpisodeRecord]:
    return [r for (v, _), r in st.session_state.get("seen", {}).items() if v == version]


# ─── Tabs ────────────────────────────────────────────────────────────────

def tab_progress(bundle: DemoBundle) -> None:
    curve = bundle.curve
    if not curve.get("atar"):
        st.info("No held-out evaluation results in this bundle yet. Evaluate checkpoints with "
                "scripts/evaluate.py, then rebuild it with scripts/build_demo_bundle.py.")
        return
    tiers = ["all"] + sorted({t for e in curve["atar"] for t in e.get("per_tier", {})})
    c1, c2 = st.columns(2)
    metric = c1.selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
    tier = c2.selectbox("Tasks", tiers, format_func=lambda t: "All tiers" if t == "all" else t.replace("_", " ").title())

    st.markdown(f"#### {METRIC_LABELS[metric]} on held-out tasks, by training step")
    chart = learning_curve_chart(curve, metric, tier)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.caption("Each point is one saved checkpoint evaluated on tasks it never trained on; the band is "
               "a 95% bootstrap confidence interval. Dashed lines are the baselines.")

    st.markdown("#### What each version chooses to do")
    mix = action_mix_chart(curve, tier)
    if mix is not None:
        st.altair_chart(mix, width="stretch")
    with st.expander("Table view"):
        table = []
        for e in curve["atar"]:
            o = e["overall"]
            table.append({"Version": version_label(e["version"]), "Success": o["accuracy"],
                          "Avg tool calls": o["avg_tool_calls"], "Efficiency": o["efficiency"],
                          "Mean latency (s)": o["mean_latency"], "Tasks": o["n_tasks"]})
        for name, e in sorted(curve.get("baselines", {}).items()):
            o = e["overall"]
            table.append({"Version": version_label(name), "Success": o["accuracy"],
                          "Avg tool calls": o["avg_tool_calls"], "Efficiency": o["efficiency"],
                          "Mean latency (s)": o["mean_latency"], "Tasks": o["n_tasks"]})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


def tab_replay(bundle: DemoBundle) -> None:
    tasks = bundle.tasks()
    atar_versions = bundle.atar_versions()
    if not tasks or not atar_versions:
        st.info("No recorded episodes in this bundle. Record them with scripts/record_replays.py.")
        return
    task = st.selectbox("Task (held-out, never seen in training)", tasks,
                        format_func=lambda t: f"Tier {t.tier} · {t.query[:90]}")
    versions = [v for v, _ in atar_versions]
    others = bundle.baseline_versions() + versions
    c1, c2, c3 = st.columns([2, 2, 1])
    left = c1.selectbox("Compare", others, index=others.index("react") if "react" in others else 0,
                        format_func=version_label)
    right = c2.selectbox("…with ATAR version", versions, index=len(versions) - 1, format_func=version_label)
    step_mode = c3.toggle("Step through", value=False)

    pair = (task.task_id, left, right)
    if st.session_state.get("replay_pair") != pair:
        st.session_state["replay_pair"] = pair
        st.session_state["replay_step"] = 1
    recs = [bundle.episode(left, task.task_id), bundle.episode(right, task.task_id)]
    longest = max((len(r.steps) for r in recs if r), default=0)

    upto: int | None = None
    if step_mode:
        b1, b2, b3, _ = st.columns([1, 1, 1, 5])
        if b1.button("◀ Prev", disabled=st.session_state["replay_step"] <= 1):
            st.session_state["replay_step"] -= 1
        if b2.button("Next ▶", disabled=st.session_state["replay_step"] >= longest):
            st.session_state["replay_step"] += 1
        if b3.button("Restart"):
            st.session_state["replay_step"] = 1
        upto = st.session_state["replay_step"]
        st.caption(f"Showing turn {upto} of {longest}")

    cols = st.columns(2)
    for col, rec, name in zip(cols, recs, (left, right)):
        with col:
            if rec is None:
                st.warning(f"No recording of {version_label(name)} on this task.")
                continue
            render_episode(rec, upto=upto)
            if upto is None or upto >= len(rec.steps):
                remember(rec)

    with st.expander("This task across every version", expanded=False):
        rows = []
        for v in bundle.baseline_versions() + versions:
            rec = bundle.episode(v, task.task_id)
            if rec is None:
                continue
            rows.append({"Version": version_label(v), "Actions": " → ".join(rec.action_trace),
                         "Answer": rec.final_answer or "—",
                         "Correct": "—" if rec.correct is None else ("yes" if rec.correct else "no"),
                         "Reward": round(rec.total_reward, 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Ground truth: {task.ground_truth}")

    render_session_metrics({version_label(left): seen_for(left), version_label(right): seen_for(right)})


def tab_live(bundle: DemoBundle) -> None:
    st.caption("Runs the real policy and ReAct on your question. Needs a GPU machine that can load the model "
               "the checkpoint was trained on (the policy reads that model's hidden states).")
    runner = st.session_state.get("live_runner")
    if runner is None:
        with st.form("load_live"):
            model = st.text_input("Model", value=bundle.manifest.get("model_name", "Qwen/Qwen2.5-7B-Instruct"))
            ckpt = st.text_input("Checkpoint (.pt)", value="")
            device = st.selectbox("Device", ["cuda", "cpu"])
            if st.form_submit_button("Load model and checkpoint"):
                try:
                    from atar.demo.live import LiveRunner
                    with st.spinner("Loading the model (this can take minutes)…"):
                        st.session_state["live_runner"] = LiveRunner(model, ckpt, device, REPO_ROOT / "data")
                    st.rerun()
                except Exception as exc:  # shown in the UI, never a crash
                    st.error(f"Could not load: {exc}")
        return

    from atar.demo.live import QueryError, make_live_task
    st.success(f"Loaded {runner.model_name}, ATAR checkpoint at step {runner.checkpoint_step:,}.")
    examples = {f"Tier {t.tier} · {t.query[:80]}": t for t in bundle.tasks()}
    pick = st.selectbox("Example task (or type your own below)", ["(type my own)"] + list(examples))
    query = st.text_area("Question", value="" if pick == "(type my own)" else examples[pick].query)
    truth = st.text_input("Expected answer (optional, for grading)",
                          value="" if pick == "(type my own)" else examples[pick].ground_truth)
    c1, c2, c3 = st.columns(3)
    live_tools = c1.toggle("Live tools (real Wikipedia search)", value=False)
    max_turns = c2.slider("Max turns", 1, 5, 5)
    step_mode = c3.toggle("Step through", value=False, key="live_step_mode")
    mode_note = ("LIVE TOOLS: SEARCH_WEB queries real Wikipedia — not the simulated search used in training."
                 if live_tools else "Simulated tools: the same sandboxed tools the policy was trained with.")
    (st.warning if live_tools else st.info)(mode_note)

    if st.button("Run", type="primary"):
        try:
            task = make_live_task(query, truth)
        except QueryError as exc:
            st.error(str(exc))
        else:
            st.session_state["live_eps"] = [
                runner.start("react", task, live_tools, max_turns),
                runner.start("atar", task, live_tools, max_turns),
            ]
    episodes = st.session_state.get("live_eps")
    if not episodes:
        return
    try:
        if step_mode:
            if st.button("Next step ▶", disabled=all(e.done for e in episodes)):
                for e in episodes:
                    e.advance()
        else:
            with st.spinner("Running both agents…"):
                for e in episodes:
                    e.run_to_end()
    except Exception as exc:  # a failing LLM/tool call must not crash the app
        st.error(f"The run failed: {exc}")
        return
    cols = st.columns(2)
    for col, e in zip(cols, episodes):
        with col:
            render_episode(e.record, finished=e.done)
            if e.done:
                remember(e.record)
    render_session_metrics({version_label(e.record.version): seen_for(e.record.version) for e in episodes})


# ─── Page ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="ATAR demo", layout="wide")
    st.title("ATAR: an RL policy that learns when to use tools")
    path = st.sidebar.text_input("Demo data folder", value=default_bundle_path())
    try:
        manifest = Path(path) / "manifest.json"
        bundle = load_bundle(path, manifest.stat().st_mtime)
    except (FileNotFoundError, OSError) as exc:
        st.error(f"No demo data at {path}: {exc}. Build it with scripts/build_demo_bundle.py.")
        return

    if bundle.is_sample:
        st.warning("SAMPLE DATA — built with the small stand-in model and toy checkpoints to show how the "
                   "demo works. These numbers are not ATAR's results. " + bundle.manifest.get("note", ""))
    st.sidebar.markdown(f"**Model:** {bundle.manifest.get('model_name', '?')}")
    st.sidebar.markdown(f"**Checkpoints on the curve:** {len(bundle.curve.get('atar', []))}")
    st.sidebar.markdown(f"**Recorded episodes:** {len(bundle.episodes)}")
    st.sidebar.caption(f"Bundle built {bundle.manifest.get('created_at', '?')}")

    progress, replay, live = st.tabs(["Training progress", "Replay", "Live"])
    with progress:
        tab_progress(bundle)
    with replay:
        tab_replay(bundle)
    with live:
        tab_live(bundle)


main()
