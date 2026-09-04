# ATAR — Adaptive Tool-use Agent with Reinforcement Learning

*A frozen 7B LLM handles language; a small trainable PPO policy learns when
and how to use external tools.*

## Status
Vertical slice complete. Full module build (M0-M8) not yet started.

## Quickstart
<!-- Fill in once M0 is done: install steps, how to run tests, how to
generate tasks, how to launch training, how to launch the demo. -->

## Architecture
<!-- One paragraph + the module dependency diagram from the project
description, once it's stable. -->

---

## Module log

<!-- One section per module, added as each one completes. Template: -->

### Vertical Slice
**What:** Minimal end-to-end demo: AST-safe calculator tool, 512-d state
vector builder, 3-action Gymnasium env (ANSWER_DIRECTLY / CALCULATOR /
TERMINATE), hardcoded rule-based policy, 10 arithmetic tasks, and a
`demo.py` entry point.

**Why this approach:** Proves the data flow (Task → State → Action →
ToolResult → RewardComponents) works before fanning out to parallel module
builds. The calculator uses `ast.parse` with an operator whitelist (no
`eval()`) because AGENTS.md rule 8 requires fail-closed validation.

**What actually happened when run:** 37/37 pytest tests pass (0.22s), 10/10
tasks solved correctly, average total reward = 1.00.

**Deviations from the MSD/spec, if any:** None — this precedes the module
build. The `step()` return type is `RewardComponents` (dataclass), not a
bare float, matching interfaces.md's M5 signature.

---

### M0 Scaffold
**What:** Created the core project directory structure (`atar/`), implemented global config loader (`base_config.py` using `pydantic-settings`), established centralized logging (`logging_setup.py`), configured a fully pinned `pyproject.toml`, initialized the action registry skeleton, and set up a basic GitHub Actions workflow.

**Why this approach:** Adopted a proper Python package structure by nesting `atar/atar/` to ensure absolute imports like `atar.shared.shared_types` work from anywhere. Kept `slice/` as reference material but updated its imports to use the new `atar.shared` package. Used relaxed version bounds in `pyproject.toml` instead of strict pinning to ensure compatibility with remote training GPUs (Kaggle/rented instances).

**What actually happened when run:** `pip install -e .[dev]` successfully installed the package in editable mode. 37/37 pytest tests pass (0.33s), `demo.py` solves 10/10 tasks correctly, and `get_logger("test")` writes correctly to both console and `logs/atar.log`.

**Deviations from the MSD/spec, if any:** Used relaxed version ranges (e.g., `torch>=2.1.0,<2.12.0`) in `pyproject.toml` instead of strict pins, per the user's explicit request to support future execution on varied GPU instances.

---

### M1 Tool Sandbox
**What:** Implemented `execute_tool` entry point wrapping 5 sandboxed simulated tools (`search_web`, `calculator`, `sql_query`, `python_exec`, `clarify`). Enforced robust security (e.g. failing closed with restricted AST parsing, avoiding `eval()`, preventing dangerous imports/calls in `python_exec`, checking regex whitelist for `sql_query`).

**Why this approach:** Ensuring tools run deterministically and safely is a critical foundational requirement. AST validation protects `calculator` and `python_exec`, with subprocess isolation for Python exec guaranteeing timeout boundaries. Search uses `rank-bm25` on a small in-memory Wikipedia corpus, with 20% random noise injection to simulate imperfect tool reliability for the RL agent.

**What actually happened when run:** `pytest atar/tests/tools/ -v` passed perfectly (16/16). Noise injection correctly returns non-top documents ~20% of the time, and all security rejection tests trigger successfully for restricted inputs.

**Deviations from the MSD/spec, if any:** None, explicitly handled `__import__` and similar dynamic evaluation calls using `ast.Call` validation in `python_exec` as requested by the architect.

---

### M2 Task Generator
**What:** Implemented `generate_tasks` and `generate_dataset` to produce 1000 synthetic tasks across 5 difficulty tiers (20% T1, 15% T2, 20% T3, 15% T4, 10% T5). Used Jinja2 templates and programmatic ground truth computation by hooking into M1's `execute_tool` (with a new `deterministic=True` flag to bypass web search noise). Outputs are serialized to JSONL files in `data/`.

**Why this approach:** Using M1 to compute the ground truth during generation guarantees absolute correctness, even as randomized inputs are fed into the Jinja2 templates. To prevent test-set leakage, generated a 6-character random ticket ID for every query to ensure that exact string matches never collide.

**What actually happened when run:** `pytest atar/tests/tasks/ -v` passes (3/3). Tested against overlap and distribution invariants. The updated `execute_tool` in M1 runs deterministically.

**Deviations from the MSD/spec, if any:** Adjusted `interfaces.md` and M1's `execute_tool` to include `deterministic: bool = False`, ensuring the Task Generator gets reliable answers while M1 preserves its original stochastic behavior for the RL agent. Tier 2 Distractors redesigned to ask direct general knowledge questions matching Tier 1 formatting.

## Results
<!-- Filled in only after the real evaluation run (Day 22-24). Table of
accuracy / avg tool calls / recovery rate for ATAR vs Random vs ReAct vs
Zero-Shot baselines, with confidence intervals. -->

## Ablations
<!-- No-curriculum and no-projection-cotraining results, once run. -->

## Known limitations
<!-- Be honest here — this is what gets asked about in interviews. -->
