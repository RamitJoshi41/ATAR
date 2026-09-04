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

## Results
<!-- Filled in only after the real evaluation run (Day 22-24). Table of
accuracy / avg tool calls / recovery rate for ATAR vs Random vs ReAct vs
Zero-Shot baselines, with confidence intervals. -->

## Ablations
<!-- No-curriculum and no-projection-cotraining results, once run. -->

## Known limitations
<!-- Be honest here — this is what gets asked about in interviews. -->
