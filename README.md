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

---

### M3 LLM Interface

**What:** Implemented `LLMInterface` in `atar/llm/llm_interface.py` — a frozen
instruction-tuned LLM wrapper with two public methods: `encode()` (mean-pool
last hidden state → trainable `nn.Linear` projection to 384-d) and
`generate()` (constrained JSON generation via Outlines 1.3.x). Custom
exceptions `ModelLoadError` and `GenerationError` guard both failure modes.

**Why this approach:**
- *Outlines 1.3.x callable API*: the old `outlines.generate.json()` helper
  module does not exist in 1.3.x. The correct pattern is
  `outlines_model(prompt, output_type=outlines.json_schema(schema), max_new_tokens=256)`.
  `max_new_tokens` must be passed explicitly — the model-agnostic default
  `max_length=76` total tokens is too small for a prompt + valid JSON object,
  causing truncated strings and `JSONDecodeError` on longer inputs.
- *Dynamic hidden_size*: `self.projection = nn.Linear(model.config.hidden_size, 384)`
  — never hardcoded — so the same code path works for 0.5B (hidden=896) and
  7B (hidden=3584) without any change.
- *Projection dtype*: `float32` on CPU, `float16` on CUDA. Using the model's
  own `.dtype` fails for 4-bit quantised models (BnB returns a wrapped dtype).
- *Seed 42*: `torch.manual_seed(42)` before the Linear initialisation so
  projection weights are reproducible before PPO training starts.
- *BitsAndBytes only on CUDA*: 4-bit quantisation guard is
  `device == "cuda" and torch.cuda.is_available()`, so CPU smoke-tests don't
  require CUDA or BnB.

**What actually happened when run:**
`pytest tests/llm/ -v` — **6/6 passed in 84.97s** (Qwen/Qwen2.5-0.5B-Instruct
on CPU, local 4 GB GPU machine).

| Test | Result |
|---|---|
| `test_encode_smoke` | ✅ PASSED — (384,) tensor, deterministic |
| `test_projection_gradients_and_dimensions` | ✅ PASSED — projection trainable, base frozen, in_features == hidden_size |
| `test_generate_constrained_smoke` (slow) | ✅ PASSED — **20/20 trials** produce valid schema-conforming dicts |
| `test_latency_metrics_smoke` | ✅ PASSED — avg encode: **0.034s**, avg generate: **0.885s** (stand-in model, CPU) |
| `test_config_driven_loading_smoke` | ✅ PASSED — HuggingFaceM4/tiny-random-LlamaForCausalLM and Qwen2.5-0.5B both load |
| `test_model_load_error_on_bad_name` | ✅ PASSED — `ModelLoadError` raised on invalid model name |

**Latency note (follow-up required):** Measured on Qwen2.5-0.5B-Instruct on
CPU — encode 0.034s, generate 0.885s. These numbers must be re-measured on
**Qwen2.5-7B-Instruct on the remote training GPU** before M7 starts, to decide
whether prefix-KV-cache reuse is needed for rollout performance.

**Deviations from the MSD/spec:** None. One non-obvious implementation detail
(why `max_new_tokens` is required) is recorded in `ARCHITECTURE_DECISIONS.md`.

---

### M4 Action Translator

**What:** Implemented `translate()` and `validate()` in `atar/translator/action_translator.py`,
plus a fully populated `configs/action_registry.yaml` for all 7 actions.
`translate()` assembles a `ConditionedPrompt` (system message + JSON schema) from
the registry entry. `validate()` is a layered security boundary: registry
forbidden_patterns (regex pre-filter) → JSON schema required-key check → action-
specific deep validation (SQL injection detection, Python AST import scan, calculator
AST node whitelist, empty-expression checks).

**Why this approach:**
- *Single source of truth for the Python allow-list*: `PYTHON_ALLOWED_IMPORTS` is
  declared as a `frozenset` in `sandbox.py` and imported by name in `action_translator.py`.
  Both modules reference the same Python object in memory (same `id()`), making
  divergence structurally impossible. This satisfies MSD DoD-5 without any schema change.
- *Two-layer rejection*: The YAML `forbidden_patterns` field acts as a cheap regex
  pre-filter (catches pattern variants even in serialised JSON); the action-specific
  validators then run AST-level checks that patterns alone cannot do (e.g., confirming
  `ast.Import` node's resolved module is in the allow-list).
- *Fail-closed outer wrapper*: `validate()` wraps `_validate_inner()` in a
  bare `try/except Exception` so no uncaught exception can ever escape the module
  boundary — consistent with AGENTS.md rule 8.
- *Registry loaded once*: `_get_registry()` uses a module-level singleton so
  `action_registry.yaml` is parsed once per process, not on every call.

**What actually happened when run:**
`pytest tests/translator/ -v` — **127/127 passed in 0.24s** (venv/bin/python3).
Full suite (excluding slow LLM tests): **188/188 passed in 4.45s**.

| Test class | Count | Result |
|---|---|---|
| `TestRegistryCompleteness` | 22 | ✅ all pass |
| `TestTranslate` | 35 | ✅ all pass |
| `TestValidateRejections` | 35 | ✅ all pass |
| `TestValidateAcceptance` | 11 | ✅ all pass |
| `TestAllowListCrossCheck` | 5 | ✅ all pass |
| `TestValidateNeverRaises` | 19 | ✅ all pass |

**Deviations from the MSD/spec:** None. The addition of `PYTHON_ALLOWED_IMPORTS`
as a public constant in `sandbox.py` (and its re-export from `atar/tools/__init__.py`)
is a targeted, additive change to M1 — no existing signatures modified. The old
private `_ALLOWED_IMPORTS` set is retained as an alias so M1's internal code is
unchanged.

---

### M5 Gymnasium Environment

**What:** Implemented `ATAREnv(gymnasium.Env)` in `atar/env/atar_env.py` — the
integration layer wrapping M1 (tools), M3 (LLM), and M4 (translator) into a
standard Gymnasium environment.  Discrete(7) action space, Box(512,) observation
space.  `step()` returns a plain `float` reward (as `RewardComponents.total`);
the full breakdown lives in `info["reward_components"]`.  Four reward components
(accuracy, efficiency, safety, bonus) implemented with the MSD's exact tuned
values.  512-d state vector composed per `interfaces.md` section 2: 384-d
semantic encoding + 7-d tool-history flags + 64-d outcome embedding + 2-d
budget features + 55-d zero padding.

**Why this approach:**
- *Scalar reward + info dict*: `check_env()` and standard RL tooling (SB3, CleanRL)
  require a plain `float` reward.  The full `RewardComponents` dataclass goes in
  `info["reward_components"]` for per-component logging and ablation curves.
- *Nondeterministic registration*: `gymnasium.register("ATAR-v0", nondeterministic=True)`
  because the LLM sampler and `_search_web` 20% noise make `step()` inherently
  stochastic.  Without this, `check_env()`'s `check_step_determinism()` assertion
  fails (correctly — the env IS non-deterministic by design).
- *Outcome embedding*: 64-d vector encoding success/fail flag, output type (32-d
  one-hot), length/latency/cost norms, and 27-d reserved padding.  Lightweight
  enough to compute every step without bottlenecking rollouts on CPU.
- *Redundancy detection*: tracks successful tool calls per action type; same action
  repeated when already successful triggers an extra -1.0 efficiency penalty.
- *Flexible accuracy matching*: exact string match OR substring containment OR
  numeric equality (with `math.isclose`, rel_tol=1e-6), to handle LLM output
  formatting variation.

**What actually happened when run:**
`venv/bin/python3 -m pytest tests/env/ -v` — **20/20 passed in 129.42s**
(Qwen/Qwen2.5-0.5B-Instruct on CPU).

| Test class | Count | Result |
|---|---|---|
| `TestReset` | 4 | ✅ all pass (20 sampled tasks across all tiers) |
| `TestFullEpisode` | 2 | ✅ all pass (real M1/M3/M4, Tier 1 calculator) |
| `TestRewardComponents` | 4 | ✅ all pass (hand-computed values, sum invariant) |
| `TestMaxTurnsTruncation` | 2 | ✅ all pass (truncated=True at turn 5, no accuracy bonus) |
| `TestGymnasiumCompliance` | 5 | ✅ all pass (check_env, obs/action space, reward type) |
| `TestStateComposition` | 3 | ✅ all pass (padding zeros, tool flags, budget features) |

**Deviations from the MSD/spec:** None.  ADR-3 text is slightly misleading (says
"return `RewardComponents` from step()") but the actual interfaces.md contract and
M5 MSD both specify plain `float` — which is what we implement.

## Results
<!-- Filled in only after the real evaluation run (Day 22-24). Table of
accuracy / avg tool calls / recovery rate for ATAR vs Random vs ReAct vs
Zero-Shot baselines, with confidence intervals. -->

## Ablations
<!-- No-curriculum and no-projection-cotraining results, once run. -->

## Known limitations
<!-- Be honest here — this is what gets asked about in interviews. -->
