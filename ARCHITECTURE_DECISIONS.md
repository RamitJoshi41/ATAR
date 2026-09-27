# Architecture Decision Log

<!--
Lightweight ADR format. One entry per non-trivial decision — anything an
interviewer might reasonably ask "why did you do it that way?" about.
Include decisions already made in the original project spec once you've
actually implemented and can speak to them, plus any new ones made during
the build (e.g. the vertical-slice-first sequencing, the NIM tradeoff,
any deviation from planned hyperparameters).

Template per entry:

## ADR-<N>: <short title>
**Date:** YYYY-MM-DD
**Decision:** <what was decided>
**Context:** <what problem this addresses>
**Alternatives considered:** <what else was on the table, and why not chosen>
**Consequence:** <what this decision costs or enables going forward>
-->

## ADR-1: Vertical slice before parallel module build
**Date:** 2026-09-03
**Decision:** Built a minimal end-to-end path (one tool, dummy policy)
before fanning out M1-M3 to parallel agents.
**Context:** Multi-agent parallel builds risk agents producing
individually-plausible code that doesn't integrate, discovered late.
**Alternatives considered:** Fan out immediately per the original spec's
Phase 1 plan.
**Consequence:** Slightly slower start (1-2 extra days), but locks the
real interface contract before 3 agents build against it, reducing
integration risk at M5.

## ADR-2: Calculator uses AST whitelist, not eval()
**Date:** 2026-09-03
**Decision:** Calculator parses with `ast.parse(expr, mode="eval")` and
walks the tree with a whitelist of `ast.Add`, `ast.Sub`, etc. Anything
not in the whitelist is rejected.
**Context:** AGENTS.md rule 8 requires fail-closed validation for
security-critical modules (M1 Tool Sandbox).
**Alternatives considered:** `eval()` with `__builtins__={}` — known to
be bypassable via `__subclasses__`. `simpleeval` library — external dep.
**Consequence:** Slightly more code, but zero import surface and provably
safe (only numeric literals and arithmetic operators can execute).

## ADR-3: Reward from step() is a plain float; RewardComponents goes in info dict
**Date:** 2026-09-03 (corrected 2026-09-13)
**Decision:** `env.step()` returns a plain `float` (= `RewardComponents.total`)
as the scalar reward.  The full `RewardComponents` dataclass is placed in
`info["reward_components"]` for per-component logging and ablation curves.
**Context:** The original vertical-slice prototype returned the
`RewardComponents` dataclass directly as the reward.  This broke
`gymnasium.utils.env_checker.check_env()` (which asserts the reward is a
scalar) and is incompatible with standard RL tooling (SB3, CleanRL) that
expects `float`.  interfaces.md section 4 was updated to codify the
scalar-reward contract before M5 was built.
**Alternatives considered:** Return `RewardComponents` directly — more
ergonomic for logging, but breaks the Gymnasium API contract and every
downstream consumer that expects a numeric reward.
**Consequence:** M7's rollout buffer stores a plain float for the reward.
Per-component logging reads `info["reward_components"]` from the step
output, which is always available.
## ADR-4: Deterministic override in M1 execute_tool
**Date:** 2026-09-04
**Decision:** Added an optional `deterministic: bool = False` flag to M1's `execute_tool` signature (and internally to `_search_web`).
**Context:** M2 Task Generator must compute ground truth programmatically via M1. However, M1's `_search_web` injects noise 20% of the time. We needed a way for M2 to get reliable, noise-free truth without stripping the stochastic behavior required for RL agent training.
**Alternatives considered:** 1) Re-running the tool in a loop until it succeeded (brittle). 2) Globally patching Python's `random` module during task generation (dangerous side-effects, test leakage).
**Consequence:** A minor change to `interfaces.md` that safely isolates task generation requirements from RL training requirements.

## ADR-5: M3 generate() must pass max_new_tokens to Outlines
**Date:** 2026-09-09
**Decision:** `LLMInterface.generate()` always passes `max_new_tokens=256`
(default, caller-overridable) to the Outlines 1.3.x model callable instead of
relying on HuggingFace's default `max_length`.
**Context:** HuggingFace sets a model-agnostic fallback of `max_length=76`
*total* tokens when neither `max_new_tokens` nor `max_length` is supplied by
the caller. The Outlines wrapper forwards this limit unchanged. For the 0.5B
stand-in the prompt alone is 50–60 tokens, leaving only 16–26 tokens for the
JSON body — not enough to close all string values before hitting the limit.
This caused `JSONDecodeError: Unterminated string` on the 20/20 constrained-
generation test (failed on trial 8 of 20 in the first run). The Outlines
constrained-decoding FSM was working correctly; only the token budget was wrong.
**Alternatives considered:** Letting callers always supply `max_new_tokens` —
too error-prone for downstream code. Setting a very large number (e.g. 2048) —
wastes KV-cache memory on CUDA for short responses.
**Consequence:** 256 new tokens is enough for any realistic tool-call JSON
object while keeping memory overhead reasonable. Callers that need longer
responses (unlikely for M3's schemas) can pass a higher value.

## ADR-6: Always use venv/bin/python3 for pytest; system Python is not acceptable
**Date:** 2026-09-09
**Decision:** All pytest invocations must use `venv/bin/python3 -m pytest` (or
`venv/bin/pytest`) from the repo root. Bare `pytest` or `/usr/bin/python3 -m
pytest` are banned. AGENTS.md rule 4a codifies this permanently.
**Context:** During M3 acceptance, the architect noticed the M3 pytest run
used `/usr/bin/python3` (system Python), not the project venv. Investigation
revealed: the venv was created during M0 (`python3 -m venv venv`) but
`pip install -e .[dev]` was never run *inside* it. The venv accumulated only
a few packages that were explicitly installed into it session-by-session
(outlines, pydantic, gymnasium, rank-bm25, wandb). Everything else
— `torch`, `transformers`, `accelerate`, `bitsandbytes`, `pytest`, `ruff`,
`black`, and the `atar` editable install — lived only under
`~/.local/lib/python3.10/site-packages` (system user site).

Pytest appeared to work because `pyproject.toml` sets `pythonpath = ["."]`,
so `import atar` resolved via CWD rather than an installed editable package.
Tests passed but against uncontrolled system-site package versions — a
reproducibility failure that would be invisible until something diverged.

The gap was fixed in one command: `venv/bin/pip install -e .[dev]`, which
installed `torch 2.11.0`, `transformers 5.16.1`, `accelerate 1.14.0`,
`bitsandbytes 0.50.2`, `pytest 9.0.3`, `pytest-cov 7.1.0`, `black 24.10.0`,
`ruff 0.8.6`, `streamlit 1.63.0`, and the editable `atar 0.1.0` package into
the venv.
**Alternatives considered:** Relying on CI to catch this — would work but
only after pushing, not during local DoD verification. Using
`include-system-site-packages = true` in pyvenv.cfg — defeats the purpose
of isolation entirely.
**Consequence:** Any agent that runs `pytest` without the venv will get
caught immediately because the venv's `pytest` binary is separate from
the system one. The rule is explicit in AGENTS.md (rule 4a) and will be
verified at the start of every future module's DoD run.

## ADR-7: M4 uses frozenset import from M1 (not a duplicate list) for Python allow-list
**Date:** 2026-09-12
**Decision:** Promote `_ALLOWED_IMPORTS` in `sandbox.py` to a public
`PYTHON_ALLOWED_IMPORTS: frozenset[str]` constant. M4 imports it by name
from `atar.tools.sandbox`. No schema change; backward-compat alias retained.
**Context:** MSD DoD-5 requires that the forbidden-imports list in `validate()`
comes from the *same source* as M1's sandbox uses — "not duplicated". If the
lists were stored in two places, they could silently diverge: someone adding
`fractions` to the sandbox allow-list might forget to update M4, making M4
reject valid code. The MSD explicitly flags "import the same list from a
shared config, don't hardcode it twice."
**Alternatives considered:**
- Store the list in a `configs/python_allowlist.yaml` file and have both M1
  and M4 read it — adds an extra I/O call on every import and introduces a
  new config file that could itself be mis-edited.
- Store it in `shared_types.py` — that file is the locked contract; adding
  runtime configuration there would blur the boundary between types and config.
- Leave it as a private `_ALLOWED_IMPORTS` in `sandbox.py` and ask M4 to
  import the private name — valid Python but fragile (private names are a
  convention, not a contract; a future refactor could rename without notice).
**Consequence:** `PYTHON_ALLOWED_IMPORTS` is now part of M1's public surface
(exported from `atar/tools/__init__.py`). Any future change to the allowed
set requires editing exactly one place: line 124 of `sandbox.py`. The
`TestAllowListCrossCheck.test_same_object_identity` test enforces this
structurally — if someone somehow re-declares the set in M4, the `is` check
will fail.

## ADR-8: M4 validate() uses two-layer rejection (registry patterns then AST)
**Date:** 2026-09-12
**Decision:** `validate()` runs two sequential checks: (1) fast regex scan of
`json.dumps(raw_params)` against `entry.forbidden_patterns` from the YAML
registry; (2) action-specific deep validators (SQL keyword check, Python AST
walk, calculator AST node whitelist, etc.).
**Context:** The security invariant (AGENTS.md rule 8, MSD section
"Validation responsibilities") requires catching SQL injection, forbidden
Python imports, malformed JSON, and empty expressions *before anything
reaches M1*. A single flat list of regex patterns cannot handle all these
cases (e.g., `import os` inside a `from X import Y` form, or an arithmetic
expression that parses as valid Python but contains a function call).
Conversely, AST validation alone doesn't catch patterns embedded in strings
or multi-line YAML serialisations before deserialisation.
**Alternatives considered:**
- Regex-only: simpler but not rigorous (can be bypassed by e.g. `from
  os import path` if the pattern only matches `import os`).
- AST-only: misses text-level patterns in string fields before they reach
  the parser.
- Full JSON Schema library (jsonschema package): adds a dependency not in
  the tech stack; the MSD doesn't require full schema compliance because
  Outlines already enforces schema at generation time. The lightweight
  required-keys + type check is sufficient for the security gate.
**Consequence:** Some inputs are rejected by the registry pre-filter rather
than the specific validator, so the error message says "Forbidden pattern
matched: ..." rather than "Forbidden import: ...". This is acceptable — the
caller receives (False, reason) either way. The tests check `ok is False`
(security invariant) and also that "Forbidden" appears in the reason
(actionable signal), without requiring the exact source of rejection.

## ADR-9: ATAREnv registered as nondeterministic
**Date:** 2026-09-13
**Decision:** Register `ATAREnv` as `ATAR-v0` with `nondeterministic=True` in
Gymnasium's env registry. In test, manually attach `env.spec = gym.spec("ATAR-v0")`
so `check_env()` sees the flag and skips `check_step_determinism()`.
**Context:** `check_env()` calls `check_step_determinism()`, which resets the env
with the same seed twice, takes the same action, and asserts the observations are
identical. ATAREnv's `step()` is inherently stochastic: M3's LLM sampler produces
different JSON params on each call (even with the same seed, the HuggingFace
sampling path is not fully deterministic), and M1's `_search_web` has 20% noise.
There is no way to make `step()` deterministic without fundamentally changing M1/M3.
**Alternatives considered:**
- Seed the LLM's RNG via `torch.manual_seed()` on each reset: breaks the stochastic
  training property the RL agent needs (it must experience varied tool outputs).
- Bypass `check_env()` entirely: loses validation of all other gymnasium contracts
  (observation/action spaces, reward type, reset return type, etc.).
- Use `env.spec = None` (don't register): `check_step_determinism` still runs when
  spec is None, so this doesn't help.
**Consequence:** The determinism check is explicitly skipped; all other `check_env()`
validations still run and pass. The env is correctly marked as stochastic in the
registry, which is accurate for an RL training environment with a neural-network
policy and noisy tools.

## ADR-10: Batched/vectorized rollout collection for M7
**Date:** <today>
**Decision:** M7 collects rollouts from multiple parallel ATAREnv instances
simultaneously, batching encode()/generate() calls across episodes rather
than running one sequential episode at a time.
**Context:** Measured on Kaggle T4 with the real Qwen2.5-7B-Instruct model:
~2.9s/step sequential, making a single 2048-step rollout ~1.6 hours and a
full multi-rollout training run 25+ hours — incompatible with Kaggle's
12-hour session cap and weekly quota.
**Alternatives considered:** Accepting the sequential cost and splitting
training across many resumed sessions (rejected — impractically slow
iteration); reducing rollout_steps below the spec's 2048 (rejected —
changes a tuned hyperparameter rather than fixing the actual bottleneck).
**Consequence:** M7's rollout scaffolding is inherently more complex
(managing N parallel episodes, independent resets, a batched buffer) than
a single sequential loop. May also require extending M3's encode()/
generate() to accept batched input if not already batch-capable.

## ADR-11: generate_batch() loops internally (Outlines 1.3.x limitation)
**Date:** 2026-09-13
**Decision:** `LLMInterface.generate_batch()` exposes a batched API but
internally calls `self.generate()` in a loop, because Outlines 1.3.x does
not support true GPU-batched constrained decoding.
**Context:** ADR-10's throughput analysis assumed both encode and generate
would be genuinely batched. After investigation, only `encode_batch()` is
truly batched (single GPU forward pass). Constrained JSON generation via
Outlines must receive one prompt at a time.
**Alternatives considered:**
- Bypass Outlines and implement batched constrained decoding directly with
  the HuggingFace model (rejected — reimplementing Outlines' trie-based
  logit masking is a major undertaking outside scope).
- Drop `generate_batch` and expose a raw loop in the collector (rejected —
  breaks API stability; future Outlines versions may add batch support, so
  a stable API is valuable).
- Call generate() sequentially in the collector itself (rejected — forces
  callers to own the loop and makes upgrading to true batch support require
  collector changes).
**Consequence:** The `generate` half of each collection cycle is sequential
(~N×2.7s on Kaggle T4). The `encode` half is genuinely batched (~0.08s for
N=8). Net effect: batching provides real speedup for encode but not generate.
This changes ADR-10's original throughput estimate. Actual per-cycle latency
must be measured on Kaggle after the first training run. The API is forward-
compatible — switching Outlines versions to gain true batch support requires
no M7 code changes.

## ADR-12: ATAREnv skip_internal_encode flag (M5 patch for M7 batching)
**Date:** 2026-09-13
**Decision:** Added `skip_internal_encode: bool = False` to `ATAREnv.__init__`.
When True, `_build_state()` skips the internal `llm.encode()` call and leaves
the semantic slice (state[0:384]) as zeros. M7's `VectorizedCollector` sets
this flag and replaces the semantic slice with batched encoding output.
**Context:** `ATAREnv.step()` calls `llm.encode()` internally to build the
next state. Without this flag, the collector would need to run N sequential
`encode()` calls (one per env, from inside each `step()`) plus the intended
1 `encode_batch()` call — paying double the encode cost.
**Alternatives considered:**
- Accept the double encode cost (rejected — N sequential encode() calls
  inside env.step() wastes the primary GPU throughput gain).
- Have the collector monkeypatch the LLM's encode() to a no-op (rejected —
  fragile and surprising; breaks if M5 caches the result).
- Move state construction entirely outside ATAREnv (rejected — requires
  a more invasive M5 refactor and breaks the M5 public interface).
**Consequence:** M5 has a new optional parameter that is transparent to all
existing callers (default=False). Existing 20/20 M5 tests pass unchanged.
The parameter is documented in the class docstring with the ARCHITECTURE_DECISIONS.md
reference. Collector must set this flag or else the semantic slice will be
zero in the stored states.

## ADR-13: Collector semantic slice management — state overwrite pattern
**Date:** 2026-09-13
**Decision:** `VectorizedCollector` overwrites `states[:, 0:384]` with the
output of `encode_batch()` before storing states in the rollout buffer and
before the policy forward pass. The `next_state` returned by `env.step()`
is stored in `EpisodeStep.next_state` as-is (semantic slice = zeros) since
it is not used by the PPO update (only the current state and the value
bootstrap at the terminal step matter).
**Context:** With `skip_internal_encode=True`, env.step() returns states
with zeros in the semantic slice. The collector must manage the semantic
content of all states it stores.
**Alternatives considered:**
- Compute a separate `encode_batch()` for next_states at each step
  (rejected — doubles encode cost; next_states become current_states at
  the next step anyway, where they'll be freshly encoded).
- Store `next_state` with the semantic slice filled (rejected — requires
  an extra encode_batch pass per step with no training benefit, since GAE
  only needs the current-state value estimates).
**Consequence:** `EpisodeStep.next_state` stored in the rollout buffer has
zeros in the semantic slice. Code that reads `next_state` from the buffer
(e.g. a bootstrapped value head) must re-encode it. The architect should
note this if implementing the GAE bootstrap step using buffered next_states.

## ADR-14: encode_batch() numerical equivalence tolerances and CPU vs GPU environment
**Date:** 2026-09-13
**Decision:** `test_encode_batch_matches_single_smoke` verifies semantic equivalence
using cosine similarity > 0.99 and max absolute difference < 0.25 rather than strict
floating-point equality. A tighter non-padded test requires cosine similarity > 0.9999
and max difference < 0.12.
**Context:** Diagnostic investigation revealed that even without padding, running
sequences individually vs in a batch produces non-zero residual differences (~0.086
on CPU). On CPU development environments (without CUDA), BitsAndBytes 4-bit
quantization is inactive; the difference stems from floating-point reduction order
variations across batch dimensions in CPU BLAS (MKL/oneDNN) and attention softmax
reductions. Left-padding introduces additional minor rounding variances.
**Alternatives considered:**
- Requiring strict `torch.allclose` with tight tolerances (rejected — fails due to
  standard floating-point non-associativity across batch dimension reductions).
- Widening tolerance blindly without root-causing (rejected — diagnostic proved that
  padding is correctly masked and model forward is called only once).
**Consequence:** Tests reliably pass locally on CPU while preventing masking bugs.
However, once running on the target remote training environment (Kaggle T4 GPU) with
actual BitsAndBytes 4-bit NF4 quantization and float16 fused attention kernels
enabled, the numerical noise profile and magnitude may differ. The tolerance must be
re-checked on Kaggle, not assumed identical to CPU.