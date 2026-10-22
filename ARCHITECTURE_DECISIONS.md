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
## ADR-15: Per-env GAE using side-arrays in RolloutBuffer
**Date:** 2026-10-13
**Decision:** `RolloutBuffer.add()` takes an `env_id` (and `pooled`, see ADR-16).
`PPOTrainer._fill_gae()` runs `compute_gae()` separately on each env's
time-ordered transitions, bootstrapping a cut-off trajectory from
`V(current state)` (computed by the collector after each rollout). Returns are
`values + raw advantages`; advantages are normalised only inside
`compute_ppo_loss`, per minibatch.
**Context:** The collector fills the buffer round-robin (env0, env1, …,
envN-1, env0, …). The original `_fill_gae` ran GAE over the flat buffer, so
`V(s_{t+1})` came from a different env's episode and the ±10 terminal reward
was almost never credited to the tool calls that led to it. It also built
returns from *normalised* advantages, putting value targets on the wrong
scale. The first Kaggle run (12 updates, checkpoints up to step 24576) was
trained with both bugs.
**Alternatives considered:**
- Add `env_id` to `EpisodeStep` (rejected: it's a locked shared type).
- Infer env from index (`k % n_envs`) (rejected: silently breaks if the
  collector's stepping order ever changes).
**Consequence:** The buffer-to-env mapping is explicit and covered by tests
(`tests/policy/test_ppo_fixes.py`, hand-computed GAE). One extra batched
encode + policy forward per rollout to get the bootstrap values (negligible
next to 2048 generate() calls).

## ADR-16: Projection co-training through stored pooled features
**Date:** 2026-10-13
**Decision:** M3 gains two additive public methods:
`encode_pooled_batch()` (frozen-LLM mean-pooled hidden states, no grad) and
`project()` (applies the trainable projection, *with* grad).
`encode_batch()` is now `project(encode_pooled_batch())` under `no_grad`.
The collector stores each transition's pooled features; the PPO update
re-applies `project()` to them (`PPOTrainer._with_projected_states`) so the
loss back-propagates into the projection. The projection is always float32
and seeded with a local `torch.Generator(42)`; checkpoints now include
`projection_state_dict` (and `wandb_run_id`), and evaluation loads it.
**Context:** The MSD's "co-adapted projection" was never actually trained:
`encode()` returned detached tensors and states were stored as numpy, so the
projection received no gradient (confirmed: the Kaggle checkpoints hold Adam
state for the 12 policy tensors and 0 for the 2 projection tensors), and
checkpoints didn't save it. The `--freeze-projection` ablation was therefore
identical to the main run.
**Alternatives considered:**
- Re-run the 7B forward pass inside the PPO update (rejected: 10 epochs ×
  2048 samples of 7B forward passes — far too slow on a T4).
- Store token ids and recompute (rejected: same cost).
- Keep the projection frozen and document it (rejected by the architect,
  2026-10-13 — co-training is the project's headline design choice).
**Consequence:** ~14 KB/transition of extra CPU memory (3584 floats for 7B;
~29 MB per 2048-step rollout). Initial projection values differ from earlier
runs (float32 init instead of float16), so pre-fix checkpoints should not be
resumed for the new run. Global torch RNG is no longer reset by
`LLMInterface()`.

## ADR-17: One shared answer grader for reward and evaluation
**Date:** 2026-10-13
**Decision:** `atar/env/answer_matching.py::is_answer_correct()` is used by
both `ATAREnv._compute_accuracy_reward()` and `atar.evaluation`. Numeric
truths compare the *last* number in the answer (tolerance rel 1e-6 / abs
0.005); short text truths (≤3 content words) need every content word as a
whole word; long reference sentences need ≥60% content-word recall or an
answer whose content words are all in the truth.
**Context:** The previous grader used bidirectional substring matching:
`"6"` matched `61559.0`, `"I do not know"` matched `no`, any answer containing
"2" matched the 9 test tasks whose truth is `2` — while
`"2660 + 58899 = 61559"` was marked wrong. The env and eval each had a copy.
**Alternatives considered:**
- Exact match only (rejected: Tier 1 search / Tier 4 truths are whole
  abstract sentences; "Paris" must match "The capital of France is Paris…").
- LLM-as-judge (rejected: adds a 7B generate() per episode during training).
- First number instead of last (rejected: "employee #2 earns 95000" pattern).
**Consequence:** The training reward changed (approved by the architect), so
pre-fix checkpoints were trained against a different reward. Known
limitation: an answer giving the result first and then other numbers
("61559 (2660 + 58899)") is graded on 58899 — documented in tests.

## ADR-18: Evaluation efficiency metric and resume fingerprints
**Date:** 2026-10-13
**Decision:** `efficiency = accuracy / (1 + avg_tool_calls)`. Each agent's
per-task file gets a `.meta.json` fingerprint (model name, seed, grader
version, checkpoint SHA-256 for `atar`); resuming into a directory whose
fingerprint differs, or that has results but no fingerprint, raises.
**Context:** The original `accuracy / avg_tool_calls` gave 0 to an agent that
answers correctly without tools and explodes as tool calls → 0. Resume
previously skipped any task id already on disk regardless of which
checkpoint produced it, so evaluating checkpoint B into A's directory would
silently report A's results.
**Alternatives considered:** Keep the old ratio and special-case zero tools
(rejected: still discontinuous near zero).
**Consequence:** Old `eval_smoke*/` directories can't be resumed (no
fingerprint) — use a fresh `--out-dir`.

## ADR-19: Step budget = 24,576 planned + optional extension; demo = recorded replays
**Date:** 2026-10-13
**Decision:** Retrain with `--total-steps 24576 --rollout-steps 1024` (24
updates). After it finishes, evaluate checkpoints on the held-out set; only
if held-out success is still rising, continue with
`--extend-to <N>` (`PPOTrainer.start_extension`): the curriculum boundaries
stay those of the original plan (so every extra step samples all 5 tiers),
the entropy coefficient stays at its end value, and the LR restarts at
`--extend-lr` (1e-4) with a fresh cosine decay to 1e-5 over the extra
updates. The M8 demo will replay recorded episodes per checkpoint
("version"), with an optional live mode on a GPU host.
**Context:** Gemini had told the architect 100k steps were needed; that
figure is `train.py`'s default, not a measurement. At the measured
4.0–4.9 s/step, 100k steps ≈ 125 GPU-hours ≈ 4+ weeks of Kaggle quota.
`--total-steps` fixes the curriculum and LR schedule up front, so a long
plan can't simply be stopped early (Phase 3 / tier 5 would be skipped).
Hosted APIs such as NVIDIA NIM return text only, but the policy's state
needs the 7B model's hidden states, so the policy cannot run on NIM alone.
**Alternatives considered:**
- 100k planned up front (rejected: no evidence it's needed; 4+ weeks).
- 49,152 planned up front (rejected: commits 2 weeks before seeing a curve).
- Switch the encoder to a small CPU model so a live demo could run free with
  NIM for generation (rejected by the architect 2026-10-13; would change
  the design and require deciding before the retrain).
**Consequence:** The decision to train longer is made from the held-out
learning curve, not guessed. Every per-update checkpoint is kept as a
demo "version". The M8 MSD's "not a mocked/precomputed pair" DoD item will
need amending to allow clearly-labelled recorded replays.

## ADR-20: CLARIFY bonus only on ambiguous tasks, once; restart after the pilot
**Date:** 2026-10-14
**Decision:** The +1 CLARIFY safety bonus is paid only when the task's
solution requires CLARIFY (`ActionType.CLARIFY in task.required_tools` —
the 150 tier-4 tasks) and only for the first CLARIFY of the episode. Any
other CLARIFY costs the normal −0.5 tool-call efficiency penalty. The env's
`info` also gains `params_valid`, and training logs per-action
`step_reward/*` and `invalid_param_rate/*` plus a `metrics.jsonl` history
file next to the checkpoints. The real run restarts from scratch.
**Context:** The 3-update Kaggle pilot (2026-10-14) chose CLARIFY for 14.6%
of actions during Phase 1, where no task is ambiguous. The env paid +1 for
every successful CLARIFY on any task with no tool cost — free reward a
policy can farm (a CLARIFY every turn beats any real tool call), contrary
to the M5 spec ("+1 for correctly recognizing and executing a CLARIFY action
on an ambiguous task"). The pilot also showed CALCULATOR at 3% with no way
to tell whether that was rational (the 7B answers arithmetic well directly)
or caused by rejected parameters — hence the per-action diagnostics.
**Alternatives considered:**
- Keep +1 per CLARIFY but cap it (rejected: still rewards asking on tasks
  that need no clarification).
- Resume the main run from the pilot checkpoint (rejected: 3 updates trained
  under the old reward; a reward change mid-run muddies the results for
  ~3 GPU-hours saved).
**Consequence:** CLARIFY now has to be learned as tier-4-specific behaviour.
The pilot otherwise confirmed the fixes: value_loss ≈ 33 (real return scale,
vs. stuck at ≈ 0.9 before ADR-15), entropy 1.76 of a 1.95 maximum (still
exploring), LR and entropy-coefficient schedules exactly as computed, and
3.7 s/step throughput.

## ADR-21: M8 demo = data bundle + one episode-record format for replay and live
**Date:** 2026-10-14
**Decision:** The demo reads a *bundle* folder (`manifest.json`,
`curve.json`, `replays.jsonl`) built offline by
`scripts/build_demo_bundle.py` from `evaluate.py` outputs (learning curve,
per-version action mix) and `scripts/record_replays.py` outputs (episodes).
Replays and live runs share one record type (`atar.demo.replay.StepRecord` /
`EpisodeRecord`) produced by one generator, `iter_episode`, so a recorded
episode renders exactly like a live one and step-through works the same in
both. Supporting changes: ATAREnv's step `info` gains `params`,
`param_error` and `cumulative_tokens`; ATAREnv takes `search_backend`
(live Wikipedia search) and `max_turns` (may only shorten episodes; the
budget feature stays normalised by the training value 5); ReAct now emits a
one-sentence Thought before each action (as in Yao et al.), shared by the
evaluation and the demo; `load_atar_agent()` is the single checkpoint loader
(evaluate, record, live); `scripts/make_untrained_checkpoint.py` provides
the step-0 point; `--max-tasks` now takes a tier-stratified subset.
**Context:** The policy can only run where the training model runs (hidden
states), the architect wants versions over training steps, and the M8 DoD
requires step-through, side-by-side ReAct vs ATAR and a live-tools toggle.
**Alternatives considered:**
- Live only (rejected: needs a GPU host for every viewer).
- Precompute screenshots/videos (rejected: no step-through, no inspection
  of probabilities or params).
- Separate record formats for replay and live (rejected: two renderers to
  keep in sync).
**Consequence:** The demo runs anywhere (laptop, free CPU host) from a
small folder; live mode is an add-on for GPU hosts. Bumping the ReAct
prompt changed the eval fingerprint (`agent_version`), so old ReAct results
can't be resumed into new ones. Discovered while building it: the test set
is sorted by tier, so the old `--max-tasks N` smoke runs evaluated tier 1
only.

## ADR-22: No answer is graded as a wrong answer (−5), not 0
**Date:** 2026-10-16
**Decision:** An episode that reaches the turn limit without a final answer
(or ends with an empty one) gets the same −5 accuracy reward as a wrong
answer (`_ACCURACY_TRUNCATED = _ACCURACY_INCORRECT` in `atar_env.py`). This
amends the M5 MSD, which specified 0.
**Context:** Greedy replays of `checkpoint_step_15360` (4 held-out tasks per
tier, 2026-10-16) answered 10 of 20 correctly. **All 10 failures** were the
policy repeating CLARIFY until the 5-turn limit (CLARIFY probability ~0.94
each turn); not one was a wrong answer. Under the M5 reward that is
rational: five CLARIFYs cost 5 × −0.5 = −2.5, a wrong answer costs −5, so
whenever the policy's chance of being right is below ~1/6, stalling has the
higher expected return. The critic had learned exactly this (value ≈ −3.4
falling to −0.5 along a stall). The behaviour even appeared with the answer
in hand: on a tier-5 task the policy read salary 80000 from SQL, then
stalled instead of answering.
**Alternatives considered:**
- Raise the per-CLARIFY cost (rejected: treats the symptom; any cheap
  non-answer action could be used to stall).
- A per-turn time penalty (rejected: it penalises legitimate multi-step
  tasks as much as stalling, and still leaves "no answer" cheaper than a
  wrong one at the end of the episode).
**Consequence:** Stalling to the turn limit now costs at least −5 −0.5 × turns,
which is always worse than committing to an answer, so the policy has to
answer when unsure (an answer can still be right; silence never is). The
evaluation metrics are unaffected (accuracy counts correct answers).

## ADR-23: CLARIFY may be used once per episode (action mask)
**Date:** 2026-10-16
**Decision:** `ATARPolicy.forward` masks CLARIFY (logit → dtype minimum,
probability 0) once the state's CLARIFY tool-history flag (state index
384 + 5) is set. The env and its `Discrete(7)` action space are unchanged.
**Context:** The env answers every CLARIFY of an episode with the same text
(the task's `clarification_answer`, or "No further information available."),
so a second CLARIFY can never add information. Yet in the diagnostic the
policy repeated it up to 5 times, including on tier-4 tasks right after
the first CLARIFY had returned exactly what was missing ("14 percent of
902"). ADR-22 removes the incentive to stall; the mask removes a dead-end
action, so exploration after a CLARIFY is spent on actions that can work.
**Alternatives considered:**
- Mask in the env (reject the action) — rejected: the env would have to
  return something for an illegal action, and the baselines share the env.
- Rely on ADR-22 alone — rejected: the collapsed policy needs many updates
  to unlearn a 0.94-probability habit, while repeating CLARIFY is never
  useful by construction.
**Consequence:** The mask is computed from the policy's own input, so the
rollout sampler, the PPO loss (log-probs recomputed from stored states),
greedy evaluation and the demo's probability charts all apply it with no
extra plumbing; old checkpoints still load (no new parameters). Masked
log-probs/entropy stay finite (`tests/policy/test_action_mask.py`).

## ADR-24: Restart the real run under a versioned training setup
**Date:** 2026-10-16
**Decision:** The real run restarts from step 0 with ADR-22, ADR-23 and the
fixed tier-3 data. Checkpoints now record `training_setup_version`
(`TRAINING_SETUP_VERSION = "2026-10-16"` in `atar/policy/types.py`);
`PPOTrainer.restore_from_checkpoint` and `scripts/train.py` refuse to
resume a checkpoint from another setup (fail closed, before the model
loads). Evaluation fingerprints gain the test set's SHA-256.
**Context:** Besides the stalling reward, the diagnostic exposed a data bug:
the tier-3 template "...capital of France and write a python script that
prints its length" computed its ground truth as the length of the whole
search snippet ("The capital of France is Paris, located in Europe." → 50)
instead of the capital's name (5). 53 of 200 tier-3 training tasks (27%) and
14 of 50 tier-3 test tasks were impossible for a correct agent — part of
why tier 3 sat at ~0.57. The generator is deterministic (regenerating with
the old code reproduced both files byte for byte), so the fix changes
exactly those 67 rows (`ground_truth` and the chain in `metadata`).
**Alternatives considered:**
- Resume from `checkpoint_step_15360` (2 sessions instead of 4) —
  rejected: that policy spent 7 updates collapsing into the stall
  (entropy 0.35, CLARIFY ~0.9 on tiers 3–5), its critic is calibrated to the
  old reward, and the cosine LR is already down to ~1e-4 of 3e-4, so the
  remaining 9 updates would mostly be spent unlearning. It would also make
  the demo's learning curve mix two different problems. Same reasoning as
  ADR-20's restart after the pilot.
- Rely on the KAGGLE.md checkpoint search alone to avoid picking an old-run
  checkpoint — rejected: the old run's step 15,360 outranks a new run's
  early checkpoints in a "newest step" search; a code-level guard can't be
  forgotten.
**Consequence:** ~4 Kaggle sessions (8 + 7 + 7 + 2 updates, ≈31 GPU-hours at
4.5 s/step) for the 24,576-step plan. Old-run checkpoints remain usable for
replays/evaluation (loading for inference is not blocked), but not for
resuming. Training logs also gain up to 3 rejected-parameter examples per
action per rollout in `metrics.jsonl` (`param_error_examples`) to diagnose
the 50–78% CALCULATOR rejection rate seen in session 2.

## ADR-25: The single-statement SQL check runs on the raw query only
**Date:** 2026-10-22
**Decision:** Remove `';\s*\S'` from SQL_QUERY's `forbidden_patterns` in
`atar/configs/action_registry.yaml`. M4's `_validate_sql_query` keeps the
same regex (`_SQL_MULTI_STMT_RE`) on the raw query string, and M1's SQLite
`execute()` refuses more than one statement on its own.
**Context:** Run 2, session 1's `param_error_examples` showed single
queries ending in `;` (`SELECT * FROM articles WHERE title LIKE '%Linux%';`)
rejected as "multiple statements". Registry patterns are matched against
`json.dumps(params)`, where a trailing `;` is followed by `"}` — a
non-whitespace character. M4's raw-query check accepts these correctly.
**Alternatives considered:**
- Match registry patterns against each string value instead of the
  serialised dict — rejected for now: it changes how every action's
  patterns behave in the middle of a run, for a bug only this pattern has.
- Change the pattern to exclude `"` after `;` — rejected: a pattern written
  against JSON escaping is fragile, and the raw-query check already exists.
**Consequence:** multi-statement SQL is still rejected (M4 raw-query check,
then SQLite), now with the "single statement" message instead of
"Forbidden pattern matched". New tests: two plain SELECTs rejected (no
forbidden keyword, so only the single-statement check can stop them), a
newline-separated pair rejected, a trailing `;` (with or without whitespace)
accepted. The reward rules are unchanged, so run 2 resumes under the same
`TRAINING_SETUP_VERSION`; from session 2 on, the few SQL calls ending in
`;` (0–6% of SQL calls in session 1) execute instead of costing the
invalid-parameter penalty.
