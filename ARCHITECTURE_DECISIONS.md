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

## ADR-3: RewardComponents returned from step(), not bare float
**Date:** 2026-09-03
**Decision:** `env.step()` returns `RewardComponents` (a dataclass with
`accuracy`, `efficiency`, `safety`, `bonus`, `total`) instead of
Gymnasium's standard `float`.
**Context:** interfaces.md section 4 specifies the M5 return type as
`RewardComponents`, and the spec requires logging each component
individually for training-curve decomposition.
**Alternatives considered:** Return `float` and log components via `info`
dict — compatible with standard Gymnasium wrappers but loses type safety.
**Consequence:** PPO rollout buffer (M7) will need to store the full
dataclass or convert to float for the value loss. Worth it for
debuggability.
## ADR-4: Deterministic override in M1 execute_tool
**Date:** 2026-09-04
**Decision:** Added an optional `deterministic: bool = False` flag to M1's `execute_tool` signature (and internally to `_search_web`).
**Context:** M2 Task Generator must compute ground truth programmatically via M1. However, M1's `_search_web` injects noise 20% of the time. We needed a way for M2 to get reliable, noise-free truth without stripping the stochastic behavior required for RL agent training.
**Alternatives considered:** 1) Re-running the tool in a loop until it succeeded (brittle). 2) Globally patching Python's `random` module during task generation (dangerous side-effects, test leakage).
**Consequence:** A minor change to `interfaces.md` that safely isolates task generation requirements from RL training requirements.
