# MSD — M5: Gymnasium Environment

## Purpose
Wrap M1 (tools) + M3 (LLM) + M4 (translator) into a standard Gymnasium
environment interface, so M7's PPO trainer can interact with it generically.

**Dependency note:** this module is the highest integration risk in the
project — it's where M1/M3/M4's interfaces actually get exercised together
for the first time. Before starting, re-read all three modules' MSDs and
confirm their actual (not just planned) function signatures match
shared/interfaces.md. If any drifted during implementation, flag it before
writing M5 against the drifted version.

## Public interface (matches shared/interfaces.md section 4)
```python
class ATAREnv(gymnasium.Env):
    observation_space: gymnasium.spaces.Box  # shape (512,)
    action_space: gymnasium.spaces.Discrete  # n=7

    def reset(self, task: Task | None = None) -> tuple[State, dict]:
        # if task is None, sample randomly from the loaded task set
        ...

    def step(self, action: ActionType) -> tuple[State, float, bool, bool, dict]:
        # returns (next_state, reward, terminated, truncated, info)
        # reward MUST be a plain float (RewardComponents.total) — confirmed
        # via the vertical slice (ADR-3): returning the dataclass directly
        # breaks gymnasium's check_env() and standard RL tooling. Put the
        # full breakdown in info["reward_components"] instead.
        ...
```

## Episode mechanics
- Max episode length: 5 turns (a "turn" = one step() call). Exceeding this
  triggers `truncated=True` with no accuracy bonus.
- `terminated=True` when the policy selects `TERMINATE`, or when a final
  answer is produced.
- `info` dict must include `info["reward_components"]` (the full
  `RewardComponents` instance, not just the total) and the raw `ToolResult`
  (if any) — this is how per-component logging happens without breaking
  the standard scalar-reward contract.

## Reward function (4 components — implement exactly, these are tuned)
1. **Accuracy**: +10 correct final answer, -5 incorrect final answer, 0 if
   episode truncated without an answer.
2. **Efficiency**: -0.5 per tool call, -0.1 per 1,000 tokens consumed, -1.0
   extra for a redundant tool call (same action already used successfully
   this episode with no new information since).
3. **Safety**: -10 for invalid/rejected parameters (from M4's `validate()`
   returning False), -3 for a tool runtime error, +1 for correctly
   recognizing and executing a CLARIFY action on an ambiguous task.
4. **Bonus**: +2 for a correct final answer using ≤1 tool call total, +1 for
   successfully recovering after an earlier tool failure in the same
   episode.

Compute all four components every step, store the full `RewardComponents`
in `info["reward_components"]`, and return `.total` as the scalar reward —
this is required for the ablations planned later, and for check_env()
compliance.

## State composition
Combine M3's `encode()` output (384-d) with tool-history flags (7-d,
tracked by this module), an outcome embedding (64-d, encode the most recent
`ToolResult` — success flag + a small hash/embedding of the output type),
budget features (2-d: turns remaining / 5, cumulative tokens / a fixed
normalizer), and 55-d zero padding, into the full 512-d vector per
shared/interfaces.md section 2.

## Definition of Done
- [ ] `ATAREnv().reset()` returns a `(512,)` state and an info dict; runs
  without error for at least 20 different sampled tasks across all 5 tiers.
- [ ] One full episode (`reset` → repeated `step` until `terminated` or
  `truncated`) runs end-to-end for a Tier 1 (single-tool) task, calling real
  M1/M3/M4 code (not mocks), and produces a sensible reward.
- [ ] Reward components tested individually: a test asserting a correct
  Tier 1 answer with 1 tool call yields the expected accuracy + efficiency +
  bonus, computed by hand and compared to the environment's output.
- [ ] Max-turns truncation tested explicitly (force an episode past 5 steps,
  confirm `truncated=True` and no accuracy bonus awarded).
- [ ] `gymnasium.utils.env_checker.check_env(ATAREnv())` passes with no
  warnings.
- [ ] `pytest tests/env/ -v` passes, report actual count.
