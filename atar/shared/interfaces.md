# Shared Interfaces — ATAR

This is the locked contract every module is built against. Nothing here
changes without the human architect's explicit approval. All types are
defined in `shared_types.py` in this same folder; this document explains
the *why* behind each shape.

## 1. Action space — `ActionType` (Discrete(7))

| id | name | meaning |
|----|------|---------|
| 0 | ANSWER_DIRECTLY | Use the LLM's internal knowledge, no tool call |
| 1 | SEARCH_WEB | BM25 retrieval over Wikipedia abstracts |
| 2 | CALCULATOR | AST-safe arithmetic evaluation |
| 3 | SQL_QUERY | SELECT-only query against synthetic SQLite tables |
| 4 | PYTHON_EXEC | Restricted, timed, import-filtered Python execution |
| 5 | CLARIFY | No-op placeholder — ask the (simulated) user a question |
| 6 | TERMINATE | End the episode and return the current answer as final |

## 2. Observation space — `State` (Box(512,))

The 512-d vector is partitioned into four fixed regions. Order matters —
downstream code slices by index range, not by name.

| Range | Size | Content |
|-------|------|---------|
| [0:384] | 384 | Semantic content — mean-pooled last hidden state of the frozen LLM over the conversation, passed through a trainable Linear(hidden_dim, 384) projection |
| [384:391] | 7 | Binary tool-history flags — one bit per action type, 1 if already used this episode |
| [391:455] | 64 | Outcome embedding — encodes the result of the most recent tool call (success/fail, result type) |
| [455:457] | 2 | Budget/cost features — turns remaining (normalized), cumulative token cost (normalized) |
| [457:512] | 55 | Reserved padding, zero-filled — do not repurpose without architect approval |

## 3. Core data types (see `shared_types.py` for exact code)

- **`ToolResult`** — `success: bool`, `output: Any`, `error: str | None`,
  `latency_ms: float`, `token_cost: int`. Every tool in M1 returns exactly
  this shape, no exceptions raised across the module boundary.
- **`Task`** — `id: str`, `tier: int (1-5)`, `query: str`,
  `required_tools: list[ActionType]`, `ground_truth: str`,
  `metadata: dict`. Produced by M2, consumed by M5.
- **`RewardComponents`** — `accuracy: float`, `efficiency: float`,
  `safety: float`, `bonus: float`, `total: float`. Computed by M5 each step;
  logged individually (not just `total`) so training curves can be
  decomposed by reward source.
- **`RegistryEntry`** — one per action in `configs/action_registry.yaml`:
  `system_suffix: str`, `output_schema: dict`, `few_shot_examples: list[str]`,
  `constraint_rules: list[str]`, `forbidden_patterns: list[str]`. Used by M4.

## 4. Key function signatures modules must implement

```python
# M1 — Tool Sandbox
def execute_tool(action: ActionType, params: dict) -> ToolResult: ...

# M2 — Task Generator
def generate_tasks(tier: int, count: int, seed: int) -> list[Task]: ...

# M3 — LLM Interface
def encode(messages: list[dict]) -> torch.Tensor:  # returns (384,) before concat
    ...
def generate(messages: list[dict], json_schema: dict) -> dict: ...

# M4 — Action Translator
def translate(action: ActionType, state: State) -> ConditionedPrompt: ...
def validate(action: ActionType, raw_params: dict) -> tuple[bool, dict | str]:
    # returns (True, validated_params) or (False, error_message)
    ...

# M5 — Gymnasium Environment
class ATAREnv(gymnasium.Env):
    def reset(self, task: Task | None = None) -> tuple[State, dict]: ...
    def step(self, action: ActionType) -> tuple[State, float, bool, bool, dict]:
        # returns (next_state, reward, terminated, truncated, info)
        # reward is RewardComponents.total (a plain float) — required for
        # gymnasium.utils.env_checker.check_env() compliance and for standard
        # RL tooling to work unmodified. The full breakdown is available at
        # info["reward_components"] (a RewardComponents instance) for
        # per-component logging (confirmed via vertical slice, ADR-3).
        ...

# M7 — RL Policy
class ATARPolicy(nn.Module):
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (action_logits[7], value_estimate[1])
        ...
```

## 5. Cross-module rule

No module calls another module's private/internal functions — only the
signatures listed above (or defined in each module's own MSD as "public
interface"). If M5 needs something from M1 that isn't listed here, that's a
signal the interface is incomplete — flag it, don't route around it.
