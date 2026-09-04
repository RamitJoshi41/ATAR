# MSD — M1: Tool Sandbox

## Purpose
Provide deterministic, safe execution for 5 simulated tools. This is a
security-critical module — see AGENTS.md rule 8: fail closed, test the
rejection paths as thoroughly as the happy paths.

## Public interface (matches shared/interfaces.md section 4)
```python
def execute_tool(action: ActionType, params: dict) -> ToolResult: ...
```
Internally, one private function per tool is fine, but `execute_tool` is the
only entry point other modules should call.

## Tools to implement

**`search_web`** — BM25 retrieval (via `rank-bm25`) over a local Wikipedia
abstracts corpus (~a few thousand articles is enough; do not attempt to
download full Wikipedia). Inject synthetic noise: 20% of the time, return a
plausible-but-irrelevant result instead of the true top match, to teach the
policy that search isn't always reliable. `params`: `{"query": str}`.

**`calculator`** — Evaluate arithmetic expressions. Do NOT use `eval()`.
Parse with Python's `ast` module, walk the tree, and only allow
`ast.Num`/`ast.Constant`, `ast.BinOp`, `ast.UnaryOp`, and the operators
`+ - * / ** ()`. Reject anything else (names, calls, strings, attribute
access) with a `ToolResult(success=False, error="disallowed expression")`.
`params`: `{"expression": str}`.

**`sql_query`** — Execute against an in-memory SQLite DB pre-loaded with 5
synthetic tables (e.g. employees, departments, salaries — design a small
coherent schema). Whitelist: the query string must match `^\s*SELECT`
case-insensitively after stripping; reject anything containing
`DROP|DELETE|INSERT|UPDATE|ATTACH|PRAGMA` (case-insensitive). `params`:
`{"query": str}`.

**`python_exec`** — Restricted Python execution. Parse with `ast` first and
reject any `Import`/`ImportFrom` node referencing `os, sys, subprocess,
socket, urllib, shutil` (or any module not on an explicit allow-list of
`math, statistics, json, re, collections`). Run in a subprocess with a
2-second wall-clock timeout (`signal.alarm` or `subprocess` timeout param —
prefer subprocess for true isolation). `params`: `{"code": str}`.

**`clarify`** — No-op placeholder. Returns
`ToolResult(success=True, output=<simulated user response from a small
template bank keyed by task metadata>, error=None, latency_ms=0,
token_cost=0)`. The task's `metadata` field (from M2) should carry what the
clarification answer ought to be.

## Output contract
Every tool call returns exactly a `ToolResult` — never raises an exception
across the module boundary. Internal exceptions are caught and converted to
`ToolResult(success=False, error=str(exc), ...)`.

## Definition of Done
- [ ] All 5 tools implemented behind `execute_tool`.
- [ ] Calculator: unit tests proving `os.system`, `__import__`, function
  calls, and string literals are all rejected, not just that `2+2` works.
- [ ] Python exec: unit test proving `import os` is rejected, and a
  deliberately infinite loop (`while True: pass`) times out within 3 seconds
  wall-clock rather than hanging the test suite.
- [ ] SQL: unit test proving `DROP TABLE` and `'; DELETE FROM` style
  injection attempts are rejected.
- [ ] search_web: unit test confirming noise injection triggers on
  approximately 20% of calls across 100 trials (statistical, not exact).
- [ ] Every tool has at least one success-path and one failure-path test.
- [ ] `pytest tests/tools/ -v` — 100% pass, report actual count in summary.
