# MSD — M2: Task Generator

## Purpose
Generate synthetic tasks with verifiable ground truth, stratified across 5
difficulty tiers, for both training (800) and held-out evaluation (200).

## Public interface (matches shared/interfaces.md section 4)
```python
def generate_tasks(tier: int, count: int, seed: int) -> list[Task]: ...
def generate_dataset(train_count=800, test_count=200, seed=42) -> dict:
    # returns {"train": list[Task], "test": list[Task]}, no overlap between them
    ...
```

## Task tiers
| Tier | Name | Count (of 1000 total) | Teaches |
|------|------|------|---------|
| 1 | Single-tool | 200 | Basic tool identification — exactly one tool call solves it |
| 2 | Distractor | 150 | Efficiency — looks like it needs a tool but doesn't (answer via ANSWER_DIRECTLY) |
| 3 | Multi-hop | 200 | Composition — needs 2-3 tool calls in sequence |
| 4 | Ambiguous | 150 | When to ask — requires a CLARIFY action before answering |
| 5 | Error-recovery | 100 | Resilience — first tool call is designed to fail; must recover via a different tool |

Split proportionally 80/20 train/test within each tier (so all 5 tiers are
represented in both splits).

## Generation method
Jinja2 templates with variable substitution (e.g. a template like `"What is
{pct}% of {number}?"` with randomized `pct`/`number` fills). **Ground truth
must be computed programmatically** by actually executing the correct tool
chain against M1's data — never hand-typed — so correctness is guaranteed
even as templates are randomized. This means M2 has a dependency on M1's
`execute_tool` at generation time (generation-time dependency only; the
`Task` objects it produces have no runtime dependency on M1).

## Output format
JSONL, one `Task` per line, written to `data/tasks_train.jsonl` and
`data/tasks_test.jsonl`.

## Definition of Done
- [ ] All 5 tiers implemented with at least 3 distinct templates per tier
  (avoid producing near-duplicate tasks from a single template pattern).
- [ ] `generate_dataset()` produces exactly 800 train / 200 test tasks with
  correct tier proportions in both.
- [ ] Zero overlap in `id` or `query` text between train and test sets —
  test with a set-intersection assertion.
- [ ] Every generated task's `ground_truth` is verified against actually
  running the correct tool chain via M1 (write a test that does this for a
  sample of at least 20 tasks per tier).
- [ ] `pytest tests/tasks/ -v` passes, report actual count.
