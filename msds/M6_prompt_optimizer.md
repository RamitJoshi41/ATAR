# MSD — M6: Prompt Optimizer

## Purpose
Empirically improve M4's registry prompts based on observed training
failures, rather than guessing at improvements.

## Dependency
Needs M4 (to modify registry entries) and M7 (to read failure logs from a
completed or in-progress training run). Build this after M7 has produced at
least one checkpoint and a failure log to analyze — don't build against
synthetic/imagined failure data.

## Public interface
```python
def analyze_failures(log_path: str) -> dict[ActionType, list[FailureCase]]:
    ...

def propose_variant(action: ActionType, failures: list[FailureCase]) -> RegistryEntry:
    # produces a candidate replacement registry entry — more examples,
    # stronger constraints, or clearer instructions, based on the observed
    # failure pattern
    ...

def ab_test(action: ActionType, original: RegistryEntry, variant: RegistryEntry,
            held_out_tasks: list[Task], n_trials: int = 50) -> ABTestResult:
    # runs n_trials on held-out tasks with each registry entry, returns
    # comparative success rates
    ...
```

## Method (implement exactly as specified)
1. Read training failure logs (from M7's checkpointing/logging).
2. Categorize failures by action type and error mode (e.g. "SQL_QUERY:
   malformed JSON" vs "SQL_QUERY: wrong table referenced").
3. Generate a proposed prompt variant per category (more few-shot examples,
   stronger constraint wording, clearer instructions).
4. A/B test the variant against the original on 50 held-out tasks.
5. Adopt the variant only if it improves success rate by ≥5 percentage
   points — otherwise discard it and log why.

## Definition of Done
- [ ] `analyze_failures()` correctly categorizes a synthetic log fixture
  with known failure types (write the fixture yourself for testing).
- [ ] `ab_test()` runs both variants on the same held-out task set and
  reports success rates with a stated sample size.
- [ ] At least one real optimization cycle run against actual M7 training
  logs (not just fixtures), with the outcome (adopted or discarded, and
  why) recorded in README.md.
- [ ] `pytest tests/optimizer/ -v` passes, report actual count.
