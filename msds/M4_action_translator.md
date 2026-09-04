# MSD — M4: Action Translator

## Purpose
Convert a discrete RL action into a conditioned LLM prompt, and validate
the LLM's resulting parameters before they reach the Tool Sandbox. Security
critical — see AGENTS.md rule 8.

## Public interface (matches shared/interfaces.md section 4)
```python
def translate(action: ActionType, state: State) -> ConditionedPrompt: ...
def validate(action: ActionType, raw_params: dict) -> tuple[bool, dict | str]:
    # (True, validated_params) on success, (False, error_message) on failure
    ...
```

## Registry
Reads `configs/action_registry.yaml`, one `RegistryEntry` per action (see
shared_types.py). Fields per entry:
- `system_suffix` — instruction text injected for this action
- `output_schema` — JSONSchema the LLM's output must satisfy
- `few_shot_examples` — 1-3 example (input, output) pairs for actions where
  the LLM tends to struggle (SQL_QUERY and PYTHON_EXEC especially benefit)
- `constraint_rules` — human-readable list of post-generation checks to run
- `forbidden_patterns` — regexes to reject before even attempting execution

Populate this registry for all 7 actions as part of this module's work —
it isn't provided pre-filled by M0.

## Validation responsibilities (this is the security boundary)
`validate()` must catch, before anything reaches M1:
- SQL injection patterns (anything besides a single well-formed SELECT)
- Forbidden Python imports (cross-check against M1's tool sandbox
  allow-list so the two modules agree — import the same list from a shared
  config, don't hardcode it twice)
- Malformed JSON / schema violations (missing required keys, wrong types)
- Empty or whitespace-only expressions for the calculator

On any failure, return `(False, "<specific reason>")` — never raise, never
pass through with a warning.

## Definition of Done
- [ ] `action_registry.yaml` has complete, non-empty entries for all 7 actions.
- [ ] `translate()` produces a `ConditionedPrompt` for every action type —
  test all 7.
- [ ] `validate()` rejection tests: SQL injection string, forbidden Python
  import, malformed JSON, missing required schema key — each must return
  `(False, ...)`, not raise.
- [ ] `validate()` acceptance tests: one valid example per action type
  returns `(True, params)` with `params` matching the schema.
- [ ] Cross-check test: the forbidden-imports list used here is imported
  from the same source as M1's, not duplicated (test that changing one
  location changes both, or that both reference a single shared config).
- [ ] `pytest tests/translator/ -v` passes, report actual count.
