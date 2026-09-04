# MSD — M0: Scaffold

## Purpose
Set up the shared project skeleton every other module builds inside. This is
the only module with no dependencies, and every other module depends on it.

## Scope
- Directory structure:
  ```
  atar/
    shared/           # interfaces.md, shared_types.py (already provided — do not regenerate)
    configs/           # action_registry.yaml, training_config.yaml, task_config.yaml
    tools/              # M1
    tasks/              # M2
    llm/                 # M3
    translator/       # M4
    env/                 # M5
    optimizer/        # M6
    policy/             # M7
    demo/                # M8
    tests/               # mirrors module structure
    scripts/            # train.py, evaluate.py, generate_tasks.py entry points
  ```
- `configs/base_config.py` — a Pydantic `BaseSettings` class for global config
  (paths, random seed, device, log level), loadable from environment
  variables or a `.env` file.
- `logging_setup.py` — a single `get_logger(name)` function used by every
  module, configured to log to both console and `logs/atar.log`.
- `pyproject.toml` — dependency list matching the tech stack in AGENTS.md,
  pinned versions, ruff + black config.
- A skeleton `configs/action_registry.yaml` with all 7 actions listed with
  empty fields (M4 fills in the actual content).
- `.github/workflows/test.yml` (optional but recommended) — run pytest on
  every push.

## Inputs / Outputs
No runtime inputs/outputs — this module produces project structure and
config-loading utilities, not runtime behavior.

## Error handling
`base_config.py` should raise a clear `ConfigError` (not a generic
exception) if required environment variables are missing, naming exactly
which one.

## Definition of Done
- [ ] Directory structure exists exactly as specified above.
- [ ] `pyproject.toml` installs cleanly with `pip install -e .`
- [ ] `from atar.shared.shared_types import ActionType, Task, ToolResult` works
  from any subdirectory.
- [ ] `get_logger("test")` produces console + file output.
- [ ] `pytest` runs (even with zero tests collected) without import errors.
- [ ] README.md and TIMELINE.md exist at root, initialized from the
  templates provided.
