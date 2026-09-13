# AGENTS.md — ATAR Project Rules

You are one of several AI agents building **ATAR** (Adaptive Tool-use Agent with
Reinforcement Learning): a system where a frozen 7B LLM handles language, and a
small trainable RL policy (PPO) decides which of 7 discrete actions to take
(answer directly, call one of 5 tools, or terminate). Full context is in
`/docs/project_description.md`.

## Non-negotiable rules

1. **Never modify `/shared/interfaces.md` or `/shared/shared_types.py` without
   stopping and flagging it in your response first.** These are the locked
   contract every module is built against. If your module seems to need a
   change to a shared type, describe the exact change and why, and wait for
   approval — do not just make the edit and continue.

2. **Only work within your assigned module's spec.** Your task will reference
   a specific file in `/msds/`. Read only that MSD plus `/shared/` — do not
   read or depend on another module's internal implementation, only its
   public interface as defined in `/shared/interfaces.md`.

3. **Every function must have type hints.** Every module must have a
   corresponding `test_<module>.py` using pytest, covering every item in
   that module's MSD "Definition of Done" section.

4. **Do not report a module as complete until you have actually run its
   tests and they pass.** Show the real test output (pass/fail counts) in
   your summary, not a description of what the tests are supposed to check.

4a. **Always run pytest via the project venv, never system Python.**
    Use `venv/bin/python3 -m pytest ...` (or equivalently
    `venv/bin/pytest ...`) from the repo root — never bare `pytest` or
    `/usr/bin/python3 -m pytest`, which resolve to system-site packages
    and bypass the isolated environment entirely. If `venv/` does not
    yet exist or is missing packages, run
    `python3 -m venv venv && venv/bin/pip install -e .[dev]` first.
    This is non-negotiable: a test run against system Python is not a
    valid DoD verification.

5. **No claimed results without a run to back them.** Never write "achieves
   91% accuracy" or similar performance claims in code comments, docstrings,
   or your summary unless you have actually executed the relevant evaluation
   and are reporting its real output.

6. **If an MSD is ambiguous or seems to conflict with another module's
   interface, stop and ask** rather than guessing and proceeding.

7. **After finishing a module, update three files** (create them if they
   don't exist yet, using the templates in `/templates/`):
   - `/README.md` — add a section for this module: what was built, why this
     approach was chosen over alternatives, what actually happened when you
     ran it, and any deviation from the plan.
   - `/TIMELINE.md` — append a dated entry: what was started/finished, any
     issues hit and how they were resolved, what's blocked, what's next.
   - `/ARCHITECTURE_DECISIONS.md` — only if you made a non-trivial design
     choice not already dictated by the MSD (e.g. how you structured an
     internal helper). One entry: decision, context, alternatives
     considered, consequence.

8. **Security-critical modules (M1 Tool Sandbox, M4 Action Translator)
   require extra care.** Any validation/filtering logic must fail closed
   (reject on doubt, not allow). Write tests for the *rejection* cases, not
   just the happy path.

## Hardware & execution environment (important — affects M3 and M7)

The human architect develops on a laptop with a 4GB GPU — not enough to load
Qwen2.5-7B, even 4-bit. Actual training runs happen separately, manually, on
a rented or free-tier GPU (Kaggle T4, RunPod, etc.), not inside this
Antigravity session. This means:

- **M3's LLM model name must be a config value, not hardcoded.** The
  architect needs to swap between a tiny stand-in model (e.g.
  Qwen2.5-0.5B-Instruct, which fits in 4GB, for local testing) and the real
  Qwen2.5-7B-Instruct (used only on the remote training GPU) by changing a
  config value, not editing code.
- **M7's training script must be a standalone CLI entry point**
  (`scripts/train.py` with argparse), runnable independently of this
  Antigravity workspace — it will be copied to and executed inside a Kaggle
  notebook or rented GPU instance, not run from here.
- **M7 must support resuming from a checkpoint** (model weights, optimizer
  state, curriculum phase, step count) via a command-line flag. Free-tier
  GPU sessions cap out around 12 hours; training that can't resume risks
  losing significant progress on session cutoff.
- Local Definition-of-Done testing for M3/M7 uses the tiny stand-in model —
  this proves the code path works, not that training will succeed on the
  real model. Note this distinction explicitly in test names/comments
  (e.g. `test_encode_smoke` vs anything claiming real-model behavior).

## Tech stack (do not substitute without flagging)

PyTorch, HuggingFace Transformers, Accelerate + BitsAndBytes (4-bit
quantization), Gymnasium, Outlines (constrained JSON generation), Pydantic,
Jinja2, rank-bm25, SQLite, Weights & Biases, pytest + pytest-cov, Streamlit,
ruff + black.

## Communication style

When you finish a task, structure your summary as: what you built, what you
tested and the actual results, any deviations from the MSD and why, and what
(if anything) you need from the human architect before this is truly done.
