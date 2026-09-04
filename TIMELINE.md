# ATAR Build Timeline

<!--
Append one entry per day of active work. Do not backdate or batch-write
entries after the fact — this document's value is that it's written
incrementally, as proof the project was actually built over time and as a
real debugging aid ("what changed the day this broke").

Template per entry:

## YYYY-MM-DD (Day N)
- Started: <module(s), which agent conversation>
- Finished: <module(s), with test pass count>
- Issues hit: <what went wrong and how it was resolved, or "none">
- Blocked: <anything waiting on architect decision, or "none">
- Next: <what's planned next>
-->

## Day 0 — Project setup
- Started: Repo initialized, AGENTS.md / interfaces.md / shared_types.py /
  MSDs placed at workspace root.
- Next: Vertical slice (solo, before fanning out to parallel agents).

## 2026-09-03 (Day 1)
- Started: Vertical slice — calculator tool, state builder, env, dummy policy, tasks, demo.py
- Finished: All 6 files in `slice/`, plus `tests/test_slice.py` (37/37 pass), `demo.py` (10/10 correct)
- Issues hit: Expression extractor regex had wrong priority order — "X + Y% of Z" was partially matched by "Y% of Z" pattern. Fixed by reordering.
- Blocked: none
- Next: Architect reviews type/shape report against interfaces.md, then fan out M0-M3.

## 2026-09-04 (Day 2)
- Started: M0 Scaffold (Directory structure, pyproject.toml, configs, logging)
- Finished: M0 Scaffold (pip install -e . works, 37/37 tests pass, demo.py passes)
- Issues hit: `pip install -e .` initially failed with system Python's pip (version 22.0.2) due to PEP 660 compatibility and Ubuntu's `dist-packages` permissions. Resolved by creating a dedicated virtual environment and upgrading pip/setuptools to support modern editable installs.
- Blocked: none
- Next: Proceed with subsequent module builds (M1, M2, etc.).
