# Testing & Completion Rules

Place this file at `.agents/rules/testing.md` in the workspace root.

## Before reporting any module complete
1. Every item in that module's MSD "Definition of Done" checklist must be
   checked off with actual evidence (test output), not just described.
2. Run `pytest tests/<module>/ -v` and paste the real summary line (e.g.
   `14 passed, 0 failed`) into your response. Do not paraphrase this as
   "all tests pass."
3. Update `/README.md`: add or update this module's section (what, why,
   what actually happened, deviations).
4. Update `/TIMELINE.md`: append a dated entry per the template in
   `/templates/TIMELINE_template.md`.
5. If you made a non-trivial design choice not dictated by the MSD, add an
   entry to `/ARCHITECTURE_DECISIONS.md`.

## If you get stuck or the MSD is ambiguous
Stop. Do not guess and proceed silently. State exactly what's ambiguous and
what your best-guess interpretation would be, and wait for the human
architect's response.

## If a shared interface seems wrong or incomplete
Do not modify `/shared/interfaces.md` or `/shared/shared_types.py`
yourself. Describe the exact change you believe is needed, why, and which
other modules it might affect, then wait for approval.

## Security-critical modules (M1, M4)
Write and run explicit rejection-path tests (malicious/malformed input),
not just happy-path tests. A security module with only happy-path tests is
not considered done.
