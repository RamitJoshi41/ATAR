"""
Tests for M4 — Action Translator
==================================
Covers every item in the MSD Definition of Done:

DoD-1  action_registry.yaml has complete, non-empty entries for all 7 actions.
DoD-2  translate() produces a ConditionedPrompt for every action type — all 7.
DoD-3  validate() rejection tests: SQL injection, forbidden Python import,
       malformed JSON structure, missing required schema key.
DoD-4  validate() acceptance tests: one valid example per action type returns
       (True, params).
DoD-5  Cross-check: PYTHON_ALLOWED_IMPORTS is imported from the same object as
       M1's sandbox uses (not duplicated).
DoD-6  pytest tests/translator/ -v passes (shown by running this file).

Security note (AGENTS.md rule 8): rejection tests are explicit and cover
adversarial inputs — the happy-path tests alone are insufficient for a
security-critical module.
"""

from __future__ import annotations

import pytest

from atar.shared.shared_types import ActionType, ConditionedPrompt
from atar.translator.action_translator import (
    _get_registry,
    translate,
    validate,
)
from atar.tools.sandbox import PYTHON_ALLOWED_IMPORTS as SANDBOX_ALLOWED_IMPORTS
from atar.translator.action_translator import PYTHON_ALLOWED_IMPORTS as M4_ALLOWED_IMPORTS


# ===========================================================================
# DoD-1: Registry completeness
# ===========================================================================

class TestRegistryCompleteness:
    """action_registry.yaml has complete, non-empty entries for all 7 actions."""

    def test_all_seven_actions_present(self) -> None:
        registry = _get_registry()
        assert set(registry.keys()) == set(ActionType), (
            f"Registry is missing actions: {set(ActionType) - set(registry.keys())}"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_system_suffix_non_empty(self, action: ActionType) -> None:
        entry = _get_registry()[action]
        assert entry.system_suffix.strip(), (
            f"{action.name}: system_suffix must not be empty"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_output_schema_non_empty(self, action: ActionType) -> None:
        entry = _get_registry()[action]
        assert entry.output_schema, (
            f"{action.name}: output_schema must not be empty"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_output_schema_has_required_keys(self, action: ActionType) -> None:
        entry = _get_registry()[action]
        schema = entry.output_schema
        assert "type" in schema, f"{action.name}: schema must have 'type'"
        assert "properties" in schema, f"{action.name}: schema must have 'properties'"
        assert "required" in schema, f"{action.name}: schema must have 'required'"

    @pytest.mark.parametrize("action", list(ActionType))
    def test_constraint_rules_present(self, action: ActionType) -> None:
        entry = _get_registry()[action]
        assert isinstance(entry.constraint_rules, list), (
            f"{action.name}: constraint_rules must be a list"
        )


# ===========================================================================
# DoD-2: translate() produces ConditionedPrompt for every action
# ===========================================================================

class TestTranslate:
    """translate() produces a ConditionedPrompt for every ActionType."""

    import numpy as np
    _DUMMY_STATE = np.zeros(512)

    @pytest.mark.parametrize("action", list(ActionType))
    def test_returns_conditioned_prompt(self, action: ActionType) -> None:
        import numpy as np
        result = translate(action, np.zeros(512))
        assert isinstance(result, ConditionedPrompt), (
            f"{action.name}: translate() must return ConditionedPrompt"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_action_field_matches(self, action: ActionType) -> None:
        import numpy as np
        result = translate(action, np.zeros(512))
        assert result.action == action, (
            f"{action.name}: ConditionedPrompt.action must match input"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_system_message_non_empty(self, action: ActionType) -> None:
        import numpy as np
        result = translate(action, np.zeros(512))
        assert result.system_message.strip(), (
            f"{action.name}: system_message must not be empty"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_json_schema_is_dict(self, action: ActionType) -> None:
        import numpy as np
        result = translate(action, np.zeros(512))
        assert isinstance(result.json_schema, dict), (
            f"{action.name}: json_schema must be a dict"
        )

    @pytest.mark.parametrize("action", list(ActionType))
    def test_action_name_in_system_message(self, action: ActionType) -> None:
        """The action name should appear somewhere in the system message."""
        import numpy as np
        result = translate(action, np.zeros(512))
        assert action.name in result.system_message, (
            f"{action.name}: action name should appear in system_message"
        )


# ===========================================================================
# DoD-3: validate() REJECTION tests (security-critical — fail-closed)
# ===========================================================================

class TestValidateRejections:
    """
    Every test here must return (False, <reason_string>) — never raise,
    never return (True, ...).  AGENTS.md rule 8 requires explicit rejection
    tests for security-critical modules.
    """

    # ---- SQL injection attempts -------------------------------------------

    def test_sql_injection_drop_table(self) -> None:
        ok, reason = validate(ActionType.SQL_QUERY, {"query": "DROP TABLE employees"})
        assert ok is False
        assert isinstance(reason, str) and reason

    def test_sql_injection_delete(self) -> None:
        ok, reason = validate(ActionType.SQL_QUERY, {"query": "DELETE FROM employees WHERE 1=1"})
        assert ok is False
        assert isinstance(reason, str)

    def test_sql_injection_insert(self) -> None:
        ok, reason = validate(
            ActionType.SQL_QUERY,
            {"query": "INSERT INTO employees VALUES (99, 'hacker', 1, 999999)"},
        )
        assert ok is False

    def test_sql_injection_update(self) -> None:
        ok, reason = validate(
            ActionType.SQL_QUERY,
            {"query": "UPDATE employees SET salary = 999999 WHERE 1=1"},
        )
        assert ok is False

    def test_sql_inline_comment_injection(self) -> None:
        """A query with inline comments should be rejected (bypass attempt)."""
        ok, reason = validate(
            ActionType.SQL_QUERY,
            {"query": "SELECT * FROM employees -- DROP TABLE employees"},
        )
        assert ok is False
        # The query may be caught by the registry's forbidden_patterns (DROP keyword)
        # or by the inline-comment check — either way it must be rejected.
        assert isinstance(reason, str) and reason

    def test_sql_multi_statement_injection(self) -> None:
        ok, reason = validate(
            ActionType.SQL_QUERY,
            {"query": "SELECT * FROM employees; DROP TABLE employees"},
        )
        assert ok is False

    def test_sql_non_select(self) -> None:
        ok, reason = validate(ActionType.SQL_QUERY, {"query": "PRAGMA table_info(employees)"})
        assert ok is False

    def test_sql_attach_injection(self) -> None:
        ok, reason = validate(
            ActionType.SQL_QUERY,
            {"query": "ATTACH DATABASE '/etc/passwd' AS shadow"},
        )
        assert ok is False

    # ---- Forbidden Python imports ------------------------------------------

    def test_python_forbidden_import_os(self) -> None:
        ok, reason = validate(ActionType.PYTHON_EXEC, {"code": "import os\nprint(os.listdir('/'))" })
        assert ok is False
        # May be caught by registry pattern or AST visitor — both say Forbidden
        assert "orbidden" in reason  # 'Forbidden' from either source

    def test_python_forbidden_import_sys(self) -> None:
        ok, reason = validate(ActionType.PYTHON_EXEC, {"code": "import sys\nprint(sys.argv)"})
        assert ok is False
        assert "orbidden" in reason  # pattern or AST visitor both say Forbidden

    def test_python_forbidden_import_subprocess(self) -> None:
        ok, reason = validate(
            ActionType.PYTHON_EXEC,
            {"code": "import subprocess\nsubprocess.run(['ls', '-la'])"},
        )
        assert ok is False

    def test_python_forbidden_import_from_os(self) -> None:
        ok, reason = validate(
            ActionType.PYTHON_EXEC,
            {"code": "from os import path\nprint(path.exists('/etc/passwd'))"},
        )
        assert ok is False

    def test_python_forbidden_eval_call(self) -> None:
        ok, reason = validate(ActionType.PYTHON_EXEC, {"code": "x = eval('1+1')"})
        assert ok is False
        assert "orbidden" in reason  # pattern or AST visitor both say Forbidden

    def test_python_forbidden_exec_call(self) -> None:
        ok, reason = validate(ActionType.PYTHON_EXEC, {"code": "exec('import os')"})
        assert ok is False

    def test_python_forbidden_open_call(self) -> None:
        ok, reason = validate(ActionType.PYTHON_EXEC, {"code": "f = open('/etc/passwd')"})
        assert ok is False

    def test_python_forbidden_dunder_import(self) -> None:
        ok, reason = validate(
            ActionType.PYTHON_EXEC,
            {"code": "os = __import__('os')\nprint(os.getcwd())"},
        )
        assert ok is False

    # ---- Malformed / invalid structure ------------------------------------

    def test_malformed_non_dict_params(self) -> None:
        """raw_params is not even a dict."""
        ok, reason = validate(ActionType.ANSWER_DIRECTLY, "this is a string")  # type: ignore[arg-type]
        assert ok is False

    def test_malformed_empty_dict(self) -> None:
        """Empty dict is missing all required keys."""
        ok, reason = validate(ActionType.ANSWER_DIRECTLY, {})
        assert ok is False

    def test_malformed_wrong_type_for_key(self) -> None:
        """answer must be a string, not an integer."""
        ok, reason = validate(ActionType.ANSWER_DIRECTLY, {"answer": 42, "confidence": "high"})
        assert ok is False

    # ---- Missing required schema keys ------------------------------------

    def test_missing_required_key_answer_directly(self) -> None:
        ok, reason = validate(ActionType.ANSWER_DIRECTLY, {"confidence": "high"})
        assert ok is False
        assert "answer" in reason

    def test_missing_required_key_search_web(self) -> None:
        ok, reason = validate(ActionType.SEARCH_WEB, {})
        assert ok is False
        assert "query" in reason

    def test_missing_required_key_calculator(self) -> None:
        ok, reason = validate(ActionType.CALCULATOR, {})
        assert ok is False
        assert "expression" in reason

    def test_missing_required_key_sql_query(self) -> None:
        ok, reason = validate(ActionType.SQL_QUERY, {})
        assert ok is False
        assert "query" in reason

    def test_missing_required_key_python_exec(self) -> None:
        ok, reason = validate(ActionType.PYTHON_EXEC, {})
        assert ok is False
        assert "code" in reason

    def test_missing_required_key_clarify(self) -> None:
        ok, reason = validate(ActionType.CLARIFY, {})
        assert ok is False
        assert "question" in reason

    def test_missing_required_key_terminate(self) -> None:
        ok, reason = validate(ActionType.TERMINATE, {"final_answer": "42"})
        assert ok is False
        assert "reasoning" in reason

    # ---- Calculator-specific rejections -----------------------------------

    def test_calculator_empty_expression(self) -> None:
        ok, reason = validate(ActionType.CALCULATOR, {"expression": "   "})
        assert ok is False

    def test_calculator_function_call_injection(self) -> None:
        ok, reason = validate(ActionType.CALCULATOR, {"expression": "math.sqrt(4)"})
        assert ok is False

    def test_calculator_import_injection(self) -> None:
        ok, reason = validate(ActionType.CALCULATOR, {"expression": "import os"})
        assert ok is False

    # ---- Search-specific rejections ----------------------------------------

    def test_search_empty_query(self) -> None:
        ok, reason = validate(ActionType.SEARCH_WEB, {"query": ""})
        assert ok is False

    def test_search_query_too_long(self) -> None:
        ok, reason = validate(ActionType.SEARCH_WEB, {"query": "x" * 201})
        assert ok is False


# ===========================================================================
# DoD-4: validate() ACCEPTANCE tests (one valid example per action)
# ===========================================================================

class TestValidateAcceptance:
    """One valid example per action type returns (True, params)."""

    def test_answer_directly_valid(self) -> None:
        params = {"answer": "Paris", "confidence": "high"}
        ok, result = validate(ActionType.ANSWER_DIRECTLY, params)
        assert ok is True
        assert result == params

    def test_search_web_valid(self) -> None:
        params = {"query": "capital of France"}
        ok, result = validate(ActionType.SEARCH_WEB, params)
        assert ok is True
        assert result == params

    def test_calculator_valid_simple(self) -> None:
        params = {"expression": "3 + 4 * 2"}
        ok, result = validate(ActionType.CALCULATOR, params)
        assert ok is True
        assert result == params

    def test_calculator_valid_power(self) -> None:
        params = {"expression": "2 ** 10"}
        ok, result = validate(ActionType.CALCULATOR, params)
        assert ok is True

    def test_sql_query_valid_select(self) -> None:
        params = {"query": "SELECT name FROM employees WHERE salary > 90000"}
        ok, result = validate(ActionType.SQL_QUERY, params)
        assert ok is True
        assert result == params

    def test_sql_query_valid_aggregate(self) -> None:
        params = {"query": "SELECT SUM(salary) AS total FROM employees"}
        ok, result = validate(ActionType.SQL_QUERY, params)
        assert ok is True

    def test_python_exec_valid_math_import(self) -> None:
        params = {"code": "import math\nprint(math.sqrt(16))"}
        ok, result = validate(ActionType.PYTHON_EXEC, params)
        assert ok is True
        assert result == params

    def test_python_exec_valid_statistics(self) -> None:
        params = {"code": "import statistics\nprint(statistics.mean([1, 2, 3, 4, 5]))"}
        ok, result = validate(ActionType.PYTHON_EXEC, params)
        assert ok is True

    def test_python_exec_valid_re(self) -> None:
        params = {"code": "import re\nprint(len(re.findall(r'[aeiou]', 'hello world')))"}
        ok, result = validate(ActionType.PYTHON_EXEC, params)
        assert ok is True

    def test_clarify_valid(self) -> None:
        params = {"question": "Which city are you asking about?", "clarification_answer": ""}
        ok, result = validate(ActionType.CLARIFY, params)
        assert ok is True
        assert result == params

    def test_terminate_valid(self) -> None:
        params = {"final_answer": "42", "reasoning": "The calculator returned an unambiguous result."}
        ok, result = validate(ActionType.TERMINATE, params)
        assert ok is True
        assert result == params


# ===========================================================================
# DoD-5: Cross-check — shared allow-list is the SAME object/source
# ===========================================================================

class TestAllowListCrossCheck:
    """
    The forbidden-imports list used in validate() must come from the SAME source
    as M1's sandbox uses (AGENTS.md / MSD DoD-5).

    This test verifies that:
      1. Both modules import from atar.tools.sandbox.PYTHON_ALLOWED_IMPORTS.
      2. They are the identical frozenset object (same id in memory, since Python
         caches module-level constants in sys.modules).
      3. A mutation to one is not possible (frozenset is immutable), guaranteeing
         they cannot silently diverge.
    """

    def test_same_object_identity(self) -> None:
        """
        Both M4 and sandbox reference the same frozenset object.
        Python module imports are cached, so re-importing yields the same id.
        """
        assert M4_ALLOWED_IMPORTS is SANDBOX_ALLOWED_IMPORTS, (
            "PYTHON_ALLOWED_IMPORTS in M4 and M1 sandbox must be the identical "
            "object, not two separate copies."
        )

    def test_same_contents(self) -> None:
        """Belt-and-suspenders: even if identity check passes, contents match."""
        assert M4_ALLOWED_IMPORTS == SANDBOX_ALLOWED_IMPORTS

    def test_is_frozenset(self) -> None:
        """frozenset guarantees the constant cannot be mutated at runtime."""
        assert isinstance(SANDBOX_ALLOWED_IMPORTS, frozenset), (
            "PYTHON_ALLOWED_IMPORTS should be a frozenset so it cannot be mutated."
        )
        assert isinstance(M4_ALLOWED_IMPORTS, frozenset)

    def test_known_allowed_modules_present(self) -> None:
        """Smoke-check: the expected modules are in the allow-list."""
        expected = {"math", "statistics", "json", "re", "collections"}
        assert expected == set(SANDBOX_ALLOWED_IMPORTS), (
            f"Allow-list mismatch: expected {expected}, got {set(SANDBOX_ALLOWED_IMPORTS)}"
        )

    def test_validate_uses_shared_list(self) -> None:
        """
        Empirical proof: a module that IS in the allow-list is accepted, and a
        module that ISN'T is rejected.  If M4 had a separate hardcoded list,
        adding a new module to sandbox's list but not M4's would cause this
        test to fail, surfacing the divergence.
        """
        # 'math' is in both lists → should accept
        ok, _ = validate(ActionType.PYTHON_EXEC, {"code": "import math\nprint(math.pi)"})
        assert ok is True, "'math' should be in the shared allow-list"

        # 'os' is in neither list → should reject
        ok2, reason2 = validate(ActionType.PYTHON_EXEC, {"code": "import os\nprint('hi')"})
        assert ok2 is False, "'os' should not be in the shared allow-list"
        assert "orbidden" in reason2  # pattern or AST visitor both say Forbidden

    def test_ast_validator_fires_independently_of_patterns(self) -> None:
        """
        Isolation test: verify the AST import check via PYTHON_ALLOWED_IMPORTS
        fires WITHOUT the registry forbidden_patterns pre-filter catching it first.

        'fractions' is NOT named in the PYTHON_EXEC forbidden_patterns regex
        (which lists only: os|sys|subprocess|socket|shutil|pathlib|importlib|
        builtins|ctypes|pickle).  It IS absent from PYTHON_ALLOWED_IMPORTS
        ({math, statistics, json, re, collections}).

        Therefore:
          - The regex pre-filter passes 'import fractions' through (no match).
          - _validate_python_exec() catches it via the AST walk against
            PYTHON_ALLOWED_IMPORTS and returns "Forbidden import: 'fractions'...".

        If this test fails with ok=True, the AST validator is broken and the
        shared constant import is not actually being exercised.
        If the reason does NOT start with "Forbidden import", the regex
        pre-filter somehow started matching 'fractions', meaning it no longer
        isolates the AST path — the test would need to be updated.
        """
        ok, reason = validate(
            ActionType.PYTHON_EXEC,
            {"code": "import fractions\nprint(fractions.Fraction(1, 3))"},
        )
        assert ok is False, (
            "'fractions' is not in PYTHON_ALLOWED_IMPORTS; validate() must reject it"
        )
        # This assertion specifically confirms the AST validator fired, NOT the
        # registry regex pre-filter.  The pre-filter reason would say
        # "Forbidden pattern matched: '...'" — that string does NOT contain
        # "Forbidden import".
        assert reason.startswith("Forbidden import"), (
            f"Expected the AST validator to fire (reason starts with 'Forbidden import'), "
            f"but got: {reason!r}\n"
            "If this says 'Forbidden pattern matched', the regex pre-filter now "
            "covers 'fractions' and this test needs a different module to isolate "
            "the AST path."
        )


# ===========================================================================
# Error safety: validate() must never raise — always return (bool, ...)
# ===========================================================================

class TestValidateNeverRaises:
    """validate() must not raise for any input — it always returns (bool, ...)."""

    @pytest.mark.parametrize("action", list(ActionType))
    def test_empty_dict_never_raises(self, action: ActionType) -> None:
        result = validate(action, {})
        assert isinstance(result, tuple) and len(result) == 2

    @pytest.mark.parametrize("action", list(ActionType))
    def test_garbage_values_never_raises(self, action: ActionType) -> None:
        result = validate(action, {"query": None, "code": 123, "answer": [], "expression": {}})
        assert isinstance(result, tuple) and len(result) == 2

    def test_none_input_never_raises(self) -> None:
        result = validate(ActionType.ANSWER_DIRECTLY, None)  # type: ignore[arg-type]
        ok, reason = result
        assert ok is False
        assert isinstance(reason, str)

    def test_list_input_never_raises(self) -> None:
        result = validate(ActionType.SQL_QUERY, ["SELECT 1"])  # type: ignore[arg-type]
        ok, reason = result
        assert ok is False
