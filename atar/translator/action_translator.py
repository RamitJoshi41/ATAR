"""
M4 — Action Translator
======================
Public interface (matches shared/interfaces.md section 4):

    translate(action, state) -> ConditionedPrompt
    validate(action, raw_params) -> (True, validated_params) | (False, error_str)

Security-critical — see AGENTS.md rule 8.  validate() is the boundary that
catches bad LLM-generated parameters *before* they reach M1's sandbox.
All failures are returned as (False, reason); this function never raises.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from atar.shared.shared_types import ActionType, ConditionedPrompt, RegistryEntry
from atar.tools.sandbox import PYTHON_ALLOWED_IMPORTS  # single source of truth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(__file__).parent.parent / "configs" / "action_registry.yaml"


def _load_registry() -> dict[ActionType, RegistryEntry]:
    """Load action_registry.yaml and return a dict keyed by ActionType."""
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    registry: dict[ActionType, RegistryEntry] = {}
    for int_key, entry in raw["actions"].items():
        action = ActionType(int(int_key))
        registry[action] = RegistryEntry(
            action=action,
            system_suffix=entry.get("system_suffix", ""),
            output_schema=entry.get("output_schema", {}),
            few_shot_examples=entry.get("few_shot_examples", []),
            constraint_rules=entry.get("constraint_rules", []),
            forbidden_patterns=entry.get("forbidden_patterns", []),
        )
    return registry


# Module-level singleton — load once on first import.
_REGISTRY: dict[ActionType, RegistryEntry] | None = None


def _get_registry() -> dict[ActionType, RegistryEntry]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_registry()
    return _REGISTRY


# ---------------------------------------------------------------------------
# translate()
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = (
    "You are a reasoning assistant operating inside an RL training loop. "
    "At each step you receive a question and context, then output structured JSON "
    "for the chosen action. Follow the action-specific instructions exactly."
)


def translate(action: ActionType, state: np.ndarray) -> ConditionedPrompt:
    """
    Convert a discrete RL action into a ConditionedPrompt the LLM can act on.

    Parameters
    ----------
    action:
        The discrete action chosen by the RL policy.
    state:
        The current 512-d state vector (np.ndarray, shape (STATE_DIM,)).
        Currently used only to extract budget features for the system message;
        future versions may include semantic summary.

    Returns
    -------
    ConditionedPrompt
        Contains the assembled system_message and the json_schema for constrained
        generation (M3 will pass json_schema to Outlines).
    """
    registry = _get_registry()
    entry = registry[action]

    # Build system message: base + optional few-shot + action-specific suffix
    sections: list[str] = [_BASE_SYSTEM_PROMPT]

    if entry.few_shot_examples:
        examples_block = "\n\n### Examples\n" + "\n---\n".join(
            ex.strip() for ex in entry.few_shot_examples
        )
        sections.append(examples_block)

    sections.append(f"\n### Your action: {action.name}\n{entry.system_suffix.strip()}")

    system_message = "\n".join(sections)

    return ConditionedPrompt(
        action=action,
        system_message=system_message,
        json_schema=entry.output_schema,
    )


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def validate(
    action: ActionType, raw_params: dict[str, Any]
) -> tuple[bool, dict[str, Any] | str]:
    """
    Validate LLM-generated parameters before they reach M1's tool sandbox.

    This is the security boundary.  The function:
      1. Checks forbidden_patterns from the registry (fast regex gate).
      2. Runs action-specific logic (SQL injection, Python import checks, etc.).
      3. Validates that required JSON schema keys are present with correct types.

    Returns
    -------
    (True, validated_params)   on success
    (False, error_message_str) on any failure — never raises.

    Fails *closed*: on any doubt, reject.
    """
    try:
        return _validate_inner(action, raw_params)
    except Exception as exc:  # pragma: no cover  — safety net
        logger.exception("Unexpected error in validate(); failing closed.")
        return False, f"Internal validation error: {exc}"


def _validate_inner(
    action: ActionType, raw_params: dict[str, Any]
) -> tuple[bool, dict[str, Any] | str]:
    """Inner (may raise) — wrapped by validate() for fail-closed guarantee."""

    # ------------------------------------------------------------------
    # 0. Basic type check on the params container
    # ------------------------------------------------------------------
    if not isinstance(raw_params, dict):
        return False, "raw_params must be a dict"

    registry = _get_registry()
    entry = registry[action]

    # ------------------------------------------------------------------
    # 1. Registry forbidden_patterns (fast regex pre-filter)
    # ------------------------------------------------------------------
    serialised = json.dumps(raw_params)
    for pattern in entry.forbidden_patterns:
        try:
            if re.search(pattern, serialised):
                return False, f"Forbidden pattern matched: {pattern!r}"
        except re.error as exc:
            # Broken regex in the registry — fail closed.
            return False, f"Registry regex error: {exc}"

    # ------------------------------------------------------------------
    # 2. JSON schema — required keys + basic type check
    # ------------------------------------------------------------------
    schema_ok, schema_err = _check_json_schema(entry.output_schema, raw_params)
    if not schema_ok:
        return False, schema_err

    # ------------------------------------------------------------------
    # 3. Action-specific deep validation
    # ------------------------------------------------------------------
    if action == ActionType.SEARCH_WEB:
        ok, err = _validate_search_web(raw_params)
    elif action == ActionType.CALCULATOR:
        ok, err = _validate_calculator(raw_params)
    elif action == ActionType.SQL_QUERY:
        ok, err = _validate_sql_query(raw_params)
    elif action == ActionType.PYTHON_EXEC:
        ok, err = _validate_python_exec(raw_params)
    elif action == ActionType.ANSWER_DIRECTLY:
        ok, err = _validate_answer_directly(raw_params)
    elif action == ActionType.CLARIFY:
        ok, err = _validate_clarify(raw_params)
    elif action == ActionType.TERMINATE:
        ok, err = _validate_terminate(raw_params)
    else:
        return False, f"Unknown action: {action}"

    if not ok:
        return False, err

    return True, raw_params


# ---------------------------------------------------------------------------
# JSON schema helper
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _check_json_schema(
    schema: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, str]:
    """
    Lightweight JSON Schema validator (required keys + type check only).
    We do not implement the full JSON Schema spec; this is deliberately minimal
    because the LLM uses Outlines to guarantee schema conformance at generation
    time.  validate() is the *security* gate, not a full schema verifier.
    """
    required: list[str] = schema.get("required", [])
    properties: dict[str, Any] = schema.get("properties", {})

    for key in required:
        if key not in params:
            return False, f"Missing required parameter: {key!r}"

    for key, value in params.items():
        if key not in properties:
            # additionalProperties: false — reject unknown keys
            if schema.get("additionalProperties") is False:
                return False, f"Unexpected parameter: {key!r}"
            continue
        prop_schema = properties[key]
        expected_type_name = prop_schema.get("type")
        if expected_type_name and expected_type_name in _TYPE_MAP:
            expected_type = _TYPE_MAP[expected_type_name]
            # Special case: JSON integers satisfy "number"
            if not isinstance(value, expected_type):
                return False, (
                    f"Parameter {key!r} must be of type {expected_type_name!r}, "
                    f"got {type(value).__name__!r}"
                )
        # Enum check
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            return False, (
                f"Parameter {key!r} must be one of {prop_schema['enum']!r}, "
                f"got {value!r}"
            )

    return True, ""


# ---------------------------------------------------------------------------
# Action-specific validators
# ---------------------------------------------------------------------------

# -- ANSWER_DIRECTLY ---------------------------------------------------------

def _validate_answer_directly(params: dict[str, Any]) -> tuple[bool, str]:
    answer = params.get("answer", "")
    if not isinstance(answer, str) or not answer.strip():
        return False, "answer must be a non-empty string"
    confidence = params.get("confidence", "")
    if confidence not in ("high", "medium", "low"):
        return False, f"confidence must be 'high', 'medium', or 'low'; got {confidence!r}"
    return True, ""


# -- SEARCH_WEB --------------------------------------------------------------

_SEARCH_MAX_CHARS = 200

def _validate_search_web(params: dict[str, Any]) -> tuple[bool, str]:
    query = params.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return False, "query must be a non-empty string"
    if len(query) > _SEARCH_MAX_CHARS:
        return False, f"query exceeds {_SEARCH_MAX_CHARS} characters"
    return True, ""


# -- CALCULATOR --------------------------------------------------------------

# Inline SQL/code injection in arithmetic expressions is caught here as well.
_CALC_FORBIDDEN_RE = re.compile(
    r"(?i)(import|exec|eval|open|__)|[a-zA-Z_][a-zA-Z0-9_]*\s*\("
)


def _validate_calculator(params: dict[str, Any]) -> tuple[bool, str]:
    expression = params.get("expression", "")
    if not isinstance(expression, str) or not expression.strip():
        return False, "expression must be a non-empty, non-whitespace string"
    if _CALC_FORBIDDEN_RE.search(expression):
        return False, "expression contains forbidden tokens (function calls, imports, etc.)"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return False, f"expression is not valid Python arithmetic: {exc}"
    # Verify only numeric/arithmetic nodes
    _ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Num,  # Python < 3.8 compat node
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    }
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return False, f"expression contains disallowed AST node: {type(node).__name__}"
    return True, ""


# -- SQL_QUERY ---------------------------------------------------------------

_SQL_FORBIDDEN_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ATTACH|PRAGMA|CREATE|ALTER|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
_SQL_MUST_START_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_SQL_COMMENT_RE = re.compile(r"(--|/\*)", re.IGNORECASE)
_SQL_MULTI_STMT_RE = re.compile(r";\s*\S")


def _validate_sql_query(params: dict[str, Any]) -> tuple[bool, str]:
    """
    Fail closed on anything besides a single well-formed SELECT.
    Checks performed (in order):
      1. Must be a non-empty string.
      2. Must start with SELECT.
      3. Must not contain forbidden DML/DDL keywords.
      4. Must not contain comment sequences (potential injection vector).
      5. Must not contain multiple statements (semicolon followed by non-whitespace).
    """
    query = params.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return False, "query must be a non-empty string"
    if not _SQL_MUST_START_RE.match(query):
        return False, "SQL query must start with SELECT"
    if _SQL_FORBIDDEN_RE.search(query):
        return False, "SQL query contains forbidden keyword (only SELECT is allowed)"
    if _SQL_COMMENT_RE.search(query):
        return False, "SQL query contains comment sequence (-- or /*)"
    if _SQL_MULTI_STMT_RE.search(query):
        return False, "SQL query must be a single statement (no multi-statement batches)"
    return True, ""


# -- PYTHON_EXEC -------------------------------------------------------------

def _validate_python_exec(params: dict[str, Any]) -> tuple[bool, str]:
    """
    Validates Python code before it reaches M1's subprocess sandbox.

    Import allow-list is sourced directly from PYTHON_ALLOWED_IMPORTS (imported
    from atar.tools.sandbox) — the same set M1 uses.  This is deliberately a
    single shared constant, not a second hardcoded copy (MSD cross-check DoD).
    """
    code = params.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return False, "code must be a non-empty string"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"code has a syntax error: {exc}"

    for node in ast.walk(tree):
        # Check import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base not in PYTHON_ALLOWED_IMPORTS:
                    return False, f"Forbidden import: {alias.name!r} (not in allow-list)"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                if base not in PYTHON_ALLOWED_IMPORTS:
                    return False, f"Forbidden import: {node.module!r} (not in allow-list)"
        # Check forbidden function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"__import__", "eval", "exec", "compile", "open"}:
                    return False, f"Forbidden call: {node.func.id!r}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {"__import__", "eval", "exec", "compile", "open"}:
                    return False, f"Forbidden attribute call: {node.func.attr!r}"

    return True, ""


# -- CLARIFY -----------------------------------------------------------------

def _validate_clarify(params: dict[str, Any]) -> tuple[bool, str]:
    question = params.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return False, "question must be a non-empty string"
    # clarification_answer is optional / injected by the environment
    return True, ""


# -- TERMINATE ---------------------------------------------------------------

def _validate_terminate(params: dict[str, Any]) -> tuple[bool, str]:
    final_answer = params.get("final_answer", "")
    if not isinstance(final_answer, str) or not final_answer.strip():
        return False, "final_answer must be a non-empty string"
    reasoning = params.get("reasoning", "")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False, "reasoning must be a non-empty string"
    return True, ""
