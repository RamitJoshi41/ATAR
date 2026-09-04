"""
Calculator tool — AST-safe arithmetic evaluation.

Supports: +, -, *, /, ** (power), unary negation, parentheses, and
integer/float literals.  No eval(), no exec(), no imports.
Returns a ToolResult per shared_types.py.
"""

from __future__ import annotations

import ast
import operator
import time
from typing import Any

from atar.shared.shared_types import ToolResult

# Allowed binary and unary operators (whitelist approach — fail closed).
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node containing only arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def execute_calculator(expression: str) -> ToolResult:
    """Evaluate an arithmetic expression safely via AST parsing.

    Args:
        expression: A string containing an arithmetic expression,
                    e.g. "2 + 3 * (4 - 1)".

    Returns:
        ToolResult with the numeric result on success, or an error message
        on failure.
    """
    start = time.perf_counter_ns()
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return ToolResult(
            success=True,
            output=result,
            error=None,
            latency_ms=elapsed_ms,
            token_cost=0,  # calculator uses no LLM tokens
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return ToolResult(
            success=False,
            output=None,
            error=str(exc),
            latency_ms=elapsed_ms,
            token_cost=0,
        )
