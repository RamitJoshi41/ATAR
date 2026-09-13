import ast
import json
import operator
import random
import re
import sqlite3
import subprocess
import tempfile
import time
import os
from typing import Any

from atar.shared.shared_types import ActionType, ToolResult


# --- Calculator AST definitions ---
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Num,  # for python < 3.8 compatibility, though we are >=3.10
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
}

_AST_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Disallowed expression")
        return float(node.value)
    elif isinstance(node, ast.Num):
        return float(node.n)
    elif isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _AST_OPERATORS:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        op = _AST_OPERATORS[type(node.op)]
        return op(_eval_ast(node.operand))
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _AST_OPERATORS:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        op = _AST_OPERATORS[type(node.op)]
        return op(_eval_ast(node.left), _eval_ast(node.right))
    else:
        raise ValueError("Disallowed expression")

def _calculator(params: dict) -> ToolResult:
    expression = params.get("expression", "")
    try:
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_AST_NODES:
                raise ValueError("Disallowed expression")
        result = _eval_ast(tree)
        return ToolResult(success=True, output=result, error=None, latency_ms=0.0, token_cost=0)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=0.0, token_cost=0)


# --- SQL Query definitions ---
_SQL_FORBIDDEN_REGEX = re.compile(r'\b(DROP|DELETE|INSERT|UPDATE|ATTACH|PRAGMA)\b', re.IGNORECASE)
_SQL_SELECT_REGEX = re.compile(r'^\s*SELECT\b', re.IGNORECASE)

_sql_conn = None

def _get_sql_conn():
    global _sql_conn
    if _sql_conn is None:
        _sql_conn = sqlite3.connect(":memory:")
        cursor = _sql_conn.cursor()
        # Create 5 synthetic tables
        cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, salary INTEGER)")
        cursor.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER)")
        cursor.execute("CREATE TABLE employee_projects (employee_id INTEGER, project_id INTEGER)")
        cursor.execute("CREATE TABLE locations (id INTEGER PRIMARY KEY, city TEXT)")
        
        # Insert some dummy data
        cursor.executemany("INSERT INTO departments VALUES (?, ?)", [(1, "Engineering"), (2, "Sales"), (3, "HR")])
        cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", [(1, "Alice", 1, 100000), (2, "Bob", 1, 95000), (3, "Charlie", 2, 80000)])
        cursor.executemany("INSERT INTO projects VALUES (?, ?, ?)", [(1, "Alpha", 500000), (2, "Beta", 200000)])
        cursor.executemany("INSERT INTO employee_projects VALUES (?, ?)", [(1, 1), (2, 1), (3, 2)])
        cursor.executemany("INSERT INTO locations VALUES (?, ?)", [(1, "New York"), (2, "London")])
        _sql_conn.commit()
    return _sql_conn

def _sql_query(params: dict) -> ToolResult:
    query = params.get("query", "")
    if _SQL_FORBIDDEN_REGEX.search(query):
        return ToolResult(success=False, output=None, error="Forbidden SQL command detected", latency_ms=0.0, token_cost=0)
    if not _SQL_SELECT_REGEX.match(query):
        return ToolResult(success=False, output=None, error="Only SELECT queries are allowed", latency_ms=0.0, token_cost=0)
    
    conn = _get_sql_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        output = [dict(zip(columns, row)) for row in rows]
        return ToolResult(success=True, output=output, error=None, latency_ms=0.0, token_cost=0)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=0.0, token_cost=0)


# --- Python Exec definitions ---
# Public constant — imported by M4 (Action Translator) so both modules share
# the same allow-list from a single source.  Never hardcode this in M4.
PYTHON_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {"math", "statistics", "json", "re", "collections"}
)
_ALLOWED_IMPORTS = PYTHON_ALLOWED_IMPORTS  # internal alias kept for backward compat
_FORBIDDEN_CALLS = {"__import__", "eval", "exec", "compile", "open"}

class PythonSecurityVisitor(ast.NodeVisitor):
    def visit_Import(self, node):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module not in _ALLOWED_IMPORTS:
                raise ValueError(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module not in _ALLOWED_IMPORTS:
                raise ValueError(f"Forbidden import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden call: {node.func.id}")
        self.generic_visit(node)

def _python_exec(params: dict) -> ToolResult:
    code = params.get("code", "")
    try:
        tree = ast.parse(code)
        visitor = PythonSecurityVisitor()
        visitor.visit(tree)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=0.0, token_cost=0)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f_name = f.name
    
    try:
        result = subprocess.run(
            ['python3', f_name],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode != 0:
            return ToolResult(success=False, output=result.stdout, error=result.stderr, latency_ms=0.0, token_cost=0)
        return ToolResult(success=True, output=result.stdout, error=None, latency_ms=0.0, token_cost=0)
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output=None, error="Execution timed out", latency_ms=0.0, token_cost=0)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=0.0, token_cost=0)
    finally:
        try:
            os.remove(f_name)
        except OSError:
            pass


# --- Web Search definitions ---
_search_corpus = [
    "Machine learning is a field of study in artificial intelligence.",
    "Python is a high-level, general-purpose programming language.",
    "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment.",
    "The capital of France is Paris, located in Europe.",
    "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance.",
    "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.",
    "A neural network is a network or circuit of biological neurons, or, in a modern sense, an artificial neural network.",
    "A database is an organized collection of data, generally stored and accessed electronically from a computer system.",
    "Linux is a family of open-source Unix-like operating systems based on the Linux kernel.",
    "Space exploration is the use of astronomy and space technology to explore outer space."
]

_bm25 = None

def _get_bm25():
    global _bm25
    if _bm25 is None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is not installed")
        tokenized_corpus = [doc.lower().split() for doc in _search_corpus]
        _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25

def _search_web(params: dict, deterministic: bool = False) -> ToolResult:
    query = params.get("query", "")
    if not query:
        return ToolResult(success=False, output=None, error="Empty query", latency_ms=0.0, token_cost=0)
    
    try:
        bm25 = _get_bm25()
        if not deterministic and random.random() < 0.20:
            # Inject noise: pick a random document that is NOT the top match
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            best_idx = max(range(len(scores)), key=scores.__getitem__) if any(s > 0 for s in scores) else 0
            other_indices = [i for i in range(len(_search_corpus)) if i != best_idx]
            if other_indices:
                noisy_idx = random.choice(other_indices)
                return ToolResult(success=True, output=_search_corpus[noisy_idx], error=None, latency_ms=0.0, token_cost=0)
        
        tokenized_query = query.lower().split()
        top_docs = bm25.get_top_n(tokenized_query, _search_corpus, n=1)
        if top_docs:
            return ToolResult(success=True, output=top_docs[0], error=None, latency_ms=0.0, token_cost=0)
        else:
            return ToolResult(success=True, output="", error=None, latency_ms=0.0, token_cost=0)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=0.0, token_cost=0)


# --- Clarify definition ---
def _clarify(params: dict) -> ToolResult:
    """
    Returns a simulated response for the clarify action.
    Note: The caller of `execute_tool` is responsible for extracting the expected
    clarification answer from Task.metadata and passing it via `params['clarification_answer']`.
    """
    answer = params.get("clarification_answer", "I cannot provide further clarification.")
    return ToolResult(success=True, output=answer, error=None, latency_ms=0.0, token_cost=0)


# --- Main Entry Point ---

def execute_tool(action: ActionType, params: dict, deterministic: bool = False) -> ToolResult:
    start_time = time.time()
    try:
        if action == ActionType.CALCULATOR:
            result = _calculator(params)
        elif action == ActionType.SQL_QUERY:
            result = _sql_query(params)
        elif action == ActionType.PYTHON_EXEC:
            result = _python_exec(params)
        elif action == ActionType.SEARCH_WEB:
            result = _search_web(params, deterministic=deterministic)
        elif action == ActionType.CLARIFY:
            result = _clarify(params)
        else:
            result = ToolResult(success=False, output=None, error=f"Unsupported action: {action}", latency_ms=0.0, token_cost=0)
    except Exception as e:
        result = ToolResult(success=False, output=None, error=str(e), latency_ms=0.0, token_cost=0)
    
    end_time = time.time()
    result.latency_ms = (end_time - start_time) * 1000.0
    return result
