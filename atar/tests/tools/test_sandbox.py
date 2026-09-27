import time

from atar.shared.shared_types import ActionType
from atar.tools.sandbox import execute_tool


def test_calculator_happy_path():
    result = execute_tool(ActionType.CALCULATOR, {"expression": "2 + 2 * (3 ** 2)"})
    assert result.success is True
    assert result.output == 20.0
    assert result.error is None

def test_calculator_reject_os_system():
    result = execute_tool(ActionType.CALCULATOR, {"expression": "os.system('echo 1')"})
    assert result.success is False
    assert result.error is not None
    assert "Disallowed expression" in result.error

def test_calculator_reject_import():
    result = execute_tool(ActionType.CALCULATOR, {"expression": "__import__('os').system('echo 1')"})
    assert result.success is False
    assert result.error is not None
    assert "Disallowed expression" in result.error

def test_calculator_reject_string_literals():
    result = execute_tool(ActionType.CALCULATOR, {"expression": "'hello' + 'world'"})
    assert result.success is False
    assert result.error is not None
    assert "Disallowed expression" in result.error

def test_calculator_reject_booleans():
    result = execute_tool(ActionType.CALCULATOR, {"expression": "True + 1"})
    assert result.success is False
    assert result.error is not None
    assert "Disallowed expression" in result.error


def test_sql_query_happy_path():
    result = execute_tool(ActionType.SQL_QUERY, {"query": "SELECT count(*) as count FROM employees"})
    assert result.success is True
    assert result.output == [{"count": 3}]

def test_sql_query_reject_drop():
    result = execute_tool(ActionType.SQL_QUERY, {"query": "DROP TABLE employees"})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden SQL command detected" in result.error

def test_sql_query_reject_injection():
    result = execute_tool(ActionType.SQL_QUERY, {"query": "SELECT * FROM employees; DELETE FROM employees;"})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden SQL command detected" in result.error

def test_sql_query_reject_non_select():
    # Attempting to execute a CREATE statement
    result = execute_tool(ActionType.SQL_QUERY, {"query": "CREATE TABLE hack (id INT)"})
    assert result.success is False
    assert result.error is not None
    assert "Only SELECT queries are allowed" in result.error


def test_python_exec_happy_path():
    code = "import math\nprint(math.sqrt(16))"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is True
    assert result.output.strip() == "4.0"

def test_python_exec_reject_import_os():
    code = "import os\nprint(os.environ)"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden import: os" in result.error

def test_python_exec_reject_importfrom_sys():
    code = "from sys import exit\nexit(0)"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden import: sys" in result.error

def test_python_exec_reject_import_os_call():
    code = "__import__('os').system('ls')"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden call: __import__" in result.error

def test_python_exec_reject_eval():
    code = "eval('1 + 1')"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden call: eval" in result.error

def test_python_exec_reject_exec():
    code = "exec('x = 1')"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden call: exec" in result.error

def test_python_exec_reject_compile():
    code = "compile('1 + 1', '<string>', 'eval')"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden call: compile" in result.error

def test_python_exec_reject_open():
    code = "open('/etc/passwd', 'r')"
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    assert result.success is False
    assert result.error is not None
    assert "Forbidden call: open" in result.error

def test_python_exec_timeout():
    code = "while True: pass"
    start = time.time()
    result = execute_tool(ActionType.PYTHON_EXEC, {"code": code})
    duration = time.time() - start
    assert result.success is False
    assert result.error is not None
    assert "Execution timed out" in result.error
    assert duration < 3.0  # Should be ~2.0 seconds


def test_search_web_happy_path():
    result = execute_tool(ActionType.SEARCH_WEB, {"query": "capital of France"})
    assert result.success is True
    assert isinstance(result.output, str)

def test_search_web_noise_distribution():
    # To check noise injection, we'll run it 100 times.
    # The best document for "capital of France" should be "The capital of France is Paris, located in Europe."
    best_doc = "The capital of France is Paris, located in Europe."
    
    noise_count = 0
    for _ in range(100):
        result = execute_tool(ActionType.SEARCH_WEB, {"query": "capital of France"})
        assert result.success is True
        if result.output != best_doc:
            noise_count += 1
            
    # Allow some statistical variance: 20% expected, so 5 to 40 should be fine for 100 trials.
    assert 5 <= noise_count <= 40, f"Noise count was {noise_count}, expected ~20"


def test_clarify():
    result = execute_tool(ActionType.CLARIFY, {"clarification_answer": "This is a test clarification."})
    assert result.success is True
    assert result.output == "This is a test clarification."
    assert result.error is None

def test_search_web_deterministic():
    best_doc = "The capital of France is Paris, located in Europe."
    for _ in range(25):
        result = execute_tool(ActionType.SEARCH_WEB, {"query": "capital of France"}, deterministic=True)
        assert result.success is True
        assert result.output == best_doc
