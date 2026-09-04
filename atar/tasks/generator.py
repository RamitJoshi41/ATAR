import json
import random
import uuid
from pathlib import Path
from jinja2 import Template

from atar.shared.shared_types import Task, ActionType
from atar.tools.sandbox import execute_tool

def _syn(r: random.Random, words: list[str]) -> str:
    return r.choice(words)

def _prefix(r: random.Random) -> str:
    return r.choice(["", "Hi, ", "Please ", "Can you ", "Could you ", "I need you to ", "Quickly ", "Hey there, ", "Assistant, ", "For this task, "])

def _postfix(r: random.Random) -> str:
    return r.choice(["", " now", " for me", " today", " as soon as possible", " right away", " please"])

# --- Tier 1 Templates (Single-tool) ---
T1_TEMPLATES = [
    {
        "template": "{{ pre }}{{ query }} {{ a }} + {{ b }}{{ post }}{{ punc }}",
        "action": ActionType.CALCULATOR,
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["What is", "Calculate", "Compute", "Tell me", "Find", "Evaluate"]),
            "a": r.randint(1, 100000), 
            "b": r.randint(1, 100000),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_params_gen": lambda vars: {"expression": f"{vars['a']} + {vars['b']}"}
    },
    {
        "template": "{{ pre }}{{ query }} the {{ salary }} of employee {{ with_id }} {{ emp_id }}{{ post }}{{ punc }}",
        "action": ActionType.SQL_QUERY,
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Find", "Get", "What is", "Show me", "Retrieve", "Query"]),
            "salary": _syn(r, ["salary", "pay", "wage", "compensation", "earnings"]),
            "with_id": _syn(r, ["with ID", "ID", "number", "#"]),
            "emp_id": r.choice([1, 2, 3]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_params_gen": lambda vars: {"query": f"SELECT salary FROM employees WHERE id = {vars['emp_id']}"},
        "output_extractor": lambda out: str(out[0]["salary"]) if out else "Unknown"
    },
    {
        "template": "{{ pre }}{{ query }} information about {{ topic }}{{ post }}{{ punc }}",
        "action": ActionType.SEARCH_WEB,
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Search for", "Find", "Look up", "Get me", "Retrieve", "I need"]),
            "topic": _syn(r, ["Python", "Linux", "quantum computing", "France", "water", "machine learning", "neural networks", "databases", "space exploration", "reinforcement learning"]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_params_gen": lambda vars: {"query": vars["topic"]}
    }
]

# --- Tier 2 Templates (Distractor) ---
CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Brazil": "Brasilia", "Canada": "Ottawa", "Australia": "Canberra", "India": "New Delhi", "China": "Beijing", "Russia": "Moscow", "Mexico": "Mexico City", "Egypt": "Cairo", "UK": "London", "USA": "Washington D.C.", "Argentina": "Buenos Aires", "Chile": "Santiago", "Peru": "Lima", "Colombia": "Bogota", "South Korea": "Seoul"}
SUBSTANCES = {"water": "100", "ethanol": "78", "methanol": "65", "acetone": "56", "glycerol": "290", "mercury": "356", "benzene": "80", "acetic acid": "118", "chloroform": "61", "sulfuric acid": "337"}

T2_TEMPLATES = [
    {
        "template": "{{ pre }}{{ query }} the number of hours in {{ days }} days{{ post }}{{ punc }}",
        "ground_truth": lambda vars: str(vars["days"] * 24),
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Calculate", "Compute", "Tell me", "Find", "Evaluate", "What is"]),
            "days": r.randint(2, 10000),
            "punc": _syn(r, ["?", ".", ""])
        }
    },
    {
        "template": "{{ pre }}{{ query }} the capital city of {{ country }}{{ post }}{{ punc }}",
        "ground_truth": lambda vars: CAPITALS[vars["country"]],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Find", "Search for", "Look up", "Tell me", "What is"]),
            "country": r.choice(list(CAPITALS.keys())),
            "punc": _syn(r, ["?", ".", ""])
        }
    },
    {
        "template": "{{ pre }}{{ query }} the boiling point of {{ substance }} in Celsius{{ post }}{{ punc }}",
        "ground_truth": lambda vars: SUBSTANCES[vars["substance"]],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Query", "Search", "Find", "Tell me", "What is"]),
            "substance": r.choice(list(SUBSTANCES.keys())),
            "punc": _syn(r, ["?", ".", ""])
        }
    }
]

# --- Tier 3 Templates (Multi-hop) ---
T3_TEMPLATES = [
    {
        "template": "{{ pre }}{{ query }} the {{ salary }} of employee {{ with_id }} {{ emp_id }} and divide it by {{ divisor }}{{ post }}{{ punc }}",
        "actions": [ActionType.SQL_QUERY, ActionType.CALCULATOR],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Find", "Get", "What is", "Show me", "Retrieve"]),
            "salary": _syn(r, ["salary", "pay", "wage", "compensation"]),
            "with_id": _syn(r, ["with ID", "ID", "number", "#"]),
            "emp_id": r.choice([1, 2, 3]), 
            "divisor": r.choice([2, 4, 5, 10, 20, 50, 100]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.SQL_QUERY, {"query": f"SELECT salary FROM employees WHERE id = {vars['emp_id']}"}),
            lambda vars, prev_out: (ActionType.CALCULATOR, {"expression": f"{prev_out[0]['salary']} / {vars['divisor']}"})
        ],
        "output_extractor": lambda out: str(out)
    },
    {
        "template": "{{ pre }}{{ query }} the capital of France and write a python script that prints its length{{ post }}{{ punc }}",
        "actions": [ActionType.SEARCH_WEB, ActionType.PYTHON_EXEC],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Search for", "Find", "Look up", "Get"]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.SEARCH_WEB, {"query": "capital of France"}),
            lambda vars, prev_out: (ActionType.PYTHON_EXEC, {"code": f"print(len({repr(prev_out)}))"})
        ],
        "output_extractor": lambda out: out.strip()
    },
    {
        "template": "{{ pre }}{{ query }} the department_id for employee ID {{ emp_id }}, then find the name of that department{{ post }}{{ punc }}",
        "actions": [ActionType.SQL_QUERY, ActionType.SQL_QUERY],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Find", "Get", "Look up", "Retrieve"]),
            "emp_id": r.choice([1, 2, 3]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.SQL_QUERY, {"query": f"SELECT department_id FROM employees WHERE id = {vars['emp_id']}"}),
            lambda vars, prev_out: (ActionType.SQL_QUERY, {"query": f"SELECT name FROM departments WHERE id = {prev_out[0]['department_id']}"})
        ],
        "output_extractor": lambda out: str(out[0]["name"])
    }
]

# --- Tier 4 Templates (Ambiguous) ---
T4_TEMPLATES = [
    {
        "template": "{{ pre }}{{ query }} the salary of the employee{{ post }}{{ punc }}",
        "clarification_answer": "ID {{ emp_id }}",
        "actions": [ActionType.CLARIFY, ActionType.SQL_QUERY],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Find", "Get", "What is", "Show me", "Retrieve"]),
            "emp_id": r.choice([1, 2, 3]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.CLARIFY, {"clarification_answer": f"ID {vars['emp_id']}"}),
            lambda vars, prev_out: (ActionType.SQL_QUERY, {"query": f"SELECT salary FROM employees WHERE id = {vars['emp_id']}"})
        ],
        "output_extractor": lambda out: str(out[0]["salary"])
    },
    {
        "template": "{{ pre }}{{ query }} the percentage{{ post }}{{ punc }}",
        "clarification_answer": "{{ a }} percent of {{ b }}",
        "actions": [ActionType.CLARIFY, ActionType.CALCULATOR],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Calculate", "Compute", "Find", "What is"]),
            "a": r.randint(10, 99), 
            "b": r.randint(100, 999),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.CLARIFY, {"clarification_answer": f"{vars['a']} percent of {vars['b']}"}),
            lambda vars, prev_out: (ActionType.CALCULATOR, {"expression": f"({vars['a']} / 100.0) * {vars['b']}"})
        ],
        "output_extractor": lambda out: str(out)
    },
    {
        "template": "{{ pre }}{{ query }} the capital{{ post }}{{ punc }}",
        "clarification_answer": "France",
        "actions": [ActionType.CLARIFY, ActionType.SEARCH_WEB],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Search for", "Find", "Look up", "What is"]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.CLARIFY, {"clarification_answer": "France"}),
            lambda vars, prev_out: (ActionType.SEARCH_WEB, {"query": "capital of France"})
        ]
    }
]

# --- Tier 5 Templates (Error-recovery) ---
T5_TEMPLATES = [
    {
        "template": "{{ pre }}{{ query }} the salary of employee {{ emp_id }} to {{ new_salary }}. If you don't have permission to do that, just tell me their current salary{{ post }}{{ punc }}",
        "actions": [ActionType.SQL_QUERY, ActionType.SQL_QUERY],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Update", "Change", "Modify", "Set"]),
            "emp_id": r.choice([1, 2, 3]), 
            "new_salary": r.randint(100, 200) * 1000,
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.SQL_QUERY, {"query": f"UPDATE employees SET salary = {vars['new_salary']} WHERE id = {vars['emp_id']}"}),
            lambda vars, prev_out: (ActionType.SQL_QUERY, {"query": f"SELECT salary FROM employees WHERE id = {vars['emp_id']}"})
        ],
        "output_extractor": lambda out: str(out[0]["salary"])
    },
    {
        "template": "{{ pre }}Run a python script using eval() to calculate {{ a }} + {{ b }}, and if eval is blocked, write a normal python script without eval to print it{{ post }}{{ punc }}",
        "actions": [ActionType.PYTHON_EXEC, ActionType.PYTHON_EXEC],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "a": r.randint(1, 1000), "b": r.randint(1, 1000),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.PYTHON_EXEC, {"code": f"print(eval('{vars['a']} + {vars['b']}'))"}),
            lambda vars, prev_out: (ActionType.PYTHON_EXEC, {"code": f"print({vars['a']} + {vars['b']})"})
        ],
        "output_extractor": lambda out: out.strip()
    },
    {
        "template": "{{ pre }}{{ query }} the projects table. If it's forbidden, count the number of projects instead{{ post }}{{ punc }}",
        "actions": [ActionType.SQL_QUERY, ActionType.SQL_QUERY],
        "params_gen": lambda r: {
            "pre": _prefix(r), "post": _postfix(r),
            "query": _syn(r, ["Drop", "Delete", "Remove", "Clear out"]),
            "punc": _syn(r, ["?", ".", ""])
        },
        "tool_chain": [
            lambda vars: (ActionType.SQL_QUERY, {"query": "DROP TABLE projects"}),
            lambda vars, prev_out: (ActionType.SQL_QUERY, {"query": "SELECT count(*) as cnt FROM projects"})
        ],
        "output_extractor": lambda out: str(out[0]["cnt"])
    }
]


def generate_tasks(tier: int, count: int, seed: int, seen_queries: set = None) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    
    if tier == 1:
        templates = T1_TEMPLATES
    elif tier == 2:
        templates = T2_TEMPLATES
    elif tier == 3:
        templates = T3_TEMPLATES
    elif tier == 4:
        templates = T4_TEMPLATES
    elif tier == 5:
        templates = T5_TEMPLATES
    else:
        raise ValueError("Tier must be between 1 and 5")

    if seen_queries is None:
        seen_queries = set()

    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError(f"Unable to generate {count} unique queries for tier {tier}. Max attempts reached.")
            
        t_def = rng.choice(templates)
        template_idx = templates.index(t_def)
        vars = t_def["params_gen"](rng)
        query = Template(t_def["template"]).render(**vars)
        
        # Replace multiple spaces with a single space cleanly in case pre/post are empty
        query = " ".join(query.split())
        # Fix punctuation issues like " ?" -> "?"
        query = query.replace(" ?", "?").replace(" .", ".")
        
        if query in seen_queries:
            continue
            
        seen_queries.add(query)
        task_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        metadata = {"template_idx": template_idx}

        if tier == 2:
            if callable(t_def["ground_truth"]):
                ground_truth = str(t_def["ground_truth"](vars))
            else:
                ground_truth = str(t_def["ground_truth"])
            req_tools = [ActionType.ANSWER_DIRECTLY]
        elif tier == 1:
            req_tools = [t_def["action"]]
            tool_params = t_def["tool_params_gen"](vars)
            metadata["expected_chain"] = [{"action": t_def["action"], "params": tool_params}]
            result = execute_tool(t_def["action"], tool_params, deterministic=True)
            if not result.success:
                raise RuntimeError(f"Tool execution failed for Tier 1: {result.error}")
            
            if "output_extractor" in t_def:
                ground_truth = str(t_def["output_extractor"](result.output))
            else:
                ground_truth = str(result.output)
        else:
            req_tools = t_def["actions"]
            if "clarification_answer" in t_def:
                rendered_clarif = Template(t_def["clarification_answer"]).render(**vars)
                metadata["clarification_answer"] = rendered_clarif
                vars["clarification_answer"] = rendered_clarif
            
            prev_out = None
            expected_chain = []
            for i, step_fn in enumerate(t_def["tool_chain"]):
                if i == 0:
                    action, tool_params = step_fn(vars)
                else:
                    action, tool_params = step_fn(vars, prev_out)
                
                expected_chain.append({"action": action, "params": tool_params})
                result = execute_tool(action, tool_params, deterministic=True)
                
                if tier == 5 and i == 0:
                    if result.success:
                        raise RuntimeError("Tier 5 first step was expected to fail but succeeded.")
                    prev_out = result.error
                else:
                    if not result.success:
                        raise RuntimeError(f"Tool execution failed in tool chain: {result.error} (Action: {action}, Params: {tool_params})")
                    prev_out = result.output
            
            metadata["expected_chain"] = expected_chain
            if "output_extractor" in t_def:
                ground_truth = str(t_def["output_extractor"](prev_out))
            else:
                ground_truth = str(prev_out)
        
        tasks.append(Task(
            id=task_id,
            tier=tier,
            query=query,
            required_tools=req_tools,
            ground_truth=ground_truth,
            metadata=metadata
        ))
        
    return tasks


def generate_dataset(train_count=800, test_count=200, seed=42) -> dict:
    rng = random.Random(seed)
    
    # Train distribution (800)
    train_dist = {1: 200, 2: 150, 3: 200, 4: 150, 5: 100}
    
    # Test distribution (200)
    test_dist = {1: 50, 2: 38, 3: 50, 4: 37, 5: 25}
    
    dataset = {"train": [], "test": []}
    seen_queries = set()
    
    # Generate Train
    for tier, count in train_dist.items():
        dataset["train"].extend(generate_tasks(tier, count, rng.randint(0, 1000000), seen_queries))
        
    # Generate Test
    for tier, count in test_dist.items():
        dataset["test"].extend(generate_tasks(tier, count, rng.randint(0, 1000000), seen_queries))
        
    # Ensure no overlap
    train_ids = {t.id for t in dataset["train"]}
    test_ids = {t.id for t in dataset["test"]}
    if train_ids.intersection(test_ids):
        raise RuntimeError("Overlap found in train and test IDs")
        
    train_queries = {t.query for t in dataset["train"]}
    test_queries = {t.query for t in dataset["test"]}
    if train_queries.intersection(test_queries):
        raise RuntimeError("Overlap found in train and test queries")
        
    # Write to files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    def write_jsonl(path: Path, tasks: list[Task]):
        with path.open("w") as f:
            for t in tasks:
                d = {
                    "id": t.id,
                    "tier": t.tier,
                    "query": t.query,
                    "required_tools": [a.value for a in t.required_tools],
                    "ground_truth": t.ground_truth,
                    "metadata": t.metadata
                }
                f.write(json.dumps(d) + "\n")
                
    write_jsonl(data_dir / "tasks_train.jsonl", dataset["train"])
    write_jsonl(data_dir / "tasks_test.jsonl", dataset["test"])
    
    return dataset
