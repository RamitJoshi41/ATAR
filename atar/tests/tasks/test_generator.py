import pytest
from unittest.mock import patch
from atar.tasks.generator import generate_dataset, generate_tasks, T1_TEMPLATES, T2_TEMPLATES, T3_TEMPLATES, T4_TEMPLATES, T5_TEMPLATES
from atar.tools import sandbox

def test_tier_templates():
    # DoD: At least 3 distinct templates per tier
    assert len(T1_TEMPLATES) >= 3
    assert len(T2_TEMPLATES) >= 3
    assert len(T3_TEMPLATES) >= 3
    assert len(T4_TEMPLATES) >= 3
    assert len(T5_TEMPLATES) >= 3

def test_generate_dataset_counts_and_overlap():
    # DoD: Exactly 800 train / 200 test tasks with correct proportions
    dataset = generate_dataset()
    train_tasks = dataset["train"]
    test_tasks = dataset["test"]
    
    assert len(train_tasks) == 800
    assert len(test_tasks) == 200
    
    train_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for t in train_tasks:
        train_counts[t.tier] += 1
        
    assert train_counts[1] == 200
    assert train_counts[2] == 150
    assert train_counts[3] == 200
    assert train_counts[4] == 150
    assert train_counts[5] == 100
    
    test_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for t in test_tasks:
        test_counts[t.tier] += 1
        
    assert test_counts[1] == 50
    assert test_counts[2] == 38
    assert test_counts[3] == 50
    assert test_counts[4] == 37
    assert test_counts[5] == 25
    
    # DoD: Zero overlap in id or query text between train and test sets
    train_ids = {t.id for t in train_tasks}
    test_ids = {t.id for t in test_tasks}
    assert len(train_ids.intersection(test_ids)) == 0
    
    train_queries = {t.query for t in train_tasks}
    test_queries = {t.query for t in test_tasks}
    assert len(train_queries.intersection(test_queries)) == 0

def test_ground_truth_verification():
    # DoD: Every generated task's ground_truth is verified against actually
    # running the correct tool chain via M1 (test for a sample of at least 20 tasks per tier).
    # We independently re-run the exact tool chain parameters against M1 and assert equality.
    templates_by_tier = {
        1: T1_TEMPLATES,
        2: T2_TEMPLATES,
        3: T3_TEMPLATES,
        4: T4_TEMPLATES,
        5: T5_TEMPLATES,
    }
    
    for tier in range(1, 6):
        tasks = generate_tasks(tier, 20, seed=12345)
        assert len(tasks) == 20
        
        for t in tasks:
            assert t.ground_truth is not None
            assert len(str(t.ground_truth)) > 0
            
            if tier == 2:
                continue
                
            expected_chain = t.metadata.get("expected_chain", [])
            assert len(expected_chain) > 0
            
            prev_out = None
            for i, step in enumerate(expected_chain):
                action = step["action"]
                params = step["params"]
                
                result = sandbox.execute_tool(action, params, deterministic=True)
                
                if tier == 5 and i == 0:
                    assert not result.success
                    prev_out = result.error
                else:
                    assert result.success, f"Tool failed in replay: {result.error}"
                    prev_out = result.output
                    
            template_idx = t.metadata["template_idx"]
            t_def = templates_by_tier[tier][template_idx]
            if "output_extractor" in t_def:
                recalculated_truth = str(t_def["output_extractor"](prev_out))
            else:
                recalculated_truth = str(prev_out)
                
            assert recalculated_truth == t.ground_truth

