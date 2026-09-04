"""
10 hand-written arithmetic tasks with ground truth.
All tasks are tier-1 (simplest) and require only CALCULATOR.
"""

from __future__ import annotations

from atar.shared.shared_types import ActionType, Task

TASKS: list[Task] = [
    Task(
        id="arith-001",
        tier=1,
        query="What is 2 + 3?",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="5",
    ),
    Task(
        id="arith-002",
        tier=1,
        query="Calculate 15% of 200.",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="30.0",
    ),
    Task(
        id="arith-003",
        tier=1,
        query="What is 144 / 12?",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="12.0",
    ),
    Task(
        id="arith-004",
        tier=1,
        query="Compute 7 * 8 - 6.",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="50",
    ),
    Task(
        id="arith-005",
        tier=1,
        query="What is 2 ** 10?",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="1024",
    ),
    Task(
        id="arith-006",
        tier=1,
        query="If I have 250 and spend 37%, how much do I spend?",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="92.5",
    ),
    Task(
        id="arith-007",
        tier=1,
        query="What is (100 - 30) * 2?",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="140",
    ),
    Task(
        id="arith-008",
        tier=1,
        query="Divide 999 by 3.",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="333.0",
    ),
    Task(
        id="arith-009",
        tier=1,
        query="What is 50 + 25% of 50?",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="62.5",
    ),
    Task(
        id="arith-010",
        tier=1,
        query="Calculate 3.14 * 5 * 5.",
        required_tools=[ActionType.CALCULATOR],
        ground_truth="78.5",
    ),
]
