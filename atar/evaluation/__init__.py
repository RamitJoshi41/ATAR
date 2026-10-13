"""
Evaluation harness for ATAR — metrics, runner, and output formatting.

Public exports:
    compute_accuracy       — fraction of correct answers
    compute_success_rate   — fraction of successful episodes (correct + terminated)
    compute_efficiency     — accuracy / avg tool calls
    compute_recovery_rate  — fraction of episodes where initial tool failure → eventual success
    bootstrap_ci           — seeded bootstrap 95% confidence interval
    EvalRunner             — orchestrates evaluation across agents and tasks
"""

from atar.evaluation.metrics import (
    compute_accuracy,
    compute_success_rate,
    compute_efficiency,
    compute_recovery_rate,
    bootstrap_ci,
)
from atar.evaluation.runner import EvalRunner

__all__ = [
    "compute_accuracy",
    "compute_success_rate",
    "compute_efficiency",
    "compute_recovery_rate",
    "bootstrap_ci",
    "EvalRunner",
]
