"""
Evaluation runner for ATAR — runs agents against tasks and collects results.

Supports four agent types:
  - atar:     Trained policy (greedy argmax, no exploration, no gradient)
  - random:   Uniform random action selection
  - zeroshot: Always ANSWER_DIRECTLY then TERMINATE (no tools)
  - react:    ReAct prompted baseline via M3 generate()

All agents are evaluated on the identical task set.  Results are resumable:
if per_task.jsonl already exists in the output directory, completed tasks
are skipped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from atar.env.answer_matching import is_answer_correct
from atar.env.atar_env import ATAREnv
from atar.llm.llm_interface import LLMInterface
from atar.policy.policy_network import ATARPolicy
from atar.policy.baselines import (
    _REACT_ACTION_SCHEMA,
    _REACT_MAX_NEW_TOKENS,
    _REACT_SYSTEM_PROMPT,
)
from atar.shared.shared_types import ActionType, Task
from atar.evaluation.metrics import aggregate_metrics

logger = logging.getLogger(__name__)


# ─── Task loading ────────────────────────────────────────────────────────

def load_tasks_from_jsonl(path: Path) -> list[Task]:
    """Load tasks from a JSONL file."""
    tasks: list[Task] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(Task(
                id=raw["id"],
                tier=raw["tier"],
                query=raw["query"],
                required_tools=[ActionType(a) for a in raw["required_tools"]],
                ground_truth=raw["ground_truth"],
                metadata=raw.get("metadata", {}),
            ))
    return tasks


def check_disjoint(test_tasks: list[Task], train_path: Path) -> None:
    """
    Verify that test tasks are disjoint from training tasks.
    Fails loudly if any overlap is found (by query string AND by id).
    """
    train_tasks = load_tasks_from_jsonl(train_path)
    train_ids = {t.id for t in train_tasks}
    train_queries = {t.query for t in train_tasks}

    test_ids = {t.id for t in test_tasks}
    test_queries = {t.query for t in test_tasks}

    id_overlap = train_ids & test_ids
    query_overlap = train_queries & test_queries

    if id_overlap:
        raise RuntimeError(
            f"FATAL: {len(id_overlap)} test task IDs overlap with training set: "
            f"{list(id_overlap)[:5]}"
        )
    if query_overlap:
        raise RuntimeError(
            f"FATAL: {len(query_overlap)} test task queries overlap with training set: "
            f"{list(query_overlap)[:5]}"
        )
    logger.info(
        "Disjointness check passed: %d test tasks, %d train tasks, 0 overlap.",
        len(test_tasks), len(train_tasks),
    )


def _tier_counts(tasks: list[Task]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for t in tasks:
        counts[t.tier] = counts.get(t.tier, 0) + 1
    return counts


def verify_stratification(
    tasks: list[Task],
    reference_tasks: list[Task] | None = None,
    tolerance: float = 0.05,
) -> dict[int, int]:
    """
    Verify the test set covers all 5 tiers and (if ``reference_tasks`` is
    given, normally the training set) that each tier's share of the test set
    is within ``tolerance`` of its share of the reference set.

    Raises RuntimeError on a missing tier or a share outside tolerance.
    Returns the tier counts.
    """
    tier_counts = _tier_counts(tasks)
    logger.info("Task distribution by tier: %s", dict(sorted(tier_counts.items())))

    missing = [tier for tier in range(1, 6) if tier_counts.get(tier, 0) == 0]
    if missing:
        raise RuntimeError(f"FATAL: test set has no tasks for tier(s) {missing}")

    if reference_tasks:
        ref_counts = _tier_counts(reference_tasks)
        n, n_ref = len(tasks), len(reference_tasks)
        for tier in range(1, 6):
            share = tier_counts.get(tier, 0) / n
            ref_share = ref_counts.get(tier, 0) / n_ref
            if abs(share - ref_share) > tolerance:
                raise RuntimeError(
                    f"FATAL: tier {tier} is {share:.1%} of the test set but "
                    f"{ref_share:.1%} of the reference set (tolerance {tolerance:.0%})"
                )
    return tier_counts


def stratified_subset(tasks: list[Task], n: int) -> list[Task]:
    """
    The first ``n`` tasks taken round-robin across tiers (file order within a
    tier), so a smoke run covers every tier.  tasks_test.jsonl is sorted by
    tier, so a plain ``tasks[:n]`` would be tier 1 only.
    """
    by_tier: dict[int, list[Task]] = {}
    for t in tasks:
        by_tier.setdefault(t.tier, []).append(t)
    queues = [by_tier[tier] for tier in sorted(by_tier)]
    picked: list[Task] = []
    i = 0
    while len(picked) < n and any(i < len(q) for q in queues):
        for q in queues:
            if i < len(q) and len(picked) < n:
                picked.append(q[i])
        i += 1
    return picked


# ─── Answer correctness checking ─────────────────────────────────────────

def check_answer_correct(final_answer: object, ground_truth: str) -> bool:
    """
    Check if an answer is correct.  Delegates to the same function ATAREnv
    uses for its accuracy reward, so training and evaluation can't disagree.
    """
    if final_answer is None:
        return False
    return is_answer_correct(str(final_answer), str(ground_truth))


# ─── Episode runner (shared across agents) ────────────────────────────────

def _categorize_failure(record: dict[str, Any]) -> str:
    """Categorize why an episode failed."""
    if record.get("truncated", False):
        return "truncated"
    if record.get("final_answer") is None:
        return "no_answer"
    if record.get("had_tool_failure", False):
        return "tool_failure_unrecovered"
    return "wrong_answer"


# ─── Agent implementations ───────────────────────────────────────────────

class ATARAgent:
    """Trained ATAR policy with greedy (argmax) action selection."""

    def __init__(
        self,
        policy: ATARPolicy,
        device: str = "cpu",
    ) -> None:
        self._policy = policy
        self._device = device
        self._policy.eval()

    def select_action(self, state: np.ndarray) -> int:
        """Greedy argmax action selection — no sampling, no exploration."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).to(self._device)
            logits, _ = self._policy(state_t)
            return int(torch.argmax(logits, dim=-1).item())

    def action_distribution(self, state: np.ndarray) -> tuple[list[float], float]:
        """Softmax action probabilities (7) and the value estimate for ``state``."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).to(self._device)
            logits, value = self._policy(state_t)
            # The policy may return a batch dimension for a single state.
            probs = torch.softmax(logits.reshape(-1), dim=-1).cpu().tolist()
            return [float(p) for p in probs], float(value.reshape(-1)[0].item())


class CheckpointMismatchError(RuntimeError):
    """The checkpoint was trained on a different LLM than the one loaded."""


def load_atar_agent(
    checkpoint_path: str | Path,
    llm: LLMInterface,
    device: str = "cpu",
) -> tuple["ATARAgent", Any]:
    """
    Load a training checkpoint as a greedy ATARAgent.

    Restores the policy AND the co-trained projection into ``llm`` — the
    policy was trained on states built by that projection.  Raises
    CheckpointMismatchError when the checkpoint's projection expects a
    different hidden size (i.e. it was trained on another model, such as
    the 7B vs the local 0.5B stand-in).  Returns (agent, CheckpointState).
    """
    from atar.policy.trainer import PPOTrainer  # deferred: pulls in wandb

    ckpt = PPOTrainer.load_checkpoint(checkpoint_path)
    policy = ATARPolicy()
    policy.load_state_dict(ckpt.policy_state_dict)
    policy = policy.to(device)

    if ckpt.projection_state_dict is not None:
        ckpt_in = ckpt.projection_state_dict["weight"].shape[1]
        if ckpt_in != llm.projection.in_features:
            raise CheckpointMismatchError(
                f"Checkpoint projection expects hidden size {ckpt_in} but "
                f"{llm.model_name} has {llm.projection.in_features}. Use the "
                "same model that was used for training."
            )
        llm.projection.load_state_dict(ckpt.projection_state_dict)
    else:
        logger.warning(
            "Checkpoint %s has no projection weights (pre-co-training format); "
            "using the seeded initial projection.", checkpoint_path,
        )
    return ATARAgent(policy=policy, device=device), ckpt


class RandomAgent:
    """Uniform random action selection."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_actions = len(ActionType)

    def select_action(self, state: np.ndarray) -> int:
        return int(self._rng.integers(0, self._n_actions))


class ZeroShotAgent:
    """
    Zero-shot baseline: always ANSWER_DIRECTLY on the first step,
    then TERMINATE.  No tool usage.
    """

    def __init__(self) -> None:
        self._step_in_episode = 0

    def reset(self) -> None:
        self._step_in_episode = 0

    def select_action(self, state: np.ndarray) -> int:
        self._step_in_episode += 1
        if self._step_in_episode == 1:
            return ActionType.ANSWER_DIRECTLY
        return ActionType.TERMINATE


class ReActAgent:
    """
    ReAct baseline agent wrapper — uses the LLM to decide actions.
    Reuses the prompts from atar.policy.baselines.
    """

    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm
        self._messages: list[dict] = []
        self._query: str = ""
        self.last_thought: str = ""   # shown by the M8 demo

    def reset(self, query: str) -> None:
        self._query = query
        self._messages = [
            {"role": "system", "content": _REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {query}\n\nBegin."},
        ]

    def select_action(self, state: np.ndarray) -> int:
        try:
            result = self._llm.generate(
                messages=self._messages,
                json_schema=_REACT_ACTION_SCHEMA,
                max_new_tokens=_REACT_MAX_NEW_TOKENS,
            )
            action_int = int(result.get("action", ActionType.TERMINATE))
            action_int = max(0, min(action_int, len(ActionType) - 1))
            self.last_thought = str(result.get("thought", ""))
        except Exception as exc:
            logger.warning("ReAct generate() failed: %s; using TERMINATE", exc)
            action_int = ActionType.TERMINATE
            self.last_thought = f"(generation failed: {exc}; terminating)"
        return action_int

    def observe(self, action: int, info: dict) -> None:
        """
        Append the chosen action and its result to the conversation, so the
        next decision can see what was already tried (Action → Observation).
        """
        self._messages.append({
            "role": "assistant",
            "content": f"Thought: {self.last_thought}\nAction: {ActionType(action).name}",
        })
        tool_result = info.get("tool_result")
        if tool_result is not None:
            observation = (
                f"Observation: success={tool_result.success}, "
                f"output={str(tool_result.output)[:200]}"
            )
        else:
            observation = "Observation: no tool result (call rejected or not a tool action)."
        self._messages.append({"role": "user", "content": observation})


# ─── EvalRunner ──────────────────────────────────────────────────────────

class EvalRunner:
    """
    Orchestrates evaluation of agents against a fixed task set.

    Parameters
    ----------
    tasks : list[Task]
        The evaluation task set.
    llm : LLMInterface
        Shared LLM instance.
    data_dir : str | Path
        Data directory (for env construction).
    out_dir : str | Path
        Output directory for results.
    seed : int
        RNG seed.
    fingerprints : dict[str, dict] | None
        Per-agent identity of what is being evaluated (model name, seed,
        checkpoint hash, grader version...).  Written next to each agent's
        per-task file; resuming into a directory whose fingerprint differs
        raises instead of silently mixing results from different runs.
    """

    def __init__(
        self,
        tasks: list[Task],
        llm: LLMInterface,
        data_dir: str | Path,
        out_dir: str | Path,
        seed: int = 42,
        fingerprints: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._fingerprints = fingerprints or {}
        self._tasks = tasks
        self._llm = llm
        self._data_dir = Path(data_dir)
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._seed = seed

    def run_agent(
        self,
        agent_name: str,
        agent: Any,
        max_tasks: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run a single agent against all tasks, collecting per-task records.

        Supports resumption: if a per-task JSONL file already exists,
        completed task IDs are skipped.

        Parameters
        ----------
        agent_name : str
            Name of the agent (for output file naming).
        agent : Any
            Agent object with .select_action(state) -> int.
        max_tasks : int | None
            If set, only evaluate this many tasks, spread evenly across tiers
            (smoke-test flag).

        Returns
        -------
        list[dict] — per-task records.
        """
        per_task_path = self._out_dir / f"per_task_{agent_name}.jsonl"
        self._check_or_write_fingerprint(agent_name, per_task_path)

        # Load already-completed task IDs for resumption
        completed_ids: set[str] = set()
        existing_records: list[dict[str, Any]] = []
        if per_task_path.exists():
            with open(per_task_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    completed_ids.add(record["task_id"])
                    existing_records.append(record)
            logger.info(
                "Resuming %s: %d tasks already completed.",
                agent_name, len(completed_ids),
            )

        # Build env for this agent
        env = ATAREnv(
            llm=self._llm,
            data_dir=self._data_dir,
            seed=self._seed,
        )

        tasks_to_run = self._tasks
        if max_tasks is not None:
            tasks_to_run = stratified_subset(tasks_to_run, max_tasks)

        records = list(existing_records)

        # Open per-task file in append mode
        with open(per_task_path, "a") as f:
            for task_idx, task in enumerate(tasks_to_run):
                if task.id in completed_ids:
                    continue

                record = self._run_single_episode(
                    env, task, agent_name, agent, task_idx,
                )
                records.append(record)

                # Write to JSONL immediately (crash-safe)
                f.write(json.dumps(record) + "\n")
                f.flush()

                if (task_idx + 1) % 10 == 0:
                    logger.info(
                        "[%s] Completed %d/%d tasks",
                        agent_name, task_idx + 1, len(tasks_to_run),
                    )

        return records

    def _check_or_write_fingerprint(self, agent_name: str, per_task_path: Path) -> None:
        """
        Fail closed if this directory already holds results for ``agent_name``
        produced by something else (other checkpoint/model/seed), or results
        with no fingerprint at all (can't prove they're comparable).
        """
        meta_path = self._out_dir / f"per_task_{agent_name}.meta.json"
        fingerprint = {"agent": agent_name, **self._fingerprints.get(agent_name, {})}

        if per_task_path.exists() and per_task_path.stat().st_size > 0:
            if not meta_path.exists():
                raise RuntimeError(
                    f"{per_task_path} exists but has no {meta_path.name}; cannot "
                    "verify it came from this checkpoint/model. Use a fresh --out-dir."
                )
            existing = json.loads(meta_path.read_text())
            if existing != fingerprint:
                raise RuntimeError(
                    f"Refusing to resume {agent_name}: {per_task_path} was produced "
                    f"by {existing}, but this run is {fingerprint}. Use a fresh --out-dir."
                )
            return
        meta_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))

    def _run_single_episode(
        self,
        env: ATAREnv,
        task: Task,
        agent_name: str,
        agent: Any,
        task_idx: int,
    ) -> dict[str, Any]:
        """Run one episode and return a per-task record."""
        t0 = time.monotonic()

        # Reset agent state if applicable
        if hasattr(agent, "reset"):
            if isinstance(agent, ReActAgent):
                agent.reset(task.query)
            elif isinstance(agent, ZeroShotAgent):
                agent.reset()

        state, info = env.reset(task=task)
        done = False
        action_trace: list[str] = []
        tool_call_count = 0
        had_tool_failure = False
        final_answer = None
        terminated = False
        truncated = False

        while not done:
            action_int = agent.select_action(state)
            action = ActionType(action_int)
            action_trace.append(action.name)

            next_state, reward, term, trunc, step_info = env.step(action_int)
            done = term or trunc
            terminated = term
            truncated = trunc

            # Track tool calls (not ANSWER_DIRECTLY, not TERMINATE)
            if action not in (ActionType.ANSWER_DIRECTLY, ActionType.TERMINATE):
                tool_call_count += 1

            # Track tool failures — the env's flag also covers calls rejected
            # by M4 validation (those produce no ToolResult at all).
            if step_info.get("had_tool_failure", False):
                had_tool_failure = True
            final_answer = step_info.get("final_answer", final_answer)

            # If ReAct agent, feed back observation
            observe = getattr(agent, "observe", None)
            if observe is not None:
                observe(action_int, step_info)

            state = next_state

        latency_s = time.monotonic() - t0

        correct = check_answer_correct(final_answer, task.ground_truth)
        success = terminated and correct

        record: dict[str, Any] = {
            "task_id": task.id,
            "tier": task.tier,
            "query": task.query,
            "ground_truth": task.ground_truth,
            "agent": agent_name,
            "success": success,
            "correct": correct,
            "final_answer": final_answer,
            "tool_call_count": tool_call_count,
            "action_trace": action_trace,
            "latency_s": round(latency_s, 4),
            "had_tool_failure": had_tool_failure,
            "terminated": terminated,
            "truncated": truncated,
        }

        if not success:
            record["failure_mode"] = _categorize_failure(record)

        return record

    def evaluate_all(
        self,
        agents: dict[str, Any],
        max_tasks: int | None = None,
        bootstrap_seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run all agents and produce all output files.

        Parameters
        ----------
        agents : dict[str, agent]
            Mapping of agent name to agent object.
        max_tasks : int | None
            If set, only evaluate this many tasks per agent.
        bootstrap_seed : int
            RNG seed for bootstrap CI.

        Returns
        -------
        dict — full results structure.
        """
        all_results: dict[str, Any] = {}

        for agent_name, agent in agents.items():
            logger.info("=" * 60)
            logger.info("Evaluating agent: %s", agent_name)
            logger.info("=" * 60)

            records = self.run_agent(agent_name, agent, max_tasks=max_tasks)

            # Overall metrics
            overall = aggregate_metrics(records, bootstrap_seed=bootstrap_seed)

            # Per-tier metrics
            tiers = sorted(set(r["tier"] for r in records))
            per_tier: dict[str, Any] = {}
            for tier in tiers:
                tier_records = [r for r in records if r["tier"] == tier]
                per_tier[f"tier_{tier}"] = aggregate_metrics(
                    tier_records, bootstrap_seed=bootstrap_seed
                )

            all_results[agent_name] = {
                "overall": overall,
                "per_tier": per_tier,
            }

        # Write results.json
        results_path = self._out_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Results written to %s", results_path)

        # Merge all per-task files into a single per_task.jsonl
        merged_path = self._out_dir / "per_task.jsonl"
        with open(merged_path, "w") as out_f:
            for agent_name in agents:
                agent_path = self._out_dir / f"per_task_{agent_name}.jsonl"
                if agent_path.exists():
                    with open(agent_path, "r") as in_f:
                        for line in in_f:
                            out_f.write(line)
        logger.info("Merged per-task results to %s", merged_path)

        # Write failures.jsonl (ATAR agent failures for M6 prompt optimizer)
        failures_path = self._out_dir / "failures.jsonl"
        with open(failures_path, "w") as f:
            for agent_name in agents:
                agent_path = self._out_dir / f"per_task_{agent_name}.jsonl"
                if not agent_path.exists():
                    continue
                # Only include ATAR agent failures for M6
                if agent_name != "atar":
                    continue
                with open(agent_path, "r") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if not record.get("success", False):
                            f.write(json.dumps(record) + "\n")
        logger.info("Failures written to %s", failures_path)

        # Write comparison.md
        self._write_comparison_md(all_results)

        return all_results

    def _write_comparison_md(self, results: dict[str, Any]) -> None:
        """Write comparison.md with agents as columns, per-tier breakdown."""
        md_path = self._out_dir / "comparison.md"
        agents = list(results.keys())

        lines: list[str] = [
            "# ATAR Evaluation Comparison",
            "",
            f"Agents evaluated: {', '.join(agents)}",
            "",
            "## Overall Metrics",
            "",
        ]

        # Overall table
        header = "| Metric | " + " | ".join(agents) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(agents)) + "|"
        lines.extend([header, sep])

        metric_names = [
            ("accuracy", "Accuracy"),
            ("success_rate", "Success Rate"),
            ("avg_tool_calls", "Avg Tool Calls"),
            ("efficiency", "Efficiency"),
            ("recovery_rate", "Recovery Rate"),
            ("mean_latency", "Mean Latency (s)"),
        ]
        for key, label in metric_names:
            row = f"| {label} |"
            for agent in agents:
                val = results[agent]["overall"].get(key, 0)
                ci = results[agent]["overall"].get(f"{key}_ci")
                if ci:
                    row += f" {val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |"
                else:
                    row += f" {val:.4f} |"
            lines.append(row)

        lines.append("")
        lines.append("## Per-Tier Breakdown")
        lines.append("")

        # Per-tier tables
        all_tiers = set()
        for agent in agents:
            all_tiers.update(results[agent].get("per_tier", {}).keys())
        all_tiers_sorted = sorted(all_tiers)

        for tier_key in all_tiers_sorted:
            tier_num = tier_key.replace("tier_", "")
            lines.append(f"### Tier {tier_num}")
            lines.append("")
            header = "| Metric | " + " | ".join(agents) + " |"
            sep = "|--------|" + "|".join(["--------"] * len(agents)) + "|"
            lines.extend([header, sep])

            for key, label in metric_names:
                row = f"| {label} |"
                for agent in agents:
                    tier_data = results[agent].get("per_tier", {}).get(tier_key, {})
                    val = tier_data.get(key, 0)
                    row += f" {val:.4f} |"
                lines.append(row)
            lines.append("")

        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        logger.info("Comparison written to %s", md_path)
