"""
M8 live mode: run the real ATAR policy and the ReAct baseline on a query.

Needs a GPU host that can load the SAME model the checkpoint was trained on
(the policy's input is that model's hidden states; a hosted text-only API
cannot provide them).  Each agent gets its own ATAREnv, and episodes are
generators of StepRecords so the UI can step through them one action at a
time.
"""

from __future__ import annotations

import itertools
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from atar.demo.live_search import wikipedia_search
from atar.demo.replay import EpisodeRecord, StepRecord, iter_episode
from atar.env.atar_env import MAX_TURNS, ATAREnv
from atar.evaluation.runner import ReActAgent, load_atar_agent
from atar.llm.llm_interface import LLMInterface
from atar.shared.shared_types import Task

MAX_QUERY_CHARS = 1_000
_live_ids = itertools.count(1)


class QueryError(ValueError):
    """A free-text query the demo refuses to run (shown to the user, not raised to Streamlit)."""


def clean_query(text: object) -> str:
    """
    Validate a free-text query from the UI.  Returns it stripped; raises
    QueryError for anything that isn't a non-empty, reasonably sized string
    of printable text.
    """
    if not isinstance(text, str):
        raise QueryError("The query must be text.")
    # Drop control characters (keep ordinary whitespace).
    cleaned = "".join(
        ch for ch in text
        if ch in "\n\t " or unicodedata.category(ch)[0] != "C"
    ).strip()
    if not cleaned:
        raise QueryError("Please enter a question.")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise QueryError(f"Please keep the question under {MAX_QUERY_CHARS} characters.")
    return cleaned


def make_live_task(query: str, ground_truth: str = "") -> Task:
    """A Task for a typed query; tier 0 marks it as not from the dataset."""
    return Task(
        id=f"live-{next(_live_ids)}", tier=0, query=clean_query(query),
        required_tools=[], ground_truth=ground_truth.strip(), metadata={},
    )


@dataclass
class LiveEpisode:
    """One agent's in-progress episode: its record plus the step generator."""

    record: EpisodeRecord
    steps: Iterator[StepRecord] = field(repr=False)
    done: bool = False

    def advance(self) -> StepRecord | None:
        """Run one more step; returns None once the episode has ended."""
        if self.done:
            return None
        try:
            step = next(self.steps)
        except StopIteration:
            self.done = True
            return None
        # iter_episode marks the record when the episode ends.
        self.done = self.record.terminated or self.record.truncated
        return step

    def run_to_end(self) -> None:
        while self.advance() is not None:
            pass


class LiveRunner:
    """Loads the LLM + a checkpoint once; starts side-by-side live episodes."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        device: str = "cuda",
        data_dir: str | Path = "./data",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.data_dir = Path(data_dir)
        self.llm = LLMInterface(model_name=model_name, device=device)
        self.atar_agent, ckpt = load_atar_agent(checkpoint_path, self.llm, device)
        self.checkpoint_step = int(ckpt.step)

    def start(
        self,
        agent_name: str,
        task: Task,
        live_tools: bool = False,
        max_turns: int = MAX_TURNS,
    ) -> LiveEpisode:
        """Begin an episode for ``agent_name`` ("atar" or "react")."""
        env = ATAREnv(
            llm=self.llm, data_dir=self.data_dir,
            search_backend=wikipedia_search if live_tools else None,
            max_turns=max_turns,
        )
        if agent_name == "atar":
            agent: object = self.atar_agent
            version = f"step_{self.checkpoint_step}"
        elif agent_name == "react":
            agent = ReActAgent(llm=self.llm)
            version = "react"
        else:
            raise ValueError(f"Unknown live agent {agent_name!r}")
        record = EpisodeRecord(
            version=version, agent=agent_name,
            checkpoint_step=self.checkpoint_step if agent_name == "atar" else None,
            task_id=task.id, tier=task.tier, query=task.query,
            ground_truth=task.ground_truth, model_name=self.model_name,
            live_tools=live_tools,
        )
        return LiveEpisode(record=record, steps=iter_episode(env, agent, task, record))
