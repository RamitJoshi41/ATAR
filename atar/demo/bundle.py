"""
M8 — the demo data bundle the Streamlit app reads.

A bundle is a directory:
  manifest.json   — model, creation date, and whether this is SAMPLE data
  curve.json      — learning curve + baselines (atar.demo.curve)
  replays.jsonl   — recorded episodes (atar.demo.replay.EpisodeRecord)

Built by scripts/build_demo_bundle.py from evaluate.py outputs and
scripts/record_replays.py outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atar.demo.replay import EpisodeRecord, load_records


@dataclass
class DemoTask:
    task_id: str
    tier: int
    query: str
    ground_truth: str


class DemoBundle:
    """Read-only view of a demo bundle directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Not a demo bundle (no manifest.json): {self.path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        curve_path = self.path / "curve.json"
        self.curve: dict[str, Any] = (
            json.loads(curve_path.read_text()) if curve_path.exists()
            else {"atar": [], "baselines": {}}
        )
        replays_path = self.path / "replays.jsonl"
        self.episodes: list[EpisodeRecord] = (
            load_records(replays_path) if replays_path.exists() else []
        )

    @property
    def is_sample(self) -> bool:
        return bool(self.manifest.get("is_sample", False))

    def atar_versions(self) -> list[tuple[str, int]]:
        """(version, training step) for every ATAR version with replays, by step."""
        seen = {
            (e.version, e.checkpoint_step) for e in self.episodes
            if e.agent == "atar" and e.checkpoint_step is not None
        }
        return sorted(seen, key=lambda v: v[1])  # type: ignore[arg-type, return-value]

    def baseline_versions(self) -> list[str]:
        return sorted({e.version for e in self.episodes if e.agent != "atar"})

    def tasks(self) -> list[DemoTask]:
        """Demo tasks with at least one replay, by tier then id."""
        by_id: dict[str, DemoTask] = {}
        for e in self.episodes:
            by_id.setdefault(e.task_id, DemoTask(e.task_id, e.tier, e.query, e.ground_truth))
        return sorted(by_id.values(), key=lambda t: (t.tier, t.task_id))

    def episode(self, version: str, task_id: str) -> EpisodeRecord | None:
        for e in self.episodes:
            if e.version == version and e.task_id == task_id:
                return e
        return None
