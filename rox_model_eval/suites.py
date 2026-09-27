"""Suite loading, plus deterministic generators for inputs too long to hand-write."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import yaml

from .types import Suite

ALL_SUITES = [
    "c1_research",
    "c2_drafting",
    "c3_insights",
    "c4_grounded_qa",
    "c5_tool_calling",
    "c6_extraction",
    "c7_long_context",
    "c8_safety",
]

_SPEAKERS = ["Rep", "Customer", "Customer (IT)", "Rep (SE)"]
_FILLER = [
    "{s}: Sorry, can everyone hear me okay? My connection has been a little choppy today.",
    "{s}: We talked about the dashboard layout last time, and people liked the new filters.",
    "{s}: Let me pull up the slide from last quarter's review so we're looking at the same thing.",
    "{s}: Honestly the team has been heads-down on the migration, so bandwidth is tight.",
    "{s}: That's a good question, I think it depends on how the regional teams roll it out.",
    "{s}: We had a similar conversation with procurement, they mostly care about the timeline.",
    "{s}: I'll be out part of next month, but my manager can cover any urgent questions.",
    "{s}: The weather here has been wild, we had three snow days in a row last week.",
    "{s}: Right now most of the reporting still happens in spreadsheets, which is painful.",
    "{s}: I think the pilot group liked the email summaries more than the chat interface.",
    "{s}: We'd want to loop in security at some point, but not for this early discussion.",
    "{s}: Let's not go too deep on pricing today, I'd rather understand the workflow first.",
]


def _suite_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_transcript(spec: dict[str, Any]) -> str:
    """Deterministic long call transcript with needle facts planted at given depths.

    `spec`: {seed, target_words, needles: [{text, depth}], distractors: [{text, depth}]}
    where depth is 0..1 through the transcript.
    """
    rng = random.Random(spec.get("seed", 0))
    target_words = int(spec.get("target_words", 6000))
    lines: list[str] = []
    words = 0
    while words < target_words:
        line = rng.choice(_FILLER).format(s=rng.choice(_SPEAKERS))
        lines.append(line)
        words += len(line.split())

    planted = [*spec.get("needles", []), *spec.get("distractors", [])]
    for item in sorted(planted, key=lambda p: -float(p["depth"])):
        idx = min(len(lines), max(0, round(float(item["depth"]) * len(lines))))
        lines.insert(idx, item["text"])
    return "\n".join(f"[{i // 4:02d}:{(i % 4) * 15:02d}] {line}" for i, line in enumerate(lines))


def load_suite(suites_dir: str | Path, name: str) -> Suite:
    path = Path(suites_dir) / name / "suite.yaml"
    if not path.exists():
        raise FileNotFoundError(f"suite '{name}' not found at {path}")
    suite = Suite(**yaml.safe_load(path.read_text()))
    for task in suite.tasks:
        spec = task.inputs.get("transcript_spec")
        if isinstance(spec, dict):
            task.inputs["transcript"] = build_transcript(spec)
    return suite


def suite_hash(suites_dir: str | Path, name: str) -> str:
    return _suite_hash(Path(suites_dir) / name / "suite.yaml")


def resolve_suites(arg: str) -> list[str]:
    if arg.strip() == "all":
        return list(ALL_SUITES)
    return [s.strip() for s in arg.split(",") if s.strip()]
