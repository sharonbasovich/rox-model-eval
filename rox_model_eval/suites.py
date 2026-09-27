"""Suite loading. A capability suite is a YAML file of tasks plus a system prompt."""

from __future__ import annotations

from pathlib import Path

import yaml

from .types import Suite


def load_suite(suites_dir: str | Path, name: str) -> Suite:
    path = Path(suites_dir) / name / "suite.yaml"
    if not path.exists():
        raise FileNotFoundError(f"suite '{name}' not found at {path}")
    return Suite(**yaml.safe_load(path.read_text()))
