"""Provider-agnostic model interface.

Every candidate model reaches the harness through this one method, so adding a
newly released model is a `config/models.yaml` entry rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import ModelSpec
from ..types import ModelRequest, ModelResponse, RunOutput, Suite, Task


class ModelAdapter(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse: ...

    def run_task(
        self, suite: Suite, task: Task, rep: int
    ) -> tuple[ModelResponse, RunOutput] | None:
        """Own the whole attempt instead of the default loop; None means use the default."""
        return None
