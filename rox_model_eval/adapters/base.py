"""Provider-agnostic model interface.

Every candidate model reaches the harness through this one method, so adding a
newly released model is a `config/models.yaml` entry rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import ModelSpec
from ..types import ModelRequest, ModelResponse


class ModelAdapter(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse: ...
