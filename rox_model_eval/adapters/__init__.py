from ..config import ModelSpec
from .anthropic import AnthropicAdapter
from .base import ModelAdapter
from .mock import MockAdapter
from .openai_compatible import OpenAICompatibleAdapter

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "mock": MockAdapter,
    "openai": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}


def build_adapter(spec: ModelSpec) -> ModelAdapter:
    try:
        return ADAPTERS[spec.adapter](spec)
    except KeyError as exc:
        raise ValueError(f"unknown adapter '{spec.adapter}' for model '{spec.id}'") from exc


__all__ = ["ADAPTERS", "ModelAdapter", "build_adapter"]
