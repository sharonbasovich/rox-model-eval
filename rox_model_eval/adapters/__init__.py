from ..config import ModelSpec, load_models
from ..system_one import LayaDecider
from .anthropic import AnthropicAdapter
from .base import ModelAdapter
from .hybrid import HybridAdapter
from .mock import MockAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .openai_responses import OpenAIResponsesAdapter

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "mock": MockAdapter,
    "openai": OpenAICompatibleAdapter,
    "openai_responses": OpenAIResponsesAdapter,
    "anthropic": AnthropicAdapter,
}


def build_adapter(spec: ModelSpec) -> ModelAdapter:
    if spec.adapter == "hybrid":
        return _build_hybrid(spec)
    try:
        return ADAPTERS[spec.adapter](spec)
    except KeyError as exc:
        raise ValueError(f"unknown adapter '{spec.adapter}' for model '{spec.id}'") from exc


_DECIDERS: dict[tuple[str, str | None, str], LayaDecider] = {}


def _build_hybrid(spec: ModelSpec) -> HybridAdapter:
    """`params.writer` names another model in the same models file; `params.decider`
    configures the local System One model, loaded once and shared."""
    p = spec.params
    writer = build_adapter(load_models(p.get("models_file", "config/models.yaml"))[p["writer"]])
    d = p.get("decider", {})
    key = (d.get("model", "convaiinnovations/laya"), d.get("revision"), d.get("device", "cpu"))
    if key not in _DECIDERS:
        _DECIDERS[key] = LayaDecider(*key)
    return HybridAdapter(spec, writer, _DECIDERS[key])


__all__ = ["ADAPTERS", "HybridAdapter", "ModelAdapter", "build_adapter"]
