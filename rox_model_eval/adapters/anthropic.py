"""Anthropic Messages API adapter (streaming, for time-to-first-token)."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from ..types import ModelRequest, ModelResponse, Timings, ToolCall, Usage
from .base import ModelAdapter


class AnthropicAdapter(ModelAdapter):
    def complete(self, request: ModelRequest) -> ModelResponse:
        key_env = self.spec.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(key_env)
        if not api_key:
            return ModelResponse(error=f"missing API key env var {key_env}")

        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages = [m.model_dump() for m in request.messages if m.role != "system"]
        params = dict(self.spec.params)
        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": messages,
            "max_tokens": params.pop("max_tokens", 2048),
            "stream": True,
            **params,
        }
        if system:
            body["system"] = system
        if request.tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {"type": "object"}),
                }
                for t in request.tools
            ]

        url = (self.spec.base_url or "https://api.anthropic.com").rstrip("/")
        started = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        tools: dict[int, dict[str, str]] = {}
        usage = Usage()
        version: str | None = None
        try:
            with httpx.stream(
                "POST",
                f"{url}/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
                timeout=120,
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    return ModelResponse(error=f"HTTP {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    kind = event.get("type")
                    if kind == "message_start":
                        msg = event["message"]
                        version = msg.get("model")
                        u = msg.get("usage", {})
                        usage.prompt_tokens = u.get("input_tokens", 0) + u.get(
                            "cache_read_input_tokens", 0
                        )
                        usage.cached_prompt_tokens = u.get("cache_read_input_tokens", 0)
                    elif kind == "content_block_start":
                        block = event["content_block"]
                        if block.get("type") == "tool_use":
                            tools[event["index"]] = {"name": block["name"], "args": ""}
                    elif kind == "content_block_delta":
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        delta = event["delta"]
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta["text"])
                        elif delta.get("type") == "input_json_delta":
                            tools[event["index"]]["args"] += delta["partial_json"]
                    elif kind == "message_delta":
                        usage.completion_tokens = event.get("usage", {}).get("output_tokens", 0)
        except httpx.HTTPError as exc:
            return ModelResponse(error=f"{type(exc).__name__}: {exc}")

        tool_calls = []
        for slot in tools.values():
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": slot["args"]}
            tool_calls.append(ToolCall(name=slot["name"], arguments=args))

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            timings=Timings(ttft_s=ttft, total_s=time.perf_counter() - started),
            provider_version=version,
        )
