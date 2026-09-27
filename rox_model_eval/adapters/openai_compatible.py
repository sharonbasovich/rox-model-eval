"""OpenAI-compatible Chat Completions adapter (OpenAI, most gateways, vLLM, etc.).

Streams so time-to-first-token is measured, and requests `include_usage` so token
counts (including cached prompt tokens) come from the provider, not an estimate.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from ..types import Message, ModelRequest, ModelResponse, Timings, ToolCall, Usage
from .base import ModelAdapter


def to_openai_message(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["content"] = m.content or None
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    if m.role == "tool":
        out["tool_call_id"] = m.tool_call_id
    return out


class OpenAICompatibleAdapter(ModelAdapter):
    def complete(self, request: ModelRequest) -> ModelResponse:
        key_env = self.spec.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(key_env)
        if not api_key:
            return ModelResponse(error=f"missing API key env var {key_env}")

        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [to_openai_message(m) for m in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            **{k: v for k, v in self.spec.params.items()},
        }
        if request.tools:
            body["tools"] = request.tools
        if request.json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema},
            }

        url = (self.spec.base_url or "https://api.openai.com/v1").rstrip("/")
        started = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        tool_buf: dict[int, dict[str, str]] = {}
        usage = Usage()
        version: str | None = None
        try:
            with httpx.stream(
                "POST",
                f"{url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=120,
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    return ModelResponse(error=f"HTTP {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    version = chunk.get("model", version)
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        details = u.get("prompt_tokens_details") or {}
                        usage = Usage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            cached_prompt_tokens=details.get("cached_tokens", 0),
                        )
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if (delta.get("content") or delta.get("tool_calls")) and ttft is None:
                            ttft = time.perf_counter() - started
                        if delta.get("content"):
                            text_parts.append(delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = tool_buf.setdefault(
                                tc["index"], {"id": "", "name": "", "args": ""}
                            )
                            slot["id"] = slot["id"] or tc.get("id") or ""
                            fn = tc.get("function", {})
                            slot["name"] += fn.get("name") or ""
                            slot["args"] += fn.get("arguments") or ""
        except httpx.HTTPError as exc:
            return ModelResponse(error=f"{type(exc).__name__}: {exc}")

        tool_calls = []
        for slot in tool_buf.values():
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": slot["args"]}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            timings=Timings(ttft_s=ttft, total_s=time.perf_counter() - started),
            provider_version=version,
        )
