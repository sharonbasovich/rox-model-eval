"""OpenAI Responses API adapter (`/v1/responses`).

Needed for models that only support function tools together with reasoning on the
Responses API (e.g. GPT-6). Streams for time-to-first-token; token usage, including
cached input, comes from the provider's `response.completed` event.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from ..types import Message, ModelRequest, ModelResponse, Timings, ToolCall, Usage
from .base import ModelAdapter

_FIRST_TOKEN_EVENTS = {"response.output_text.delta", "response.function_call_arguments.delta"}


_RETRY_DELAYS_S = (2.0, 5.0, 15.0, 30.0)


def _retryable(error: str | None) -> bool:
    return bool(error) and str(error).startswith(
        ("HTTP 429", "HTTP 5", "stream ended", "transport ")
    )


def to_responses_input(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content}
            )
            continue
        if m.content or not m.tool_calls:
            items.append({"role": m.role, "content": m.content})
        for tc in m.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                }
            )
    return items


def to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Chat Completions `{type, function: {...}}` tool into Responses form."""
    fn = tool.get("function")
    return {"type": "function", **fn} if fn else tool


def parse_output(output: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
    text: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if item.get("type") == "message":
            text += [
                c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"
            ]
        elif item.get("type") == "function_call":
            try:
                args = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": item.get("arguments")}
            calls.append(
                ToolCall(id=item.get("call_id", ""), name=item.get("name", ""), arguments=args)
            )
    return "".join(text), calls


def parse_usage(u: dict[str, Any] | None) -> Usage:
    u = u or {}
    return Usage(
        prompt_tokens=u.get("input_tokens", 0),
        completion_tokens=u.get("output_tokens", 0),
        cached_prompt_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens", 0),
    )


class OpenAIResponsesAdapter(ModelAdapter):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Retries rate limits, server errors and dropped connections with backoff."""
        response = self._complete_once(request)
        for delay in _RETRY_DELAYS_S:
            if not _retryable(response.error):
                break
            time.sleep(delay)
            response = self._complete_once(request)
        return response

    def _complete_once(self, request: ModelRequest) -> ModelResponse:
        key_env = self.spec.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(key_env)
        if not api_key:
            return ModelResponse(error=f"missing API key env var {key_env}")

        body: dict[str, Any] = {
            "model": self.spec.model,
            "input": to_responses_input(request.messages),
            "stream": True,
            **self.spec.params,
        }
        if request.tools:
            body["tools"] = [to_responses_tool(t) for t in request.tools]
        if request.json_schema:
            body["text"] = {
                "format": {"type": "json_schema", "name": "output", "schema": request.json_schema}
            }

        url = (self.spec.base_url or "https://api.openai.com/v1").rstrip("/")
        started = time.perf_counter()
        ttft: float | None = None
        final: dict[str, Any] | None = None
        try:
            with httpx.stream(
                "POST",
                f"{url}/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=300,
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    return ModelResponse(error=f"HTTP {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    kind = event.get("type", "")
                    if kind in _FIRST_TOKEN_EVENTS and ttft is None:
                        ttft = time.perf_counter() - started
                    if kind in ("response.completed", "response.incomplete", "response.failed"):
                        final = event.get("response") or {}
        except httpx.HTTPError as exc:
            return ModelResponse(error=f"transport {type(exc).__name__}: {exc}")

        if final is None:
            return ModelResponse(error="stream ended without a completed response")
        timings = Timings(ttft_s=ttft, total_s=time.perf_counter() - started)
        usage = parse_usage(final.get("usage"))
        if final.get("status") == "failed":
            err = (final.get("error") or {}).get("message", "response failed")
            return ModelResponse(usage=usage, timings=timings, error=err)
        text, calls = parse_output(final.get("output") or [])
        return ModelResponse(
            text=text,
            tool_calls=calls,
            usage=usage,
            timings=timings,
            provider_version=final.get("model"),
        )
