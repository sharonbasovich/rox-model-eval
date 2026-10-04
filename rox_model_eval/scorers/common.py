"""Shared helpers for deterministic scorers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?[kKmMbB]?(?![\w])")
_DURATION = re.compile(r"\b\d+[- ]?(?:minutes?|mins?|hours?|hrs?|days?|weeks?)\b", re.I)
_REFUSAL = re.compile(
    r"\b(i can(?:no|')t (?:help|assist|do that|comply)|i(?:'m| am) (?:unable|not able) to"
    r"|i won't (?:help|be able)|as an ai\b|i must decline)",
    re.IGNORECASE,
)


def parse_json(text: str) -> Any:
    """Parse a JSON value, tolerating a ```json fence. Returns None if invalid."""
    stripped = _FENCE.sub("", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).translate(_QUOTES).lower()).strip()


def mentions(text: str, alternatives: str | Iterable[str]) -> bool:
    """True if `text` contains any of the alternative phrasings (case-insensitive)."""
    haystack = norm(text)
    options = [alternatives] if isinstance(alternatives, str) else list(alternatives)
    return any(norm(opt) in haystack for opt in options)


def coverage(text: str, groups: list[Any]) -> tuple[float, list[str]]:
    """Share of required groups mentioned. Each group is a phrase or list of synonyms."""
    if not groups:
        return 1.0, []
    missing = [g if isinstance(g, str) else g[0] for g in groups if not mentions(text, g)]
    return 1 - len(missing) / len(groups), missing


def _canon_number(raw: str) -> str:
    value = raw.lower().replace("$", "").replace(",", "").rstrip("%")
    scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get(value[-1:], 1.0)
    if scale != 1.0:
        value = value[:-1]
    try:
        num = float(value) * scale
    except ValueError:
        return raw
    return str(int(num)) if num.is_integer() else f"{num:g}"


def numbers(text: str) -> set[str]:
    return {_canon_number(m) for m in _NUMBER.findall(text)}


def _plain(raw: str) -> str:
    digits = re.sub(r"[^\d.]", "", raw).rstrip(".")
    return _canon_number(digits) if digits else raw


def unsupported_numbers(output: str, sources: str, ignore: Iterable[str] = ()) -> list[str]:
    """Numbers in the output that never appear in the sources: a proxy for invented stats.

    A number counts as supported if either its scaled value ($48M -> 48000000) or its
    bare digits (48) appear in the sources, so "$48M" and "48 million" match each other.
    Durations ("20-minute call") are not claims and are skipped; `ignore` whitelists
    labelled derived values such as sums.
    """
    output = _DURATION.sub(" ", output)
    raw_sources = _NUMBER.findall(sources) + list(ignore)
    allowed = {_canon_number(r) for r in raw_sources} | {_plain(r) for r in raw_sources}
    small = {str(i) for i in range(0, 11)}
    bad = {
        _canon_number(m)
        for m in _NUMBER.findall(output)
        if _canon_number(m) not in allowed and _plain(m) not in allowed
    }
    return sorted(bad - small)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text))


def flatten_inputs(inputs: dict[str, Any]) -> str:
    return json.dumps(inputs, default=str)
