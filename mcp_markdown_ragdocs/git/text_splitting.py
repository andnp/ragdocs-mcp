"""Generic token-bounded, overlap-tail text splitting.

Splits arbitrary line-oriented text into chunks that stay under an
estimated token budget, carrying a trailing-line overlap into the next
chunk for context continuity. Contains no domain-specific formatting.
"""

from __future__ import annotations

import math

_ESTIMATED_CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without downloading a model tokenizer."""

    return max(1, math.ceil(len(text) / _ESTIMATED_CHARS_PER_TOKEN))


def split_text_by_tokens(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into line-bounded chunks within a token budget.

    Each chunk after the first carries a trailing-line overlap from the
    previous chunk, up to ``overlap_tokens``.
    """

    max_chars = max_tokens * _ESTIMATED_CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _ESTIMATED_CHARS_PER_TOKEN
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for line in lines:
        line_parts = _split_long_line(line, max_chars)
        for part in line_parts:
            part_chars = len(part)
            if current and current_chars + part_chars + 1 > max_chars:
                chunk = "\n".join(current).strip()
                if chunk:
                    chunks.append(chunk)
                overlap = _overlap_tail(current, overlap_chars)
                current = overlap
                current_chars = sum(len(item) for item in current) + max(
                    len(current) - 1, 0
                )
            current.append(part)
            current_chars += part_chars + (1 if len(current) > 1 else 0)

    chunk = "\n".join(current).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _split_long_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    return [line[start : start + max_chars] for start in range(0, len(line), max_chars)]


def _overlap_tail(lines: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    selected: list[str] = []
    total = 0
    for line in reversed(lines):
        if total + len(line) > overlap_chars and selected:
            break
        selected.append(line)
        total += len(line)
    return list(reversed(selected))


__all__ = [
    "estimate_tokens",
    "split_text_by_tokens",
]
