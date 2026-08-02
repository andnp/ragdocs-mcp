"""Structure-aware chunking for Git commit embeddings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from mcp_markdown_ragdocs.git.commit_parser import CommitData

CommitChunkSection = Literal["summary", "body", "diff"]

DEFAULT_MAX_TOKENS = 448
DEFAULT_OVERLAP_TOKENS = 32
_ESTIMATED_CHARS_PER_TOKEN = 3
_DIFF_FILE_HEADER = re.compile(r"(?m)(?=^diff --(?:git|cc) )")


@dataclass(frozen=True)
class CommitChunk:
    """A bounded, semantically labeled portion of one commit."""

    section: CommitChunkSection
    section_index: int
    text: str
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without downloading a model tokenizer."""

    return max(1, math.ceil(len(text) / _ESTIMATED_CHARS_PER_TOKEN))


def chunk_commit(
    commit: CommitData,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> tuple[CommitChunk, ...]:
    """Split a commit by semantic section within a conservative token budget."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and less than max_tokens")

    chunks: list[CommitChunk] = []
    chunks.extend(
        _build_chunks(
            "summary",
            _summary_text(commit),
            max_tokens=max_tokens,
            overlap_tokens=0,
        )
    )
    chunks.extend(
        _build_chunks(
            "body",
            commit.message,
            max_tokens=max_tokens,
            overlap_tokens=0,
        )
    )
    for diff_file in _split_diff_files(commit.delta_truncated):
        chunks.extend(
            _build_chunks(
                "diff",
                diff_file,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return tuple(chunks)


def _summary_text(commit: CommitData) -> str:
    parts = [commit.title or "(no commit message)"]
    if commit.author:
        parts.append(f"Author: {commit.author}")
    if commit.committer:
        parts.append(f"Committer: {commit.committer}")
    if commit.files_changed:
        parts.append("Files changed:\n" + "\n".join(commit.files_changed))
    return "\n\n".join(parts)


def _split_diff_files(delta: str) -> list[str]:
    if not delta.strip():
        return []
    files = [part.strip() for part in _DIFF_FILE_HEADER.split(delta) if part.strip()]
    return files or [delta.strip()]


def _build_chunks(
    section: CommitChunkSection,
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[CommitChunk]:
    normalized = text.strip()
    if not normalized:
        return []

    prefix = f"{section.title()}:\n"
    content_budget = max_tokens - estimate_tokens(prefix)
    if content_budget < 1:
        raise ValueError("max_tokens is too small for section label")

    content_chunks = _split_lines(
        normalized,
        max_tokens=content_budget,
        overlap_tokens=overlap_tokens,
    )
    return [
        CommitChunk(
            section=section,
            section_index=index,
            text=prefix + content,
            estimated_tokens=estimate_tokens(prefix + content),
        )
        for index, content in enumerate(content_chunks)
    ]


def _split_lines(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
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
    "CommitChunk",
    "CommitChunkSection",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "chunk_commit",
    "estimate_tokens",
]
