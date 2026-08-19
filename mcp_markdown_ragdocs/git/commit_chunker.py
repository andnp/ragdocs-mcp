"""Structure-aware chunking for Git commit embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from mcp_markdown_ragdocs.git.commit_parser import CommitData
from mcp_markdown_ragdocs.git.text_splitting import estimate_tokens, split_text_by_tokens

CommitChunkSection = Literal["summary", "body", "diff"]

DEFAULT_MAX_TOKENS = 448
DEFAULT_OVERLAP_TOKENS = 32
_DIFF_FILE_HEADER = re.compile(r"(?m)(?=^diff --(?:git|cc) )")


@dataclass(frozen=True)
class CommitChunk:
    """A bounded, semantically labeled portion of one commit."""

    section: CommitChunkSection
    section_index: int
    text: str
    estimated_tokens: int


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
        files_section = "Files changed:\n" + "\n".join(commit.files_changed)
        omitted = commit.files_changed_total - len(commit.files_changed)
        if omitted > 0:
            files_section += f"\n(+{omitted} more files not shown)"
        parts.append(files_section)
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

    content_chunks = split_text_by_tokens(
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


__all__ = [
    "CommitChunk",
    "CommitChunkSection",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "chunk_commit",
    "estimate_tokens",
]
