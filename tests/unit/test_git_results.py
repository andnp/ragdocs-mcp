"""Tests for logical Git commit result aggregation."""

from mcp_markdown_ragdocs.git.results import aggregate_commit_results
from mcp_markdown_ragdocs.models import ChunkResult


def _result(
    chunk_id: str,
    score: float,
    section: str,
    *,
    commit_id: str = "git:abc123",
) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        doc_id=f"{commit_id}:{section}:0",
        score=score,
        header_path="",
        file_path="",
        content=f"{section} content",
        metadata={
            "source_kind": "git_commit",
            "commit_id": commit_id,
            "chunk_section": section,
        },
    )


def test_aggregate_commit_results_returns_best_chunk_per_commit() -> None:
    results = aggregate_commit_results(
        [
            _result("summary", 0.6, "summary"),
            _result("body", 0.9, "body"),
            _result("other", 0.8, "diff", commit_id="git:def456"),
        ]
    )

    assert [result.doc_id for result in results] == ["git:abc123", "git:def456"]
    assert results[0].chunk_id == "body"
    assert results[0].metadata["matched_sections"] == ["summary", "body"]
    assert results[0].metadata["matched_chunk_count"] == 2


def test_aggregate_commit_results_preserves_non_git_documents() -> None:
    result = ChunkResult(
        chunk_id="note-chunk",
        doc_id="note-1",
        score=0.5,
        header_path="",
        file_path="note.md",
        content="note",
        metadata={"source_kind": "note"},
    )

    assert aggregate_commit_results([result]) == [result]
