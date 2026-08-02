"""Git search result aggregation."""

from __future__ import annotations

from collections.abc import Sequence

from mcp_markdown_ragdocs.models import ChunkResult


def aggregate_commit_results(results: Sequence[ChunkResult]) -> list[ChunkResult]:
    """Return one best-scoring result per commit while retaining match metadata."""

    grouped: dict[str, list[ChunkResult]] = {}
    passthrough: list[ChunkResult] = []
    for result in results:
        if result.metadata.get("source_kind") != "git_commit":
            passthrough.append(result)
            continue
        commit_id = str(result.metadata.get("commit_id", result.doc_id))
        grouped.setdefault(commit_id, []).append(result)

    aggregated = list(passthrough)
    for matches in grouped.values():
        best = max(matches, key=lambda result: result.score)
        sections = list(
            dict.fromkeys(
                str(match.metadata.get("chunk_section"))
                for match in matches
                if match.metadata.get("chunk_section") is not None
            )
        )
        metadata = {
            **best.metadata,
            "matched_sections": sections,
            "matched_chunk_count": len(matches),
        }
        aggregated.append(
            ChunkResult(
                chunk_id=best.chunk_id,
                doc_id=str(best.metadata.get("commit_id", best.doc_id)),
                score=best.score,
                header_path=best.header_path,
                file_path=best.file_path,
                project_id=best.project_id,
                content=best.content,
                parent_chunk_id=best.parent_chunk_id,
                parent_content=best.parent_content,
                provenance=best.provenance,
                metadata=metadata,
            )
        )
    return sorted(aggregated, key=lambda result: result.score, reverse=True)


__all__ = ["aggregate_commit_results"]
