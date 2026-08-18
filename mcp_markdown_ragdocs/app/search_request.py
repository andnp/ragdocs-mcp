"""Transport-neutral construction of application search requests."""

from __future__ import annotations

from collections.abc import Iterable

from mcp_markdown_ragdocs.app.search import SearchQuery


def build_search_query(
    query: str,
    top_n: int,
    *,
    top_k: int | None = None,
    project_filter: Iterable[str] = (),
    source_filter: Iterable[str] = (),
    project_context: str | None = None,
    excluded_files: Iterable[str] = (),
    min_score: float | None = None,
    similarity_threshold: float | None = None,
    max_chunks_per_doc: int = 1,
) -> SearchQuery:
    """Build the canonical application request used by all transports."""

    project_filter = tuple(project_filter)
    source_filter = tuple(source_filter)
    excluded_files = frozenset(excluded_files)
    return SearchQuery(
        query=query,
        top_n=top_n,
        top_k=top_k if top_k is not None else search_top_k(top_n, project_filter),
        project_filter=project_filter,
        source_filter=source_filter,
        project_context=project_context,
        excluded_files=excluded_files,
        min_score=min_score,
        similarity_threshold=similarity_threshold,
        max_chunks_per_doc=max_chunks_per_doc,
    )


def search_top_k(top_n: int, project_filter: Iterable[str] = ()) -> int:
    """Return the existing application overfetch limit for a search request."""

    top_k = max(20, top_n * 4)
    if project_filter:
        top_k = max(top_k, top_n * 10)
    return top_k
