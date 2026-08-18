"""Tests for transport-neutral application search request mapping."""

from mcp_markdown_ragdocs.app.search_request import build_search_query


def test_build_search_query_preserves_application_search_controls() -> None:
    """Preserve application search controls.

    Filters, exclusions, thresholds, and uniqueness remain normalized.
    """

    request = build_search_query(
        "transport boundary",
        3,
        project_filter=["project-a"],
        source_filter=["git_commit"],
        project_context="project-a",
        excluded_files=["private.md"],
        min_score=0.25,
        similarity_threshold=0.9,
        max_chunks_per_doc=1,
    )

    assert request.query == "transport boundary"
    assert request.top_n == 3
    assert request.top_k == 30
    assert request.project_filter == ("project-a",)
    assert request.source_filter == ("git_commit",)
    assert request.project_context == "project-a"
    assert request.excluded_files == frozenset({"private.md"})
    assert request.min_score == 0.25
    assert request.similarity_threshold == 0.9
    assert request.max_chunks_per_doc == 1


def test_build_search_query_preserves_default_overfetch() -> None:
    """Keep the default application overfetch.

    A five-result request still retrieves the existing minimum of twenty.
    """

    request = build_search_query("default search", 5)

    assert request.top_k == 20
    assert request.project_filter == ()
    assert request.source_filter == ()
    assert request.excluded_files == frozenset()
