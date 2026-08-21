"""Boundary contracts for the lexical query classifier."""

import pytest

from mcp_markdown_ragdocs.app.search import _is_lexical_query


@pytest.mark.parametrize(
    "query",
    [
        "app/search.py",
        "mcp_markdown_ragdocs/app/search.py",
        "README.md",
        "get_user_id",
        "getUserId",
        "SearchQuery",
        "FOO-123",
        "PROJ-1234",
        "  get_user_id  ",
    ],
)
def test_classifies_lexical_shapes_as_lexical(query: str) -> None:
    assert _is_lexical_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "boundary",
        "diagnostics",
        "well-known",
        "README",
        "Hello",
        "3.1",
        "",
        "   ",
        "what does foo_bar do",
        "how do I call get_user_id?",
        "details for PROJ-1234",
    ],
)
def test_rejects_natural_language_and_bare_words_as_non_lexical(query: str) -> None:
    assert _is_lexical_query(query) is False
