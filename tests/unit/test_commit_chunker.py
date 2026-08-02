from mcp_markdown_ragdocs.git.commit_chunker import (
    DEFAULT_MAX_TOKENS,
    CommitChunk,
    chunk_commit,
)
from mcp_markdown_ragdocs.git.commit_parser import CommitData


def _commit(**overrides: object) -> CommitData:
    values: dict[str, object] = {
        "hash": "abc123",
        "timestamp": 1,
        "author": "Author <author@example.com>",
        "committer": "Committer <committer@example.com>",
        "title": "Improve search",
        "message": "Explain the search change.\n\nAdd retrieval coverage.",
        "files_changed": ["src/search.py", "tests/test_search.py"],
        "delta_truncated": (
            "diff --git a/src/search.py b/src/search.py\n"
            "@@ -1,2 +1,4 @@\n"
            "+semantic search\n"
            "diff --git a/tests/test_search.py b/tests/test_search.py\n"
            "@@ -1,1 +1,3 @@\n"
            "+retrieval coverage\n"
        ),
    }
    values.update(overrides)
    return CommitData(**values)


def test_chunk_commit_preserves_commit_structure() -> None:
    chunks = chunk_commit(_commit())

    assert [chunk.section for chunk in chunks] == [
        "summary",
        "body",
        "diff",
        "diff",
    ]
    assert chunks[0].text.startswith("Summary:\nImprove search")
    assert "Explain the search change." in chunks[1].text
    assert "src/search.py" in chunks[2].text
    assert "tests/test_search.py" in chunks[3].text


def test_chunk_commit_bounds_long_diff_chunks() -> None:
    delta = "diff --git a/src/app.py b/src/app.py\n" + "\n".join(
        f"+line {index} with implementation detail" for index in range(200)
    )

    chunks = chunk_commit(
        _commit(delta_truncated=delta),
        max_tokens=40,
        overlap_tokens=4,
    )

    diff_chunks = [chunk for chunk in chunks if chunk.section == "diff"]
    assert len(diff_chunks) > 1
    assert all(chunk.estimated_tokens <= 40 for chunk in chunks)
    assert "line 0" in diff_chunks[0].text
    assert "line 199" in diff_chunks[-1].text


def test_chunk_commit_keeps_empty_sections_out() -> None:
    chunks = chunk_commit(
        _commit(
            message="",
            author="",
            committer="",
            files_changed=[],
            delta_truncated="",
        )
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], CommitChunk)
    assert chunks[0].section == "summary"
    assert chunks[0].estimated_tokens <= DEFAULT_MAX_TOKENS
