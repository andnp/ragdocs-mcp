from datetime import UTC, datetime

from mcp_markdown_ragdocs.git.commit_chunker import (
    DEFAULT_MAX_TOKENS,
    CommitChunk,
    chunk_commit,
    is_diff_chunk_eligible,
)
from mcp_markdown_ragdocs.git.commit_parser import CommitData


def _commit(
    *,
    author: str = "Author <author@example.com>",
    committer: str = "Committer <committer@example.com>",
    delta_truncated: str = (
            "diff --git a/src/search.py b/src/search.py\n"
            "@@ -1,2 +1,4 @@\n"
            "+semantic search\n"
            "diff --git a/tests/test_search.py b/tests/test_search.py\n"
            "@@ -1,1 +1,3 @@\n"
            "+retrieval coverage\n"
        ),
    files_changed: list[str] | None = None,
    files_changed_total: int | None = None,
    message: str = "Explain the search change.\n\nAdd retrieval coverage.",
) -> CommitData:
    resolved_files_changed = (
        ["src/search.py", "tests/test_search.py"]
        if files_changed is None
        else files_changed
    )
    return CommitData(
        hash="abc123",
        timestamp=1,
        author=author,
        committer=committer,
        title="Improve search",
        message=message,
        files_changed=resolved_files_changed,
        delta_truncated=delta_truncated,
        files_changed_total=(
            len(resolved_files_changed)
            if files_changed_total is None
            else files_changed_total
        ),
    )


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


def test_chunk_commit_summary_names_omitted_files_when_truncated() -> None:
    chunks = chunk_commit(
        _commit(
            files_changed=["src/search.py", "tests/test_search.py"],
            files_changed_total=12,
        )
    )

    summary_text = chunks[0].text
    assert "(+10 more files not shown)" in summary_text


def test_chunk_commit_summary_omits_omitted_line_when_not_truncated() -> None:
    chunks = chunk_commit(_commit())

    summary_text = chunks[0].text
    assert "more files not shown" not in summary_text


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


def test_chunk_commit_include_diff_false_drops_diff_chunks() -> None:
    chunks = chunk_commit(_commit(), include_diff=False)

    assert [chunk.section for chunk in chunks] == ["summary", "body"]


def test_chunk_commit_include_diff_true_keeps_diff_chunks() -> None:
    chunks = chunk_commit(_commit(), include_diff=True)

    assert "diff" in [chunk.section for chunk in chunks]


def test_is_diff_chunk_eligible_recent_commit_is_eligible() -> None:
    reference_time = datetime(2026, 1, 10, tzinfo=UTC)
    recent_timestamp = int(datetime(2026, 1, 9, tzinfo=UTC).timestamp())

    assert is_diff_chunk_eligible(recent_timestamp, 30, reference_time) is True


def test_is_diff_chunk_eligible_old_commit_is_ineligible() -> None:
    reference_time = datetime(2026, 1, 10, tzinfo=UTC)
    old_timestamp = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())

    assert is_diff_chunk_eligible(old_timestamp, 30, reference_time) is False


def test_is_diff_chunk_eligible_zero_days_always_eligible() -> None:
    reference_time = datetime(2026, 1, 10, tzinfo=UTC)
    old_timestamp = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())

    assert is_diff_chunk_eligible(old_timestamp, 0, reference_time) is True


def test_is_diff_chunk_eligible_at_exact_boundary_is_eligible() -> None:
    reference_time = datetime(2026, 1, 31, tzinfo=UTC)
    boundary_timestamp = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())

    assert is_diff_chunk_eligible(boundary_timestamp, 30, reference_time) is True
