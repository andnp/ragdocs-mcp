"""Git commit search with glob filtering."""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.git.commit_indexer import CommitIndexer

logger = logging.getLogger(__name__)


@dataclass
class CommitResult:
    hash: str
    title: str
    author: str
    committer: str
    timestamp: int
    message: str
    files_changed: list[str]
    delta_truncated: str
    score: float
    repo_path: str


@dataclass
class GitSearchResponse:
    results: list[CommitResult]
    query: str
    total_commits_indexed: int


def search_git_history(
    commit_indexer: CommitIndexer,
    query: str,
    top_n: int = 5,
    files_glob: str | None = None,
    after_timestamp: int | None = None,
    before_timestamp: int | None = None,
) -> GitSearchResponse:
    """
    Search git commit history with optional filters.

    Args:
        commit_indexer: CommitIndexer instance
        query: Natural language query
        top_n: Maximum results to return
        files_glob: Optional glob pattern (e.g., 'src/**/*.py')
        after_timestamp: Optional Unix timestamp (commits after)
        before_timestamp: Optional Unix timestamp (commits before)

    Returns:
        GitSearchResponse with ranked commits
    """
    # Generate query embedding
    query_embedding = commit_indexer._embedding_model.get_text_embedding(query)

    # Query index (over-fetch for filtering)
    candidates = commit_indexer.query_by_embedding(
        query_embedding,
        top_k=top_n * 2 if files_glob else top_n,
        after_timestamp=after_timestamp,
        before_timestamp=before_timestamp,
    )

    # Apply glob filtering if specified
    if files_glob:
        candidates = filter_by_glob(candidates, files_glob)

    # Convert top N to CommitResult objects
    results = [
        CommitResult(
            hash=c["hash"],
            title=c["title"],
            author=c["author"],
            committer=c["committer"],
            timestamp=c["timestamp"],
            message=c["message"],
            files_changed=c["files_changed"],
            delta_truncated=c["delta_truncated"],
            score=c["score"],
            repo_path=c.get("repo_path", ""),
        )
        for c in candidates[:top_n]
    ]

    total = commit_indexer.get_total_commits()

    return GitSearchResponse(
        results=results,
        query=query,
        total_commits_indexed=total,
    )


def filter_by_glob(commits: list[dict], glob_pattern: str) -> list[dict]:
    """
    Filter commits by glob pattern matching any changed file.

    Args:
        commits: List of commit dicts with 'files_changed' key
        glob_pattern: Glob pattern (e.g., 'src/**/*.py')

    Returns:
        Filtered list of commits
    """
    return [
        commit
        for commit in commits
        if any(Path(f).match(glob_pattern) for f in commit.get("files_changed", []))
    ]
