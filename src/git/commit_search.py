"""Git commit search with glob filtering."""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import Config, resolve_project_id_for_path
from src.git.commit_indexer import CommitIndexer
from src.search.filters import matches_project_filter, normalize_project_filter

logger = logging.getLogger(__name__)

_ACTIVE_PROJECT_UPLIFT = 1.2


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
    project_id: str | None = None


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
    project_filter: list[str] | None = None,
    project_context: str | None = None,
    config: Config | None = None,
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
    normalized_project_filter = normalize_project_filter(project_filter)

    overfetch_multiplier = 1
    if files_glob:
        overfetch_multiplier = 2
    if normalized_project_filter:
        overfetch_multiplier = max(overfetch_multiplier, 10)
    candidates = commit_indexer.query_by_embedding(
        query_embedding,
        top_k=top_n * overfetch_multiplier,
        after_timestamp=after_timestamp,
        before_timestamp=before_timestamp,
    )

    # Apply glob filtering if specified
    if files_glob:
        candidates = filter_by_glob(candidates, files_glob)

    candidates = apply_project_semantics(
        candidates,
        config=config,
        project_filter=normalized_project_filter,
        project_context=project_context,
    )

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
            project_id=c.get("project_id"),
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


def apply_project_semantics(
    commits: list[dict],
    *,
    config: Config | None,
    project_filter: set[str] | None,
    project_context: str | None,
) -> list[dict]:
    enriched: list[dict] = []
    for commit in commits:
        repo_path = commit.get("repo_path")
        project_id = None
        if config is not None and isinstance(repo_path, str) and repo_path:
            project_id = resolve_project_id_for_path(Path(repo_path), config)

        if not matches_project_filter(project_id, project_filter):
            continue

        score = float(commit.get("score", 0.0))
        if project_context and project_id == project_context:
            score *= _ACTIVE_PROJECT_UPLIFT

        enriched.append({**commit, "project_id": project_id, "score": score})

    return sorted(enriched, key=lambda commit: float(commit.get("score", 0.0)), reverse=True)
