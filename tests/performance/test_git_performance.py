import subprocess
import time
from pathlib import Path

import pytest

from src.git.commit_indexer import CommitIndexer
from src.git.commit_parser import parse_commit, build_commit_document
from src.git.commit_search import search_git_history
from src.git.repository import get_commits_after_timestamp


def _init_git_repo(path: Path) -> None:
    """Initialize a git repository at path."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _create_commit(
    repo_path: Path,
    file_name: str,
    content: str,
    message: str,
) -> str:
    """Create a file and commit it, return commit hash."""
    file_path = repo_path / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_indexing_100_commits_performance(tmp_path, shared_embedding_model):
    """
    Performance test: Index 100 commits and measure time.

    Verifies that indexing completes in reasonable time.
    Target: < 30 seconds for 100 commits (with real embeddings).
    """
    repo_path = tmp_path / "large_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    # Create 100 commits
    print("\nCreating 100 commits...")
    commit_hashes = []
    start_creation = time.time()

    for i in range(100):
        hash_val = _create_commit(
            repo_path,
            f"file_{i}.py",
            f"def function_{i}():\n    return {i}\n",
            f"Add feature {i}",
        )
        commit_hashes.append(hash_val)

    creation_time = time.time() - start_creation
    print(f"Created 100 commits in {creation_time:.2f}s")

    # Index all commits
    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    git_dir = repo_path / ".git"
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    assert len(commits) == 100

    print("Indexing 100 commits...")
    start_indexing = time.time()

    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        indexer.add_commit(
            hash=commit_data.hash,
            timestamp=commit_data.timestamp,
            author=commit_data.author,
            committer=commit_data.committer,
            title=commit_data.title,
            message=commit_data.message,
            files_changed=commit_data.files_changed,
            delta_truncated=commit_data.delta_truncated,
            commit_document=doc,
            repo_path=str(git_dir),
        )

    indexing_time = time.time() - start_indexing
    print(f"Indexed 100 commits in {indexing_time:.2f}s")
    print(f"Average: {indexing_time / 100:.3f}s per commit")

    # Verify all commits were indexed
    assert indexer.get_total_commits() == 100

    # Performance assertion (generous threshold for CI)
    # With embedding generation, expect ~0.2-0.5s per commit
    assert indexing_time < 60.0, f"Indexing took {indexing_time:.2f}s, expected < 60s"


def test_search_query_latency(tmp_path, shared_embedding_model):
    """
    Performance test: Measure search query latency.

    Verifies that search queries complete quickly.
    Target: < 1 second for query over 50 commits.
    """
    repo_path = tmp_path / "search_perf_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    # Create 50 commits with diverse content
    for i in range(50):
        _create_commit(
            repo_path,
            f"module_{i % 10}.py",
            f"# Module {i}\ndef process_{i}():\n    return {i} * 2\n",
            f"Implement feature {i} for module {i % 10}",
        )

    # Index commits
    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    git_dir = repo_path / ".git"
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)

    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        indexer.add_commit(
            hash=commit_data.hash,
            timestamp=commit_data.timestamp,
            author=commit_data.author,
            committer=commit_data.committer,
            title=commit_data.title,
            message=commit_data.message,
            files_changed=commit_data.files_changed,
            delta_truncated=commit_data.delta_truncated,
            commit_document=doc,
            repo_path=str(git_dir),
        )

    # Warm up embedding model
    _ = search_git_history(indexer, "warmup query", top_n=5)

    # Measure search performance
    queries = [
        "feature implementation",
        "module changes",
        "bug fix",
        "process function",
    ]

    print("\nSearch query latency:")
    search_times = []

    for query in queries:
        start = time.time()
        response = search_git_history(indexer, query, top_n=10)
        elapsed = time.time() - start
        search_times.append(elapsed)

        print(f"  '{query}': {elapsed:.3f}s ({len(response.results)} results)")

        # Verify search worked
        assert len(response.results) > 0

    avg_search_time = sum(search_times) / len(search_times)
    print(f"Average search time: {avg_search_time:.3f}s")

    # Performance assertion
    assert avg_search_time < 2.0, f"Average search took {avg_search_time:.3f}s, expected < 2s"
    assert max(search_times) < 3.0, f"Slowest search took {max(search_times):.3f}s, expected < 3s"


@pytest.mark.skip(reason="Long-running performance test, run manually")
def test_indexing_1000_commits_performance(tmp_path, shared_embedding_model):
    """
    Performance test: Index 1000+ commits (manual run only).

    This test is skipped by default as it takes several minutes.
    Run with: pytest -k test_indexing_1000_commits -v

    Target: < 5 minutes for 1000 commits.
    """
    repo_path = tmp_path / "very_large_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    # Create 1000 commits
    print("\nCreating 1000 commits...")
    num_commits = 1000
    start_creation = time.time()

    for i in range(num_commits):
        if i % 100 == 0:
            print(f"  Created {i} commits...")

        _create_commit(
            repo_path,
            f"src/module_{i % 50}.py",
            f"# Module {i}\ndef function_{i}():\n    return {i}\n",
            f"Commit {i}: Update module {i % 50}",
        )

    creation_time = time.time() - start_creation
    print(f"Created {num_commits} commits in {creation_time:.2f}s")

    # Index all commits
    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    git_dir = repo_path / ".git"
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    assert len(commits) == num_commits

    print(f"Indexing {num_commits} commits...")
    start_indexing = time.time()

    for idx, commit_hash in enumerate(commits):
        if idx % 100 == 0:
            elapsed = time.time() - start_indexing
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  Indexed {idx}/{num_commits} ({rate:.1f} commits/s)")

        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        indexer.add_commit(
            hash=commit_data.hash,
            timestamp=commit_data.timestamp,
            author=commit_data.author,
            committer=commit_data.committer,
            title=commit_data.title,
            message=commit_data.message,
            files_changed=commit_data.files_changed,
            delta_truncated=commit_data.delta_truncated,
            commit_document=doc,
            repo_path=str(git_dir),
        )

    indexing_time = time.time() - start_indexing
    print(f"\nIndexed {num_commits} commits in {indexing_time:.2f}s")
    print(f"Average: {indexing_time / num_commits:.3f}s per commit")
    print(f"Rate: {num_commits / indexing_time:.1f} commits/s")

    # Verify
    assert indexer.get_total_commits() == num_commits

    # Performance assertion (5 minutes = 300 seconds)
    assert indexing_time < 300.0, f"Indexing took {indexing_time:.2f}s, expected < 300s"

    # Test search performance on large index
    start_search = time.time()
    response = search_git_history(indexer, "module update function", top_n=10)
    search_time = time.time() - start_search

    print(f"Search over {num_commits} commits: {search_time:.3f}s")
    assert len(response.results) == 10
    assert search_time < 3.0, f"Search took {search_time:.3f}s, expected < 3s"
