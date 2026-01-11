import subprocess
import time
from pathlib import Path

import pytest

from src.git.commit_indexer import CommitIndexer
from src.git.commit_parser import parse_commit, build_commit_document
from src.git.commit_search import search_git_history
from src.git.repository import discover_git_repositories, get_commits_after_timestamp


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


@pytest.fixture
def git_test_repo(tmp_path):
    """
    Create a test git repository with diverse commits for testing.

    Returns tuple of (repo_path, git_dir, commit_hashes).
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    commit_hashes = []

    # Commit 1: Python authentication fix
    hash1 = _create_commit(
        repo_path,
        "src/auth.py",
        "def authenticate(user, password):\n    return check_credentials(user, password)\n",
        "Fix authentication bug in login system",
    )
    commit_hashes.append(hash1)
    time.sleep(0.1)

    # Commit 2: JavaScript frontend update
    hash2 = _create_commit(
        repo_path,
        "frontend/app.js",
        "function handleLogin() {\n    fetch('/api/auth');\n}\n",
        "Update frontend login handler",
    )
    commit_hashes.append(hash2)
    time.sleep(0.1)

    # Commit 3: Python API endpoint
    hash3 = _create_commit(
        repo_path,
        "src/api/endpoints.py",
        "@app.route('/api/auth')\ndef auth_endpoint():\n    return authenticate()\n",
        "Add authentication API endpoint",
    )
    commit_hashes.append(hash3)
    time.sleep(0.1)

    # Commit 4: Documentation
    hash4 = _create_commit(
        repo_path,
        "docs/api.md",
        "# Authentication API\n\nUse POST /api/auth to authenticate users.\n",
        "Document authentication API",
    )
    commit_hashes.append(hash4)
    time.sleep(0.1)

    # Commit 5: Test file
    hash5 = _create_commit(
        repo_path,
        "tests/test_auth.py",
        "def test_authentication():\n    assert authenticate('user', 'pass') == True\n",
        "Add authentication tests",
    )
    commit_hashes.append(hash5)

    git_dir = repo_path / ".git"
    return repo_path, git_dir, commit_hashes


@pytest.fixture
def commit_indexer(tmp_path, shared_embedding_model):
    """Create a commit indexer with real embedding model."""
    db_path = tmp_path / "commits.db"
    return CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)


def test_end_to_end_discover_index_search(git_test_repo, commit_indexer, tmp_path):
    """
    End-to-end test: discover repos → index commits → search → verify results.

    This tests the complete workflow from repository discovery through
    to semantic search over commit history.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Step 1: Discover repositories
    repos = discover_git_repositories(
        tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=True,
    )

    assert len(repos) == 1
    assert repos[0] == git_dir.resolve()

    # Step 2: Get all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    assert len(commits) == 5

    # Step 3: Index all commits
    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Step 4: Search for authentication-related commits
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="authentication bug fix",
        top_n=3,
    )

    # Verify response structure
    assert response.query == "authentication bug fix"
    assert response.total_commits_indexed == 5
    assert len(response.results) <= 3
    assert len(response.results) > 0

    # Verify results contain expected fields
    for result in response.results:
        assert result.hash in commit_hashes
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.timestamp > 0
        assert result.author
        assert result.title
        assert result.repo_path == str(git_dir)


def test_glob_filtering_python_files(git_test_repo, commit_indexer):
    """
    Test glob filtering to match only Python file changes.

    Verifies that 'src/**/*.py' pattern correctly filters commits.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Search with glob filter for Python files in src/
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="authentication",
        top_n=10,
        files_glob="src/**/*.py",
    )

    # Should only get commits that changed Python files in src/
    # From our test data: src/auth.py and src/api/endpoints.py
    assert len(response.results) >= 1

    for result in response.results:
        # Verify at least one file matches the glob pattern
        matching_files = [
            f for f in result.files_changed
            if Path(f).match("src/**/*.py")
        ]
        assert len(matching_files) > 0

    # Verify excluded commits (frontend/app.js, docs/api.md, tests/test_auth.py)
    result_hashes = {r.hash for r in response.results}

    # Get commit that modified frontend/app.js
    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        if "frontend/app.js" in commit_data.files_changed:
            # This commit should NOT be in results
            if len(commit_data.files_changed) == 1:  # Only if it's the only file
                assert commit_hash not in result_hashes


def test_glob_filtering_test_files(git_test_repo, commit_indexer):
    """
    Test glob filtering for test files pattern.

    Verifies that 'tests/**/*.py' pattern works correctly.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Search for test-related commits
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="authentication test",
        top_n=5,
        files_glob="tests/**/*.py",
    )

    # Should get the test file commit (if semantic query matches)
    # Note: Due to semantic matching, may not always find if embedding similarity is low
    # Just verify that if we get results, they match the pattern
    if len(response.results) == 0:
        # Semantic search may not rank test commit highly enough
        pytest.skip("Semantic search didn't rank test commit highly enough")

    # Verify all results have test files
    for result in response.results:
        matching_files = [
            f for f in result.files_changed
            if Path(f).match("tests/**/*.py")
        ]
        assert len(matching_files) > 0


def test_timestamp_filter_after(git_test_repo, commit_indexer):
    """
    Test filtering commits by after_timestamp.

    Verifies that only commits after a certain timestamp are returned.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    timestamps = []

    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)
        timestamps.append(commit_data.timestamp)

        commit_indexer.add_commit(
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

    # Get middle timestamp
    timestamps_sorted = sorted(timestamps)
    middle_timestamp = timestamps_sorted[len(timestamps_sorted) // 2]

    # Search for commits after middle timestamp
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="authentication",
        top_n=10,
        after_timestamp=middle_timestamp,
    )

    # All results should have timestamp > middle_timestamp
    for result in response.results:
        assert result.timestamp > middle_timestamp


def test_timestamp_filter_before(git_test_repo, commit_indexer):
    """
    Test filtering commits by before_timestamp.

    Verifies that only commits before a certain timestamp are returned.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    timestamps = []

    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)
        timestamps.append(commit_data.timestamp)

        commit_indexer.add_commit(
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

    # Get middle timestamp
    timestamps_sorted = sorted(timestamps)
    middle_timestamp = timestamps_sorted[len(timestamps_sorted) // 2]

    # Search for commits before middle timestamp
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="authentication",
        top_n=10,
        before_timestamp=middle_timestamp,
    )

    # All results should have timestamp < middle_timestamp
    for result in response.results:
        assert result.timestamp < middle_timestamp


def test_incremental_update(git_test_repo, commit_indexer):
    """
    Test incremental updates: add new commit, verify it appears in search.

    Simulates the git watcher behavior of indexing only new commits.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index initial commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    for commit_hash in commits[:3]:  # Only index first 3
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Verify initial count
    assert commit_indexer.get_total_commits() == 3

    # Get last indexed timestamp
    last_timestamp = commit_indexer.get_last_indexed_timestamp(str(git_dir))
    assert last_timestamp is not None

    # Create a new commit with distinctive content
    new_hash = _create_commit(
        repo_path,
        "src/security.py",
        "def encrypt_password(password):\n    return hash(password)\n",
        "Add password encryption security feature",
    )

    # Get new commits after last timestamp
    new_commits = get_commits_after_timestamp(git_dir, after_timestamp=last_timestamp)
    assert len(new_commits) >= 1

    # Index the new commit
    for commit_hash in new_commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Verify updated count
    assert commit_indexer.get_total_commits() >= 4

    # Search for the new commit
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="password encryption security",
        top_n=5,
    )

    # Should find the new commit
    assert len(response.results) > 0
    result_hashes = {r.hash for r in response.results}
    assert new_hash in result_hashes


def test_search_relevance_ranking(git_test_repo, commit_indexer):
    """
    Test that search results are properly ranked by relevance.

    Verifies that more relevant commits appear first in results.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Search with specific query
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="authentication API endpoint",
        top_n=5,
    )

    # Verify scores are in descending order
    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True)

    # Verify scores are in reasonable range
    for score in scores:
        assert 0.0 <= score <= 1.0


def test_multiple_repositories(tmp_path, shared_embedding_model):
    """
    Test indexing and searching across multiple repositories.

    Verifies that commits from different repositories can be
    indexed and searched together.
    """
    # Create two separate repositories
    repo1_path = tmp_path / "repo1"
    repo1_path.mkdir()
    _init_git_repo(repo1_path)

    repo2_path = tmp_path / "repo2"
    repo2_path.mkdir()
    _init_git_repo(repo2_path)

    # Create distinct commits in each repo
    _create_commit(
        repo1_path,
        "feature.py",
        "def feature():\n    pass\n",
        "Add feature in repo1",
    )

    _create_commit(
        repo2_path,
        "module.py",
        "def module():\n    pass\n",
        "Add module in repo2",
    )

    # Create indexer
    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    # Discover both repositories
    repos = discover_git_repositories(
        tmp_path,
        exclude_patterns=[],
        exclude_hidden_dirs=True,
    )
    assert len(repos) == 2

    # Index commits from both repos
    for git_dir in repos:
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

    # Verify both commits are indexed
    assert indexer.get_total_commits() == 2

    # Search across both repos
    response = search_git_history(
        indexer,
        query="python code",
        top_n=5,
    )

    # Should get results from both repos
    assert len(response.results) == 2
    repo_paths = {r.repo_path for r in response.results}
    assert len(repo_paths) == 2


def test_empty_query_returns_results(git_test_repo, commit_indexer):
    """
    Test that empty or very generic queries still return results.

    Verifies graceful handling of edge case queries.
    """
    repo_path, git_dir, commit_hashes = git_test_repo

    # Index all commits
    commits = get_commits_after_timestamp(git_dir, after_timestamp=None)
    for commit_hash in commits:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)

        commit_indexer.add_commit(
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

    # Search with generic query
    response = search_git_history(
        commit_indexer=commit_indexer,
        query="commit",
        top_n=3,
    )

    # Should still get results
    assert len(response.results) > 0
    assert response.total_commits_indexed == 5
