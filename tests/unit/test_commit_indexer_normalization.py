"""Test path normalization in CommitIndexer for incremental indexing."""

from pathlib import Path

import numpy as np
import pytest

from src.git.commit_indexer import CommitIndexer


class MockEmbeddingModel:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def get_text_embedding(self, text: str):
        hash_val = hash(text) % 1000
        embedding = np.random.RandomState(hash_val).randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()


@pytest.fixture
def mock_model():
    """Create a mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def indexer(tmp_path, mock_model):
    """Create a commit indexer with temporary database."""
    db_path = tmp_path / "test_commits.db"
    return CommitIndexer(db_path=db_path, embedding_model=mock_model)


def test_normalize_repo_path_strips_git_suffix(indexer):
    """
    Verify that .git suffix is removed from repo paths.
    """
    normalized = indexer._normalize_repo_path("/home/user/repo/.git")
    assert not normalized.endswith(".git")
    assert normalized.endswith("repo")


def test_normalize_repo_path_removes_trailing_slash(indexer):
    """
    Verify that trailing slashes are removed.
    """
    normalized = indexer._normalize_repo_path("/home/user/repo/")
    assert not normalized.endswith("/")


def test_normalize_repo_path_handles_relative_paths(indexer):
    """
    Verify that relative paths are resolved to absolute.
    """
    normalized = indexer._normalize_repo_path("./repo")
    assert Path(normalized).is_absolute()


def test_normalize_repo_path_idempotent(indexer):
    """
    Verify that normalizing twice gives same result.
    """
    path = "/home/user/repo/.git"
    normalized1 = indexer._normalize_repo_path(path)
    normalized2 = indexer._normalize_repo_path(normalized1)
    assert normalized1 == normalized2


def test_path_normalization_in_storage_and_retrieval(indexer, tmp_path):
    """
    Verify that paths are normalized consistently in storage and retrieval.
    """
    # Create actual path for normalization
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    git_dir = repo_path / ".git"

    # Add commit with .git suffix
    indexer.add_commit(
        hash="commit1",
        timestamp=1000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc1",
        repo_path=str(git_dir),  # Path with .git
    )

    # Add another commit without .git suffix
    indexer.add_commit(
        hash="commit2",
        timestamp=2000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc2",
        repo_path=str(repo_path),  # Path without .git
    )

    # Both should be stored with same normalized repo_path
    # Get last timestamp should work with either form
    last_ts_with_git = indexer.get_last_indexed_timestamp(str(git_dir))
    last_ts_without_git = indexer.get_last_indexed_timestamp(str(repo_path))

    assert last_ts_with_git == last_ts_without_git == 2000


def test_path_normalization_multiple_repos(indexer, tmp_path):
    """
    Verify that different repos are kept separate after normalization.
    """
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()

    # Add commits to different repos
    indexer.add_commit(
        hash="commit1",
        timestamp=1000,
        author="A",
        committer="C",
        title="T1",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc1",
        repo_path=str(repo1 / ".git"),
    )

    indexer.add_commit(
        hash="commit2",
        timestamp=2000,
        author="A",
        committer="C",
        title="T2",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc2",
        repo_path=str(repo2 / ".git"),
    )

    # Each repo should have its own last timestamp
    last_ts1 = indexer.get_last_indexed_timestamp(str(repo1))
    last_ts2 = indexer.get_last_indexed_timestamp(str(repo2))

    assert last_ts1 == 1000
    assert last_ts2 == 2000


def test_path_with_git_in_middle_not_stripped(indexer):
    """
    Verify that .git in middle of path is not removed.
    """
    path = "/home/.git/user/repo/.git"
    normalized = indexer._normalize_repo_path(path)
    # Should only strip the trailing .git, not the one in middle
    assert "/.git/user/repo" in normalized
    assert not normalized.endswith(".git")


def test_clear_method_removes_all_commits(indexer):
    """
    Verify that clear() removes all commits regardless of repo_path.
    """
    # Add commits to different repos
    indexer.add_commit(
        hash="commit1",
        timestamp=1000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc1",
        repo_path="/repo1",
    )

    indexer.add_commit(
        hash="commit2",
        timestamp=2000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc2",
        repo_path="/repo2",
    )

    assert indexer.get_total_commits() == 2

    # Clear all
    indexer.clear()

    assert indexer.get_total_commits() == 0
    assert indexer.get_last_indexed_timestamp("/repo1") is None
    assert indexer.get_last_indexed_timestamp("/repo2") is None


def test_backward_compatibility_with_existing_db(indexer, tmp_path):
    """
    Verify that existing DBs with non-normalized paths still work.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Simulate old behavior: directly insert non-normalized path
    conn = indexer._get_connection()
    conn.execute(
        """
        INSERT INTO git_commits
        (hash, timestamp, author, committer, title, message,
         files_changed, delta_truncated, embedding, indexed_at, repo_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "old_commit",
            1000,
            "A",
            "C",
            "T",
            "M",
            "[]",
            "",
            indexer._serialize_embedding([1.0] * 384),
            1000,
            str(repo_path / ".git"),  # Non-normalized (stored as-is)
        ),
    )
    conn.commit()

    # New commits should normalize
    indexer.add_commit(
        hash="new_commit",
        timestamp=2000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc",
        repo_path=str(repo_path / ".git"),
    )

    # Query with normalized path should find both
    last_ts = indexer.get_last_indexed_timestamp(str(repo_path))
    # Should return the new commit's timestamp (normalized path match)
    assert last_ts == 2000

    # Total commits should be 2
    assert indexer.get_total_commits() == 2


def test_empty_repo_path_normalization(indexer):
    """
    Verify that empty repo_path is handled correctly.
    """
    normalized = indexer._normalize_repo_path("")
    # Should resolve to current directory
    assert Path(normalized).is_absolute()


def test_symlink_resolution(indexer, tmp_path):
    """
    Verify that symlinks are resolved during normalization.
    """
    real_repo = tmp_path / "real-repo"
    real_repo.mkdir()
    symlink_repo = tmp_path / "symlink-repo"

    # Create symlink
    try:
        symlink_repo.symlink_to(real_repo)
    except OSError:
        pytest.skip("Cannot create symlinks on this system")

    # Both should normalize to same path
    normalized_real = indexer._normalize_repo_path(str(real_repo))
    normalized_symlink = indexer._normalize_repo_path(str(symlink_repo))

    assert normalized_real == normalized_symlink
