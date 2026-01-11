import numpy as np
import pytest

from src.git.commit_indexer import CommitIndexer


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._counter = 0
    
    def get_text_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash."""
        # Use text hash to generate consistent embedding
        hash_val = hash(text) % 1000
        embedding = np.random.RandomState(hash_val).randn(self.dimension).astype(np.float32)
        # Normalize
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


def test_schema_creation(indexer):
    """Test that schema is created on initialization."""
    conn = indexer._get_connection()
    
    # Check table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='git_commits'"
    )
    assert cursor.fetchone() is not None
    
    # Check indexes exist
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'"
    )
    assert cursor.fetchone() is not None


def test_add_commit(indexer):
    """Test adding a commit."""
    indexer.add_commit(
        hash="abc123",
        timestamp=1234567890,
        author="John Doe <john@example.com>",
        committer="Jane Smith <jane@example.com>",
        title="Fix bug",
        message="Detailed description",
        files_changed=["src/file.py"],
        delta_truncated="@@ -1,3 +1,3 @@\n-old\n+new",
        commit_document="Fix bug\n\nDetailed description",
        repo_path="/test/repo",
    )
    
    # Verify stored
    conn = indexer._get_connection()
    cursor = conn.execute("SELECT * FROM git_commits WHERE hash = ?", ("abc123",))
    row = cursor.fetchone()
    
    assert row is not None
    assert row["hash"] == "abc123"
    assert row["timestamp"] == 1234567890
    assert row["author"] == "John Doe <john@example.com>"
    assert row["title"] == "Fix bug"


def test_update_commit_idempotent(indexer):
    """Test that updating an existing commit is idempotent."""
    # Add commit
    indexer.add_commit(
        hash="def456",
        timestamp=1234567890,
        author="Author 1",
        committer="Committer 1",
        title="Title 1",
        message="Message 1",
        files_changed=["file1.py"],
        delta_truncated="delta 1",
        commit_document="doc 1",
        repo_path="/repo",
    )
    
    # Update same commit
    indexer.add_commit(
        hash="def456",
        timestamp=1234567891,
        author="Author 2",
        committer="Committer 2",
        title="Title 2",
        message="Message 2",
        files_changed=["file2.py"],
        delta_truncated="delta 2",
        commit_document="doc 2",
        repo_path="/repo",
    )
    
    # Verify only one commit exists with updated values
    conn = indexer._get_connection()
    cursor = conn.execute("SELECT COUNT(*) as count FROM git_commits WHERE hash = ?", ("def456",))
    row = cursor.fetchone()
    assert row["count"] == 1
    
    cursor = conn.execute("SELECT * FROM git_commits WHERE hash = ?", ("def456",))
    row = cursor.fetchone()
    assert row["title"] == "Title 2"
    assert row["timestamp"] == 1234567891


def test_remove_commit(indexer):
    """Test removing a commit."""
    indexer.add_commit(
        hash="ghi789",
        timestamp=1234567890,
        author="Author",
        committer="Committer",
        title="Title",
        message="Message",
        files_changed=[],
        delta_truncated="",
        commit_document="doc",
        repo_path="/repo",
    )
    
    # Verify exists
    conn = indexer._get_connection()
    cursor = conn.execute("SELECT COUNT(*) as count FROM git_commits WHERE hash = ?", ("ghi789",))
    assert cursor.fetchone()["count"] == 1
    
    # Remove
    indexer.remove_commit("ghi789")
    
    # Verify deleted
    cursor = conn.execute("SELECT COUNT(*) as count FROM git_commits WHERE hash = ?", ("ghi789",))
    assert cursor.fetchone()["count"] == 0


def test_query_by_embedding(indexer):
    """Test querying by embedding similarity."""
    # Add multiple commits
    for i in range(5):
        indexer.add_commit(
            hash=f"commit{i}",
            timestamp=1234567890 + i,
            author="Author",
            committer="Committer",
            title=f"Title {i}",
            message=f"Message {i}",
            files_changed=[],
            delta_truncated="",
            commit_document=f"Document {i}",
            repo_path="/repo",
        )
    
    # Query with a document that should match "Document 2"
    query_embedding = indexer._embedding_model.get_text_embedding("Document 2")
    results = indexer.query_by_embedding(query_embedding, top_k=3)
    
    # Should get results
    assert len(results) <= 3
    assert len(results) > 0
    assert all(isinstance(r["score"], float) for r in results)
    # Cosine similarity should be in valid range, with some tolerance for floating point
    assert all(-1.01 <= r["score"] <= 1.01 for r in results)
    
    # Results should be sorted by score descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_timestamp_filter_after(indexer):
    """Test filtering by after_timestamp."""
    # Add commits with different timestamps
    for i in range(5):
        indexer.add_commit(
            hash=f"commit{i}",
            timestamp=1000 + i * 100,
            author="Author",
            committer="Committer",
            title=f"Title {i}",
            message="",
            files_changed=[],
            delta_truncated="",
            commit_document=f"doc {i}",
            repo_path="/repo",
        )
    
    query_embedding = indexer._embedding_model.get_text_embedding("doc")
    results = indexer.query_by_embedding(
        query_embedding,
        top_k=10,
        after_timestamp=1200,
    )
    
    # Should only get commits with timestamp > 1200 (i >= 3)
    assert all(r["timestamp"] > 1200 for r in results)
    assert len(results) <= 2  # commits 3 and 4


def test_timestamp_filter_before(indexer):
    """Test filtering by before_timestamp."""
    # Add commits with different timestamps
    for i in range(5):
        indexer.add_commit(
            hash=f"commit{i}",
            timestamp=1000 + i * 100,
            author="Author",
            committer="Committer",
            title=f"Title {i}",
            message="",
            files_changed=[],
            delta_truncated="",
            commit_document=f"doc {i}",
            repo_path="/repo",
        )
    
    query_embedding = indexer._embedding_model.get_text_embedding("doc")
    results = indexer.query_by_embedding(
        query_embedding,
        top_k=10,
        before_timestamp=1200,
    )
    
    # Should only get commits with timestamp < 1200 (i < 3)
    assert all(r["timestamp"] < 1200 for r in results)
    assert len(results) <= 3  # commits 0, 1, 2


def test_empty_index_query(indexer):
    """Test querying an empty index."""
    query_embedding = indexer._embedding_model.get_text_embedding("test")
    results = indexer.query_by_embedding(query_embedding, top_k=10)
    
    assert len(results) == 0


def test_embedding_roundtrip(indexer):
    """Test embedding serialization/deserialization."""
    original = [1.0, 2.0, 3.0, 4.0]
    serialized = indexer._serialize_embedding(original)
    deserialized = indexer._deserialize_embedding(serialized)
    
    np.testing.assert_array_almost_equal(original, deserialized, decimal=5)


def test_malformed_json_files(indexer):
    """Test handling of malformed JSON in files_changed."""
    # Add commit with valid JSON
    indexer.add_commit(
        hash="valid",
        timestamp=1000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=["file.py"],
        delta_truncated="",
        commit_document="doc",
        repo_path="/repo",
    )
    
    # Manually corrupt the JSON
    conn = indexer._get_connection()
    conn.execute(
        "UPDATE git_commits SET files_changed = ? WHERE hash = ?",
        ("{invalid json}", "valid"),
    )
    conn.commit()
    
    # Query should handle gracefully
    query_embedding = indexer._embedding_model.get_text_embedding("doc")
    results = indexer.query_by_embedding(query_embedding, top_k=10)
    
    assert len(results) == 1
    # Should fallback to empty list
    assert results[0]["files_changed"] == []


def test_get_last_indexed_timestamp(indexer):
    """Test getting last indexed timestamp for a repo."""
    # Add commits for different repos
    indexer.add_commit(
        hash="commit1",
        timestamp=1000,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc",
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
        commit_document="doc",
        repo_path="/repo1",
    )
    
    indexer.add_commit(
        hash="commit3",
        timestamp=1500,
        author="A",
        committer="C",
        title="T",
        message="M",
        files_changed=[],
        delta_truncated="",
        commit_document="doc",
        repo_path="/repo2",
    )
    
    # Get last timestamp for repo1
    last_ts = indexer.get_last_indexed_timestamp("/repo1")
    assert last_ts == 2000
    
    # Get last timestamp for repo2
    last_ts = indexer.get_last_indexed_timestamp("/repo2")
    assert last_ts == 1500
    
    # Non-existent repo
    last_ts = indexer.get_last_indexed_timestamp("/repo3")
    assert last_ts is None


def test_get_total_commits(indexer):
    """Test counting total commits."""
    assert indexer.get_total_commits() == 0
    
    # Add commits
    for i in range(3):
        indexer.add_commit(
            hash=f"commit{i}",
            timestamp=1000 + i,
            author="A",
            committer="C",
            title="T",
            message="M",
            files_changed=[],
            delta_truncated="",
            commit_document="doc",
            repo_path="/repo",
        )
    
    assert indexer.get_total_commits() == 3
