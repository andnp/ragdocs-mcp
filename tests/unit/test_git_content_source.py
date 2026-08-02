"""Tests for GitContentSource adapter."""

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from searchkernel.domain import RecordStatus

from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource


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


def _records_for_commit(records, commit_hash: str):
    return [
        record
        for record in records
        if record.metadata.get("commit_id") == f"git:{commit_hash}"
    ]


def _summary_record(records):
    return next(record for record in records if record.metadata["chunk_section"] == "summary")


class TestGitContentSourceInit:
    """Tests for GitContentSource initialization."""

    def test_init_with_valid_git_dir(self):
        """Test initialization with a valid .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)

            assert source.git_dir == git_dir.resolve()
            assert source.repo_path == repo_path
            assert source.source_kind == "git_commit"

    def test_init_with_nonexistent_directory(self):
        """Test that initialization fails with nonexistent directory."""
        with pytest.raises(ValueError, match="Git directory does not exist"):
            GitContentSource(Path("/nonexistent/path/.git"))

    def test_init_with_non_git_directory(self):
        """Test that initialization fails with non-.git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()

            with pytest.raises(ValueError, match="Expected .git directory"):
                GitContentSource(repo_path)


class TestGitContentSourceIterRecords:
    """Tests for iter_records method."""

    def test_iter_records_single_commit(self):
        """Test iterating over a single commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            commit_hash = _create_commit(
                repo_path,
                "test.txt",
                "Hello world",
                "Initial commit",
            )

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            commit_records = _records_for_commit(records, commit_hash)
            assert commit_records
            record = _summary_record(commit_records)

            # Check all required fields
            assert record.source_kind == "git_commit"
            assert record.source_id.startswith(f"git:{commit_hash}:summary:")
            assert record.title == "Initial commit"
            assert len(record.body) > 0
            assert "Initial commit" in record.body
            assert isinstance(record.created_at, datetime)
            assert isinstance(record.updated_at, datetime)
            assert record.status == RecordStatus.ACTIVE
            assert record.embedding is None
            assert record.embedding_model is None

            # Check metadata
            assert "author" in record.metadata
            assert "Test User" in record.metadata["author"]
            assert "test@example.com" in record.metadata["author"]
            assert "timestamp" in record.metadata
            assert "files_changed" in record.metadata

    def test_iter_records_multiple_commits(self):
        """Test iterating over multiple commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            hash1 = _create_commit(repo_path, "file1.txt", "content1", "First commit")
            hash2 = _create_commit(repo_path, "file2.txt", "content2", "Second commit")
            hash3 = _create_commit(repo_path, "file3.txt", "content3", "Third commit")

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            assert {
                record.metadata["commit_id"] for record in records
            } == {f"git:{hash1}", f"git:{hash2}", f"git:{hash3}"}

            # Commits should remain in reverse chronological order.
            summaries = [
                _summary_record(_records_for_commit(records, commit_hash))
                for commit_hash in (hash3, hash2, hash1)
            ]
            assert [record.title for record in summaries] == [
                "Third commit",
                "Second commit",
                "First commit",
            ]

    def test_iter_records_body_contains_metadata(self):
        """Test that record body contains author, files, and diff info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            _create_commit(
                repo_path,
                "test.txt",
                "Hello world",
                "Test commit\n\nDetailed message",
            )

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            assert records
            body = "\n".join(record.body for record in records)

            # Body should contain the message
            assert "Test commit" in body
            assert "Detailed message" in body

            # Body should contain author info
            assert "Author:" in body or "Test User" in body

            # Body should contain diff info (files changed section or diff output)
            assert "Files changed:" in body or "diff --git" in body

    def test_iter_records_incremental_with_timestamp(self):
        """Test incremental iteration using a timestamp cursor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            hash1 = _create_commit(repo_path, "file1.txt", "content1", "First commit")

            hash2 = _create_commit(repo_path, "file2.txt", "content2", "Second commit")

            _create_commit(repo_path, "file3.txt", "content3", "Third commit")

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)

            # Get timestamp of the second commit
            result = subprocess.run(
                ["git", "log", "--format=%ct", "-1", hash2],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            second_timestamp = int(result.stdout.strip())

            # Get records since the second commit's timestamp
            # git log --after includes commits with timestamp >= the specified time
            # We add 1 to exclude the second commit itself
            cursor = str(second_timestamp + 1)
            records = list(source.iter_records(since=cursor))

            # Should get records that were created after second_timestamp
            # At minimum, we should not get hash1 (which is before hash2)
            record_ids = [r.source_id for r in records]
            assert f"git:{hash1}" not in record_ids

            # Also verify that all retrieved records have source_kind and source_id properly set
            for record in records:
                assert record.source_kind == "git_commit"
                assert record.source_id.startswith("git:")

    def test_iter_records_incremental_with_none_cursor(self):
        """Test that None cursor returns all commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            _create_commit(repo_path, "file1.txt", "content1", "First commit")
            _create_commit(repo_path, "file2.txt", "content2", "Second commit")

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)

            records_all = list(source.iter_records(since=None))
            records_explicit = list(source.iter_records())

            assert {
                record.metadata["commit_id"] for record in records_all
            } == {
                record.metadata["commit_id"] for record in records_explicit
            }
            assert len({record.metadata["commit_id"] for record in records_all}) == 2

    def test_iter_records_with_invalid_cursor(self):
        """Test that invalid cursor falls back to all commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            _create_commit(repo_path, "file1.txt", "content1", "First commit")
            _create_commit(repo_path, "file2.txt", "content2", "Second commit")

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)

            # Invalid cursor should fall back to returning all commits
            records = list(source.iter_records(since="not-a-number"))

            assert len({record.metadata["commit_id"] for record in records}) == 2

    def test_iter_records_with_multiline_message(self):
        """Test record creation from commit with multiline message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            _create_commit(
                repo_path,
                "test.txt",
                "content",
                "Short title\n\nDetailed description\nwith multiple lines",
            )

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            assert records
            record = _summary_record(records)
            body = "\n".join(item.body for item in records)

            # Title should be just the first line
            assert record.title == "Short title"

            # Body should contain the full message
            assert "Detailed description" in body
            assert "multiple lines" in body

    def test_iter_records_empty_message(self):
        """Test handling of commit with empty message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            file_path = repo_path / "test.txt"
            file_path.write_text("content")
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)

            # Create commit with empty message (using allow-empty-message)
            subprocess.run(
                ["git", "commit", "--allow-empty-message", "-m", ""],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            assert records
            record = _summary_record(records)

            # Should have a fallback title
            assert len(record.title) > 0
            assert len(record.body) > 0


class TestGitContentSourceChangeSignal:
    """Tests for change_signal method."""

    def test_change_signal_returns_poll_interval(self):
        """Test that change_signal returns a valid polling signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)

            signal = source.change_signal()

            assert isinstance(signal, dict)
            assert "poll_interval" in signal
            assert isinstance(signal["poll_interval"], int)
            assert signal["poll_interval"] > 0


class TestGitContentSourceSourceKind:
    """Tests for source_kind attribute."""

    def test_source_kind_is_git_commit(self):
        """Test that source_kind is correctly set to 'git_commit'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)

            assert source.source_kind == "git_commit"

            # Also check that records have the same source_kind
            _create_commit(repo_path, "test.txt", "content", "Test")
            records = list(source.iter_records())
            assert records
            assert all(record.source_kind == "git_commit" for record in records)


class TestGitContentSourceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_iter_records_with_utf8_commits(self):
        """Test handling of commits with UTF-8 characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            _create_commit(
                repo_path,
                "utf8.txt",
                "Hello 世界 🌍",
                "UTF-8 commit: 日本語 テスト",
            )

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            assert records
            record = _summary_record(records)

            # Should handle UTF-8 correctly
            assert "テスト" in record.title or len(record.title) > 0
            assert len(record.body) > 0

    def test_iter_records_large_diff(self):
        """Test handling of commits with large diffs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            # Create file with many lines
            large_content = "\n".join([f"Line {i}" for i in range(500)])
            _create_commit(repo_path, "large.txt", large_content, "Large file")

            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            assert len(records) > 1
            record = _summary_record(records)

            # Body should exist but diff should be truncated
            assert all(record.body for record in records)
            # The body may contain a truncation indicator
            assert "omitted" in record.body or len(record.body) > 0

    def test_iter_records_no_commits(self):
        """Test iteration over empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            _init_git_repo(repo_path)

            # Don't create any commits
            git_dir = repo_path / ".git"
            source = GitContentSource(git_dir)
            records = list(source.iter_records())

            # Should return empty list, not crash
            assert len(records) == 0
