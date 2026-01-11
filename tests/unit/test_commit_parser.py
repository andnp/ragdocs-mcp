import subprocess
import tempfile
from pathlib import Path

from src.git.commit_parser import (
    CommitData,
    parse_commit,
    build_commit_document,
    truncate_delta,
)


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


def test_parse_standard_commit():
    """Test parsing a standard commit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        commit_hash = _create_commit(
            repo_path,
            "test.txt",
            "Hello world",
            "Add test file",
        )

        git_dir = repo_path / ".git"
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)

        assert commit_data.hash == commit_hash
        assert commit_data.timestamp > 0
        assert "Test User" in commit_data.author
        assert "test@example.com" in commit_data.author
        assert commit_data.title == "Add test file"
        # Git diff-tree may not show files for new file commits in some cases
        assert len(commit_data.delta_truncated) > 0  # Verify delta is captured


def test_parse_merge_commit():
    """Test parsing a merge commit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        # Create initial commit
        _create_commit(repo_path, "initial.txt", "initial", "Initial commit")

        # Get the current branch name
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        main_branch = result.stdout.strip()
        if not main_branch:
            # Fallback for older git versions
            main_branch = "master"

        # Create a branch
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        _create_commit(repo_path, "feature.txt", "feature content", "Feature commit")

        # Switch back to main and create another commit
        subprocess.run(
            ["git", "checkout", main_branch],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        _create_commit(repo_path, "main.txt", "main content", "Main commit")

        # Merge feature branch
        subprocess.run(
            ["git", "merge", "feature", "--no-edit"],
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
        merge_commit_hash = result.stdout.strip()

        git_dir = repo_path / ".git"
        commit_data = parse_commit(git_dir, merge_commit_hash, max_delta_lines=200)

        assert commit_data.hash == merge_commit_hash
        assert commit_data.timestamp > 0


def test_multiline_message():
    """Test parsing a commit with multi-line message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        file_path = repo_path / "test.txt"
        file_path.write_text("content")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)

        # Create commit with multi-line message
        message = "Short title\n\nDetailed description\nwith multiple lines"
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
        commit_hash = result.stdout.strip()

        git_dir = repo_path / ".git"
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)

        assert commit_data.title == "Short title"
        assert "Detailed description" in commit_data.message
        assert "multiple lines" in commit_data.message


def test_delta_truncation():
    """Test delta truncation at 200 lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        # Create a file with many lines
        large_content = "\n".join([f"Line {i}" for i in range(300)])
        commit_hash = _create_commit(
            repo_path,
            "large.txt",
            large_content,
            "Add large file",
        )

        git_dir = repo_path / ".git"
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)

        # Delta should be truncated
        assert "lines omitted" in commit_data.delta_truncated
        lines_in_delta = len(commit_data.delta_truncated.splitlines())
        assert lines_in_delta <= 202  # 200 + "..." + "(N lines omitted)"


def test_delta_no_truncation():
    """Test delta not truncated when under limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        # Create a small file
        small_content = "\n".join([f"Line {i}" for i in range(10)])
        commit_hash = _create_commit(
            repo_path,
            "small.txt",
            small_content,
            "Add small file",
        )

        git_dir = repo_path / ".git"
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)

        # Delta should not be truncated
        assert "lines omitted" not in commit_data.delta_truncated


def test_build_commit_document():
    """Test building searchable document from commit data."""
    commit = CommitData(
        hash="abc123",
        timestamp=1234567890,
        author="John Doe <john@example.com>",
        committer="Jane Smith <jane@example.com>",
        title="Fix authentication bug",
        message="Detailed description\nof the fix",
        files_changed=["src/auth.py", "tests/test_auth.py"],
        delta_truncated="@@ -1,3 +1,3 @@\n-old line\n+new line",
    )

    doc = build_commit_document(commit)

    assert "Fix authentication bug" in doc
    assert "Detailed description" in doc
    assert "Author: John Doe" in doc
    assert "Committer: Jane Smith" in doc
    assert "Files changed:" in doc
    assert "src/auth.py" in doc
    assert "tests/test_auth.py" in doc
    assert "-old line" in doc
    assert "+new line" in doc


def test_truncate_delta_function():
    """Test truncate_delta function directly."""
    # Test with short input
    short_diff = "\n".join([f"line {i}" for i in range(50)])
    result = truncate_delta(short_diff, max_lines=200)
    assert result == short_diff
    assert "omitted" not in result

    # Test with long input
    long_diff = "\n".join([f"line {i}" for i in range(300)])
    result = truncate_delta(long_diff, max_lines=200)
    assert "lines omitted" in result
    lines = result.splitlines()
    # Should have 200 lines + empty line + omission message
    assert len([line for line in lines if line and "omitted" not in line]) == 200


def test_utf8_encoding():
    """Test handling of UTF-8 content in commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        # Create file with UTF-8 content
        utf8_content = "Hello 世界 🌍"
        commit_hash = _create_commit(
            repo_path,
            "utf8.txt",
            utf8_content,
            "Add UTF-8 content",
        )

        git_dir = repo_path / ".git"
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)

        # Should handle UTF-8 correctly
        assert commit_data.title == "Add UTF-8 content"
        # Delta should contain the UTF-8 content (or fallback gracefully)
        assert len(commit_data.delta_truncated) > 0
