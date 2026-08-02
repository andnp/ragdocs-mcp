"""Unit tests for git repository discovery."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.git import repository
from mcp_markdown_ragdocs.git.repository import (
    discover_git_repositories,
    get_commits_after_timestamp,
    iter_commit_hashes_after_timestamp,
    is_git_available,
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


def _create_commit(repo_path: Path, file_name: str, content: str, message: str) -> None:
    """Create a file and commit it."""
    file_path = repo_path / file_name
    file_path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def test_discover_single_repo():
    """Test discovering a single repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo1"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        repos = discover_git_repositories(
            Path(tmpdir),
            exclude_patterns=[],
            exclude_hidden_dirs=True,
        )

        assert len(repos) == 1
        assert repos[0] == (repo_path / ".git").resolve()


def test_discover_nested_repos():
    """Test discovering nested repositories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo1_path = Path(tmpdir) / "repo1"
        repo1_path.mkdir()
        _init_git_repo(repo1_path)

        repo2_path = Path(tmpdir) / "subdir" / "repo2"
        repo2_path.mkdir(parents=True)
        _init_git_repo(repo2_path)

        repos = discover_git_repositories(
            Path(tmpdir),
            exclude_patterns=[],
            exclude_hidden_dirs=True,
        )

        assert len(repos) == 2
        repo_names = {r.parent.name for r in repos}
        assert repo_names == {"repo1", "repo2"}


def test_exclude_venv_pattern():
    """Test exclusion of .venv directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        venv_repo_path = Path(tmpdir) / ".venv" / "repo"
        venv_repo_path.mkdir(parents=True)
        _init_git_repo(venv_repo_path)

        repos = discover_git_repositories(
            Path(tmpdir),
            exclude_patterns=["**/.venv/**"],
            exclude_hidden_dirs=False,
        )

        assert len(repos) == 1
        assert repos[0].parent.name == "repo"


def test_exclude_hidden_dirs():
    """Test exclusion of hidden directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        hidden_repo_path = Path(tmpdir) / ".hidden" / "repo"
        hidden_repo_path.mkdir(parents=True)
        _init_git_repo(hidden_repo_path)

        repos = discover_git_repositories(
            Path(tmpdir),
            exclude_patterns=[],
            exclude_hidden_dirs=True,
        )

        assert len(repos) == 1
        assert repos[0].parent.name == "repo"


def test_no_repos_found():
    """Test behavior when no repositories are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repos = discover_git_repositories(
            Path(tmpdir),
            exclude_patterns=[],
            exclude_hidden_dirs=True,
        )

        assert len(repos) == 0


def test_get_all_commits():
    """Test getting all commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        # Create 3 commits
        for i in range(3):
            _create_commit(repo_path, f"file{i}.txt", f"content {i}", f"Commit {i}")

        git_dir = repo_path / ".git"
        commits = get_commits_after_timestamp(git_dir, after_timestamp=None)

        assert len(commits) == 3


def test_get_commits_after_timestamp():
    """Test getting commits after a specific timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)

        # Create first commit
        _create_commit(repo_path, "file0.txt", "content 0", "Commit 0")

        # Get timestamp
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        first_commit_time = int(result.stdout.strip())

        # Wait and create more commits
        import time

        time.sleep(1)

        for i in range(1, 3):
            _create_commit(repo_path, f"file{i}.txt", f"content {i}", f"Commit {i}")

        git_dir = repo_path / ".git"

        # Get commits after first commit (--after is inclusive in git)
        commits = get_commits_after_timestamp(
            git_dir, after_timestamp=first_commit_time
        )

        # Should get commits after the timestamp (git --after is inclusive, so all 3)
        assert len(commits) >= 2


class _FakeStream:
    def __init__(self, content: str):
        self._content = content
        self.closed = False

    def __iter__(self):
        return iter(self._content.splitlines(keepends=True))

    def read(self):
        return self._content

    def close(self):
        self.closed = True


class _FakeGitProcess:
    def __init__(self, output: str, *, wait_error=None, returncode: int = 0):
        self.stdout = _FakeStream(output)
        self.stderr = _FakeStream("git failed")
        self._wait_error = wait_error
        self._returncode = returncode
        self._running = True
        self.killed = False
        self.wait_calls: list[int | None] = []

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self._wait_error is not None and self._running:
            if isinstance(self._wait_error, subprocess.TimeoutExpired):
                raise self._wait_error
            self._running = False
            raise self._wait_error
        self._running = False
        return self._returncode

    def poll(self):
        return None if self._running else self._returncode

    def kill(self):
        self.killed = True
        self._running = False


def _patch_git_process(monkeypatch, process: _FakeGitProcess):
    commands = []

    def fake_popen(command, **kwargs):
        commands.append((command, kwargs))
        return process

    monkeypatch.setattr(repository.subprocess, "Popen", fake_popen)
    return commands


def test_iter_commit_hashes_is_lazy_and_preserves_git_order(tmp_path, monkeypatch):
    """Hash production starts on demand and preserves Git's output order."""
    process = _FakeGitProcess("new\nmiddle\nold\n")
    commands = _patch_git_process(monkeypatch, process)
    iterator = iter_commit_hashes_after_timestamp(tmp_path / ".git", 123)

    assert commands == []
    assert next(iterator) == "new"
    assert commands[0][0] == [
        "git",
        "log",
        "--all",
        "--format=%H",
        "--after=123",
    ]
    assert list(iterator) == ["middle", "old"]


@pytest.mark.parametrize(
    "wait_error",
    [
        subprocess.CalledProcessError(2, ["git", "log"], stderr="bad"),
        subprocess.TimeoutExpired(["git", "log"], timeout=30),
    ],
)
def test_iter_commit_hashes_surfaces_git_failures(tmp_path, monkeypatch, wait_error):
    """The streaming iterator exposes subprocess failures to its caller."""
    process = _FakeGitProcess("partial\n", wait_error=wait_error)
    _patch_git_process(monkeypatch, process)

    with pytest.raises(type(wait_error)):
        list(iter_commit_hashes_after_timestamp(tmp_path / ".git"))


def test_iter_commit_hashes_closes_process_on_early_close(tmp_path, monkeypatch):
    """Closing a partially consumed iterator terminates and closes Git safely."""
    process = _FakeGitProcess("first\nsecond\n")
    _patch_git_process(monkeypatch, process)
    iterator = iter_commit_hashes_after_timestamp(tmp_path / ".git")

    assert next(iterator) == "first"
    iterator.close()

    assert process.killed
    assert process.wait_calls == [1]
    assert process.stdout.closed
    assert process.stderr.closed


def test_get_commits_after_timestamp_keeps_list_contract(tmp_path, monkeypatch):
    """The compatibility wrapper materializes streamed hashes into a list."""
    git_dir = tmp_path / ".git"
    calls = []

    def fake_iterator(received_git_dir, received_timestamp):
        calls.append((received_git_dir, received_timestamp))
        yield "first"
        yield "second"

    monkeypatch.setattr(repository, "iter_commit_hashes_after_timestamp", fake_iterator)

    commits = get_commits_after_timestamp(git_dir, after_timestamp=456)

    assert commits == ["first", "second"]
    assert isinstance(commits, list)
    assert calls == [(git_dir, 456)]


def test_git_not_available(monkeypatch):
    """Test detection when git is not available."""

    def mock_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", mock_run)

    assert not is_git_available()


def test_git_available():
    """Test detection when git is available."""
    # This test assumes git is installed
    assert is_git_available()
