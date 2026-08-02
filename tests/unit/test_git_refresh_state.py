"""Tests for durable git refresh cursors."""

from pathlib import Path

from mcp_markdown_ragdocs.indexing.git_refresh_state import (
    get_cursor,
    get_progress,
    list_progress,
    load_cursors,
    load_progress,
    save_cursor,
    save_progress,
)


def test_missing_state_is_empty(tmp_path: Path) -> None:
    assert load_cursors(tmp_path) == {}
    assert get_cursor(tmp_path, tmp_path / "repo" / ".git") is None


def test_save_cursor_round_trips_by_resolved_repo_path(tmp_path: Path) -> None:
    git_dir = tmp_path / "repo" / ".git"

    save_cursor(tmp_path, git_dir, 123)

    assert get_cursor(tmp_path, git_dir) == 123
    assert load_cursors(tmp_path) == {str(git_dir.resolve()): 123}


def test_corrupt_state_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "git-refresh-state.json").write_text("not json", encoding="utf-8")

    assert load_cursors(tmp_path) == {}


def test_progress_round_trips_by_resolved_repo_path(tmp_path: Path) -> None:
    git_dir = tmp_path / "repo" / ".git"
    progress = {
        "state": "running",
        "cursor": 123,
        "processed_count": 2,
        "updated_at": "2026-08-02T22:00:00+00:00",
    }

    save_progress(tmp_path, git_dir, progress)

    assert get_progress(tmp_path, git_dir) == {
        "repository_path": str(git_dir.resolve()),
        **progress,
    }
    assert list_progress(tmp_path) == [get_progress(tmp_path, git_dir)]


def test_corrupt_progress_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "git-refresh-progress.json").write_text("not json", encoding="utf-8")

    assert load_progress(tmp_path) == {}
