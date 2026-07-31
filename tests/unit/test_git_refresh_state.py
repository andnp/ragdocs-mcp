"""Tests for durable git refresh cursors."""

from pathlib import Path

from ragdocs.indexing.git_refresh_state import (
    get_cursor,
    load_cursors,
    save_cursor,
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
