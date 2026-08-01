"""Durable cursors for task-backed git refreshes."""

from __future__ import annotations

import json
from pathlib import Path

from searchkernel.api import atomic_write_json

GIT_REFRESH_STATE_FILENAME = "git-refresh-state.json"
GIT_REFRESH_HEADS_FILENAME = "git-refresh-heads.json"


def state_path(index_root: Path) -> Path:
    """Return the path used for worker-owned git refresh cursors."""

    return index_root / GIT_REFRESH_STATE_FILENAME


def load_cursors(index_root: Path) -> dict[str, int]:
    """Load valid repository cursors, treating missing/corrupt state as empty."""

    path = state_path(index_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw, dict):
        return {}

    cursors: dict[str, int] = {}
    for repo, cursor in raw.items():
        if not isinstance(repo, str) or isinstance(cursor, bool):
            continue
        if isinstance(cursor, int):
            cursors[repo] = cursor
    return cursors


def get_cursor(index_root: Path, git_dir: Path) -> int | None:
    """Return the last successfully indexed commit timestamp for a repo."""

    return load_cursors(index_root).get(str(git_dir.resolve()))


def save_cursor(index_root: Path, git_dir: Path, cursor: int) -> None:
    """Atomically save a repository cursor after a successful refresh."""

    cursors = load_cursors(index_root)
    cursors[str(git_dir.resolve())] = int(cursor)
    atomic_write_json(state_path(index_root), cursors)


def _heads_path(index_root: Path) -> Path:
    return index_root / GIT_REFRESH_HEADS_FILENAME


def load_heads(index_root: Path) -> dict[str, str]:
    """Load successfully refreshed repository ref signatures."""

    path = _heads_path(index_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw, dict):
        return {}
    return {
        repo: head
        for repo, head in raw.items()
        if isinstance(repo, str) and isinstance(head, str)
    }


def get_head(index_root: Path, git_dir: Path) -> str | None:
    return load_heads(index_root).get(str(git_dir.resolve()))


def save_head(index_root: Path, git_dir: Path, head: str) -> None:
    """Atomically save the ref signature observed before a refresh."""

    heads = load_heads(index_root)
    heads[str(git_dir.resolve())] = head
    atomic_write_json(_heads_path(index_root), heads)
