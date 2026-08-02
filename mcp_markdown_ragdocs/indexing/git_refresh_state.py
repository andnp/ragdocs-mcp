"""Durable cursors for task-backed git refreshes."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from searchkernel.api import atomic_write_json

GIT_REFRESH_STATE_FILENAME = "git-refresh-state.json"
GIT_REFRESH_HEADS_FILENAME = "git-refresh-heads.json"
GIT_REFRESH_PROGRESS_FILENAME = "git-refresh-progress.json"
_PROGRESS_LOCK = Lock()


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


def _progress_path(index_root: Path) -> Path:
    return index_root / GIT_REFRESH_PROGRESS_FILENAME


def load_progress(index_root: Path) -> dict[str, dict[str, Any]]:
    """Load durable per-repository refresh telemetry."""

    try:
        raw = json.loads(_progress_path(index_root).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw, dict):
        return {}
    return {
        repo: dict(values)
        for repo, values in raw.items()
        if isinstance(repo, str) and isinstance(values, dict)
    }


def get_progress(index_root: Path, git_dir: Path) -> dict[str, Any] | None:
    """Return the latest durable refresh telemetry for a repository."""

    return load_progress(index_root).get(str(git_dir.resolve()))


def list_progress(index_root: Path) -> list[dict[str, Any]]:
    """Return durable refresh telemetry sorted by repository path."""

    return [
        progress
        for _, progress in sorted(load_progress(index_root).items())
    ]


def save_progress(
    index_root: Path,
    git_dir: Path,
    progress: dict[str, Any],
) -> None:
    """Atomically replace one repository's refresh telemetry."""

    repo_path = str(git_dir.resolve())
    with _PROGRESS_LOCK:
        all_progress = load_progress(index_root)
        all_progress[repo_path] = {
            **all_progress.get(repo_path, {}),
            "repository_path": repo_path,
            **progress,
        }
        atomic_write_json(_progress_path(index_root), all_progress)
