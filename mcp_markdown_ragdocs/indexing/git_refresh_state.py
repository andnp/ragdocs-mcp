"""Durable cursors for task-backed git refreshes."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from mcp_markdown_ragdocs.gdrive.json_record_store import JsonEnvelopeStore

GIT_REFRESH_STATE_FILENAME = "git-refresh-state.json"
GIT_REFRESH_HEADS_FILENAME = "git-refresh-heads.json"
GIT_REFRESH_PROGRESS_FILENAME = "git-refresh-progress.json"
_STATE_SCHEMA_VERSION = 1
_PROGRESS_LOCK = Lock()


def state_path(index_root: Path) -> Path:
    """Return the path used for worker-owned git refresh cursors."""

    return index_root / GIT_REFRESH_STATE_FILENAME


def _cursors_store(index_root: Path) -> JsonEnvelopeStore:
    return JsonEnvelopeStore(
        path=state_path(index_root),
        schema_version=_STATE_SCHEMA_VERSION,
        key="cursors",
    )


def load_cursors(index_root: Path) -> dict[str, int]:
    """Load valid repository cursors, treating missing/corrupt state as empty."""

    raw = _cursors_store(index_root).read(dict) or {}

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
    _cursors_store(index_root).write(cursors)


def _heads_path(index_root: Path) -> Path:
    return index_root / GIT_REFRESH_HEADS_FILENAME


def _heads_store(index_root: Path) -> JsonEnvelopeStore:
    return JsonEnvelopeStore(
        path=_heads_path(index_root),
        schema_version=_STATE_SCHEMA_VERSION,
        key="heads",
    )


def load_heads(index_root: Path) -> dict[str, str]:
    """Load successfully refreshed repository ref signatures."""

    raw = _heads_store(index_root).read(dict) or {}
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
    _heads_store(index_root).write(heads)


def _progress_path(index_root: Path) -> Path:
    return index_root / GIT_REFRESH_PROGRESS_FILENAME


def _progress_store(index_root: Path) -> JsonEnvelopeStore:
    return JsonEnvelopeStore(
        path=_progress_path(index_root),
        schema_version=_STATE_SCHEMA_VERSION,
        key="progress",
    )


def load_progress(index_root: Path) -> dict[str, dict[str, Any]]:
    """Load durable per-repository refresh telemetry."""

    raw = _progress_store(index_root).read(dict) or {}
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
        _progress_store(index_root).write(all_progress)
