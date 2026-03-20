from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import Config, detect_project, resolve_documents_path
from src.git.parallel_indexer import ParallelIndexingConfig, index_commits_parallel_sync
from src.git.repository import (
    discover_git_repositories,
    discover_git_repositories_multi_root,
    get_commits_after_timestamp,
    is_git_available,
)
from src.indexing.discovery import discover_files as discover_files_single_root
from src.indexing.discovery import discover_files_multi_root
from src.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

REBUILD_ACTIVE_STATUSES = {"queued", "running"}
REBUILD_TERMINAL_STATUSES = {"succeeded", "failed"}


@dataclass(frozen=True)
class RebuildScope:
    project: str | None
    project_label: str | None
    documents_roots: list[Path]

    @property
    def is_global(self) -> bool:
        return self.project is None

    @property
    def scope_label(self) -> str:
        if self.project_label is not None:
            return f"project '{self.project_label}'"
        return "global corpus"


def rebuild_status_path(runtime_root: Path) -> Path:
    return runtime_root / "rebuild-status.json"


def default_rebuild_status() -> dict[str, object]:
    return {
        "status": "idle",
        "phase": "idle",
        "request_id": None,
        "project": None,
        "project_label": None,
        "scope_label": None,
        "documents_roots": [],
        "checkpoint_interval": 0,
        "discovered_files": 0,
        "indexed_files": 0,
        "removed_documents": 0,
        "git_repositories": 0,
        "git_commits_indexed": 0,
        "messages": [],
        "error": None,
        "submitted_at": None,
        "started_at": None,
        "completed_at": None,
        "vocabulary_catch_up_scheduled": False,
    }


def read_rebuild_status(runtime_root: Path) -> dict[str, object]:
    path = rebuild_status_path(runtime_root)
    if not path.exists():
        return default_rebuild_status()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read rebuild status at %s", path, exc_info=True)
        return default_rebuild_status()

    if not isinstance(payload, dict):
        return default_rebuild_status()
    return {**default_rebuild_status(), **payload}


def write_rebuild_status(runtime_root: Path, payload: dict[str, object]) -> dict[str, object]:
    normalized = {**default_rebuild_status(), **payload}
    atomic_write_json(rebuild_status_path(runtime_root), normalized)
    return normalized


def resolve_rebuild_scope(
    config: Config,
    global_documents_roots: list[Path],
    project_override: str | None,
) -> RebuildScope:
    if project_override is None:
        if global_documents_roots:
            roots = [root.resolve() for root in global_documents_roots]
        else:
            roots = [Path(resolve_documents_path(config)).resolve()]
        return RebuildScope(project=None, project_label=None, documents_roots=roots)

    detected_project = detect_project(
        projects=config.projects,
        project_override=project_override,
    )
    if detected_project is not None:
        for project in config.projects:
            if project.name == detected_project:
                return RebuildScope(
                    project=detected_project,
                    project_label=detected_project,
                    documents_roots=[Path(project.path).resolve()],
                )

    override_path = Path(project_override).expanduser()
    if override_path.exists():
        return RebuildScope(
            project=detected_project,
            project_label=detected_project or project_override,
            documents_roots=[override_path.resolve()],
        )

    raise ValueError(f"Unknown rebuild project scope: {project_override}")


def submit_rebuild_status(
    runtime_root: Path,
    *,
    request_id: str,
    scope: RebuildScope,
) -> dict[str, object]:
    now = time.time()
    return write_rebuild_status(
        runtime_root,
        {
            "status": "queued",
            "phase": "queued",
            "request_id": request_id,
            "project": scope.project,
            "project_label": scope.project_label,
            "scope_label": scope.scope_label,
            "documents_roots": [str(root) for root in scope.documents_roots],
            "submitted_at": now,
            "started_at": None,
            "completed_at": None,
            "messages": [],
            "error": None,
            "discovered_files": 0,
            "indexed_files": 0,
            "removed_documents": 0,
            "git_repositories": 0,
            "git_commits_indexed": 0,
            "vocabulary_catch_up_scheduled": False,
        },
    )


def iter_rebuild_batches(file_paths: list[str], batch_size: int):
    normalized_batch_size = max(1, batch_size)
    for start in range(0, len(file_paths), normalized_batch_size):
        yield file_paths[start : start + normalized_batch_size]


def _discover_scope_files(config: Config, documents_roots: list[Path]) -> list[str]:
    if len(documents_roots) <= 1:
        single_root = documents_roots[0] if documents_roots else Path(
            resolve_documents_path(config)
        ).resolve()
        return discover_files_single_root(
            documents_path=str(single_root),
            include_patterns=config.indexing.include,
            exclude_patterns=config.indexing.exclude,
            exclude_hidden_dirs=config.indexing.exclude_hidden_dirs,
        )

    return discover_files_multi_root(
        [str(root) for root in documents_roots],
        include_patterns=config.indexing.include,
        exclude_patterns=config.indexing.exclude,
        exclude_hidden_dirs=config.indexing.exclude_hidden_dirs,
    )


def _discover_scope_git_repositories(config: Config, documents_roots: list[Path]) -> list[Path]:
    if len(documents_roots) <= 1:
        single_root = documents_roots[0] if documents_roots else Path(
            resolve_documents_path(config)
        ).resolve()
        return discover_git_repositories(
            single_root,
            config.indexing.exclude,
            config.indexing.exclude_hidden_dirs,
        )

    return discover_git_repositories_multi_root(
        documents_roots,
        config.indexing.exclude,
        config.indexing.exclude_hidden_dirs,
    )


def _resolve_indexed_file_path(raw_file_path: str | None, *, common_root: Path) -> Path | None:
    if not raw_file_path:
        return None

    candidate = Path(raw_file_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (common_root / candidate).resolve()


def _find_scope_document_ids(
    *,
    descriptions: list[dict[str, object]],
    documents_roots: list[Path],
    common_root: Path,
) -> list[str]:
    scope_roots = [root.resolve() for root in documents_roots]
    doc_ids: list[str] = []
    for description in descriptions:
        raw_doc_id = description.get("doc_id")
        raw_file_path = description.get("file_path")
        if not isinstance(raw_doc_id, str):
            continue
        resolved_path = _resolve_indexed_file_path(
            raw_file_path if isinstance(raw_file_path, str) else None,
            common_root=common_root,
        )
        if resolved_path is None:
            continue
        if any(_path_is_relative_to(resolved_path, root) for root in scope_roots):
            doc_ids.append(raw_doc_id)
    return doc_ids


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _append_message(runtime_root: Path, message: str) -> None:
    status = read_rebuild_status(runtime_root)
    messages = status.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    messages = [item for item in messages if isinstance(item, str)]
    messages.append(message)
    write_rebuild_status(runtime_root, {**status, "messages": messages})


def _update_rebuild_progress(
    runtime_root: Path,
    **changes: object,
) -> dict[str, object]:
    status = read_rebuild_status(runtime_root)
    status.update(changes)
    return write_rebuild_status(runtime_root, status)


def run_rebuild(
    *,
    runtime_root: Path,
    config: Config,
    index_manager,
    commit_indexer,
    global_documents_roots: list[Path],
    request_id: str,
    project_override: str | None,
    schedule_vocabulary_catch_up: Callable[[], bool] | None = None,
) -> dict[str, object]:
    scope = resolve_rebuild_scope(config, global_documents_roots, project_override)
    checkpoint_interval = max(1, config.indexing.rebuild_checkpoint_interval)
    total_files = 0
    indexed_files = 0
    removed_documents = 0
    git_repositories = 0
    git_commits_indexed = 0

    _update_rebuild_progress(
        runtime_root,
        status="running",
        phase="preparing",
        request_id=request_id,
        project=scope.project,
        project_label=scope.project_label,
        scope_label=scope.scope_label,
        documents_roots=[str(root) for root in scope.documents_roots],
        checkpoint_interval=checkpoint_interval,
        started_at=time.time(),
        completed_at=None,
        error=None,
        messages=[],
    )

    try:
        if scope.is_global:
            index_manager.clear_documents()
            manifest_path = Path(config.indexing.index_path) / "index.manifest.json"
            manifest_path.unlink(missing_ok=True)
        else:
            existing_doc_ids = _find_scope_document_ids(
                descriptions=index_manager.vector.describe_documents(),
                documents_roots=scope.documents_roots,
                common_root=Path(config.indexing.documents_path).resolve(),
            )
            if existing_doc_ids:
                index_manager.remove_documents(existing_doc_ids, persist=False)
                removed_documents = len(existing_doc_ids)
                index_manager.persist_checkpoint()

        scope_message = (
            f"Rebuild scope: {scope.scope_label} across {len(scope.documents_roots)} root(s)"
        )
        _append_message(runtime_root, scope_message)

        files_to_index = _discover_scope_files(config, scope.documents_roots)
        total_files = len(files_to_index)
        _append_message(
            runtime_root,
            (
                "Discovered "
                f"{total_files} files; persisting checkpoints every {checkpoint_interval} file(s)"
            ),
        )
        _update_rebuild_progress(
            runtime_root,
            phase="indexing_documents",
            discovered_files=total_files,
            indexed_files=0,
            removed_documents=removed_documents,
        )

        for file_batch in iter_rebuild_batches(files_to_index, checkpoint_interval):
            if file_batch:
                index_manager.index_documents(file_batch, force=True, persist=False)
                indexed_files += len(file_batch)
                index_manager.persist_checkpoint()
                checkpoint_message = (
                    f"📍 Checkpoint persisted: {indexed_files}/{total_files} documents"
                )
                _append_message(runtime_root, checkpoint_message)
                _update_rebuild_progress(
                    runtime_root,
                    phase="indexing_documents",
                    indexed_files=indexed_files,
                )

        if total_files == 0:
            index_manager.persist_checkpoint()

        _update_rebuild_progress(runtime_root, phase="finalizing")
        index_manager.finalize_derived_graph_state()

        if config.git_indexing.enabled and commit_indexer is not None:
            if not is_git_available():
                _append_message(
                    runtime_root,
                    "⚠️  Git binary not available, skipping git commit indexing",
                )
            else:
                repos = _discover_scope_git_repositories(config, scope.documents_roots)
                git_repositories = len(repos)
                _update_rebuild_progress(
                    runtime_root,
                    phase="indexing_git",
                    git_repositories=git_repositories,
                )
                if scope.is_global:
                    commit_indexer.clear()
                elif repos:
                    commit_indexer.clear_repositories(
                        [str(repo.parent) for repo in repos]
                    )

                for repo_path in repos:
                    commit_hashes = get_commits_after_timestamp(repo_path, None)
                    if not commit_hashes:
                        continue
                    git_commits_indexed += index_commits_parallel_sync(
                        commit_hashes,
                        repo_path,
                        commit_indexer,
                        ParallelIndexingConfig(),
                        200,
                    )

                if repos:
                    _append_message(
                        runtime_root,
                        (
                            "✅ Successfully indexed "
                            f"{git_commits_indexed} git commits from {git_repositories} repositories"
                        ),
                    )
                else:
                    _append_message(runtime_root, "ℹ️  No git repositories found")

        vocabulary_scheduled = False
        if schedule_vocabulary_catch_up is not None:
            try:
                vocabulary_scheduled = bool(schedule_vocabulary_catch_up())
            except Exception:
                logger.warning(
                    "Failed to schedule vocabulary catch-up after rebuild",
                    exc_info=True,
                )

        if vocabulary_scheduled:
            _append_message(
                runtime_root,
                "ℹ️  Concept vocabulary catch-up scheduled in the daemon runtime",
            )

        summary_message = (
            f"✅ Successfully rebuilt index: {total_files} documents indexed"
        )
        _append_message(runtime_root, summary_message)
        return _update_rebuild_progress(
            runtime_root,
            status="succeeded",
            phase="completed",
            discovered_files=total_files,
            indexed_files=indexed_files,
            removed_documents=removed_documents,
            git_repositories=git_repositories,
            git_commits_indexed=git_commits_indexed,
            vocabulary_catch_up_scheduled=vocabulary_scheduled,
            completed_at=time.time(),
        )
    except Exception as exc:
        logger.error("Daemon-owned rebuild failed: %s", exc, exc_info=True)
        _append_message(runtime_root, f"❌ Rebuild failed: {exc}")
        return _update_rebuild_progress(
            runtime_root,
            status="failed",
            phase="failed",
            discovered_files=total_files,
            indexed_files=indexed_files,
            removed_documents=removed_documents,
            git_repositories=git_repositories,
            git_commits_indexed=git_commits_indexed,
            error=str(exc),
            completed_at=time.time(),
        )