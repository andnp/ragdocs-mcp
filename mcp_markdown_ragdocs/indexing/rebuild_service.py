from __future__ import annotations

import json
import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
from mcp_markdown_ragdocs.config import Config, detect_project, resolve_documents_path
from mcp_markdown_ragdocs.git.repository import (
    discover_git_repositories,
    discover_git_repositories_multi_root,
    get_git_ref_signature,
    is_git_available,
)
from searchkernel.api import (
    AsyncIndexIngestor,
    Cursor,
    IngestionFailureMode,
    IngestionReceipt,
    Record,
    atomic_write_json,
    discover_files as discover_files_single_root,
    discover_files_multi_root,
)

logger = logging.getLogger(__name__)

REBUILD_ACTIVE_STATUSES = {"queued", "running"}
REBUILD_TERMINAL_STATUSES = {"succeeded", "failed"}
GIT_REBUILD_BATCH_SIZE = 25
REBUILD_CHECKPOINT_SCHEMA_VERSION = 1
REBUILD_CHECKPOINT_FILE_NAME = "rebuild.checkpoint.json"


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


class _AsyncRecordIndexManager:
    def __init__(self, index_manager) -> None:
        self._ingestor = AsyncIndexIngestor(index_manager)

    async def index_records(
        self,
        records: list[Record],
        *,
        checkpoint: Cursor | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        return await self._ingestor.index_records(
            records,
            checkpoint=checkpoint,
            failure_mode=failure_mode,
        )


def rebuild_checkpoint_path(runtime_root: Path) -> Path:
    return runtime_root / REBUILD_CHECKPOINT_FILE_NAME


def _load_rebuild_checkpoint(runtime_root: Path) -> dict[str, object] | None:
    path = rebuild_checkpoint_path(runtime_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read rebuild checkpoint at %s", path, exc_info=True)
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REBUILD_CHECKPOINT_SCHEMA_VERSION
    ):
        logger.warning("Ignoring unsupported rebuild checkpoint at %s", path)
        return None
    return payload


def _save_rebuild_checkpoint(
    runtime_root: Path,
    checkpoint: dict[str, object],
) -> None:
    atomic_write_json(rebuild_checkpoint_path(runtime_root), checkpoint)


def _file_targets(file_paths: list[str]) -> dict[str, dict[str, int]]:
    targets: dict[str, dict[str, int]] = {}
    for raw_path in file_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            stat_result = path.stat()
        except OSError as exc:
            raise RuntimeError(
                f"Rebuild source file disappeared or is unreadable: {path}"
            ) from exc
        targets[str(path)] = {
            "mtime_ns": stat_result.st_mtime_ns,
            "size": stat_result.st_size,
        }
    return targets


def _encoder_identity(index_manager) -> dict[str, object]:
    fingerprint = getattr(index_manager, "_encoder_fingerprint", None)
    if fingerprint is None:
        return {}
    return {
        field: getattr(fingerprint, field, None)
        for field in (
            "model",
            "version",
            "normalization",
            "query_instruction",
            "text_instruction",
            "dimension",
        )
    }


def _rebuild_identity(
    *,
    config: Config,
    index_manager,
    scope: RebuildScope,
    git_targets: dict[str, str],
) -> dict[str, object]:
    indexing = config.indexing
    chunking = config.chunking
    return {
        "project": scope.project,
        "documents_roots": [str(root.resolve()) for root in scope.documents_roots],
        "index_path": str(Path(indexing.index_path).expanduser().resolve()),
        "config": {
            "include": list(indexing.include),
            "exclude": list(indexing.exclude),
            "exclude_hidden_dirs": indexing.exclude_hidden_dirs,
            "chunking": {
                "strategy": chunking.strategy,
                "min_chunk_chars": chunking.min_chunk_chars,
                "max_chunk_chars": chunking.max_chunk_chars,
                "overlap_chars": chunking.overlap_chars,
                "parent_chunk_min_chars": chunking.parent_chunk_min_chars,
                "parent_chunk_max_chars": chunking.parent_chunk_max_chars,
            },
            "embedding_model": getattr(
                config.llm,
                "resolved_embedding_model",
                getattr(config.llm, "embedding_model", None),
            ),
            "store_backend": getattr(config.store, "backend", None),
        },
        "encoder": _encoder_identity(index_manager),
        "git": git_targets,
    }


def _new_rebuild_checkpoint(
    *,
    request_id: str,
    identity: dict[str, object],
    document_targets: dict[str, dict[str, int]],
    git_targets: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": REBUILD_CHECKPOINT_SCHEMA_VERSION,
        "request_id": request_id,
        "identity": identity,
        "documents": {
            "targets": document_targets,
            "completed": {},
        },
        "git": {
            repo_path: {
                "ref_signature": ref_signature,
                "completed_batches": [],
            }
            for repo_path, ref_signature in git_targets.items()
        },
    }


def _checkpoint_matches(
    checkpoint: dict[str, object] | None,
    *,
    request_id: str,
    identity: dict[str, object],
    document_targets: dict[str, dict[str, int]],
    git_targets: dict[str, str],
) -> bool:
    if checkpoint is None:
        return False
    raw_documents = checkpoint.get("documents")
    raw_git = checkpoint.get("git")
    if not isinstance(raw_documents, dict) or not isinstance(raw_git, dict):
        return False
    return (
        checkpoint.get("request_id") == request_id
        and checkpoint.get("identity") == identity
        and raw_documents.get("targets") == document_targets
        and set(raw_git) == set(git_targets)
    )


def _batch_key(records: list[Record]) -> str:
    source_ids = [
        str(getattr(record, "storage_key", getattr(record, "source_id", "")))
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(source_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_documents(
    checkpoint: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    documents = checkpoint["documents"]
    if not isinstance(documents, dict):
        raise RuntimeError("Rebuild checkpoint has invalid document progress")
    targets = documents.get("targets")
    completed = documents.get("completed")
    if not isinstance(targets, dict) or not isinstance(completed, dict):
        raise RuntimeError("Rebuild checkpoint has invalid document progress")
    return targets, completed


def _ingest_git_repository(
    *,
    runtime_root: Path,
    index_manager,
    repo_path: Path,
    git_commits_indexed: int,
    checkpoint: dict[str, object] | None = None,
    save_checkpoint: Callable[[], None] | None = None,
) -> int:
    source = GitContentSource(repo_path)
    batch: list[Record] = []
    total_indexed = git_commits_indexed
    repo_checkpoint: dict[str, object] | None = None
    completed_batches: set[str] = set()
    if checkpoint is not None:
        raw_git = checkpoint.get("git")
        if isinstance(raw_git, dict):
            raw_repo = raw_git.get(str(repo_path.resolve()))
            if isinstance(raw_repo, dict):
                repo_checkpoint = raw_repo
                raw_completed = raw_repo.get("completed_batches", [])
                if isinstance(raw_completed, list):
                    completed_batches = {
                        item for item in raw_completed if isinstance(item, str)
                    }

    def _process_batch(records: list[Record]) -> None:
        nonlocal total_indexed
        batch_key = _batch_key(records)
        if batch_key in completed_batches:
            total_indexed += len(records)
            _update_rebuild_progress(
                runtime_root,
                phase="indexing_git",
                git_commits_indexed=total_indexed,
                git_repository_path=str(repo_path),
            )
            return

        receipt = asyncio.run(
            _AsyncRecordIndexManager(index_manager).index_records(records)
        )
        if receipt.failed:
            raise RuntimeError(
                f"Git ingestion batch failed for {repo_path}: "
                f"{receipt.failed} record(s)"
            )
        total_indexed += receipt.successful
        index_manager.persist_checkpoint()
        if repo_checkpoint is not None:
            completed_batches.add(batch_key)
            repo_checkpoint["completed_batches"] = sorted(completed_batches)
            repo_checkpoint["completed_count"] = len(completed_batches)
            if save_checkpoint is not None:
                save_checkpoint()
        _update_rebuild_progress(
            runtime_root,
            phase="indexing_git",
            git_commits_indexed=total_indexed,
            git_repository_path=str(repo_path),
        )

    for record in source.iter_records():
        batch.append(record)
        if len(batch) < GIT_REBUILD_BATCH_SIZE:
            continue

        _process_batch(batch)
        batch.clear()

    if batch:
        _process_batch(batch)

    return total_indexed


def rebuild_status_path(runtime_root: Path) -> Path:
    return runtime_root / "rebuild-status.json"


def default_rebuild_status() -> dict[str, object]:
    return {
        "status": "idle",
        "phase": "idle",
        "request_id": None,
        "writer_owned": False,
        "writer_owner": None,
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
            "writer_owned": True,
            "writer_owner": request_id,
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
        writer_owned=True,
        writer_owner=request_id,
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
        scope_message = (
            f"Rebuild scope: {scope.scope_label} across {len(scope.documents_roots)} root(s)"
        )
        _append_message(runtime_root, scope_message)

        files_to_index = _discover_scope_files(config, scope.documents_roots)
        files_to_index = [
            str(Path(file_path).expanduser().resolve())
            for file_path in files_to_index
        ]
        document_targets = _file_targets(files_to_index)
        git_targets: dict[str, str] = {}
        repos: list[Path] = []
        if config.git_indexing.enabled and is_git_available():
            repos = _discover_scope_git_repositories(config, scope.documents_roots)
            for repo_path in repos:
                ref_signature = get_git_ref_signature(repo_path)
                if ref_signature is None:
                    raise RuntimeError(
                        f"Unable to fingerprint Git repository: {repo_path}"
                    )
                git_targets[str(repo_path.resolve())] = ref_signature

        identity = _rebuild_identity(
            config=config,
            index_manager=index_manager,
            scope=scope,
            git_targets=git_targets,
        )
        existing_checkpoint = _load_rebuild_checkpoint(runtime_root)
        checkpoint_is_resumable = _checkpoint_matches(
            existing_checkpoint,
            request_id=request_id,
            identity=identity,
            document_targets=document_targets,
            git_targets=git_targets,
        )
        if checkpoint_is_resumable:
            checkpoint = existing_checkpoint
            assert checkpoint is not None
            prior_status = read_rebuild_status(runtime_root)
            prior_removed_documents = prior_status.get("removed_documents")
            if isinstance(prior_removed_documents, int):
                removed_documents = prior_removed_documents
            _append_message(
                runtime_root,
                "↩️ Resuming durable rebuild progress from the last completed batch",
            )
        else:
            if existing_checkpoint is not None:
                _append_message(
                    runtime_root,
                    "⚠️ Ignoring stale or mismatched rebuild checkpoint",
                )
            checkpoint = _new_rebuild_checkpoint(
                request_id=request_id,
                identity=identity,
                document_targets=document_targets,
                git_targets=git_targets,
            )
            _save_rebuild_checkpoint(runtime_root, checkpoint)
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
            (Path(config.indexing.index_path) / "bootstrap.checkpoint.json").unlink(
                missing_ok=True
            )

        total_files = len(files_to_index)
        _, completed_documents = _checkpoint_documents(checkpoint)
        indexed_files = len(completed_documents)
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
            indexed_files=indexed_files,
            removed_documents=removed_documents,
        )

        for file_batch in iter_rebuild_batches(files_to_index, checkpoint_interval):
            if not file_batch:
                continue

            current_targets = _file_targets(file_batch)
            targets, completed_documents = _checkpoint_documents(checkpoint)
            changed_paths = [
                path
                for path, target in current_targets.items()
                if targets.get(path) != target
            ]
            if changed_paths:
                for path in changed_paths:
                    targets[path] = current_targets[path]
                    completed_documents.pop(path, None)
                _save_rebuild_checkpoint(runtime_root, checkpoint)
                _append_message(
                    runtime_root,
                    (
                        "⚠️ Reprocessing changed document source(s): "
                        + ", ".join(changed_paths)
                    ),
                )

            pending_files = [
                file_path
                for file_path in file_batch
                if str(Path(file_path).resolve()) not in completed_documents
            ]
            if pending_files:
                index_manager.index_documents(
                    pending_files,
                    force=True,
                    persist=False,
                )
                get_failed_files = getattr(index_manager, "get_failed_files", None)
                if callable(get_failed_files):
                    raw_failed_files = get_failed_files()
                    if not isinstance(raw_failed_files, list):
                        raw_failed_files = []
                    failed_paths = {
                        str(Path(item["path"]).resolve())
                        for item in raw_failed_files
                        if isinstance(item, dict) and isinstance(item.get("path"), str)
                    }
                    batch_failures = failed_paths.intersection(
                        {
                            str(Path(file_path).resolve())
                            for file_path in pending_files
                        }
                    )
                    if batch_failures:
                        raise RuntimeError(
                            "Document indexing failed for: "
                            + ", ".join(sorted(batch_failures))
                        )
                index_manager.persist_checkpoint()
                targets, completed_documents = _checkpoint_documents(checkpoint)
                for file_path in pending_files:
                    resolved_path = str(Path(file_path).resolve())
                    completed_documents[resolved_path] = targets[resolved_path]
                _save_rebuild_checkpoint(runtime_root, checkpoint)

            indexed_files = len(completed_documents)
            checkpoint_message = (
                f"📍 Checkpoint persisted: {indexed_files}/{total_files} documents"
            )
            _append_message(runtime_root, checkpoint_message)
            _update_rebuild_progress(
                runtime_root,
                phase="indexing_documents",
                indexed_files=indexed_files,
            )

        current_document_targets = _file_targets(
            [
                str(Path(file_path).expanduser().resolve())
                for file_path in _discover_scope_files(config, scope.documents_roots)
            ]
        )
        targets, completed_documents = _checkpoint_documents(checkpoint)
        if current_document_targets != targets:
            raise RuntimeError(
                "Document scope changed during rebuild; rerun to index the current corpus"
            )
        if set(completed_documents) != set(targets):
            raise RuntimeError(
                "Document rebuild checkpoint is incomplete; refusing to finalize"
            )

        if total_files == 0:
            index_manager.persist_checkpoint()

        _update_rebuild_progress(runtime_root, phase="finalizing")
        index_manager.finalize_derived_graph_state()

        if config.git_indexing.enabled:
            if not is_git_available():
                _append_message(
                    runtime_root,
                    "⚠️  Git binary not available, skipping git commit indexing",
                )
            else:
                git_repositories = len(repos)
                _update_rebuild_progress(
                    runtime_root,
                    phase="indexing_git",
                    git_repositories=git_repositories,
                )

                for repo_path in repos:
                    git_commits_indexed = _ingest_git_repository(
                        runtime_root=runtime_root,
                        index_manager=index_manager,
                        repo_path=repo_path,
                        git_commits_indexed=git_commits_indexed,
                        checkpoint=checkpoint,
                        save_checkpoint=lambda: _save_rebuild_checkpoint(
                            runtime_root,
                            checkpoint,
                        ),
                    )
                    _append_message(
                        runtime_root,
                        (
                            f"✅ Indexed git repository {repo_path} "
                            f"({git_commits_indexed} total commits)"
                        ),
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
        logger.exception("Daemon-owned rebuild failed")
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