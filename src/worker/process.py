from __future__ import annotations

import asyncio
import glob
import logging
import signal
from dataclasses import dataclass, field
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Config, load_config
from src.ipc.commands import (
    HealthCheckCommand,
    HealthStatusResponse,
    IndexUpdatedNotification,
    InitCompleteNotification,
    ReindexDocumentCommand,
    ShutdownCommand,
)
from src.ipc.index_sync import IndexSyncPublisher
from src.ipc.queue_manager import QueueManager

if TYPE_CHECKING:
    from src.git.commit_indexer import CommitIndexer
    from src.git.watcher import GitWatcher
    from src.indexing.manager import IndexManager
    from src.indexing.watcher import FileWatcher
    from src.indices.graph import GraphStore
    from src.indices.keyword import KeywordIndex
    from src.indices.vector import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    config: Config
    vector: VectorIndex
    keyword: KeywordIndex
    graph: GraphStore
    index_manager: IndexManager
    sync_publisher: IndexSyncPublisher
    command_queue: QueueManager
    response_queue: QueueManager
    shutdown_event: EventType
    file_watcher: FileWatcher | None = None
    git_watcher: GitWatcher | None = None
    commit_indexer: CommitIndexer | None = None
    last_index_time: float | None = None
    last_publish_time: float | None = None
    docs_indexed_since_publish: int = 0
    _watcher_callback_registered: bool = field(default=False, repr=False)


def worker_main(
    config_dict: dict,
    command_queue: Queue,
    response_queue: Queue,
    shutdown_event: EventType,
    snapshot_base: Path,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[WORKER %(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(_worker_async_main(
        config_dict,
        command_queue,
        response_queue,
        shutdown_event,
        snapshot_base,
    ))


async def _worker_async_main(
    config_dict: dict,
    command_queue: Queue,
    response_queue: Queue,
    shutdown_event: EventType,
    snapshot_base: Path,
) -> None:
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    cmd_manager = QueueManager(command_queue, "command")
    resp_manager = QueueManager(response_queue, "response")

    try:
        state = await _initialize_worker(
            config_dict,
            cmd_manager,
            resp_manager,
            shutdown_event,
            snapshot_base,
        )
    except Exception as e:
        logger.error("Worker initialization failed: %s", e, exc_info=True)
        return

    try:
        await _run_command_loop(state)
    except Exception as e:
        logger.error("Worker command loop failed: %s", e, exc_info=True)
    finally:
        await _shutdown_worker(state)


async def _initialize_worker(
    config_dict: dict,
    command_queue: QueueManager,
    response_queue: QueueManager,
    shutdown_event: EventType,
    snapshot_base: Path,
) -> WorkerState:
    logger.info("Initializing worker process")

    config = _config_from_dict(config_dict)

    vector, keyword, graph = await _create_indices(config)

    from src.indexing.manager import IndexManager
    index_manager = IndexManager(config, vector, keyword, graph)

    sync_publisher = IndexSyncPublisher(snapshot_base)

    state = WorkerState(
        config=config,
        vector=vector,
        keyword=keyword,
        graph=graph,
        index_manager=index_manager,
        sync_publisher=sync_publisher,
        command_queue=command_queue,
        response_queue=response_queue,
        shutdown_event=shutdown_event,
    )

    await _run_initial_indexing(state)

    _publish_snapshot(state)

    import time
    state.last_publish_time = time.time()
    state.docs_indexed_since_publish = 0

    await _start_watchers(state)

    doc_count = index_manager.get_document_count()
    notification = InitCompleteNotification(
        version=sync_publisher.version,
        doc_count=doc_count,
    )
    response_queue.put_nowait(notification)

    logger.info(
        "Worker initialized: %d documents, snapshot v%d",
        doc_count,
        sync_publisher.version,
    )

    return state


def _config_from_dict(config_dict: dict) -> Config:
    config = load_config()

    if "indexing" in config_dict:
        for key, value in config_dict["indexing"].items():
            if hasattr(config.indexing, key):
                setattr(config.indexing, key, value)

    if "documents_path" in config_dict:
        config.indexing.documents_path = config_dict["documents_path"]

    if "index_path" in config_dict:
        config.indexing.index_path = config_dict["index_path"]

    return config


async def _create_indices(config: Config) -> tuple[VectorIndex, KeywordIndex, GraphStore]:
    from src.indices.graph import GraphStore
    from src.indices.keyword import KeywordIndex
    from src.indices.vector import VectorIndex

    embedding_model_name = config.llm.embedding_model
    if embedding_model_name == "local":
        embedding_model_name = "BAAI/bge-small-en-v1.5"

    logger.info("Loading embedding model: %s", embedding_model_name)
    vector = VectorIndex(
        embedding_model_name=embedding_model_name,
        embedding_workers=config.indexing.embedding_workers,
    )
    await asyncio.to_thread(vector.warm_up)

    keyword = KeywordIndex()
    graph = GraphStore()

    return vector, keyword, graph


async def _run_initial_indexing(state: WorkerState) -> None:
    index_path = Path(state.config.indexing.index_path)

    from src.indexing.manifest import load_manifest, should_rebuild

    current_manifest = _build_manifest(state.config)
    saved_manifest = load_manifest(index_path)
    needs_rebuild = should_rebuild(current_manifest, saved_manifest)

    if needs_rebuild:
        logger.info("Full index rebuild required")
        await _full_index(state, current_manifest)
    else:
        logger.info("Loading existing indices")
        await asyncio.to_thread(state.index_manager.load)
        await _startup_reconciliation(state, current_manifest)


def _build_manifest(config: Config):
    from src.indexing.manifest import IndexManifest

    return IndexManifest(
        spec_version="1.0.0",
        embedding_model=config.llm.embedding_model,
        parsers=config.parsers,
        chunking_config={
            "strategy": config.document_chunking.strategy,
            "min_chunk_chars": config.document_chunking.min_chunk_chars,
            "max_chunk_chars": config.document_chunking.max_chunk_chars,
            "overlap_chars": config.document_chunking.overlap_chars,
        },
    )


def _discover_files(config: Config) -> list[str]:
    from src.utils import should_include_file

    docs_path = Path(config.indexing.documents_path)
    all_files: set[str] = set()

    for pattern in config.parsers.keys():
        glob_pattern = str(docs_path / pattern)
        files = glob.glob(glob_pattern, recursive=config.indexing.recursive)
        all_files.update(files)

    return [
        f for f in sorted(all_files)
        if should_include_file(
            f,
            config.indexing.include,
            config.indexing.exclude,
            config.indexing.exclude_hidden_dirs,
        )
    ]


async def _full_index(state: WorkerState, manifest) -> None:
    from src.indexing.manifest import save_manifest
    from src.indexing.reconciler import build_indexed_files_map

    files_to_index = _discover_files(state.config)
    docs_path = Path(state.config.indexing.documents_path)
    index_path = Path(state.config.indexing.index_path)

    logger.info("Indexing %d documents", len(files_to_index))

    for file_path in files_to_index:
        await asyncio.to_thread(state.index_manager.index_document, file_path)

    await asyncio.to_thread(state.index_manager.persist)

    manifest.indexed_files = build_indexed_files_map(files_to_index, docs_path)
    await asyncio.to_thread(save_manifest, index_path, manifest)

    logger.info("Initial indexing complete: %d documents", len(files_to_index))


async def _startup_reconciliation(state: WorkerState, manifest) -> None:
    from src.indexing.manifest import save_manifest
    from src.indexing.reconciler import build_indexed_files_map

    docs_path = Path(state.config.indexing.documents_path)
    index_path = Path(state.config.indexing.index_path)

    discovered_files = _discover_files(state.config)

    result = await asyncio.to_thread(
        state.index_manager.reconcile_indices,
        discovered_files,
        docs_path,
    )

    if result.added_count > 0 or result.removed_count > 0 or result.moved_count > 0:
        await asyncio.to_thread(state.index_manager.persist)
        manifest.indexed_files = build_indexed_files_map(discovered_files, docs_path)
        await asyncio.to_thread(save_manifest, index_path, manifest)
        logger.info(
            "Reconciliation complete: added=%d, removed=%d, moved=%d, failed=%d",
            result.added_count,
            result.removed_count,
            result.moved_count,
            result.failed_count,
        )
    else:
        logger.info("Reconciliation complete: no changes needed")


def _publish_snapshot(state: WorkerState) -> int:
    def persist_callback(snapshot_dir: Path) -> None:
        state.vector.persist_to(snapshot_dir / "vector")
        state.keyword.persist_to(snapshot_dir / "keyword")
        state.graph.persist_to(snapshot_dir / "graph")

    version = state.sync_publisher.publish(persist_callback)

    import time
    state.last_index_time = time.time()

    return version


def _should_publish_snapshot(
    state: WorkerState,
    pending_count: int,
    last_sync: str | None,
) -> bool:
    if last_sync is None:
        return False

    from datetime import datetime
    last_sync_dt = datetime.fromisoformat(last_sync)
    last_sync_ts = last_sync_dt.timestamp()

    if pending_count == 0:
        if state.last_index_time is None or last_sync_ts > state.last_index_time:
            return True
        return False

    if state.last_publish_time is None:
        return False

    import time
    config = state.config.worker
    elapsed = time.time() - state.last_publish_time

    if elapsed >= config.progressive_snapshot_interval:
        return True

    if state.docs_indexed_since_publish >= config.progressive_snapshot_doc_count:
        return True

    return False


async def _start_watchers(state: WorkerState) -> None:
    from src.indexing.watcher import FileWatcher

    state.file_watcher = FileWatcher(
        documents_path=state.config.indexing.documents_path,
        index_manager=state.index_manager,
    )

    if not state._watcher_callback_registered:
        original_batch_process = state.file_watcher._batch_process

        async def batch_process_with_tracking(events: dict):
            await original_batch_process(events)
            create_or_modify = sum(1 for event_type in events.values() if event_type in ("created", "modified"))
            state.docs_indexed_since_publish += create_or_modify

        state.file_watcher._batch_process = batch_process_with_tracking
        state._watcher_callback_registered = True

    state.file_watcher.start()
    logger.info("File watcher started")

    if state.config.git_indexing.enabled:
        await _start_git_indexer(state)


async def _start_git_indexer(state: WorkerState) -> None:
    from src.git.repository import discover_git_repositories, is_git_available

    if not is_git_available():
        logger.warning("Git binary not found - git indexing disabled")
        return

    from src.git.commit_indexer import CommitIndexer

    index_path = Path(state.config.indexing.index_path)
    db_path = index_path / "git_commits.db"

    state.commit_indexer = CommitIndexer(
        db_path=db_path,
        embedding_model=state.vector,
    )

    await _index_git_commits(state)

    if state.config.git_indexing.watch_enabled:
        from src.git.watcher import GitWatcher

        git_repos = discover_git_repositories(
            Path(state.config.indexing.documents_path),
            state.config.indexing.exclude,
            state.config.indexing.exclude_hidden_dirs,
        )

        if git_repos:
            state.git_watcher = GitWatcher(
                git_repos=git_repos,
                commit_indexer=state.commit_indexer,
                config=state.config,
                cooldown=state.config.git_indexing.watch_cooldown,
            )
            state.git_watcher.start()
            logger.info("Git watcher started for %d repositories", len(git_repos))


async def _index_git_commits(state: WorkerState) -> None:
    if state.commit_indexer is None:
        return

    from src.git.parallel_indexer import ParallelIndexingConfig, index_commits_parallel
    from src.git.repository import discover_git_repositories, get_commits_after_timestamp

    repos = discover_git_repositories(
        Path(state.config.indexing.documents_path),
        state.config.indexing.exclude,
        state.config.indexing.exclude_hidden_dirs,
    )

    parallel_config = ParallelIndexingConfig(
        max_workers=state.config.git_indexing.parallel_workers,
        batch_size=state.config.git_indexing.batch_size,
        embed_batch_size=state.config.git_indexing.embed_batch_size,
    )

    total_indexed = 0
    for repo_path in repos:
        try:
            last_timestamp = state.commit_indexer.get_last_indexed_timestamp(str(repo_path.parent))
            commit_hashes = get_commits_after_timestamp(repo_path, last_timestamp)

            if not commit_hashes:
                continue

            logger.info(
                "Repository %s: indexing %d commits",
                repo_path.parent,
                len(commit_hashes),
            )

            indexed = await index_commits_parallel(
                commit_hashes,
                repo_path,
                state.commit_indexer,
                parallel_config,
                state.config.git_indexing.delta_max_lines,
            )
            total_indexed += indexed
        except Exception as e:
            logger.error("Failed to index repository %s: %s", repo_path, e, exc_info=True)

    if total_indexed > 0:
        logger.info("Git commit indexing complete: %d commits", total_indexed)


async def _run_command_loop(state: WorkerState) -> None:
    logger.info("Entering command processing loop")

    while not state.shutdown_event.is_set():
        message = await state.command_queue.get(timeout=0.5)

        if message is None:
            await _check_for_index_updates(state)
            continue

        await _handle_command(state, message)


async def _check_for_index_updates(state: WorkerState) -> None:
    if state.file_watcher is None:
        return

    pending = state.file_watcher.get_pending_queue_size()
    last_sync = state.file_watcher.get_last_sync_time()

    should_publish = _should_publish_snapshot(state, pending, last_sync)

    if should_publish:
        logger.info("Publishing progressive snapshot (pending: %d)", pending)
        await asyncio.to_thread(state.index_manager.persist)
        version = _publish_snapshot(state)

        import time
        state.last_publish_time = time.time()
        state.docs_indexed_since_publish = 0

        doc_count = state.index_manager.get_document_count()
        notification = IndexUpdatedNotification(
            version=version,
            doc_count=doc_count,
        )
        state.response_queue.put_nowait(notification)


async def _handle_command(state: WorkerState, message) -> None:
    if isinstance(message, ShutdownCommand):
        logger.info("Received shutdown command (graceful=%s)", message.graceful)
        state.shutdown_event.set()

    elif isinstance(message, HealthCheckCommand):
        response = _build_health_response(state)
        state.response_queue.put_nowait(response)

    elif isinstance(message, ReindexDocumentCommand):
        logger.info("Reindexing document: %s (reason: %s)", message.doc_id, message.reason)
        success = await asyncio.to_thread(
            state.index_manager.reindex_document,
            message.doc_id,
            message.reason,
        )

        if success:
            state.docs_indexed_since_publish += 1
            await asyncio.to_thread(state.index_manager.persist)
            version = _publish_snapshot(state)

            import time
            state.last_publish_time = time.time()
            state.docs_indexed_since_publish = 0

            doc_count = state.index_manager.get_document_count()
            notification = IndexUpdatedNotification(
                version=version,
                doc_count=doc_count,
            )
            state.response_queue.put_nowait(notification)

    else:
        logger.warning("Unknown command type: %s", type(message).__name__)


def _build_health_response(state: WorkerState) -> HealthStatusResponse:
    queue_depth = 0
    if state.file_watcher:
        queue_depth = state.file_watcher.get_pending_queue_size()

    return HealthStatusResponse(
        healthy=True,
        queue_depth=queue_depth,
        last_index_time=state.last_index_time,
        doc_count=state.index_manager.get_document_count(),
    )


async def _shutdown_worker(state: WorkerState) -> None:
    logger.info("Shutting down worker")

    if state.file_watcher:
        try:
            await asyncio.wait_for(state.file_watcher.stop(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("File watcher stop timed out")

    if state.git_watcher:
        try:
            await asyncio.wait_for(state.git_watcher.stop(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Git watcher stop timed out")

    try:
        await asyncio.to_thread(state.index_manager.persist)
        _publish_snapshot(state)
    except Exception as e:
        logger.error("Failed to persist indices during shutdown: %s", e)

    if state.commit_indexer:
        try:
            await asyncio.to_thread(state.commit_indexer.close)
        except Exception as e:
            logger.error("Failed to close commit indexer: %s", e)

    logger.info("Worker shutdown complete")
