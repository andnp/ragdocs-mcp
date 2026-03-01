from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.ipc.index_sync import IndexSyncReceiver
from src.memory.search import MemorySearchOrchestrator
from src.search.orchestrator import SearchOrchestrator

if TYPE_CHECKING:
    from src.git.commit_indexer import CommitIndexer
    from src.memory.manager import MemoryIndexManager

logger = logging.getLogger(__name__)


@dataclass
class ReadOnlyContext:
    config: Config
    vector: VectorIndex
    keyword: KeywordIndex
    graph: GraphStore
    orchestrator: SearchOrchestrator
    memory_search: MemorySearchOrchestrator | None
    sync_receiver: IndexSyncReceiver
    commit_indexer: CommitIndexer | None = None
    memory_manager: MemoryIndexManager | None = None
    _sync_task: asyncio.Task | None = field(default=None, repr=False)

    @classmethod
    async def create(cls, config: Config, snapshot_base: Path) -> ReadOnlyContext:
        embedding_model_name = config.llm.resolved_embedding_model

        vector = VectorIndex(
            embedding_model_name=embedding_model_name,
            embedding_workers=config.indexing.embedding_workers,
        )
        keyword = KeywordIndex()
        graph = GraphStore()

        loaded_version: int | None = None
        latest_snapshot = _find_latest_snapshot(snapshot_base)
        if latest_snapshot is not None:
            logger.info("Loading indices from snapshot: %s", latest_snapshot)
            await asyncio.to_thread(_load_indices_from_snapshot, vector, keyword, graph, latest_snapshot)
            # Extract version number from snapshot directory name (e.g., "v42" -> 42)
            try:
                loaded_version = int(latest_snapshot.name[1:])  # Strip "v" prefix
            except (ValueError, IndexError):
                logger.warning("Could not parse version from snapshot: %s", latest_snapshot)
        else:
            logger.info("No snapshot found, starting with empty indices")

        orchestrator = SearchOrchestrator(
            vector,
            keyword,
            graph,
            config,
            index_manager=None,
            documents_path=Path(config.indexing.documents_path),
        )

        memory_search = None
        memory_manager = None
        if config.memory.enabled:
            from src.config import resolve_memory_path
            from src.memory.init import (
                create_memory_system,
                load_or_rebuild_memory_indices,
            )

            memory_path = resolve_memory_path(config, config.detected_project, config.projects)

            # Memory system is owned by main process, indices at memory_path/indices/
            # NOT in document snapshots (see ADR-022)
            memory_manager, memory_search = create_memory_system(
                config=config,
                memory_path=memory_path,
                embedding_model_name=embedding_model_name,
            )

            await asyncio.to_thread(load_or_rebuild_memory_indices, memory_manager)

        def reload_callback(snapshot_dir: Path, version: int) -> None:
            _load_indices_from_snapshot(vector, keyword, graph, snapshot_dir)
            logger.info("Reloaded indices from snapshot v%d", version)

        sync_receiver = IndexSyncReceiver(snapshot_base, reload_callback)

        # If we loaded indices from snapshot, sync the version so is_ready() returns True
        if loaded_version is not None:
            sync_receiver.initialize_from_loaded_version(loaded_version)

        commit_indexer = None
        if config.git_indexing.enabled:
            from src.git.repository import is_git_available

            if is_git_available():
                from src.git.commit_indexer import CommitIndexer

                index_path = Path(config.indexing.index_path)
                db_path = index_path / "git_commits.db"
                if db_path.exists():
                    commit_indexer = CommitIndexer(
                        db_path=db_path,
                        embedding_model=vector,
                    )
                    logger.info("Commit indexer loaded for read-only access")

        ctx = cls(
            config=config,
            vector=vector,
            keyword=keyword,
            graph=graph,
            orchestrator=orchestrator,
            memory_search=memory_search,
            sync_receiver=sync_receiver,
            commit_indexer=commit_indexer,
            memory_manager=memory_manager,
        )

        return ctx

    def reload_indices(self, snapshot_dir: Path, version: int) -> None:
        _load_indices_from_snapshot(self.vector, self.keyword, self.graph, snapshot_dir)
        logger.info("Reloaded indices from snapshot v%d", version)

    async def start_sync_watcher(self) -> None:
        if self._sync_task is not None:
            return

        self._sync_task = asyncio.create_task(self.sync_receiver.watch())
        logger.info("Index sync watcher started")

    async def stop(self) -> None:
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        logger.info("ReadOnlyContext stopped")

    def is_ready(self) -> bool:
        return self.sync_receiver.current_version > 0


def _find_available_snapshots(snapshot_base: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (version, path) tuples for existing snapshots, newest first."""
    if not snapshot_base.exists():
        return []

    snapshots = []
    for item in snapshot_base.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            try:
                version = int(item.name[1:])
                snapshots.append((version, item))
            except ValueError:
                continue

    return sorted(snapshots, key=lambda x: x[0], reverse=True)


def _find_latest_snapshot(snapshot_base: Path) -> Path | None:
    """Find the latest valid snapshot, with fallback if version.bin is inconsistent."""
    if not snapshot_base.exists():
        return None

    import struct

    version_file = snapshot_base / "version.bin"
    pointed_version: int | None = None

    # Try to read the version pointer
    if version_file.exists():
        try:
            data = version_file.read_bytes()
            if len(data) >= 4:
                pointed_version = struct.unpack("<I", data[:4])[0]
                snapshot_dir = snapshot_base / f"v{pointed_version}"
                if snapshot_dir.exists():
                    return snapshot_dir
        except (OSError, struct.error):
            pass

    # Fallback: find highest available snapshot on disk
    available = _find_available_snapshots(snapshot_base)
    if not available:
        return None

    highest_version, highest_path = available[0]

    if pointed_version is not None:
        logger.warning(
            "version.bin points to v%d but directory missing. "
            "Available snapshots: %s. Falling back to v%d.",
            pointed_version,
            [v for v, _ in available],
            highest_version,
        )
    else:
        logger.info("No version.bin found, using highest available snapshot v%d", highest_version)

    return highest_path


def _load_indices_from_snapshot(
    vector: VectorIndex,
    keyword: KeywordIndex,
    graph: GraphStore,
    snapshot_dir: Path,
) -> None:
    vector.load_from(snapshot_dir / "vector")
    keyword.load_from(snapshot_dir / "keyword")
    graph.load_from(snapshot_dir / "graph")
