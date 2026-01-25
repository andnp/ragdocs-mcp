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
        embedding_model_name = config.llm.embedding_model
        if embedding_model_name == "local":
            embedding_model_name = "BAAI/bge-small-en-v1.5"

        vector = VectorIndex(
            embedding_model_name=embedding_model_name,
            embedding_workers=config.indexing.embedding_workers,
        )
        keyword = KeywordIndex()
        graph = GraphStore()

        latest_snapshot = _find_latest_snapshot(snapshot_base)
        if latest_snapshot is not None:
            logger.info("Loading indices from snapshot: %s", latest_snapshot)
            await asyncio.to_thread(_load_indices_from_snapshot, vector, keyword, graph, latest_snapshot)
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
        if config.memory.enabled:
            from src.config import resolve_memory_path
            from src.memory.manager import MemoryIndexManager
            from src.memory.search import MemorySearchOrchestrator

            memory_path = resolve_memory_path(config)

            memory_vector = VectorIndex(
                embedding_model_name=embedding_model_name,
                embedding_workers=config.indexing.embedding_workers,
            )
            memory_keyword = KeywordIndex()
            memory_graph = GraphStore()

            memory_index_path = snapshot_base / "memory"
            if memory_index_path.exists():
                await asyncio.to_thread(
                    _load_indices_from_snapshot,
                    memory_vector,
                    memory_keyword,
                    memory_graph,
                    memory_index_path,
                )

            memory_manager = MemoryIndexManager(
                config=config,
                memory_path=memory_path,
                vector=memory_vector,
                keyword=memory_keyword,
                graph=memory_graph,
            )

            memory_search = MemorySearchOrchestrator(
                vector=memory_vector,
                keyword=memory_keyword,
                graph=memory_graph,
                config=config,
                manager=memory_manager,
                documents_path=Path(memory_path),
            )

        def reload_callback(snapshot_dir: Path, version: int) -> None:
            _load_indices_from_snapshot(vector, keyword, graph, snapshot_dir)
            logger.info("Reloaded indices from snapshot v%d", version)

        sync_receiver = IndexSyncReceiver(snapshot_base, reload_callback)

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

        memory_manager_instance = None
        if config.memory.enabled and memory_search is not None:
            memory_manager_instance = memory_search._manager

        ctx = cls(
            config=config,
            vector=vector,
            keyword=keyword,
            graph=graph,
            orchestrator=orchestrator,
            memory_search=memory_search,
            sync_receiver=sync_receiver,
            commit_indexer=commit_indexer,
            memory_manager=memory_manager_instance,
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


def _find_latest_snapshot(snapshot_base: Path) -> Path | None:
    if not snapshot_base.exists():
        return None

    version_file = snapshot_base / "version.bin"
    if not version_file.exists():
        return None

    import struct

    try:
        data = version_file.read_bytes()
        if len(data) >= 4:
            version = struct.unpack("<I", data[:4])[0]
            snapshot_dir = snapshot_base / f"v{version}"
            if snapshot_dir.exists():
                return snapshot_dir
    except (OSError, struct.error):
        pass

    return None


def _load_indices_from_snapshot(
    vector: VectorIndex,
    keyword: KeywordIndex,
    graph: GraphStore,
    snapshot_dir: Path,
) -> None:
    vector.load_from(snapshot_dir / "vector")
    keyword.load_from(snapshot_dir / "keyword")
    graph.load_from(snapshot_dir / "graph")
