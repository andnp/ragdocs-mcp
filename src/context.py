from __future__ import annotations

import asyncio
import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Config, load_config, detect_project, resolve_index_path, resolve_documents_path, resolve_memory_path
from src.coordination import SingletonGuard
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, load_manifest, save_manifest, should_rebuild
from src.indexing.reconciler import build_indexed_files_map, reconcile_indices
from src.indexing.watcher import FileWatcher
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import SearchOrchestrator
from src.utils import should_include_file

if TYPE_CHECKING:
    from src.git.commit_indexer import CommitIndexer
    from src.memory.manager import MemoryIndexManager
    from src.memory.search import MemorySearchOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ApplicationContext:
    config: Config
    index_manager: IndexManager
    orchestrator: SearchOrchestrator
    watcher: FileWatcher | None = None
    commit_indexer: CommitIndexer | None = None
    memory_manager: MemoryIndexManager | None = None
    memory_search: MemorySearchOrchestrator | None = None
    index_path: Path = field(default_factory=lambda: Path(".index_data"))
    current_manifest: IndexManifest | None = None
    reconciliation_task: asyncio.Task | None = field(default=None, repr=False)
    _background_index_task: asyncio.Task | None = field(default=None, repr=False)
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _init_error: Exception | None = field(default=None, repr=False)
    _singleton_guard: SingletonGuard | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        project_override: str | None = None,
        enable_watcher: bool = True,
        lazy_embeddings: bool = True,
    ) -> ApplicationContext:
        config = load_config()

        detected_project = detect_project(
            projects=config.projects,
            project_override=project_override,
        )

        if detected_project and project_override:
            config = load_config()

        index_path = resolve_index_path(config, detected_project)
        documents_path = resolve_documents_path(config, detected_project, config.projects)

        config.indexing.index_path = str(index_path)
        config.indexing.documents_path = documents_path

        embedding_model_name = config.llm.embedding_model
        if embedding_model_name == "local":
            embedding_model_name = "BAAI/bge-small-en-v1.5"

        if lazy_embeddings:
            vector = VectorIndex(embedding_model_name=embedding_model_name)
        else:
            vector = VectorIndex(embedding_model_name=embedding_model_name)
            vector.warm_up()

        keyword = KeywordIndex()
        graph = GraphStore()

        manager = IndexManager(config, vector, keyword, graph)
        orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

        watcher = None
        if enable_watcher:
            watcher = FileWatcher(
                documents_path=config.indexing.documents_path,
                index_manager=manager,
            )

        # Initialize commit indexer if enabled and git available
        commit_indexer = None
        if config.git_indexing.enabled:
            from src.git.repository import is_git_available

            if is_git_available():
                from src.git.commit_indexer import CommitIndexer

                db_path = index_path / "git_commits.db"
                commit_indexer = CommitIndexer(
                    db_path=db_path,
                    embedding_model=vector,
                )
                logger.info(f"Git commit indexer initialized: {db_path}")
            else:
                logger.warning("Git binary not found - git history search disabled")

        memory_manager = None
        memory_search = None
        if config.memory.enabled:
            from src.memory.manager import MemoryIndexManager
            from src.memory.search import MemorySearchOrchestrator

            memory_path = resolve_memory_path(config, detected_project, config.projects)

            memory_vector = VectorIndex(embedding_model_name=embedding_model_name)
            memory_keyword = KeywordIndex()
            memory_graph = GraphStore()

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
            )

            logger.info(f"Memory system initialized: {memory_path}")

        return cls(
            config=config,
            index_manager=manager,
            orchestrator=orchestrator,
            watcher=watcher,
            commit_indexer=commit_indexer,
            memory_manager=memory_manager,
            memory_search=memory_search,
            index_path=index_path,
            current_manifest=None,
            reconciliation_task=None,
        )

    def _build_manifest(self) -> IndexManifest:
        return IndexManifest(
            spec_version="1.0.0",
            embedding_model=self.config.llm.embedding_model,
            parsers=self.config.parsers,
            chunking_config={
                "strategy": self.config.document_chunking.strategy,
                "min_chunk_chars": self.config.document_chunking.min_chunk_chars,
                "max_chunk_chars": self.config.document_chunking.max_chunk_chars,
                "overlap_chars": self.config.document_chunking.overlap_chars,
            },
        )

    def discover_files(self) -> list[str]:
        docs_path = Path(self.config.indexing.documents_path)

        # Collect all files matching parser patterns
        all_files = set()
        for pattern in self.config.parsers.keys():
            # Keep the full pattern including ** for recursive matching
            glob_pattern = str(docs_path / pattern)

            files = glob.glob(glob_pattern, recursive=self.config.indexing.recursive)
            all_files.update(files)

        return [
            f for f in sorted(all_files)
            if should_include_file(
                f,
                self.config.indexing.include,
                self.config.indexing.exclude,
                self.config.indexing.exclude_hidden_dirs,
            )
        ]

    def _check_and_rebuild_if_needed(self) -> bool:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.current_manifest = self._build_manifest()
        saved_manifest = load_manifest(self.index_path)
        return should_rebuild(self.current_manifest, saved_manifest)

    async def start(self, background_index: bool = False) -> None:
        coordination_mode_str = self.config.indexing.coordination_mode.lower()

        if coordination_mode_str == "singleton":
            self._singleton_guard = SingletonGuard(self.index_path)
            try:
                self._singleton_guard.acquire()
            except RuntimeError as e:
                logger.error(f"Failed to acquire singleton lock: {e}")
                raise
            logger.warning(
                "Using singleton mode - only one instance can run at a time. "
                "For multi-instance support, set coordination_mode='file_lock' in config."
            )

        needs_rebuild = self._check_and_rebuild_if_needed()

        if needs_rebuild:
            logger.info("Index rebuild required - indexing all documents")
            if background_index:
                self._background_index_task = asyncio.create_task(self._background_index())
            else:
                self._full_index()
                self._ready_event.set()
                # Build vocabulary after full index (in background)
                asyncio.create_task(self._build_initial_vocabulary())
        else:
            logger.info("Loading existing indices")
            self.index_manager.load()
            self._ready_event.set()
            await self._startup_reconciliation()
            # Build vocabulary if empty (migration from old index)
            if not self.index_manager.vector._concept_vocabulary:
                asyncio.create_task(self._build_initial_vocabulary())

        if self.watcher:
            self.watcher.start()
            logger.info("File watcher started")

        # Index git commits after document indexing
        if self.commit_indexer is not None:
            if background_index:
                asyncio.create_task(self._index_git_commits_initial_with_timeout())
            else:
                self._index_git_commits_initial_sync()

        if self.config.indexing.reconciliation_interval_seconds > 0:
            self.reconciliation_task = asyncio.create_task(self._periodic_reconciliation())
            logger.info(
                f"Periodic reconciliation enabled (interval: "
                f"{self.config.indexing.reconciliation_interval_seconds}s)"
            )

        if self.memory_manager is not None:
            try:
                self.memory_manager.load()
                reindexed = self.memory_manager.reconcile()
                if reindexed > 0:
                    self.memory_manager.persist()
                logger.info("Memory system loaded")
            except Exception as e:
                logger.warning(f"Failed to load memory indices: {e}")

    def _full_index(self) -> None:
        files_to_index = self.discover_files()
        docs_path = Path(self.config.indexing.documents_path)

        for file_path in files_to_index:
            self.index_manager.index_document(file_path)

        self.index_manager.persist()

        if self.current_manifest:
            self.current_manifest.indexed_files = build_indexed_files_map(files_to_index, docs_path)
            save_manifest(self.index_path, self.current_manifest)

        logger.info(f"Initial indexing complete: {len(files_to_index)} documents indexed")

    async def _background_index(self) -> None:
        files_to_index: list[str] = []
        try:
            logger.info("Starting background indexing")
            files_to_index = self.discover_files()
            docs_path = Path(self.config.indexing.documents_path)

            for file_path in files_to_index:
                await asyncio.to_thread(self.index_manager.index_document, file_path)

            await asyncio.to_thread(self.index_manager.persist)

            if self.current_manifest:
                self.current_manifest.indexed_files = build_indexed_files_map(files_to_index, docs_path)
                await asyncio.to_thread(save_manifest, self.index_path, self.current_manifest)

            logger.info(f"Background indexing complete: {len(files_to_index)} documents indexed")
            self._ready_event.set()
        except Exception as e:
            logger.error(
                f"Background indexing failed after processing some of {len(files_to_index)} files: {e}",
                exc_info=True
            )
            self._init_error = e
            self._ready_event.set()  # Unblock waiters so they can see the error

    async def _startup_reconciliation(self) -> None:
        logger.info("Running startup reconciliation")
        docs_path = Path(self.config.indexing.documents_path)
        discovered_files = self.discover_files()

        saved_manifest = load_manifest(self.index_path)
        if saved_manifest is None:
            raise RuntimeError("Manifest should exist when needs_rebuild is False")

        files_to_add, doc_ids_to_remove = reconcile_indices(
            discovered_files,
            saved_manifest,
            docs_path,
        )

        for doc_id in doc_ids_to_remove:
            self.index_manager.remove_document(doc_id)

        for file_path in files_to_add:
            self.index_manager.index_document(file_path)

        if files_to_add or doc_ids_to_remove:
            self.index_manager.persist()
            if self.current_manifest:
                self.current_manifest.indexed_files = build_indexed_files_map(discovered_files, docs_path)
                save_manifest(self.index_path, self.current_manifest)
            logger.info(f"Reconciliation complete: added {len(files_to_add)}, removed {len(doc_ids_to_remove)}")
        else:
            logger.info("Reconciliation complete: no changes needed")

    async def _periodic_reconciliation(self) -> None:
        interval = self.config.indexing.reconciliation_interval_seconds

        while True:
            try:
                await asyncio.sleep(interval)
                logger.info("Starting periodic reconciliation")

                docs_path = Path(self.config.indexing.documents_path)
                discovered_files = self.discover_files()

                saved_manifest = load_manifest(self.index_path)
                if not saved_manifest:
                    logger.warning("No manifest found during reconciliation, skipping")
                    continue

                files_to_add, doc_ids_to_remove = reconcile_indices(
                    discovered_files,
                    saved_manifest,
                    docs_path,
                )

                for doc_id in doc_ids_to_remove:
                    self.index_manager.remove_document(doc_id)

                for file_path in files_to_add:
                    self.index_manager.index_document(file_path)

                if files_to_add or doc_ids_to_remove:
                    self.index_manager.persist()
                    if self.current_manifest:
                        self.current_manifest.indexed_files = build_indexed_files_map(discovered_files, docs_path)
                        save_manifest(self.index_path, self.current_manifest)
                    logger.info(
                        f"Periodic reconciliation complete: added {len(files_to_add)}, "
                        f"removed {len(doc_ids_to_remove)}"
                    )
                else:
                    logger.debug("Periodic reconciliation: no changes needed")

                # Incrementally build vocabulary in background
                await self._update_vocabulary_incremental()

            except asyncio.CancelledError:
                logger.info("Periodic reconciliation task cancelled")
                raise
            except Exception as e:
                logger.error(f"Error during periodic reconciliation: {e}", exc_info=True)

    async def _update_vocabulary_incremental(self) -> None:
        """Update concept vocabulary incrementally in background."""
        vector = self.index_manager.vector
        pending = vector.get_pending_vocabulary_count()
        if pending == 0:
            return

        logger.debug(f"Updating vocabulary: {pending} pending terms")
        # Process in batches to avoid blocking
        total_embedded = 0
        while True:
            embedded = await asyncio.to_thread(
                vector.update_vocabulary_incremental,
                batch_size=50,
            )
            if embedded == 0:
                break
            total_embedded += embedded
            # Yield to event loop between batches
            await asyncio.sleep(0)

        if total_embedded > 0:
            logger.info(f"Vocabulary update: embedded {total_embedded} new terms")
            # Persist after vocabulary update
            await asyncio.to_thread(self.index_manager.persist)

    async def _build_initial_vocabulary(self) -> None:
        """Build concept vocabulary from scratch in background."""
        if not self.config.search.query_expansion_enabled:
            logger.info("Query expansion disabled, skipping vocabulary build")
            return

        try:
            logger.info("Building concept vocabulary in background...")
            await asyncio.to_thread(
                self.index_manager.vector.build_concept_vocabulary,
                max_terms=self.config.search.query_expansion_max_terms,
                min_frequency=self.config.search.query_expansion_min_frequency,
            )
            await asyncio.to_thread(self.index_manager.persist)
            logger.info("Concept vocabulary built and persisted")
        except asyncio.CancelledError:
            logger.info("Vocabulary building cancelled")
        except Exception as e:
            logger.error(f"Failed to build vocabulary: {e}", exc_info=True)

    def is_ready(self) -> bool:
        """Check if initialization is complete and indices are ready."""
        return self._ready_event.is_set() and self._init_error is None and self.index_manager.is_ready()

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        """Wait for initialization to complete. Call before first query."""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Index initialization timed out after {timeout}s") from None

        if self._init_error is not None:
            raise RuntimeError(f"Index initialization failed: {self._init_error}") from self._init_error

    async def stop(self) -> None:
        logger.info("Stopping ApplicationContext")

        if self._singleton_guard is not None:
            try:
                self._singleton_guard.release()
            except Exception as e:
                logger.error(f"Failed to release singleton lock: {e}")

        tasks_to_cancel: list[asyncio.Task] = []
        if self._background_index_task and not self._background_index_task.done():
            self._background_index_task.cancel()
            tasks_to_cancel.append(self._background_index_task)

        if self.reconciliation_task and not self.reconciliation_task.done():
            self.reconciliation_task.cancel()
            tasks_to_cancel.append(self.reconciliation_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        self._background_index_task = None
        self.reconciliation_task = None

        if self.watcher:
            try:
                await asyncio.wait_for(self.watcher.stop(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("FileWatcher stop timed out")

        try:
            await asyncio.to_thread(self.index_manager.persist)
        except Exception as e:
            logger.error(f"Failed to persist indices during stop: {e}")

        if self.commit_indexer:
            try:
                await asyncio.to_thread(self.commit_indexer.close)
            except Exception as e:
                logger.error(f"Failed to close commit indexer: {e}")

        if self.memory_manager:
            try:
                await asyncio.to_thread(self.memory_manager.persist)
            except Exception as e:
                logger.error(f"Failed to persist memory indices: {e}")

        logger.info("ApplicationContext stopped")

    def _index_git_commits_initial_sync(self) -> None:
        """Index all commits in discovered repositories (synchronous)."""
        if self.commit_indexer is None:
            return

        from src.git.parallel_indexer import (
            ParallelIndexingConfig,
            index_commits_parallel_sync,
        )
        from src.git.repository import discover_git_repositories, get_commits_after_timestamp

        logger.info("Starting initial git commit indexing (parallel)")

        repos = discover_git_repositories(
            Path(self.config.indexing.documents_path),
            self.config.indexing.exclude,
            self.config.indexing.exclude_hidden_dirs,
        )

        parallel_config = ParallelIndexingConfig(
            max_workers=self.config.git_indexing.parallel_workers,
            batch_size=self.config.git_indexing.batch_size,
            embed_batch_size=self.config.git_indexing.embed_batch_size,
        )

        total_indexed = 0
        for repo_path in repos:
            try:
                last_timestamp = self.commit_indexer.get_last_indexed_timestamp(str(repo_path.parent))
                commit_hashes = get_commits_after_timestamp(repo_path, last_timestamp)

                if last_timestamp is not None:
                    from datetime import datetime
                    last_indexed_dt = datetime.fromtimestamp(last_timestamp).isoformat()
                    logger.info(f"Repository {repo_path.parent}: Last indexed at {last_indexed_dt}, found {len(commit_hashes)} new commits")
                else:
                    logger.info(f"Repository {repo_path.parent}: First-time indexing, found {len(commit_hashes)} commits")

                if len(commit_hashes) == 0:
                    logger.debug(f"No new commits to index for {repo_path.parent}")
                    continue

                indexed = index_commits_parallel_sync(
                    commit_hashes,
                    repo_path,
                    self.commit_indexer,
                    parallel_config,
                    self.config.git_indexing.delta_max_lines,
                )
                total_indexed += indexed

            except Exception as e:
                logger.error(f"Failed to index repository {repo_path}: {e}", exc_info=True)

        logger.info(f"Initial git commit indexing complete: {total_indexed} commits")

    async def _index_git_commits_initial(self) -> None:
        """Index all commits in discovered repositories (async wrapper)."""
        await asyncio.to_thread(self._index_git_commits_initial_sync)

    async def _index_git_commits_initial_with_timeout(self) -> None:
        """Index git commits with timeout protection.

        Prevents git indexing from hanging indefinitely during startup.
        If timeout is reached, logs warning and continues without blocking.
        """
        try:
            await asyncio.wait_for(
                self._index_git_commits_initial(),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Git commit indexing timed out after 30s. "
                "Consider reducing repository size or increasing timeout."
            )
        except Exception as e:
            logger.error(f"Git commit indexing failed: {e}", exc_info=True)
