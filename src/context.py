from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.config import (
    Config,
    load_config,
    detect_project,
    resolve_index_path,
    resolve_documents_path,
)
from src.git.commit_indexer import CommitIndexer
from src.indexing.discovery import (
    discover_files as _discover_files,
    get_parser_suffixes,
)
from src.indexing.manager import IndexManager
from src.indexing.manifest import (
    IndexManifest,
    load_manifest,
    save_manifest,
    should_rebuild,
)
from src.indexing.reconciler import build_indexed_files_map
from src.indexing.watcher import FileWatcher
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import SearchOrchestrator
from src.storage.db import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class IndexState:
    """Tracks the current state of background indexing."""

    status: Literal["uninitialized", "indexing", "partial", "ready", "failed"]
    indexed_count: int = 0
    total_count: int = 0
    last_error: str | None = None


@dataclass
class ApplicationContext:
    config: Config
    index_manager: IndexManager
    orchestrator: SearchOrchestrator
    watcher: FileWatcher | None = None
    commit_indexer: CommitIndexer | None = None
    index_path: Path = field(default_factory=lambda: Path(".index_data"))
    db_manager: DatabaseManager | None = None
    current_manifest: IndexManifest | None = None
    reconciliation_task: asyncio.Task | None = field(default=None, repr=False)
    _background_index_task: asyncio.Task | None = field(default=None, repr=False)
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _init_error: Exception | None = field(default=None, repr=False)
    _index_state: IndexState = field(
        default_factory=lambda: IndexState(status="uninitialized"),
        repr=False,
    )
    _freshness_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _loaded_index_state_version: float = field(default=0.0, repr=False)

    @classmethod
    def create(
        cls,
        project_override: str | None = None,
        enable_watcher: bool = True,
        lazy_embeddings: bool = True,
        use_tasks: bool = False,
        index_path_override: Path | None = None,
    ) -> ApplicationContext:
        config = load_config()

        detected_project = detect_project(
            projects=config.projects,
            project_override=project_override,
        )

        if detected_project and project_override:
            config = load_config()

        index_path = index_path_override or resolve_index_path(config, detected_project)
        documents_path = resolve_documents_path(
            config, detected_project, config.projects
        )

        config.indexing.index_path = str(index_path)
        config.indexing.documents_path = documents_path
        config.detected_project = detected_project

        embedding_model_name = config.llm.resolved_embedding_model

        if lazy_embeddings:
            vector = VectorIndex(
                embedding_model_name=embedding_model_name,
                embedding_workers=config.indexing.embedding_workers,
            )
        else:
            vector = VectorIndex(
                embedding_model_name=embedding_model_name,
                embedding_workers=config.indexing.embedding_workers,
            )
            vector.warm_up()

        from src.indexing.migration import detect_and_migrate_legacy_index

        detect_and_migrate_legacy_index(index_path)

        db_manager = DatabaseManager(index_path / "index.db")
        keyword = KeywordIndex(db_manager)
        graph = GraphStore(db_manager)

        manager = IndexManager(config, vector, keyword, graph)
        orchestrator = SearchOrchestrator(
            vector,
            keyword,
            graph,
            config,
            manager,
            documents_path=Path(documents_path),
        )

        watcher = None
        if enable_watcher:
            watcher = FileWatcher(
                documents_path=config.indexing.documents_path,
                index_manager=manager,
                include_patterns=config.indexing.include,
                exclude_patterns=config.indexing.exclude,
                exclude_hidden_dirs=config.indexing.exclude_hidden_dirs,
                parser_suffixes=get_parser_suffixes(),
                use_tasks=use_tasks,
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

        return cls(
            config=config,
            index_manager=manager,
            orchestrator=orchestrator,
            watcher=watcher,
            commit_indexer=commit_indexer,
            index_path=index_path,
            db_manager=db_manager,
            current_manifest=None,
            reconciliation_task=None,
        )

    def _build_manifest(self) -> IndexManifest:
        return IndexManifest(
            spec_version="1.0.0",
            embedding_model=self.config.llm.embedding_model,
            chunking_config={
                "strategy": self.config.chunking.strategy,
                "min_chunk_chars": self.config.chunking.min_chunk_chars,
                "max_chunk_chars": self.config.chunking.max_chunk_chars,
                "overlap_chars": self.config.chunking.overlap_chars,
            },
        )

    def discover_files(self) -> list[str]:
        return _discover_files(
            documents_path=self.config.indexing.documents_path,
            include_patterns=self.config.indexing.include,
            exclude_patterns=self.config.indexing.exclude,
            exclude_hidden_dirs=self.config.indexing.exclude_hidden_dirs,
        )

    def _check_and_rebuild_if_needed(self) -> bool:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.current_manifest = self._build_manifest()
        saved_manifest = load_manifest(self.index_path)
        return should_rebuild(self.current_manifest, saved_manifest)

    def _compute_index_state_version(self) -> float:
        candidates = [
            self.index_path / "index.manifest.json",
            self.index_path / "index.db",
            self.index_path / "index.db-wal",
            self.index_path / "vector" / "docstore.json",
            self.index_path / "vector" / "faiss_index.bin",
            self.index_path / "graph" / "graph.json",
        ]
        version = 0.0
        for candidate in candidates:
            try:
                if candidate.exists():
                    version = max(version, candidate.stat().st_mtime)
            except OSError:
                continue
        return version

    def _mark_index_state_loaded(self) -> None:
        self._loaded_index_state_version = self._compute_index_state_version()

    async def start(self, background_index: bool = False) -> None:
        needs_rebuild = await asyncio.to_thread(self._check_and_rebuild_if_needed)

        if needs_rebuild:
            logger.info("Index rebuild required - indexing all documents")
            if background_index:
                self._background_index_task = asyncio.create_task(
                    self._background_index()
                )
            else:
                self._full_index()
                self._mark_index_state_loaded()
                self._ready_event.set()
                # Build vocabulary after full index (in background)
                asyncio.create_task(self._build_initial_vocabulary())
        else:
            logger.info("Loading existing indices")
            if background_index:
                self._index_state = IndexState(status="indexing")
                self._background_index_task = asyncio.create_task(
                    self._load_existing_indices_background()
                )
            else:
                await asyncio.to_thread(self.index_manager.load)
                self._mark_index_state_loaded()
                self._index_state = IndexState(status="ready")
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
            self.reconciliation_task = asyncio.create_task(
                self._periodic_reconciliation()
            )
            logger.info(
                f"Periodic reconciliation enabled (interval: "
                f"{self.config.indexing.reconciliation_interval_seconds}s)"
            )

    def _full_index(self) -> None:
        files_to_index = self.discover_files()
        docs_path = Path(self.config.indexing.documents_path)

        for file_path in files_to_index:
            self.index_manager.index_document(file_path)

        self.index_manager.persist()

        if self.current_manifest:
            self.current_manifest.indexed_files = build_indexed_files_map(
                files_to_index, docs_path
            )
            save_manifest(self.index_path, self.current_manifest)

        logger.info(
            f"Initial indexing complete: {len(files_to_index)} documents indexed"
        )

    async def _background_index(self) -> None:
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            files_to_index: list[str] = []
            indexed_count = 0
            try:
                logger.info(
                    f"Starting background indexing (attempt {attempt + 1}/{max_retries})"
                )
                files_to_index = await asyncio.to_thread(self.discover_files)
                docs_path = Path(self.config.indexing.documents_path)

                self._index_state = IndexState(
                    status="indexing",
                    indexed_count=0,
                    total_count=len(files_to_index),
                )

                for file_path in files_to_index:
                    await asyncio.to_thread(
                        self.index_manager.index_document, file_path
                    )
                    indexed_count += 1
                    self._index_state.indexed_count = indexed_count

                await asyncio.to_thread(self.index_manager.persist)
                self._mark_index_state_loaded()

                if self.current_manifest:
                    self.current_manifest.indexed_files = build_indexed_files_map(
                        files_to_index, docs_path
                    )
                    await asyncio.to_thread(
                        save_manifest, self.index_path, self.current_manifest
                    )

                logger.info(
                    f"Background indexing complete: {len(files_to_index)} documents indexed"
                )
                self._index_state = IndexState(
                    status="ready",
                    indexed_count=indexed_count,
                    total_count=len(files_to_index),
                )
                self._ready_event.set()
                return  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)
                self._index_state = IndexState(
                    status="partial" if indexed_count > 0 else "failed",
                    indexed_count=indexed_count,
                    total_count=len(files_to_index),
                    last_error=error_msg,
                )

                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Background indexing failed after {indexed_count}/{len(files_to_index)} files "
                        f"(attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s",
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Background indexing failed after {indexed_count}/{len(files_to_index)} files "
                        f"(exhausted {max_retries} retries): {e}",
                        exc_info=True,
                    )
                    self._init_error = e
                    self._ready_event.set()  # Unblock waiters so they can see the error

    async def _load_existing_indices_background(self) -> None:
        try:
            await asyncio.to_thread(self.index_manager.load)
            self._mark_index_state_loaded()
            await self._startup_reconciliation()

            if not self.index_manager.vector._concept_vocabulary:
                asyncio.create_task(self._build_initial_vocabulary())

            self._index_state = IndexState(status="ready")
            self._ready_event.set()
        except Exception as e:
            self._index_state = IndexState(
                status="failed",
                last_error=str(e),
            )
            self._init_error = e
            self._ready_event.set()

    async def _startup_reconciliation(self) -> None:
        logger.info("Running startup reconciliation")
        docs_path = Path(self.config.indexing.documents_path)
        discovered_files = await asyncio.to_thread(self.discover_files)

        result = await asyncio.to_thread(
            self.index_manager.reconcile_indices,
            discovered_files,
            docs_path,
        )

        if result.added_count > 0 or result.removed_count > 0 or result.moved_count > 0:
            self.index_manager.persist()
            self._mark_index_state_loaded()
            if self.current_manifest:
                self.current_manifest.indexed_files = build_indexed_files_map(
                    discovered_files, docs_path
                )
                save_manifest(self.index_path, self.current_manifest)
            logger.info(
                f"Reconciliation complete: "
                f"added={result.added_count}, "
                f"removed={result.removed_count}, "
                f"moved={result.moved_count}, "
                f"failed={result.failed_count}"
            )
        else:
            logger.info("Reconciliation complete: no changes needed")

    async def _periodic_reconciliation(self) -> None:
        interval = self.config.indexing.reconciliation_interval_seconds

        while True:
            try:
                await asyncio.sleep(interval)
                logger.info("Starting periodic reconciliation")

                docs_path = Path(self.config.indexing.documents_path)
                discovered_files = await asyncio.to_thread(self.discover_files)

                result = await asyncio.to_thread(
                    self.index_manager.reconcile_indices,
                    discovered_files,
                    docs_path,
                )

                if (
                    result.added_count > 0
                    or result.removed_count > 0
                    or result.moved_count > 0
                ):
                    self.index_manager.persist()
                    self._mark_index_state_loaded()
                    if self.current_manifest:
                        self.current_manifest.indexed_files = build_indexed_files_map(
                            discovered_files, docs_path
                        )
                        save_manifest(self.index_path, self.current_manifest)
                    logger.info(
                        f"Periodic reconciliation: "
                        f"added={result.added_count}, "
                        f"removed={result.removed_count}, "
                        f"moved={result.moved_count}, "
                        f"failed={result.failed_count}"
                    )
                else:
                    logger.debug("Periodic reconciliation: no changes needed")

                # Register inotify watches for any directories that appeared since startup
                if self.watcher:
                    self.watcher.refresh_watches()

                # Incrementally build vocabulary in background
                await self._update_vocabulary_incremental()

            except asyncio.CancelledError:
                logger.info("Periodic reconciliation task cancelled")
                raise
            except Exception as e:
                logger.error(
                    f"Error during periodic reconciliation: {e}", exc_info=True
                )

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
            self._mark_index_state_loaded()

    async def _build_initial_vocabulary(self) -> None:
        """Build concept vocabulary from scratch in background."""
        try:
            logger.info("Building concept vocabulary in background...")
            await asyncio.to_thread(
                self.index_manager.vector.build_concept_vocabulary,
                max_terms=2000,
                min_frequency=3,
            )
            await asyncio.to_thread(self.index_manager.persist)
            self._mark_index_state_loaded()
            logger.info("Concept vocabulary built and persisted")
        except asyncio.CancelledError:
            logger.info("Vocabulary building cancelled")
        except Exception as e:
            logger.error(f"Failed to build vocabulary: {e}", exc_info=True)

    def is_ready(self) -> bool:
        """Check if initialization is complete and indices are ready.

        Returns True for both 'ready' and 'partial' states, allowing
        queries on partially indexed data.
        """
        if not self._ready_event.is_set():
            return False
        if self._init_error is not None:
            return False
        if self._index_state.status in ("ready", "partial"):
            return self.index_manager.is_ready()
        return self.index_manager.is_ready()

    def is_fully_ready(self) -> bool:
        """Check if initialization succeeded completely.

        Returns True only when all documents were indexed successfully.
        Use is_ready() if partial results are acceptable.
        """
        return (
            self._ready_event.is_set()
            and self._init_error is None
            and self._index_state.status == "ready"
            and self.index_manager.is_ready()
        )

    def get_index_state(self) -> IndexState:
        """Get current index state for health checks."""
        return self._index_state

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        """Wait for initialization to complete. Call before first query."""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Index initialization timed out after {timeout}s"
            ) from None

        if self._init_error is not None:
            raise RuntimeError(
                f"Index initialization failed: {self._init_error}"
            ) from self._init_error

    async def ensure_fresh_indices(self) -> None:
        if not self._ready_event.is_set() or self._init_error is not None:
            return

        current_version = await asyncio.to_thread(self._compute_index_state_version)
        if current_version <= self._loaded_index_state_version:
            return

        async with self._freshness_lock:
            current_version = await asyncio.to_thread(self._compute_index_state_version)
            if current_version <= self._loaded_index_state_version:
                return

            await asyncio.to_thread(self.index_manager.load)
            self._loaded_index_state_version = current_version

    async def stop(self) -> None:
        logger.info("Stopping ApplicationContext")

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
            self._mark_index_state_loaded()
        except Exception as e:
            logger.error(f"Failed to persist indices during stop: {e}")

        if self.commit_indexer:
            try:
                await asyncio.to_thread(self.commit_indexer.close)
            except Exception as e:
                logger.error(f"Failed to close commit indexer: {e}")

        logger.info("ApplicationContext stopped")

    def _index_git_commits_initial_sync(self) -> None:
        """Index all commits in discovered repositories (synchronous)."""
        if self.commit_indexer is None:
            return

        from src.git.parallel_indexer import (
            ParallelIndexingConfig,
            index_commits_parallel_sync,
        )
        from src.git.repository import (
            discover_git_repositories,
            get_commits_after_timestamp,
        )

        logger.info("Starting initial git commit indexing (parallel)")

        repos = discover_git_repositories(
            Path(self.config.indexing.documents_path),
            self.config.indexing.exclude,
            self.config.indexing.exclude_hidden_dirs,
        )

        parallel_config = ParallelIndexingConfig()

        total_indexed = 0
        for repo_path in repos:
            try:
                last_timestamp = self.commit_indexer.get_last_indexed_timestamp(
                    str(repo_path.parent)
                )
                commit_hashes = get_commits_after_timestamp(repo_path, last_timestamp)

                if last_timestamp is not None:
                    from datetime import datetime

                    last_indexed_dt = datetime.fromtimestamp(last_timestamp).isoformat()
                    logger.info(
                        f"Repository {repo_path.parent}: Last indexed at {last_indexed_dt}, found {len(commit_hashes)} new commits"
                    )
                else:
                    logger.info(
                        f"Repository {repo_path.parent}: First-time indexing, found {len(commit_hashes)} commits"
                    )

                if len(commit_hashes) == 0:
                    logger.debug(f"No new commits to index for {repo_path.parent}")
                    continue

                indexed = index_commits_parallel_sync(
                    commit_hashes,
                    repo_path,
                    self.commit_indexer,
                    parallel_config,
                    200,
                )
                total_indexed += indexed

            except Exception as e:
                logger.error(
                    f"Failed to index repository {repo_path}: {e}", exc_info=True
                )

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
            await asyncio.wait_for(self._index_git_commits_initial(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Git commit indexing timed out after 30s. "
                "Consider reducing repository size or increasing timeout."
            )
        except Exception as e:
            logger.error(f"Git commit indexing failed: {e}", exc_info=True)
