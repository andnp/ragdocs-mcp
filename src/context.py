from __future__ import annotations

import asyncio
import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Config, load_config, detect_project, resolve_index_path, resolve_documents_path
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

logger = logging.getLogger(__name__)


@dataclass
class ApplicationContext:
    config: Config
    index_manager: IndexManager
    orchestrator: SearchOrchestrator
    watcher: FileWatcher | None = None
    commit_indexer: CommitIndexer | None = None
    index_path: Path = field(default_factory=lambda: Path(".index_data"))
    current_manifest: IndexManifest | None = None
    reconciliation_task: asyncio.Task | None = field(default=None, repr=False)
    _background_index_task: asyncio.Task | None = field(default=None, repr=False)
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _init_error: Exception | None = field(default=None, repr=False)

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

        return cls(
            config=config,
            index_manager=manager,
            orchestrator=orchestrator,
            watcher=watcher,
            commit_indexer=commit_indexer,
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
                "strategy": self.config.chunking.strategy,
                "min_chunk_chars": self.config.chunking.min_chunk_chars,
                "max_chunk_chars": self.config.chunking.max_chunk_chars,
                "overlap_chars": self.config.chunking.overlap_chars,
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
                asyncio.create_task(self._index_git_commits_initial())
            else:
                self._index_git_commits_initial_sync()

        if self.config.indexing.reconciliation_interval_seconds > 0:
            self.reconciliation_task = asyncio.create_task(self._periodic_reconciliation())
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

        logger.info("ApplicationContext stopped")

    def _index_git_commits_initial_sync(self) -> None:
        """Index all commits in discovered repositories (synchronous)."""
        if self.commit_indexer is None:
            return

        from src.git.repository import discover_git_repositories, get_commits_after_timestamp
        from src.git.commit_parser import parse_commit, build_commit_document

        logger.info("Starting initial git commit indexing")

        repos = discover_git_repositories(
            Path(self.config.indexing.documents_path),
            self.config.indexing.exclude,
            self.config.indexing.exclude_hidden_dirs,
        )

        total_indexed = 0
        for repo_path in repos:
            try:
                # Get last indexed timestamp for this repo
                last_timestamp = self.commit_indexer.get_last_indexed_timestamp(str(repo_path))

                # Get new commits
                commit_hashes = get_commits_after_timestamp(repo_path, last_timestamp)

                logger.info(f"Indexing {len(commit_hashes)} commits from {repo_path.parent}")

                # Batch process
                for i in range(0, len(commit_hashes), self.config.git_indexing.batch_size):
                    batch = commit_hashes[i:i + self.config.git_indexing.batch_size]

                    for hash in batch:
                        try:
                            commit = parse_commit(
                                repo_path,
                                hash,
                                self.config.git_indexing.delta_max_lines,
                            )
                            doc = build_commit_document(commit)

                            self.commit_indexer.add_commit(
                                hash=commit.hash,
                                timestamp=commit.timestamp,
                                author=commit.author,
                                committer=commit.committer,
                                title=commit.title,
                                message=commit.message,
                                files_changed=commit.files_changed,
                                delta_truncated=commit.delta_truncated,
                                commit_document=doc,
                                repo_path=str(repo_path),
                            )
                            total_indexed += 1
                        except Exception as e:
                            logger.error(f"Failed to index commit {hash}: {e}")

            except Exception as e:
                logger.error(f"Failed to index repository {repo_path}: {e}")

        logger.info(f"Initial git commit indexing complete: {total_indexed} commits")

    async def _index_git_commits_initial(self) -> None:
        """Index all commits in discovered repositories (async wrapper)."""
        await asyncio.to_thread(self._index_git_commits_initial_sync)
