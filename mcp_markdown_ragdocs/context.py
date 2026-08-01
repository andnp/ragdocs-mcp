from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from searchkernel.api import (
    AsyncIndexIngestor,
    DatabaseManager,
    GraphIndex as GraphStore,
    IndexManifest,
    KeywordIndex,
    PublicIndexStateSnapshot,
    Record,
    SearchAvailability,
    VectorIndex,
    build_file_stamps,
    build_indexed_files_map,
    can_refresh_loaded_indices,
    can_serve_queries,
    derive_loaded_index_state_snapshot,
    detect_and_migrate_legacy_index,
    discover_files,
    discover_files_multi_root,
    get_parser_suffixes,
    has_incomplete_bootstrap_checkpoint,
    CURRENT_MANIFEST_SPEC_VERSION,
    load_manifest,
    save_manifest,
    should_rebuild,
    reconcile_indices,
    semantic_tier_from_progress,
    is_fully_ready as runtime_is_fully_ready,
)

from mcp_markdown_ragdocs.config import (
    Config,
    detect_project,
    load_config,
    resolve_documents_path,
    resolve_index_path,
)
from mcp_markdown_ragdocs.indexing.bootstrap_session import BootstrapSession
from mcp_markdown_ragdocs.indexing.manager import IndexManager
from mcp_markdown_ragdocs.indexing.watcher import FileWatcher
from mcp_markdown_ragdocs.indexing.watcher_lifecycle import WatcherLifecycle
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter

logger = logging.getLogger(__name__)


@dataclass
class IndexState:
    """Tracks the current state of background indexing."""

    status: Literal["uninitialized", "indexing", "partial", "ready", "failed"]
    indexed_count: int = 0
    total_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "last_error": self.last_error,
        }


@dataclass
class ApplicationContext:
    config: Config
    index_manager: IndexManager
    orchestrator: CanonicalSearchAdapter
    record_ingestor: AsyncIndexIngestor | None = None
    use_tasks: bool = False
    _watcher_lifecycle: WatcherLifecycle = field(
        default_factory=WatcherLifecycle, repr=False
    )
    git_indexing_enabled: bool = False
    index_path: Path = field(default_factory=lambda: Path(".index_data"))
    fallback_index_path: Path | None = None
    documents_roots: list[Path] = field(default_factory=list)
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
    _freshness_task: asyncio.Task | None = field(default=None, repr=False)
    _embedding_warmup_task: asyncio.Task | None = field(default=None, repr=False)
    _vocabulary_catch_up_task: asyncio.Task | None = field(default=None, repr=False)
    _loaded_index_state_version: float = field(default=0.0, repr=False)
    _is_virgin_startup: bool = field(default=False, repr=False)
    _bootstrap_session: BootstrapSession | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        project_override: str | None = None,
        enable_watcher: bool = True,
        lazy_embeddings: bool = True,
        use_tasks: bool = False,
        index_path_override: Path | None = None,
        documents_path_override: Path | None = None,
        global_runtime: bool = False,
    ) -> ApplicationContext:
        config = load_config()

        detected_project = None
        if not global_runtime:
            detected_project = detect_project(
                projects=config.projects,
                project_override=project_override,
            )

        if detected_project and project_override:
            config = load_config()

        configured_index_path = resolve_index_path(config)
        index_path = index_path_override or configured_index_path
        fallback_index_path = None
        if index_path_override is not None and configured_index_path != index_path:
            fallback_index_path = configured_index_path

        documents_roots = cls._resolve_documents_roots(
            config,
            detected_project=detected_project,
            project_override=project_override,
            documents_path_override=documents_path_override,
            global_runtime=global_runtime,
        )
        documents_root = cls._compute_common_documents_root(documents_roots)
        documents_path = str(documents_root)

        config.indexing.index_path = str(index_path)
        config.indexing.documents_path = documents_path
        config.detected_project = None if global_runtime else detected_project

        embedding_model_name = config.llm.resolved_embedding_model

        vector = cls._build_vector_store(config, embedding_model_name)
        if not lazy_embeddings:
            vector.warm_up()

        detect_and_migrate_legacy_index(index_path)

        db_manager = DatabaseManager(index_path / "index.db")
        keyword = KeywordIndex(db_manager)
        graph = GraphStore(db_manager)

        manager = IndexManager(
            config,
            vector,
            keyword,
            graph,
            documents_roots=documents_roots,
        )
        orchestrator = CanonicalSearchAdapter(
            manager,
            documents_path=Path(documents_path),
        )
        record_ingestor = AsyncIndexIngestor(manager)

        watcher_lifecycle = WatcherLifecycle.create(
            enabled=enable_watcher,
            documents_path=config.indexing.documents_path,
            documents_roots=documents_roots,
            index_manager=manager,
            cooldown=config.indexing.debounce_window_seconds,
            include_patterns=config.indexing.include,
            exclude_patterns=config.indexing.exclude,
            exclude_hidden_dirs=config.indexing.exclude_hidden_dirs,
            parser_suffixes=set(get_parser_suffixes()),
            use_tasks=use_tasks,
            task_backpressure_limit=config.indexing.task_backpressure_limit,
        )

        # Enable git commit indexing if configured and git is available
        git_indexing_enabled = False
        if config.git_indexing.enabled:
            from mcp_markdown_ragdocs.git.repository import is_git_available

            if is_git_available():
                git_indexing_enabled = True
                logger.info("Git commit indexing enabled")
            else:
                logger.warning("Git binary not found - git history search disabled")

        return cls(
            config=config,
            index_manager=manager,
            orchestrator=orchestrator,
            record_ingestor=record_ingestor,
            use_tasks=use_tasks,
            _watcher_lifecycle=watcher_lifecycle,
            git_indexing_enabled=git_indexing_enabled,
            index_path=index_path,
            fallback_index_path=fallback_index_path,
            documents_roots=documents_roots,
            db_manager=db_manager,
            current_manifest=None,
            reconciliation_task=None,
        )

    @property
    def watcher(self) -> FileWatcher | None:
        """The underlying FileWatcher, if enabled (delegates to `WatcherLifecycle`)."""
        return self._watcher_lifecycle.watcher

    @watcher.setter
    def watcher(self, value: FileWatcher | None) -> None:
        self._watcher_lifecycle = WatcherLifecycle(watcher=value)

    @staticmethod
    def _build_vector_store(config: Config, embedding_model_name: str) -> Any:
        """Build the live vector store per `config.store.backend`.

        FAISS (`VectorIndex`) is the default; `store.backend = "pgvector"`
        selects the Postgres+pgvector-backed adapter instead. Both present
        the same method surface the live index/search path calls.
        """
        if config.store.backend == "pgvector":
            from searchkernel.adapters.stores.pgvector_index import PGVectorIndex

            return PGVectorIndex(
                pg_dsn=config.store.pg_dsn,
                embedding_model_name=embedding_model_name,
                truncate_dim=config.embedding.truncate_dim,
            )

        return VectorIndex(
            embedding_model_name=embedding_model_name,
            embedding_workers=config.indexing.embedding_workers,
            torch_num_threads=config.indexing.torch_num_threads,
            truncate_dim=config.embedding.truncate_dim,
        )

    @staticmethod
    def _resolve_documents_roots(
        config: Config,
        *,
        detected_project: str | None,
        project_override: str | None,
        documents_path_override: Path | None,
        global_runtime: bool,
    ) -> list[Path]:
        if documents_path_override is not None:
            return [documents_path_override.expanduser().resolve()]

        if global_runtime:
            if config.projects:
                return [Path(project.path).resolve() for project in config.projects]

            return [Path(resolve_documents_path(config)).resolve()]

        if project_override:
            override_path = Path(project_override).expanduser()
            if override_path.exists():
                return [override_path.resolve()]

        if detected_project:
            for project in config.projects:
                if project.name == detected_project:
                    return [Path(project.path).resolve()]

        if config.projects:
            return [Path(project.path).resolve() for project in config.projects]

        return [Path(resolve_documents_path(config)).resolve()]

    @staticmethod
    def _compute_common_documents_root(documents_roots: list[Path]) -> Path:
        if not documents_roots:
            return Path.cwd().resolve()
        if len(documents_roots) == 1:
            return documents_roots[0]
        common = os.path.commonpath([str(root) for root in documents_roots])
        return Path(common).resolve()

    def _build_manifest(self) -> IndexManifest:
        return IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model=self.config.llm.embedding_model,
            chunking_config={
                "strategy": self.config.chunking.strategy,
                "min_chunk_chars": self.config.chunking.min_chunk_chars,
                "max_chunk_chars": self.config.chunking.max_chunk_chars,
                "overlap_chars": self.config.chunking.overlap_chars,
            },
        )

    def discover_files(self) -> list[str]:
        if len(self.documents_roots) <= 1:
            return discover_files(
                documents_path=self.config.indexing.documents_path,
                include_patterns=self.config.indexing.include,
                exclude_patterns=self.config.indexing.exclude,
                exclude_hidden_dirs=self.config.indexing.exclude_hidden_dirs,
            )
        return discover_files_multi_root(
            [str(root) for root in self.documents_roots],
            include_patterns=self.config.indexing.include,
            exclude_patterns=self.config.indexing.exclude,
            exclude_hidden_dirs=self.config.indexing.exclude_hidden_dirs,
        )

    def discover_git_repositories(self) -> list[Path]:
        from mcp_markdown_ragdocs.git.repository import (
            discover_git_repositories,
            discover_git_repositories_multi_root,
        )

        if len(self.documents_roots) <= 1:
            return discover_git_repositories(
                Path(self.config.indexing.documents_path),
                self.config.indexing.exclude,
                self.config.indexing.exclude_hidden_dirs,
            )

        return discover_git_repositories_multi_root(
            self.documents_roots,
            self.config.indexing.exclude,
            self.config.indexing.exclude_hidden_dirs,
        )

    def _check_and_rebuild_if_needed(self) -> bool:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._hydrate_index_path_from_fallback()
        self.current_manifest = self._build_manifest()
        saved_manifest = load_manifest(self.index_path)
        self._is_virgin_startup = saved_manifest is None
        return should_rebuild(self.current_manifest, saved_manifest)

    def _has_persisted_index_state(self, index_path: Path) -> bool:
        return any(
            candidate.exists()
            for candidate in [
                index_path / "index.manifest.json",
                index_path / "vector" / "docstore.json",
                index_path / "vector" / "faiss_index.bin",
                index_path / "graph" / "graph.json",
            ]
        )

    def _hydrate_index_path_from_fallback(self) -> None:
        if self.fallback_index_path is None:
            return
        if self.index_path == self.fallback_index_path:
            return
        if self._has_persisted_index_state(self.index_path):
            return
        if not self._has_persisted_index_state(self.fallback_index_path):
            return

        logger.info(
            "Hydrating runtime index at %s from existing persisted index at %s",
            self.index_path,
            self.fallback_index_path,
        )

        for directory_name in ["vector", "keyword", "graph"]:
            source_dir = self.fallback_index_path / directory_name
            if source_dir.exists():
                shutil.copytree(
                    source_dir,
                    self.index_path / directory_name,
                    dirs_exist_ok=True,
                )

        for file_name in [
            "index.manifest.json",
            "index.db",
            "index.db-shm",
            "index.db-wal",
            "chunk_hashes.json",
        ]:
            source_file = self.fallback_index_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, self.index_path / file_name)

    def _compute_index_state_version(self) -> float:
        candidates = [
            self.index_path / "bootstrap.checkpoint.json",
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
            except OSError as e:
                logger.debug("Failed to stat %s while computing index state version: %s", candidate, e)
                continue
        return version

    def _mark_index_state_loaded(self) -> None:
        self._loaded_index_state_version = self._compute_index_state_version()

    @staticmethod
    def _index_state_from_snapshot(
        snapshot: PublicIndexStateSnapshot,
    ) -> IndexState:
        return IndexState(
            status=snapshot.status,
            indexed_count=snapshot.indexed_count,
            total_count=snapshot.total_count,
        )

    def _refresh_index_state_from_loaded_indices(self) -> None:
        snapshot = derive_loaded_index_state_snapshot(
            total_targets=self._index_state.total_count,
            loaded_indexed_count=self.index_manager.get_document_count(),
        )
        self._index_state = self._index_state_from_snapshot(snapshot)

    async def _preload_existing_indices_for_background_bootstrap(
        self,
        *,
        rebuild_pending: bool,
    ) -> bool:
        session = self._get_bootstrap_session()
        return await session.preload_persisted_state(rebuild_pending=rebuild_pending)

    async def _enqueue_initial_git_refresh_tasks(self) -> None:
        if not self.git_indexing_enabled:
            return

        from mcp_markdown_ragdocs.indexing.tasks import submit_refresh_git_batch

        repos = await asyncio.to_thread(self.discover_git_repositories)
        if not repos:
            logger.info("No git repositories found for task-driven startup refresh")
            return

        submission = submit_refresh_git_batch([str(repo) for repo in repos])
        logger.info(
            "Enqueued %d startup git refresh task(s) for %d repositories (%d already pending)",
            submission.enqueued_count,
            len(repos),
            submission.already_pending_count,
        )

    async def _bootstrap_via_tasks(self) -> None:
        session = self._get_bootstrap_session()
        try:
            await session.run()
        finally:
            if self._bootstrap_session is session:
                self._bootstrap_session = None

    async def start(self, background_index: bool = False) -> None:
        needs_rebuild = await asyncio.to_thread(self._check_and_rebuild_if_needed)
        resume_bootstrap = False
        if background_index and self.use_tasks:
            resume_bootstrap = await asyncio.to_thread(
                has_incomplete_bootstrap_checkpoint,
                self.index_path,
            )
        startup_git_refresh_enqueued = False

        if needs_rebuild or resume_bootstrap:
            if resume_bootstrap and not needs_rebuild:
                logger.info("Resuming interrupted task bootstrap from checkpoint")
            else:
                logger.info("Index rebuild required - indexing all documents")
            if background_index:
                if self.use_tasks:
                    await self._preload_existing_indices_for_background_bootstrap(
                        rebuild_pending=needs_rebuild,
                    )
                    startup_git_refresh_enqueued = self.git_indexing_enabled
                    self._background_index_task = asyncio.create_task(
                        self._bootstrap_via_tasks()
                    )
                else:
                    self._background_index_task = asyncio.create_task(
                        self._background_index()
                    )
            else:
                self._full_index()
                self._mark_index_state_loaded()
                self._ready_event.set()
                self.schedule_embedding_model_warmup()
                self.schedule_vocabulary_catch_up()
        else:
            logger.info("Loading existing indices")
            if background_index:
                if self.use_tasks:
                    await asyncio.to_thread(self.index_manager.load)
                    self._mark_index_state_loaded()
                    self._refresh_index_state_from_loaded_indices()
                    self._ready_event.set()
                    self.schedule_embedding_model_warmup()
                    self._background_index_task = asyncio.create_task(
                        self._reconcile_existing_indices_background()
                    )
                else:
                    self._index_state = IndexState(status="indexing")
                    self._background_index_task = asyncio.create_task(
                        self._load_existing_indices_background()
                    )
            else:
                await asyncio.to_thread(self.index_manager.load)
                self._mark_index_state_loaded()
                self._index_state = IndexState(status="ready")
                self._ready_event.set()
                self.schedule_embedding_model_warmup()
                await self._startup_reconciliation()
                self.schedule_vocabulary_catch_up()

        self._watcher_lifecycle.start(self._on_watcher_overflow)

        # Index git commits after document indexing
        if self.git_indexing_enabled:
            if background_index:
                if self.use_tasks:
                    if not startup_git_refresh_enqueued:
                        asyncio.create_task(self._enqueue_initial_git_refresh_tasks())
                else:
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

    def _bootstrap_relative_path_for_file(self, file_path: str) -> str | None:
        stamps = build_file_stamps([file_path], self.documents_roots)
        if not stamps:
            return None
        return next(iter(stamps.keys()))

    def _get_bootstrap_session(self) -> BootstrapSession:
        session = getattr(self, "_bootstrap_session", None)
        if session is not None:
            return session

        session = self._create_bootstrap_session()
        self._bootstrap_session = session
        return session

    def _create_bootstrap_session(self) -> BootstrapSession:
        async def load_persisted_indices() -> None:
            await asyncio.to_thread(self.index_manager.load)
            self._mark_index_state_loaded()

        async def persist_indices() -> None:
            await asyncio.to_thread(self.index_manager.persist)
            self._mark_index_state_loaded()

        async def compute_index_state_version() -> float:
            return await asyncio.to_thread(self._compute_index_state_version)

        return BootstrapSession(
            index_path=self.index_path,
            documents_roots=self.documents_roots,
            git_refresh_enabled=self.git_indexing_enabled,
            discover_files=self.discover_files,
            discover_git_repositories=self.discover_git_repositories,
            get_bootstrap_manifest=lambda: self.current_manifest or self._build_manifest(),
            load_persisted_indices=load_persisted_indices,
            persist_indices=persist_indices,
            compute_index_state_version=compute_index_state_version,
            get_loaded_index_state_version=lambda: self._loaded_index_state_version,
            get_loaded_document_count=self.index_manager.get_document_count,
            is_queryable=self.index_manager.is_ready,
            publish_public_state=self._publish_bootstrap_public_state,
            mark_ready=self._ready_event.set,
            schedule_embedding_warmup=self.schedule_embedding_model_warmup,
            schedule_vocabulary_catch_up=self.schedule_vocabulary_catch_up,
            report_failure=self._report_bootstrap_failure,
        )

    def _publish_bootstrap_public_state(
        self,
        snapshot: PublicIndexStateSnapshot,
    ) -> None:
        self._index_state = self._index_state_from_snapshot(snapshot)

    def schedule_vocabulary_catch_up(self) -> bool:
        if self._init_error is not None:
            return False
        if not self.index_manager.is_ready():
            return False

        vector = self.index_manager.vector
        needs_catch_up = getattr(vector, "needs_vocabulary_catch_up", None)
        if not callable(needs_catch_up) or not needs_catch_up():
            return False

        current_task = getattr(self, "_vocabulary_catch_up_task", None)
        if current_task is not None and not current_task.done():
            return False

        task = asyncio.create_task(self._run_vocabulary_catch_up())
        self._vocabulary_catch_up_task = task
        task.add_done_callback(self._clear_vocabulary_catch_up_task)
        return True

    def _report_bootstrap_failure(
        self,
        error: Exception,
        indexed_count: int,
        total_count: int,
    ) -> None:
        self._index_state = IndexState(
            status="failed",
            indexed_count=indexed_count,
            total_count=total_count,
            last_error=str(error),
        )
        self._init_error = error
        self._ready_event.set()

    def _full_index(self) -> None:
        files_to_index = self.discover_files()
        docs_path = Path(self.config.indexing.documents_path)

        for file_path in files_to_index:
            self.index_manager.index_document(file_path)

        self.index_manager.persist()

        if self.current_manifest:
            self.current_manifest.indexed_files = build_indexed_files_map(
                files_to_index,
                docs_path,
                self.documents_roots,
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
                        files_to_index,
                        docs_path,
                        self.documents_roots,
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
                self.schedule_embedding_model_warmup()
                self.schedule_vocabulary_catch_up()
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
                    logger.exception(
                        f"Background indexing failed after {indexed_count}/{len(files_to_index)} files "
                        f"(exhausted {max_retries} retries)"
                    )
                    self._init_error = e
                    self._ready_event.set()  # Unblock waiters so they can see the error

    async def _load_existing_indices_background(self) -> None:
        try:
            await asyncio.to_thread(self.index_manager.load)
            self._mark_index_state_loaded()
            self.schedule_embedding_model_warmup()
            await self._startup_reconciliation()

            self.schedule_vocabulary_catch_up()

            self._index_state = IndexState(status="ready")
            self._ready_event.set()
        except Exception as e:
            logger.exception("Failed to load existing indices in background")
            self._index_state = IndexState(
                status="failed",
                last_error=str(e),
            )
            self._init_error = e
            self._ready_event.set()

    async def _reconcile_existing_indices_background(self) -> None:
        try:
            await self._startup_reconciliation()
            self.schedule_vocabulary_catch_up()
        except Exception:
            logger.exception(
                "Startup reconciliation failed after loading existing indices"
            )

    async def _startup_reconciliation(self) -> None:
        if self.use_tasks:
            await self._enqueue_reconciliation_tasks()
            return

        logger.info("Running startup reconciliation")
        docs_path = Path(self.config.indexing.documents_path)
        discovered_files = await asyncio.to_thread(self.discover_files)

        result = await asyncio.to_thread(
            self.index_manager.reconcile_indices,
            discovered_files,
            docs_path,
            self.documents_roots,
        )

        if result.added_count > 0 or result.removed_count > 0 or result.moved_count > 0:
            await asyncio.to_thread(
                self._persist_reconciliation_state,
                discovered_files,
                docs_path,
            )
            logger.info(
                f"Reconciliation complete: "
                f"added={result.added_count}, "
                f"removed={result.removed_count}, "
                f"moved={result.moved_count}, "
                f"failed={result.failed_count}"
            )
        else:
            logger.info("Reconciliation complete: no changes needed")

    def _persist_reconciliation_state(
        self,
        discovered_files: list[str],
        docs_path: Path,
    ) -> None:
        self.index_manager.persist()
        self._mark_index_state_loaded()
        if self.current_manifest:
            self.current_manifest.indexed_files = build_indexed_files_map(
                discovered_files,
                docs_path,
                self.documents_roots,
            )
            save_manifest(self.index_path, self.current_manifest)

    async def _enqueue_reconciliation_tasks(self) -> None:
        """Reconcile through the worker process without blocking query serving.

        Move detection parses and chunks every added file against every removed
        document. That work is useful for foreground reconciliation, but it can
        monopolize the daemon process during startup. Task-backed daemons leave
        move handling to the normal add/remove tasks so the worker owns all
        CPU-heavy indexing and the daemon keeps serving the loaded snapshot.
        """
        logger.info("Running task-backed startup reconciliation")
        docs_path = Path(self.config.indexing.documents_path)
        discovered_files = await asyncio.to_thread(self.discover_files)
        saved_manifest = await asyncio.to_thread(load_manifest, self.index_path)
        if saved_manifest is None:
            logger.warning("No manifest found during task-backed reconciliation")
            return

        files_to_add, doc_ids_to_remove, _ = await asyncio.to_thread(
            reconcile_indices,
            discovered_files,
            saved_manifest,
            docs_path,
            self.documents_roots,
            self.config.indexing.include,
            self.config.indexing.exclude,
            self.config.indexing.exclude_hidden_dirs,
        )
        if not files_to_add and not doc_ids_to_remove:
            logger.info("Task-backed reconciliation complete: no changes needed")
            return

        from mcp_markdown_ragdocs.indexing.tasks import (
            submit_index_batch,
            submit_remove_request_batch,
        )

        index_submission, remove_submission = await asyncio.gather(
            asyncio.to_thread(submit_index_batch, files_to_add),
            asyncio.to_thread(submit_remove_request_batch, doc_ids_to_remove),
        )
        if not index_submission.all_represented:
            logger.warning(
                "Task-backed reconciliation could not represent %d added file(s): %s",
                len(files_to_add),
                index_submission,
            )
        if not remove_submission.all_represented:
            logger.warning(
                "Task-backed reconciliation could not represent %d removed document(s): %s",
                len(doc_ids_to_remove),
                remove_submission,
            )
        logger.info(
            "Task-backed reconciliation enqueued %d file(s) and %d removal(s)",
            index_submission.enqueued_count,
            remove_submission.enqueued_count,
        )

    async def _on_watcher_overflow(self) -> None:
        """Handle inotify queue overflow detected by the file watcher.

        Triggers an immediate reconciliation pass to ensure the index state
        is consistent after dropped file-change events.
        """
        logger.info("File watcher detected inotify queue overflow, triggering reconciliation")
        if self.use_tasks:
            await self._enqueue_reconciliation_tasks()
            return

        docs_path = Path(self.config.indexing.documents_path)
        discovered_files = await asyncio.to_thread(self.discover_files)
        result = await asyncio.to_thread(
            self.index_manager.reconcile_indices,
            discovered_files,
            docs_path,
            self.documents_roots,
        )

        if result.added_count > 0 or result.removed_count > 0 or result.moved_count > 0:
            await asyncio.to_thread(
                self._persist_reconciliation_state,
                discovered_files,
                docs_path,
            )
            logger.info(
                f"Overflow reconciliation complete: "
                f"added={result.added_count}, "
                f"removed={result.removed_count}, "
                f"moved={result.moved_count}"
            )
        else:
            logger.debug("Overflow reconciliation: no changes needed")

        self._watcher_lifecycle.refresh_watches()

    async def _periodic_reconciliation(self) -> None:
        interval = self.config.indexing.reconciliation_interval_seconds

        while True:
            try:
                await asyncio.sleep(interval)
                logger.info("Starting periodic reconciliation")
                if self.use_tasks:
                    await self._enqueue_reconciliation_tasks()
                    continue

                docs_path = Path(self.config.indexing.documents_path)
                discovered_files = await asyncio.to_thread(self.discover_files)
                result = await asyncio.to_thread(
                    self.index_manager.reconcile_indices,
                    discovered_files,
                    docs_path,
                    self.documents_roots,
                )

                if (
                    result.added_count > 0
                    or result.removed_count > 0
                    or result.moved_count > 0
                ):
                    await asyncio.to_thread(
                        self._persist_reconciliation_state,
                        discovered_files,
                        docs_path,
                    )
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
                self._watcher_lifecycle.refresh_watches()

                self.schedule_vocabulary_catch_up()

            except asyncio.CancelledError:
                logger.info("Periodic reconciliation task cancelled")
                raise
            except Exception:
                logger.exception("Error during periodic reconciliation")

    async def _update_vocabulary_incremental(self) -> None:
        """Catch up concept vocabulary without blocking search readiness."""
        vector = self.index_manager.vector
        if not vector.needs_vocabulary_catch_up():
            return

        before_state = vector.get_vocabulary_state()
        total_embedded = 0
        while vector.needs_vocabulary_catch_up():
            embedded = await asyncio.to_thread(
                vector.update_vocabulary_incremental,
                max_terms=2000,
                min_frequency=3,
                batch_size=50,
            )
            if embedded == 0 and not vector.needs_vocabulary_catch_up():
                break
            if embedded == 0:
                break
            total_embedded += embedded
            await asyncio.sleep(0)

        after_state = vector.get_vocabulary_state()
        if total_embedded > 0 or before_state != after_state:
            logger.info(
                "Vocabulary catch-up status=%s embedded=%d",
                after_state["status"],
                total_embedded,
            )
            await asyncio.to_thread(self.index_manager.persist_checkpoint)
            self._mark_index_state_loaded()

    async def _run_vocabulary_catch_up(self) -> None:
        try:
            await self._update_vocabulary_incremental()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Background vocabulary catch-up failed; continuing to serve baseline search",
                exc_info=True,
            )

    def _clear_vocabulary_catch_up_task(self, task: asyncio.Task) -> None:
        if getattr(self, "_vocabulary_catch_up_task", None) is task:
            self._vocabulary_catch_up_task = None

    async def _build_initial_vocabulary(self) -> None:
        """Build concept vocabulary from scratch in background."""
        try:
            logger.info("Building concept vocabulary in background...")
            await asyncio.to_thread(
                self.index_manager.vector.build_concept_vocabulary,
                max_terms=2000,
                min_frequency=3,
            )
            await asyncio.to_thread(self.index_manager.persist_checkpoint)
            self._mark_index_state_loaded()
            logger.info("Concept vocabulary built and persisted")
        except asyncio.CancelledError:
            logger.info("Vocabulary building cancelled")
        except Exception:
            logger.exception("Failed to build vocabulary")

    def is_ready(self) -> bool:
        """Check if initialization is complete and indices are ready.

        Returns True when the active indices are queryable.

        On first-ever startup, queries stay blocked until initialization
        finishes. On later background rebuilds, queries are allowed once the
        underlying indices are queryable, even if indexing is still ongoing.
        """
        # Derive semantic tier from actual indexing progress
        # In main's synchronous architecture, indexed_count reflects documents
        # that have completed the full pipeline including embeddings
        semantic_tier = semantic_tier_from_progress(
            self._index_state.indexed_count, self._index_state.total_count
        )
        availability = SearchAvailability(
            lexical="available" if self.index_manager.is_ready() else "unavailable",
            graph="available" if self.index_manager.is_ready() else "unavailable",
            semantic_coarse=semantic_tier,
            semantic_fine=semantic_tier,
        )
        return can_serve_queries(
            init_error=self._init_error,
            ready_event_set=self._ready_event.is_set(),
            is_virgin_startup=self._is_virgin_startup,
            indices_queryable=self.index_manager.is_ready(),
            availability=availability,
        )

    def is_fully_ready(self) -> bool:
        """Check if initialization succeeded completely.

        Returns True only when all documents were indexed successfully.
        Use is_ready() if partial results are acceptable.
        """
        # Derive semantic tier from actual indexing progress
        semantic_tier = semantic_tier_from_progress(
            self._index_state.indexed_count, self._index_state.total_count
        )
        availability = SearchAvailability(
            lexical="available" if self.index_manager.is_ready() else "unavailable",
            graph="available" if self.index_manager.is_ready() else "unavailable",
            semantic_coarse=semantic_tier,
            semantic_fine=semantic_tier,
        )
        return runtime_is_fully_ready(
            init_error=self._init_error,
            ready_event_set=self._ready_event.is_set(),
            index_status=self._index_state.status,
            indices_queryable=self.index_manager.is_ready(),
            availability=availability,
        )

    def get_index_state(self) -> IndexState:
        """Get current index state for health checks."""
        return self._index_state

    async def ingest_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: str | None = None,
        failure_mode: Literal["strict", "lenient"] = "strict",
    ):
        """Ingest records through searchkernel's async batch boundary."""
        if self.record_ingestor is None:
            raise RuntimeError("record ingestion is not configured")
        return await self.record_ingestor.index_records(
            records,
            checkpoint=checkpoint,
            failure_mode=failure_mode,
        )

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        """Wait for initialization to complete. Call before first query."""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except TimeoutError:
            raise RuntimeError(
                f"Index initialization timed out after {timeout}s"
            ) from None

        if self._init_error is not None:
            raise RuntimeError(
                f"Index initialization failed: {self._init_error}"
            ) from self._init_error

    async def ensure_fresh_indices(self) -> None:
        if not can_refresh_loaded_indices(
            ready_event_set=self._ready_event.is_set(),
            init_error=self._init_error,
        ):
            return

        current_version = await asyncio.to_thread(self._compute_index_state_version)
        if current_version <= self._loaded_index_state_version:
            return

        async with self._freshness_lock:
            current_version = await asyncio.to_thread(self._compute_index_state_version)
            if current_version <= self._loaded_index_state_version:
                return

            try:
                await asyncio.to_thread(self.index_manager.load)
            except TimeoutError:
                logger.warning(
                    "Freshness reload timed out acquiring shared index lock; "
                    "continuing to serve existing in-memory indices"
                )
                return
            self._loaded_index_state_version = current_version
            self._refresh_index_state_from_loaded_indices()

    def schedule_freshness_refresh(self) -> bool:
        if not can_refresh_loaded_indices(
            ready_event_set=self._ready_event.is_set(),
            init_error=self._init_error,
        ):
            return False

        current_task = getattr(self, "_freshness_task", None)
        if current_task is not None and not current_task.done():
            return False

        task = asyncio.create_task(self._run_freshness_refresh())
        self._freshness_task = task
        task.add_done_callback(self._clear_freshness_task)
        return True

    async def _run_freshness_refresh(self) -> None:
        try:
            await self.ensure_fresh_indices()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Background freshness refresh failed; continuing to serve existing in-memory indices",
                exc_info=True,
            )

    def _clear_freshness_task(self, task: asyncio.Task) -> None:
        if getattr(self, "_freshness_task", None) is task:
            self._freshness_task = None

    def schedule_embedding_model_warmup(self) -> bool:
        if self._init_error is not None:
            return False
        if not self.index_manager.is_ready():
            return False
        if self.index_manager.vector.model_ready():
            return False

        current_task = getattr(self, "_embedding_warmup_task", None)
        if current_task is not None and not current_task.done():
            return False

        task = asyncio.create_task(self._run_embedding_model_warmup())
        self._embedding_warmup_task = task
        task.add_done_callback(self._clear_embedding_warmup_task)
        return True

    async def _run_embedding_model_warmup(self) -> None:
        try:
            await asyncio.to_thread(self.index_manager.vector.warm_up)
            logger.info("Background embedding model warmup complete")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Background embedding warmup failed; semantic search will lazy-load on demand",
                exc_info=True,
            )

    def _clear_embedding_warmup_task(self, task: asyncio.Task) -> None:
        if getattr(self, "_embedding_warmup_task", None) is task:
            self._embedding_warmup_task = None

    async def stop(self) -> None:
        logger.info("Stopping ApplicationContext")

        tasks_to_cancel: list[asyncio.Task] = []
        if self._background_index_task and not self._background_index_task.done():
            self._background_index_task.cancel()
            tasks_to_cancel.append(self._background_index_task)

        if self.reconciliation_task and not self.reconciliation_task.done():
            self.reconciliation_task.cancel()
            tasks_to_cancel.append(self.reconciliation_task)

        freshness_task = getattr(self, "_freshness_task", None)
        if freshness_task and not freshness_task.done():
            freshness_task.cancel()
            tasks_to_cancel.append(freshness_task)

        embedding_warmup_task = getattr(self, "_embedding_warmup_task", None)
        if embedding_warmup_task and not embedding_warmup_task.done():
            embedding_warmup_task.cancel()
            tasks_to_cancel.append(embedding_warmup_task)

        vocabulary_catch_up_task = getattr(self, "_vocabulary_catch_up_task", None)
        if vocabulary_catch_up_task and not vocabulary_catch_up_task.done():
            vocabulary_catch_up_task.cancel()
            tasks_to_cancel.append(vocabulary_catch_up_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        self._background_index_task = None
        self.reconciliation_task = None
        self._freshness_task = None
        self._embedding_warmup_task = None
        self._vocabulary_catch_up_task = None
        self._bootstrap_session = None

        await self._watcher_lifecycle.stop()

        try:
            await asyncio.to_thread(self.index_manager.persist)
            self._mark_index_state_loaded()
        except Exception as e:  # noqa: BLE001 -- shutdown boundary; must not crash the process
            logger.error(f"Failed to persist indices during stop: {e}")

        logger.info("ApplicationContext stopped")

    def _index_git_commits_initial_sync(self) -> None:
        """Index all commits in discovered repositories (synchronous)."""
        if not self.git_indexing_enabled:
            return

        logger.info("Starting initial git commit indexing")
        repos = self.discover_git_repositories()
        self._ingest_git_records_into_kernel_index(repos)
        logger.info("Initial git commit indexing complete")

    def get_total_git_commits_indexed(self) -> int:
        """Count git commits discoverable in the live index (for status/stats display)."""
        if not self.git_indexing_enabled:
            return 0
        return self.index_manager.vector.count_documents_by_source_kind("git_commit")

    def _ingest_git_records_into_kernel_index(self, repos: list[Path]) -> None:
        """Ingest discovered git commits into the live IndexManager as Records.

        Commits land in the same vector/keyword/graph store as documents and
        become discoverable via SearchOrchestrator.query(source_filter=["git_commit"]).
        """
        from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource

        for repo_path in repos:
            try:
                source = GitContentSource(repo_path)
                for record in source.iter_records():
                    self.index_manager.index_record(record)
            except Exception:
                logger.exception(
                    f"Failed to ingest git records for {repo_path} into kernel index"
                )

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
        except TimeoutError:
            logger.warning(
                "Git commit indexing timed out after 30s. "
                "Consider reducing repository size or increasing timeout."
            )
        except Exception:
            logger.exception("Git commit indexing failed")
