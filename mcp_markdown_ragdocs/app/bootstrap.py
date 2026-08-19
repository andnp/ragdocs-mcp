"""Bootstrap and loaded-index freshness coordination for the application."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from searchkernel.api import (
    IndexManifest,
    SearchAvailability,
    can_refresh_loaded_indices,
    can_serve_queries,
    is_fully_ready as runtime_is_fully_ready,
    semantic_tier_from_progress,
)

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing.watcher_lifecycle import WatcherLifecycle

logger = logging.getLogger(__name__)


@dataclass
class IndexState:
    """Tracks the current state of background indexing."""

    status: Literal["uninitialized", "indexing", "partial", "ready", "failed"]
    indexed_count: int = 0
    total_count: int = 0
    last_error: str | None = None
    availability: SearchAvailability | None = None

    def to_dict(self) -> dict[str, object]:
        status = self.status
        if status == "ready" and self.availability is not None:
            if not self.availability.is_fully_ready():
                status = "partial"
        payload: dict[str, object] = {
            "status": status,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "last_error": self.last_error,
        }
        if self.availability is not None:
            payload["availability"] = self.availability.to_dict()
        return payload


class BootstrapHost(Protocol):
    """Context operations used by bootstrap coordination."""

    config: Config
    index_manager: Any
    index_path: Path
    current_manifest: IndexManifest | None
    use_tasks: bool
    git_indexing_enabled: bool
    _watcher_lifecycle: WatcherLifecycle
    reconciliation_task: asyncio.Task | None
    _background_index_task: asyncio.Task | None
    _ready_event: asyncio.Event
    _init_error: Exception | None
    _index_state: IndexState
    _availability: SearchAvailability | None
    _freshness_lock: asyncio.Lock
    _freshness_task: asyncio.Task | None
    _loaded_index_state_version: float
    _is_virgin_startup: bool
    _active_model_identity: tuple[str, int] | None

    def _check_and_rebuild_if_needed(self) -> bool: ...
    async def _preload_existing_indices_for_background_bootstrap(
        self,
        *,
        rebuild_pending: bool,
    ) -> bool: ...
    async def _bootstrap_via_tasks(self) -> None: ...
    def _full_index(self) -> None: ...
    async def _background_index(self) -> None: ...
    async def _load_existing_indices_background(self) -> None: ...
    async def _reconcile_existing_indices_background(self) -> None: ...
    async def _startup_reconciliation(self) -> None: ...
    async def _enqueue_initial_git_refresh_tasks(self) -> None: ...
    def _index_git_commits_initial_sync(self) -> None: ...
    async def _index_git_commits_initial_with_timeout(self) -> None: ...
    async def _periodic_reconciliation(self) -> None: ...
    async def _on_watcher_overflow(self) -> None: ...
    def _mark_index_state_loaded(self) -> None: ...
    def _publish_bootstrap_availability(
        self,
        availability: SearchAvailability,
    ) -> None: ...
    def _refresh_index_state_from_loaded_indices(self) -> None: ...
    def _compute_index_state_version(self) -> float: ...
    def _build_vector_store(self, config: Config, embedding_model_name: str) -> Any: ...
    def schedule_embedding_model_warmup(self) -> bool: ...
    async def _run_freshness_refresh(self) -> None: ...
    def _clear_freshness_task(self, task: asyncio.Task) -> None: ...


@dataclass
class BootstrapCoordinator:
    """Coordinate startup, readiness, and freshness without owning indices."""

    host: BootstrapHost
    _logger: logging.Logger = field(default=logger, repr=False)

    async def start(self, background_index: bool = False) -> None:
        host = self.host
        needs_rebuild = await asyncio.to_thread(host._check_and_rebuild_if_needed)
        resume_bootstrap = False
        if background_index and host.use_tasks:
            from searchkernel.api import has_incomplete_bootstrap_checkpoint

            resume_bootstrap = await asyncio.to_thread(
                has_incomplete_bootstrap_checkpoint,
                host.index_path,
            )
        startup_git_refresh_enqueued = False

        if needs_rebuild or resume_bootstrap:
            if resume_bootstrap and not needs_rebuild:
                self._logger.info("Resuming interrupted task bootstrap from checkpoint")
            else:
                self._logger.info("Index rebuild required - indexing all documents")
            if background_index:
                if host.use_tasks:
                    await host._preload_existing_indices_for_background_bootstrap(
                        rebuild_pending=needs_rebuild,
                    )
                    startup_git_refresh_enqueued = host.git_indexing_enabled
                    host._background_index_task = asyncio.create_task(
                        host._bootstrap_via_tasks()
                    )
                else:
                    host._background_index_task = asyncio.create_task(
                        host._background_index()
                    )
            else:
                host._full_index()
                host._mark_index_state_loaded()
                host._publish_bootstrap_availability(_fully_available())
                host._ready_event.set()
                host.schedule_embedding_model_warmup()
        else:
            self._logger.info("Loading existing indices")
            if background_index:
                if host.use_tasks:
                    await asyncio.to_thread(host.index_manager.load)
                    host._mark_index_state_loaded()
                    host._refresh_index_state_from_loaded_indices()
                    host._publish_bootstrap_availability(_fully_available())
                    host._ready_event.set()
                    host.schedule_embedding_model_warmup()
                    host._background_index_task = asyncio.create_task(
                        host._reconcile_existing_indices_background()
                    )
                else:
                    host._index_state = IndexState(status="indexing")
                    host._background_index_task = asyncio.create_task(
                        host._load_existing_indices_background()
                    )
            else:
                await asyncio.to_thread(host.index_manager.load)
                host._mark_index_state_loaded()
                host._index_state = IndexState(status="ready")
                host._publish_bootstrap_availability(_fully_available())
                host._ready_event.set()
                host.schedule_embedding_model_warmup()
                await host._startup_reconciliation()

        host._watcher_lifecycle.start(host._on_watcher_overflow)

        if host.git_indexing_enabled:
            if background_index:
                if host.use_tasks:
                    if not startup_git_refresh_enqueued:
                        asyncio.create_task(host._enqueue_initial_git_refresh_tasks())
                else:
                    asyncio.create_task(
                        host._index_git_commits_initial_with_timeout()
                    )
            else:
                host._index_git_commits_initial_sync()

        if host.config.indexing.reconciliation_interval_seconds > 0:
            host.reconciliation_task = asyncio.create_task(
                host._periodic_reconciliation()
            )
            self._logger.info(
                "Periodic reconciliation enabled (interval: %ss)",
                host.config.indexing.reconciliation_interval_seconds,
            )

    def is_ready(self) -> bool:
        host = self.host
        semantic_tier = semantic_tier_from_progress(
            host._index_state.indexed_count,
            host._index_state.total_count,
        )
        availability = getattr(host, "_availability", None) or SearchAvailability(
            lexical="available" if host.index_manager.is_ready() else "unavailable",
            graph="available" if host.index_manager.is_ready() else "unavailable",
            semantic_coarse=semantic_tier,
            semantic_fine=semantic_tier,
        )
        return can_serve_queries(
            init_error=host._init_error,
            ready_event_set=host._ready_event.is_set(),
            is_virgin_startup=host._is_virgin_startup,
            indices_queryable=host.index_manager.is_ready(),
            availability=availability,
        )

    def is_fully_ready(self) -> bool:
        host = self.host
        semantic_tier = semantic_tier_from_progress(
            host._index_state.indexed_count,
            host._index_state.total_count,
        )
        availability = getattr(host, "_availability", None) or SearchAvailability(
            lexical="available" if host.index_manager.is_ready() else "unavailable",
            graph="available" if host.index_manager.is_ready() else "unavailable",
            semantic_coarse=semantic_tier,
            semantic_fine=semantic_tier,
        )
        return runtime_is_fully_ready(
            init_error=host._init_error,
            ready_event_set=host._ready_event.is_set(),
            index_status=host._index_state.status,
            indices_queryable=host.index_manager.is_ready(),
            availability=availability,
        )

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        host = self.host
        try:
            await asyncio.wait_for(host._ready_event.wait(), timeout=timeout)
        except TimeoutError:
            raise RuntimeError(
                f"Index initialization timed out after {timeout}s"
            ) from None

        if host._init_error is not None:
            raise RuntimeError(
                f"Index initialization failed: {host._init_error}"
            ) from host._init_error

    async def ensure_fresh_indices(self) -> None:
        host = self.host
        if not can_refresh_loaded_indices(
            ready_event_set=host._ready_event.is_set(),
            init_error=host._init_error,
        ):
            return

        current_version = await asyncio.to_thread(host._compute_index_state_version)
        if current_version <= host._loaded_index_state_version:
            return

        async with host._freshness_lock:
            await self.refresh_active_model_from_manifest()
            current_version = await asyncio.to_thread(
                host._compute_index_state_version
            )
            if current_version <= host._loaded_index_state_version:
                return

            try:
                await asyncio.to_thread(host.index_manager.load)
            except TimeoutError:
                self._logger.warning(
                    "Freshness reload timed out acquiring shared index lock; "
                    "continuing to serve existing in-memory indices"
                )
                return
            host._loaded_index_state_version = current_version
            host._refresh_index_state_from_loaded_indices()

    async def refresh_active_model_from_manifest(self) -> None:
        host = self.host
        if hasattr(host.index_manager, "kernel"):
            return
        from searchkernel.api import load_manifest

        manifest = await asyncio.to_thread(load_manifest, host.index_path)
        if manifest is None or manifest.active_model is None:
            return

        namespace = manifest.active_model.namespace
        if namespace.identity == host._active_model_identity:
            return
        if host.config.store.backend != "pgvector":
            raise RuntimeError(
                "active model metadata changed for an unsupported legacy index"
            )

        host.config.llm.embedding_model = namespace.model_name
        host.config.embedding.truncate_dim = namespace.dim
        vector = await asyncio.to_thread(
            host._build_vector_store,
            host.config,
            namespace.model_name,
        )
        host.index_manager.replace_vector_store(vector)
        host._active_model_identity = namespace.identity
        host._loaded_index_state_version = host._compute_index_state_version()

    def schedule_freshness_refresh(self) -> bool:
        host = self.host
        if not can_refresh_loaded_indices(
            ready_event_set=host._ready_event.is_set(),
            init_error=host._init_error,
        ):
            return False

        current_task = host._freshness_task
        if current_task is not None and not current_task.done():
            return False

        task = asyncio.create_task(host._run_freshness_refresh())
        host._freshness_task = task
        task.add_done_callback(host._clear_freshness_task)
        return True

    async def run_freshness_refresh(self) -> None:
        try:
            await self.ensure_fresh_indices()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.warning(
                "Background freshness refresh failed; continuing to serve existing in-memory indices",
                exc_info=True,
            )

    def clear_freshness_task(self, task: asyncio.Task) -> None:
        if self.host._freshness_task is task:
            self.host._freshness_task = None


def _fully_available() -> SearchAvailability:
    return SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="complete",
        semantic_fine="complete",
    )


__all__ = ["BootstrapCoordinator", "BootstrapHost", "IndexState"]
