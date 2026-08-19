"""Concrete runtime assembly for the local record application."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from searchkernel.api import (
    ContentSource,
    LocalRecordKernel,
    build_local_record_kernel,
    get_parser_suffixes,
)

from mcp_markdown_ragdocs.app.search import (
    ApplicationSearchUseCase,
    build_record_search_policy,
    build_reranker,
    to_record_search_config,
)
from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing.record_manager import (
    RecordIndexManager,
    build_embedding_provider,
)
from mcp_markdown_ragdocs.indexing.record_ports import LocalRecordStorage
from mcp_markdown_ragdocs.indexing.watcher_lifecycle import WatcherLifecycle
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """The provider attributes required by the application runtime."""

    model_name: str
    dim: int


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved paths shared by every component in one local runtime."""

    index_path: Path
    documents_path: Path
    documents_roots: tuple[Path, ...]


@dataclass
class RuntimeComponents:
    """The complete application runtime assembled around one record kernel."""

    config: Config
    paths: RuntimePaths
    kernel: LocalRecordKernel
    embedding_provider: EmbeddingProvider
    index_manager: RecordIndexManager
    orchestrator: CanonicalSearchAdapter
    search_use_case: ApplicationSearchUseCase
    record_ingestor: Any
    watcher_lifecycle: WatcherLifecycle
    db_manager: Any
    git_indexing_enabled: bool
    active_model_identity: tuple[str, int] | None


def build_gdrive_source(
    config: Config,
    *,
    source_root: Path,
    index_path: Path,
) -> ContentSource:
    """Compose Drive against the existing global record/index runtime."""
    from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
    from mcp_markdown_ragdocs.gdrive.client import GoogleDriveClient
    from mcp_markdown_ragdocs.gdrive.extraction import ExtractionLimits
    from mcp_markdown_ragdocs.gdrive.gate import DriveRequestGate
    from mcp_markdown_ragdocs.gdrive.session import AuthorizedUserSession
    from mcp_markdown_ragdocs.gdrive.adapter import GDriveStateRepository

    drive_config = config.gdrive
    session = AuthorizedUserSession(
        drive_config.credentials_path,
        source_root,
        scopes=drive_config.scopes,
    )
    client = GoogleDriveClient(
        session,
        max_page_size=drive_config.page_size,
        max_download_bytes=drive_config.max_download_bytes,
        request_gate=DriveRequestGate(
            index_path / "gdrive-request-gate.db",
            min_interval_seconds=drive_config.request_min_interval_seconds,
            max_concurrent=drive_config.request_max_concurrent,
        ),
    )
    state_repository = GDriveStateRepository(index_path / "gdrive-state.db")
    return GoogleDriveContentSource(
        client,
        workspace_id=drive_config.workspace_id,
        shared_drive_ids=drive_config.shared_drive_ids,
        extraction_limits=ExtractionLimits(
            max_download_bytes=drive_config.max_download_bytes,
            max_text_bytes=drive_config.max_text_bytes,
            max_items=drive_config.max_items,
            max_pages=drive_config.max_pages,
            max_seconds=drive_config.max_seconds,
        ),
        page_size=drive_config.page_size,
        state_repository=state_repository,
    )


def assemble_runtime(
    config: Config,
    paths: RuntimePaths,
    *,
    embedding_model_name: str,
    active_model_identity: tuple[str, int] | None,
    enable_watcher: bool,
    lazy_embeddings: bool,
    use_tasks: bool,
    source_builder: Callable[..., ContentSource] = build_gdrive_source,
) -> RuntimeComponents:
    """Assemble providers, indices, sources, and runtime-owned services."""
    embedding_provider = build_embedding_provider(config, embedding_model_name)
    content_sources: list[ContentSource] = []
    if config.gdrive.enabled:
        content_sources.append(
            source_builder(
                config,
                source_root=paths.documents_path,
                index_path=paths.index_path,
            )
        )

    storage_holder: dict[str, LocalRecordStorage] = {}
    search_policy = build_record_search_policy(
        lambda: storage_holder["storage"].keyword_store,
        lambda identity: storage_holder["storage"].hydrate_record(identity),
        lambda incoming: storage_holder["storage"].graph.set_direction(incoming),
        project_uplift_multiplier=config.search.project_uplift_multiplier,
    )
    local_kernel = build_local_record_kernel(
        paths.index_path / "index.db",
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_provider.model_name,
        embedding_dim=embedding_provider.dim,
        vector_engine="auto",
        reranker=build_reranker(config.search),
        search_policy=search_policy,
        search_config=to_record_search_config(config.search),
    )
    storage = LocalRecordStorage(local_kernel)
    storage_holder["storage"] = storage
    storage.tune_backend()
    if not lazy_embeddings:
        logger.info("Embedding provider is daemon-backed; no in-process warmup needed")

    manager = RecordIndexManager(
        config,
        local_kernel,
        embedding_provider,
        documents_roots=list(paths.documents_roots),
        content_sources=content_sources,
        storage=storage,
    )
    orchestrator = CanonicalSearchAdapter(
        manager,
        documents_path=paths.documents_path,
    )
    watcher_lifecycle = WatcherLifecycle.create(
        enabled=enable_watcher,
        documents_path=config.indexing.documents_path,
        documents_roots=list(paths.documents_roots),
        index_manager=manager,
        cooldown=config.indexing.debounce_window_seconds,
        include_patterns=config.indexing.include,
        exclude_patterns=config.indexing.exclude,
        exclude_hidden_dirs=config.indexing.exclude_hidden_dirs,
        parser_suffixes=set(get_parser_suffixes()),
        use_tasks=use_tasks,
        task_backpressure_limit=config.indexing.task_backpressure_limit,
    )

    git_indexing_enabled = False
    if config.git_indexing.enabled:
        from mcp_markdown_ragdocs.git.repository import is_git_available

        if is_git_available():
            git_indexing_enabled = True
            logger.info("Git commit indexing enabled")
        else:
            logger.warning("Git binary not found - git history search disabled")

    return RuntimeComponents(
        config=config,
        paths=paths,
        kernel=local_kernel,
        embedding_provider=embedding_provider,
        index_manager=manager,
        orchestrator=orchestrator,
        search_use_case=orchestrator.search_use_case,
        record_ingestor=manager.ingestor,
        watcher_lifecycle=watcher_lifecycle,
        db_manager=manager.database_manager,
        git_indexing_enabled=git_indexing_enabled,
        active_model_identity=active_model_identity,
    )


__all__ = [
    "EmbeddingProvider",
    "RuntimeComponents",
    "RuntimePaths",
    "assemble_runtime",
    "build_gdrive_source",
]
