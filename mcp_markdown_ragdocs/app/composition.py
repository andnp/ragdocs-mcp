"""Application composition contracts for the local record runtime."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from searchkernel.api import (
    ContentSource,
    LocalRecordKernel,
    RecordIdentity,
    build_local_record_kernel,
    load_manifest,
)

from mcp_markdown_ragdocs.app.search import (
    ApplicationSearchUseCase,
    build_record_search_policy,
    build_reranker,
    to_record_search_config,
)
from mcp_markdown_ragdocs.config import (
    Config,
    detect_project,
    load_config,
    resolve_documents_path,
    resolve_index_path,
)
from mcp_markdown_ragdocs.indexing.record_manager import (
    RecordIndexManager,
    build_embedding_provider,
    install_bidirectional_graph_store,
)
from mcp_markdown_ragdocs.indexing.watcher_lifecycle import WatcherLifecycle
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """The provider attributes required by the application composition root."""

    model_name: str
    dim: int


@runtime_checkable
class _DirectionalGraphStore(Protocol):
    def set_direction(self, incoming: str | bool) -> None: ...


def _set_graph_direction(
    kernel: LocalRecordKernel,
    incoming: str | bool,
) -> None:
    graph_store = kernel.pipeline._graph_store
    if not isinstance(graph_store, _DirectionalGraphStore):
        raise RuntimeError("record kernel graph store does not support direction")
    graph_store.set_direction(incoming)


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved paths shared by every component in one local runtime."""

    index_path: Path
    fallback_index_path: Path | None
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
    from mcp_markdown_ragdocs.gdrive.state import GDriveStateRepository

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
        request_gate=DriveRequestGate(index_path / "gdrive-request-gate.db"),
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


def _resolve_documents_roots(
    config: Config,
    *,
    detected_project: str | None,
    project_override: str | None,
    documents_path_override: Path | None,
    global_runtime: bool,
) -> tuple[Path, ...]:
    if documents_path_override is not None:
        return (documents_path_override.expanduser().resolve(),)

    if global_runtime:
        if config.projects:
            return tuple(Path(project.path).resolve() for project in config.projects)
        return (Path(resolve_documents_path(config)).resolve(),)

    if project_override:
        override_path = Path(project_override).expanduser()
        if override_path.exists():
            return (override_path.resolve(),)

    if detected_project:
        for project in config.projects:
            if project.name == detected_project:
                return (Path(project.path).resolve(),)

    if config.projects:
        return tuple(Path(project.path).resolve() for project in config.projects)

    return (Path(resolve_documents_path(config)).resolve(),)


def _compute_common_documents_root(documents_roots: tuple[Path, ...]) -> Path:
    if not documents_roots:
        return Path.cwd().resolve()
    if len(documents_roots) == 1:
        return documents_roots[0]
    return Path(os.path.commonpath([str(root) for root in documents_roots])).resolve()


def build_runtime_components(
    config: Config,
    *,
    project_override: str | None = None,
    enable_watcher: bool = True,
    lazy_embeddings: bool = True,
    use_tasks: bool = False,
    index_path_override: Path | None = None,
    documents_path_override: Path | None = None,
    global_runtime: bool = False,
    source_builder: Callable[..., ContentSource] = build_gdrive_source,
) -> RuntimeComponents:
    """Build all application components around one local record kernel."""
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

    documents_roots = _resolve_documents_roots(
        config,
        detected_project=detected_project,
        project_override=project_override,
        documents_path_override=documents_path_override,
        global_runtime=global_runtime,
    )
    documents_path = _compute_common_documents_root(documents_roots)
    config.indexing.index_path = str(index_path)
    config.indexing.documents_path = str(documents_path)
    config.detected_project = None if global_runtime else detected_project

    saved_manifest = load_manifest(index_path)
    active_model_identity: tuple[str, int] | None = None
    if saved_manifest is not None and saved_manifest.active_model is not None:
        active_namespace = saved_manifest.active_model.namespace
        config.llm.embedding_model = active_namespace.model_name
        config.embedding.truncate_dim = active_namespace.dim
        active_model_identity = active_namespace.identity

    embedding_model_name = config.embedding.model_name
    if active_model_identity is not None:
        embedding_model_name = config.llm.embedding_model
    config.embedding.model_name = embedding_model_name
    config.llm.embedding_model = embedding_model_name
    if config.store.backend not in {"local", "faiss+sqlite"}:
        raise ValueError(
            "canonical runtime supports store.backend = 'local'; "
            f"got {config.store.backend!r}"
        )

    embedding_provider = build_embedding_provider(config, embedding_model_name)
    content_sources: list[ContentSource] = []
    if config.gdrive.enabled:
        content_sources.append(
            source_builder(
                config,
                source_root=documents_path,
                index_path=index_path,
            )
        )

    kernel_holder: dict[str, LocalRecordKernel] = {}
    search_policy = build_record_search_policy(
        lambda: kernel_holder["kernel"].keyword_store,
        lambda identity: kernel_holder["kernel"].backend.hydrate_record(identity),
        lambda incoming: _set_graph_direction(kernel_holder["kernel"], incoming),
        project_uplift_multiplier=config.search.project_uplift_multiplier,
    )
    local_kernel = build_local_record_kernel(
        index_path / "index.db",
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_provider.model_name,
        embedding_dim=embedding_provider.dim,
        vector_engine="exact",
        reranker=build_reranker(config.search),
        search_policy=search_policy,
        search_config=to_record_search_config(config.search),
    )
    kernel_holder["kernel"] = local_kernel
    if not lazy_embeddings:
        logger.info("Embedding provider is daemon-backed; no in-process warmup needed")

    manager = RecordIndexManager(
        config,
        local_kernel,
        embedding_provider,
        documents_roots=list(documents_roots),
        content_sources=content_sources,
    )
    install_bidirectional_graph_store(
        local_kernel,
        lambda: tuple(
            RecordIdentity.from_storage_key(str(row["storage_key"]))
            for row in local_kernel.backend._record_rows()
        ),
    )
    orchestrator = CanonicalSearchAdapter(
        manager,
        documents_path=documents_path,
    )

    from searchkernel.api import get_parser_suffixes

    watcher_lifecycle = WatcherLifecycle.create(
        enabled=enable_watcher,
        documents_path=config.indexing.documents_path,
        documents_roots=list(documents_roots),
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
        paths=RuntimePaths(
            index_path=index_path,
            fallback_index_path=fallback_index_path,
            documents_path=documents_path,
            documents_roots=documents_roots,
        ),
        kernel=local_kernel,
        embedding_provider=embedding_provider,
        index_manager=manager,
        orchestrator=orchestrator,
        search_use_case=orchestrator.search_use_case,
        record_ingestor=manager.ingestor,
        watcher_lifecycle=watcher_lifecycle,
        db_manager=local_kernel.backend.db_manager,
        git_indexing_enabled=git_indexing_enabled,
        active_model_identity=active_model_identity,
    )


def build_kernel(
    config: Config | None = None,
    *,
    project_override: str | None = None,
    enable_watcher: bool = True,
    lazy_embeddings: bool = True,
    index_path_override: Path | None = None,
    documents_path_override: Path | None = None,
) -> Any:
    """Build an application context without mutating process environment."""
    from mcp_markdown_ragdocs.context import ApplicationContext

    return ApplicationContext.create(
        project_override=project_override,
        enable_watcher=enable_watcher,
        lazy_embeddings=lazy_embeddings,
        index_path_override=index_path_override,
        documents_path_override=documents_path_override,
        global_runtime=False,
        config=config or load_config(),
    )


__all__ = [
    "EmbeddingProvider",
    "RuntimeComponents",
    "RuntimePaths",
    "build_gdrive_source",
    "build_kernel",
    "build_runtime_components",
]
