"""Application composition contracts for the local record runtime."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from searchkernel.api import ContentSource, load_manifest

from mcp_markdown_ragdocs.app.runtime_factory import (
    EmbeddingProvider,
    RuntimeComponents,
    RuntimePaths,
    assemble_runtime,
    build_gdrive_source,
)
from mcp_markdown_ragdocs.config import (
    Config,
    detect_project,
    load_config,
    resolve_documents_path,
    resolve_index_path,
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
    project_detector: Callable[..., str | None] | None = None,
) -> RuntimeComponents:
    """Resolve application paths before delegating concrete runtime assembly."""
    detected_project = None
    if not global_runtime:
        detector = project_detector or detect_project
        detected_project = detector(
            projects=config.projects,
            project_override=project_override,
        )

    configured_index_path = resolve_index_path(config)
    index_path = index_path_override or configured_index_path
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

    paths = RuntimePaths(
        index_path=index_path,
        documents_path=documents_path,
        documents_roots=documents_roots,
    )
    return assemble_runtime(
        config,
        paths,
        embedding_model_name=embedding_model_name,
        active_model_identity=active_model_identity,
        enable_watcher=enable_watcher,
        lazy_embeddings=lazy_embeddings,
        use_tasks=use_tasks,
        source_builder=source_builder,
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
    "assemble_runtime",
    "build_gdrive_source",
    "build_kernel",
    "build_runtime_components",
]
