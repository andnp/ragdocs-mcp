"""Project/context resolution helpers shared by CLI commands."""

import logging
from pathlib import Path

from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads
from mcp_markdown_ragdocs.context import ApplicationContext


def _create_query_context(project: str | None) -> ApplicationContext:
    logging.getLogger().setLevel(logging.WARNING)
    configure_runtime_threads()
    ctx = ApplicationContext.create(
        project_override=project,
        enable_watcher=False,
        lazy_embeddings=False,
    )
    return ctx


def _apply_project_detection(config, project_override: str | None = None):
    from mcp_markdown_ragdocs.config import (
        detect_project,
        resolve_documents_path,
        resolve_index_path,
    )

    detected_project = detect_project(
        projects=config.projects, project_override=project_override
    )
    index_path = resolve_index_path(config)

    explicit_documents_path: Path | None = None
    if project_override:
        override_path = Path(project_override).expanduser()
        if override_path.exists():
            explicit_documents_path = override_path.resolve()
        elif detected_project:
            for project in config.projects:
                if project.name == detected_project:
                    explicit_documents_path = Path(project.path).resolve()
                    break

    documents_path = (
        str(explicit_documents_path)
        if explicit_documents_path is not None
        else resolve_documents_path(config)
    )

    config.indexing.index_path = str(index_path)
    config.indexing.documents_path = documents_path
    config.detected_project = detected_project
    return config
