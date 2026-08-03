"""Composition root for the searchkernel library.

This module provides a clean, library-friendly entry point for building
the kernel without triggering daemon initialization or global mutation.
"""

from pathlib import Path

from mcp_markdown_ragdocs.config import Config, load_config
from mcp_markdown_ragdocs.context import ApplicationContext


def build_kernel(
    config: Config | None = None,
    *,
    project_override: str | None = None,
    enable_watcher: bool = True,
    lazy_embeddings: bool = True,
    index_path_override: Path | None = None,
    documents_path_override: Path | None = None,
) -> ApplicationContext:
    """Build a searchkernel ApplicationContext for library usage.

    This is the composition root for the searchkernel library. It creates and
    configures an ApplicationContext without triggering daemon initialization
    or mutating global environment variables.

    The caller is responsible for environment setup via configure_runtime_threads()
    if needed, allowing library consumers to avoid global mutations.

    Args:
        config: Optional pre-loaded configuration. If not provided, loads from defaults.
        project_override: Override the detected project.
        enable_watcher: Whether to enable file watching for changes.
        lazy_embeddings: Whether to defer embedding model warmup.
        index_path_override: Override the index storage path.
        documents_path_override: Override the documents root path.

    Returns:
        A configured ApplicationContext ready for search operations.
    """
    if config is None:
        config = load_config()

    return ApplicationContext.create(
        project_override=project_override,
        enable_watcher=enable_watcher,
        lazy_embeddings=lazy_embeddings,
        index_path_override=index_path_override,
        documents_path_override=documents_path_override,
        global_runtime=False,
        config=config,
    )
