"""Application-level composition and runtime setup for searchkernel."""

from typing import Any

__all__ = ["build_kernel", "configure_runtime_threads"]


def build_kernel(*args: Any, **kwargs: Any) -> Any:
    from mcp_markdown_ragdocs.app.composition import build_kernel as _build_kernel

    return _build_kernel(*args, **kwargs)


def configure_runtime_threads(*args: Any, **kwargs: Any) -> Any:
    from mcp_markdown_ragdocs.app.runtime import (
        configure_runtime_threads as _configure_runtime_threads,
    )

    return _configure_runtime_threads(*args, **kwargs)
