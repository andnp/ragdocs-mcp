"""Application-level composition and runtime setup for the searchkernel library."""

from mcp_markdown_ragdocs.app.composition import build_kernel
from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads

__all__ = ["build_kernel", "configure_runtime_threads"]
