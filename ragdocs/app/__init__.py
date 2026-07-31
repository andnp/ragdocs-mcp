"""Application-level composition and runtime setup for the searchkernel library."""

from searchkernel.app.composition import build_kernel
from searchkernel.app.runtime import configure_runtime_threads

__all__ = ["build_kernel", "configure_runtime_threads"]
