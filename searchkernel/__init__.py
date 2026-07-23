"""searchkernel: A composable search and indexing framework.

This library provides:
- ApplicationContext: The core application context for building a search system
- build_kernel: Composition root for library usage (no daemon, no global mutations)
- configure_runtime_threads: Runtime configuration for threading environment variables
"""

from searchkernel.app.composition import build_kernel
from searchkernel.app.runtime import configure_runtime_threads
from searchkernel.context import ApplicationContext

__all__ = [
    "ApplicationContext",
    "build_kernel",
    "configure_runtime_threads",
]
