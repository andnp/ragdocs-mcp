"""Runtime configuration for the searchkernel application.

This module handles runtime setup that should only be done at the application level,
not when the library is imported. This includes environment variable configuration
for threading and numerical libraries.
"""

import os

from searchkernel.config import Config, load_config


def configure_runtime_threads(config: Config | None = None) -> None:
    """Configure threading environment variables for numerical libraries.

    This function sets the number of threads for OMP (OpenMP), MKL (Intel MKL),
    and PyTorch to enable proper parallelization. This should be called at
    application startup (daemon, CLI, FastAPI server) but NOT during library
    import, to avoid mutating global state for library consumers.

    Must be called BEFORE building the application context, because the context
    initializes torch/embedding models that read these variables at load time.

    Args:
        config: The application configuration containing torch_num_threads. If
            omitted, the config is loaded so callers can invoke this before the
            context (and thus the config) exists.
    """
    if config is None:
        config = load_config()
    num_threads = str(config.indexing.torch_num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["TORCH_NUM_THREADS"] = num_threads
