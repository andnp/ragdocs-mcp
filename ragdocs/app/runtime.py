"""Runtime configuration for the searchkernel application.

This module handles runtime setup that should only be done at the application level,
not when the library is imported. This includes environment variable configuration
for threading and numerical libraries.
"""

import logging
import os

from searchkernel.config import Config, load_config

logger = logging.getLogger(__name__)


def raise_file_descriptor_limit() -> None:
    """Raise the soft descriptor limit to the process hard limit when possible.

    Daemon workers may need more descriptors than the common interactive-shell
    default while watching multiple repositories. This is a best-effort process
    setup step and is intentionally a no-op on platforms without ``resource``
    or when the operating system refuses the adjustment.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= hard:
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        logger.info("Raised file descriptor soft limit from %s to %s", soft, hard)
    except (ImportError, OSError, ValueError) as exc:
        logger.debug("Could not raise file descriptor limit: %s", exc)


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
    raise_file_descriptor_limit()

    if config is None:
        config = load_config()
    num_threads = str(config.indexing.torch_num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["TORCH_NUM_THREADS"] = num_threads
