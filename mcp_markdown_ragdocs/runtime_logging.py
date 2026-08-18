from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mcp_markdown_ragdocs.config import LoggingConfig


class _OutsideRootWarningFilter(logging.Filter):
    """Coalesce repeated path-root warnings from indexing dependencies."""

    _MARKERS = (
        "configured document roots",
        "outside docs root",
        "outside documents path",
    )

    def __init__(self, interval_seconds: float = 60.0) -> None:
        super().__init__()
        self._interval_seconds = interval_seconds
        self._last_emitted: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.levelno < logging.WARNING or not any(
            marker in message for marker in self._MARKERS
        ):
            return True

        key = f"{record.name}:{next(marker for marker in self._MARKERS if marker in message)}"
        now = time.monotonic()
        last_emitted = self._last_emitted.get(key)
        if last_emitted is not None and now - last_emitted < self._interval_seconds:
            return False
        self._last_emitted[key] = now
        return True


def configure_file_logging(log_path: Path, config: LoggingConfig) -> None:
    """Replace inherited stream handlers with a bounded file handler."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    handler.addFilter(_OutsideRootWarningFilter())
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    # One line per embedding request otherwise dominates the indexing log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
