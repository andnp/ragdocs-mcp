from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mcp_markdown_ragdocs.config import LoggingConfig


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
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
