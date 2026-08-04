from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.config import LoggingConfig
from mcp_markdown_ragdocs.runtime_logging import configure_file_logging


def test_logging_config_rejects_invalid_retention() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        LoggingConfig(max_bytes=0)
    with pytest.raises(ValueError, match="backup_count"):
        LoggingConfig(backup_count=0)


def test_file_logging_retains_configured_backups(tmp_path: Path) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    log_path = tmp_path / "worker.log"

    try:
        configure_file_logging(
            log_path,
            LoggingConfig(max_bytes=128, backup_count=2),
        )
        test_logger = logging.getLogger("rotation-test")
        for index in range(10):
            test_logger.info("entry-%d %s", index, "x" * 180)
        for handler in root_logger.handlers:
            handler.flush()

        log_files = sorted(tmp_path.glob("worker.log*"))
        assert len(log_files) == 3
        assert log_path.exists()
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
