"""Task queue backed by SQLite via Huey."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from huey import SqliteHuey

logger = logging.getLogger(__name__)

# Module-level Huey instance, lazily initialized
_huey: SqliteHuey | None = None


@dataclass(frozen=True)
class QueueRuntime:
    """Own the queue instance and its durable database identity."""

    huey: SqliteHuey
    db_path: Path


def get_huey(db_path: Path | None = None) -> SqliteHuey:
    """Get or create the module-level SqliteHuey instance.

    Args:
        db_path: Path to the SQLite database for the queue.
                 Required on first call, optional after that.
    """
    global _huey
    if _huey is not None:
        return _huey

    if db_path is None:
        raise RuntimeError("db_path required on first call to get_huey()")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    _huey = SqliteHuey(
        name="ragdocs",
        filename=str(db_path),
        immediate=False,  # Tasks go to queue, not executed inline
    )
    logger.info("Task queue initialized: %s", db_path)
    return _huey


def build_queue_runtime(db_path: Path) -> QueueRuntime:
    """Create the explicit queue runtime used by application composition."""
    return QueueRuntime(huey=get_huey(db_path), db_path=db_path)


def reset_huey() -> None:
    """Reset the module-level Huey instance. Used in tests."""
    global _huey
    _huey = None
