"""Task queue backed by SQLite via Huey."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from huey import SqliteHuey

logger = logging.getLogger(__name__)


class _HueyStorage(Protocol):
    filename: str


@dataclass(frozen=True)
class QueueRuntime:
    """Own the queue instance and its durable database identity."""

    huey: SqliteHuey
    db_path: Path

    def __post_init__(self) -> None:
        if not self.db_path.name:
            raise ValueError("QueueRuntime requires a database file path")
        storage = cast(_HueyStorage, self.huey.storage)
        if Path(storage.filename).resolve() != self.db_path.resolve():
            raise ValueError("QueueRuntime path must match the Huey storage path")


def build_queue_runtime(db_path: Path) -> QueueRuntime:
    """Create the explicit queue runtime used by application composition."""
    normalized_path = Path(db_path).expanduser()
    if not normalized_path.name or normalized_path.is_dir():
        raise ValueError("QueueRuntime requires a database file path")
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    huey = SqliteHuey(
        name="ragdocs",
        filename=str(normalized_path),
        immediate=False,  # Tasks go to queue, not executed inline
    )
    logger.info("Task queue initialized: %s", normalized_path)
    return QueueRuntime(huey=huey, db_path=normalized_path)
