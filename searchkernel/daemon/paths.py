from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _state_home() -> Path:
    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home)
    return Path.home() / ".local" / "state"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    index_db_path: Path
    queue_db_path: Path
    metadata_path: Path
    lock_path: Path
    socket_path: Path

    @classmethod
    def resolve(cls) -> RuntimePaths:
        root = _state_home() / "mcp-markdown-ragdocs" / "daemon"
        return cls(
            root=root,
            index_db_path=root / "index.db",
            queue_db_path=root / "queue.db",
            metadata_path=root / "daemon.json",
            lock_path=root / "daemon.lock",
            socket_path=root / "daemon.sock",
        )

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)