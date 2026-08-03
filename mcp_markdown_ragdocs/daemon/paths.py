from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# AF_UNIX socket paths are limited to sizeof(sockaddr_un.sun_path), 108 bytes
# on Linux (including the trailing NUL) - stay comfortably under that.
_MAX_SOCKET_PATH_LEN = 100


def _state_home() -> Path:
    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home)
    return Path.home() / ".local" / "state"


def _socket_path_for(root: Path) -> Path:
    """Socket path under `root`, falling back to a short tmp-dir path.

    A long $HOME/$XDG_STATE_HOME can push `root / "daemon.sock"` past the
    OS-level AF_UNIX path length limit. When that would happen, use a short
    path under the system temp directory instead, keyed by a hash of `root`
    so distinct runtime roots (projects) don't collide.
    """
    candidate = root / "daemon.sock"
    if len(str(candidate)) < _MAX_SOCKET_PATH_LEN:
        return candidate

    root_hash = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"mcp-ragdocs-{root_hash}.sock"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    index_db_path: Path
    queue_db_path: Path
    metadata_path: Path
    lock_path: Path
    socket_path: Path
    producer_metadata_path: Path | None = None

    @classmethod
    def resolve(cls) -> RuntimePaths:
        root = _state_home() / "mcp-markdown-ragdocs" / "daemon"
        return cls(
            root=root,
            index_db_path=root / "index.db",
            queue_db_path=root / "queue.db",
            metadata_path=root / "daemon.json",
            lock_path=root / "daemon.lock",
            socket_path=_socket_path_for(root),
            producer_metadata_path=root / "producer.json",
        )

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)