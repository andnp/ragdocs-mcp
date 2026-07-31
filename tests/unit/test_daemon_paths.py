from pathlib import Path

from ragdocs.daemon.paths import RuntimePaths, _socket_path_for


def test_runtime_paths_are_global_and_co_located() -> None:
    paths = RuntimePaths.resolve()

    assert paths.index_db_path.parent == paths.root
    assert paths.queue_db_path.parent == paths.root
    assert paths.metadata_path.parent == paths.root
    assert paths.lock_path.parent == paths.root
    assert paths.metadata_path.name == "daemon.json"
    assert paths.lock_path.name == "daemon.lock"

    # socket_path is normally co-located too, except it falls back to a short
    # tmp-dir path when root would push it past the AF_UNIX path length limit
    # (see test_socket_path_falls_back_when_root_is_long below).
    if len(str(paths.root / "daemon.sock")) < 100:
        assert paths.socket_path.parent == paths.root
        assert paths.socket_path.name == "daemon.sock"


def test_socket_path_short_root_is_co_located() -> None:
    root = Path("/home/user/.local/state/mcp-markdown-ragdocs/daemon")
    assert _socket_path_for(root) == root / "daemon.sock"


def test_socket_path_falls_back_when_root_is_long() -> None:
    root = Path("/home/user/" + "x" * 100 + "/.local/state/mcp-markdown-ragdocs/daemon")
    socket_path = _socket_path_for(root)

    assert socket_path.parent != root
    assert socket_path.name.startswith("mcp-ragdocs-")
    assert socket_path.name.endswith(".sock")
    assert len(str(socket_path)) < 100

    # Deterministic per root, so repeated resolution reuses the same socket.
    assert _socket_path_for(root) == socket_path