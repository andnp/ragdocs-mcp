from src.daemon.paths import RuntimePaths


def test_runtime_paths_are_global_and_co_located() -> None:
    paths = RuntimePaths.resolve()

    assert paths.index_db_path.parent == paths.root
    assert paths.queue_db_path.parent == paths.root
    assert paths.metadata_path.parent == paths.root
    assert paths.lock_path.parent == paths.root
    assert paths.socket_path.parent == paths.root
    assert paths.metadata_path.name == "daemon.json"
    assert paths.lock_path.name == "daemon.lock"
    assert paths.socket_path.name == "daemon.sock"