from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from src.daemon import RuntimePaths
from src.daemon.runtime import create_daemon_runtime


def test_create_daemon_runtime_builds_worker_health_server_and_registers_tasks(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    fake_ctx = SimpleNamespace(
        index_manager=object(),
        commit_indexer=None,
        config=SimpleNamespace(indexing=SimpleNamespace(task_backpressure_limit=7)),
        index_path=Path("/runtime/index"),
        documents_roots=[Path("/docs")],
        schedule_vocabulary_catch_up=lambda: True,
    )
    fake_worker = SimpleNamespace(is_running=False, pid=321)
    runtime_paths = RuntimePaths(
        root=Path("/runtime"),
        queue_db_path=Path("/runtime/queue.db"),
        socket_path=Path("/runtime/daemon.sock"),
        metadata_path=Path("/runtime/daemon.json"),
        index_db_path=Path("/runtime/index.db"),
        lock_path=Path("/runtime/daemon.lock"),
    )

    monkeypatch.setattr(
        "src.daemon.runtime.ApplicationContext.create",
        lambda **kwargs: calls.update({"create_kwargs": kwargs}) or fake_ctx,
    )
    monkeypatch.setattr("src.daemon.runtime.get_huey", lambda queue_db_path: "huey")
    monkeypatch.setattr(
        "src.daemon.runtime.register_tasks",
        lambda huey, index_manager, **kwargs: calls.setdefault(
            "register_tasks",
            {
                "huey": huey,
                "index_manager": index_manager,
                **kwargs,
            },
        ),
    )
    monkeypatch.setattr(
        "src.daemon.runtime.HueyWorkerProcess",
        lambda runtime_paths: fake_worker,
    )
    monkeypatch.setattr(
        "src.daemon.runtime.read_daemon_metadata",
        lambda metadata_path: {"status": "ready"},
    )
    monkeypatch.setattr(
        "src.daemon.runtime.DaemonHealthServer",
        lambda socket_path, metadata_provider, request_handler: calls.setdefault(
            "health_server",
            {
                "socket_path": socket_path,
                "metadata": metadata_provider(),
                "request_handler": request_handler,
            },
        ),
    )

    runtime = create_daemon_runtime(
        runtime_paths,
        coordinator=SimpleNamespace(state=SimpleNamespace(value="ready")),
        build_admin_overview_payload=lambda ctx, runtime_paths, worker_running, worker_pid, lifecycle: {
            "lifecycle": lifecycle,
            "worker_pid": worker_pid,
        },
        build_index_stats_payload=lambda ctx: {"indexed_documents": 0},
        build_queue_status_payload=lambda queue_path, worker_running, backpressure_limit: {
            "queue": str(queue_path),
            "worker_running": worker_running,
            "backpressure_limit": backpressure_limit,
        },
    )

    assert calls["create_kwargs"] == {
        "enable_watcher": False,
        "lazy_embeddings": True,
        "use_tasks": True,
        "index_path_override": runtime_paths.root,
        "global_runtime": True,
    }
    assert calls["register_tasks"] == {
        "huey": "huey",
        "index_manager": fake_ctx.index_manager,
        "commit_indexer": None,
        "task_backpressure_limit": 7,
        "bootstrap_index_path": fake_ctx.index_path,
        "bootstrap_documents_roots": fake_ctx.documents_roots,
        "schedule_vocabulary_catch_up": fake_ctx.schedule_vocabulary_catch_up,
    }
    health_server_args = cast(dict[str, object], calls["health_server"])
    assert health_server_args == {
        "socket_path": runtime_paths.socket_path,
        "metadata": {"status": "ready"},
        "request_handler": health_server_args["request_handler"],
    }
    assert callable(health_server_args["request_handler"])
    assert runtime.ctx is fake_ctx
    assert runtime.worker is fake_worker
    assert runtime.health_server == health_server_args
