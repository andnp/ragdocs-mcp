from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from mcp_markdown_ragdocs.daemon import RuntimePaths
from mcp_markdown_ragdocs.daemon.runtime import create_daemon_runtime


def test_create_daemon_runtime_builds_worker_health_server_and_registers_tasks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    fake_ctx = SimpleNamespace(
        index_manager=object(),
        git_indexing_enabled=False,
        config=SimpleNamespace(indexing=SimpleNamespace(task_backpressure_limit=7)),
        index_path=tmp_path / "index",
        documents_roots=[Path("/docs")],
    )
    fake_worker = SimpleNamespace(is_running=False, pid=321)
    runtime_paths = RuntimePaths(
        root=tmp_path,
        queue_db_path=tmp_path / "queue.db",
        socket_path=tmp_path / "daemon.sock",
        metadata_path=tmp_path / "daemon.json",
        index_db_path=tmp_path / "index.db",
        lock_path=tmp_path / "daemon.lock",
    )
    fake_queue_runtime = SimpleNamespace(
        huey="huey",
        db_path=runtime_paths.queue_db_path,
    )
    fake_ctx.attach_task_runtime = lambda task_runtime: calls.update(
        {"task_runtime": task_runtime}
    )

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.runtime.ApplicationContext.create",
        lambda **kwargs: calls.update({"create_kwargs": kwargs}) or fake_ctx,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.runtime.build_queue_runtime",
        lambda queue_db_path: fake_queue_runtime,
    )
    def fake_register_tasks(huey, index_manager, task_lease_store, work_intent_store, **kwargs):
        calls["register_tasks"] = {
            "huey": huey,
            "index_manager": index_manager,
            "task_lease_store": task_lease_store,
            "work_intent_store": work_intent_store,
            **kwargs,
        }
        return SimpleNamespace(
            queue_runtime=fake_queue_runtime,
            submission=SimpleNamespace(submit_record_batch=lambda records: None),
        )

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.runtime.register_tasks",
        fake_register_tasks,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.runtime.HueyWorkerProcess",
        lambda runtime_paths: fake_worker,
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.runtime.read_daemon_metadata",
        lambda metadata_path: {"status": "ready"},
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.runtime.DaemonHealthServer",
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
        build_admin_overview_payload=lambda ctx, runtime_paths, queue_runtime, worker_running, worker_pid, lifecycle: {
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
    register_tasks_call = cast(dict[str, object], calls["register_tasks"])
    assert {
        "huey": "huey",
        "index_manager": fake_ctx.index_manager,
        "task_lease_store": register_tasks_call["task_lease_store"],
        "work_intent_store": register_tasks_call["work_intent_store"],
        "task_backpressure_limit": 7,
        "bootstrap_index_path": fake_ctx.index_path,
        "bootstrap_documents_roots": fake_ctx.documents_roots,
    } == {
        key: value
        for key, value in register_tasks_call.items()
        if key != "schedule_vocabulary_catch_up"
    }
    schedule_vocabulary_catch_up = register_tasks_call[
        "schedule_vocabulary_catch_up"
    ]
    assert callable(schedule_vocabulary_catch_up)
    assert schedule_vocabulary_catch_up() is False
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
