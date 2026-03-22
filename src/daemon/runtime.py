from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.context import ApplicationContext
from src.coordination.queue import get_huey
from src.daemon import RuntimePaths, read_daemon_metadata
from src.daemon.health import DaemonHealthServer
from src.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from src.indexing.tasks import register_tasks
from src.worker.process import HueyWorkerProcess


BuildAdminOverviewPayload = Callable[[ApplicationContext, RuntimePaths, bool, int | None, str], dict[str, object]]
BuildIndexStatsPayload = Callable[[object], dict[str, object]]
BuildQueueStatusPayload = Callable[[Path, bool, int | None], dict[str, object]]


@dataclass(frozen=True)
class DaemonRuntime:
    ctx: ApplicationContext
    worker: HueyWorkerProcess
    health_server: DaemonHealthServer


def create_daemon_runtime(
    runtime_paths: RuntimePaths,
    *,
    coordinator,
    build_admin_overview_payload: BuildAdminOverviewPayload,
    build_index_stats_payload: BuildIndexStatsPayload,
    build_queue_status_payload: BuildQueueStatusPayload,
) -> DaemonRuntime:
    ctx = ApplicationContext.create(
        enable_watcher=False,
        lazy_embeddings=True,
        use_tasks=True,
        index_path_override=runtime_paths.root,
        global_runtime=True,
    )
    huey = get_huey(runtime_paths.queue_db_path)
    register_tasks(
        huey,
        ctx.index_manager,
        commit_indexer=ctx.commit_indexer,
        task_backpressure_limit=ctx.config.indexing.task_backpressure_limit,
        bootstrap_index_path=ctx.index_path,
        bootstrap_documents_roots=ctx.documents_roots,
        schedule_vocabulary_catch_up=ctx.schedule_vocabulary_catch_up,
    )
    worker = HueyWorkerProcess(runtime_paths=runtime_paths)
    health_server = DaemonHealthServer(
        socket_path=runtime_paths.socket_path,
        metadata_provider=lambda: read_daemon_metadata(runtime_paths.metadata_path),
        request_handler=build_daemon_request_handler(
            DaemonRequestRouterDependencies(
                ctx=ctx,
                coordinator=coordinator,
                runtime_root=runtime_paths.root,
                queue_db_path=runtime_paths.queue_db_path,
                socket_path=runtime_paths.socket_path,
                index_db_path=runtime_paths.index_db_path,
                get_worker_running=lambda: worker.is_running,
                get_worker_pid=lambda: worker.pid,
                build_admin_overview_payload=lambda current_ctx, runtime_root, worker_running, worker_pid, lifecycle: build_admin_overview_payload(
                    cast(ApplicationContext, current_ctx),
                    RuntimePaths(
                        root=runtime_root,
                        index_db_path=runtime_paths.index_db_path,
                        queue_db_path=runtime_paths.queue_db_path,
                        metadata_path=runtime_paths.metadata_path,
                        lock_path=runtime_paths.lock_path,
                        socket_path=runtime_paths.socket_path,
                    ),
                    worker_running,
                    worker_pid,
                    lifecycle,
                ),
                build_index_stats_payload=build_index_stats_payload,
                build_queue_status_payload=build_queue_status_payload,
            )
        ),
    )
    return DaemonRuntime(ctx=ctx, worker=worker, health_server=health_server)
