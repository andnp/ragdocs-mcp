from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads
from mcp_markdown_ragdocs.context import ApplicationContext
from mcp_markdown_ragdocs.coordination.queue import get_huey
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.daemon import RuntimePaths, read_daemon_metadata
from mcp_markdown_ragdocs.daemon.health import DaemonHealthServer
from mcp_markdown_ragdocs.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from mcp_markdown_ragdocs.indexing.tasks import register_tasks, submit_record_batch
from mcp_markdown_ragdocs.worker.process import HueyWorkerProcess

BuildAdminOverviewPayload = Callable[[ApplicationContext, RuntimePaths, bool, int | None, str], dict[str, object]]
BuildIndexStatsPayload = Callable[[Any], dict[str, object]]
BuildQueueStatusPayload = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class DaemonRuntime:
    ctx: Any
    worker: Any
    health_server: Any


def create_daemon_runtime(
    runtime_paths: RuntimePaths,
    *,
    coordinator,
    build_admin_overview_payload: BuildAdminOverviewPayload,
    build_index_stats_payload: BuildIndexStatsPayload,
    build_queue_status_payload: BuildQueueStatusPayload,
) -> DaemonRuntime:
    configure_runtime_threads()
    ctx = ApplicationContext.create(
        enable_watcher=False,
        lazy_embeddings=True,
        use_tasks=True,
        index_path_override=runtime_paths.root,
        global_runtime=True,
    )
    huey = get_huey(runtime_paths.queue_db_path)
    task_lease_store = TaskLeaseStore(runtime_paths.queue_db_path)
    work_intent_store = WorkIntentStore(runtime_paths.queue_db_path)
    register_tasks(
        huey,
        ctx.index_manager,
        task_lease_store,
        work_intent_store,
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
                submit_record_batch=submit_record_batch,
            )
        ),
    )
    return DaemonRuntime(ctx=ctx, worker=worker, health_server=health_server)
