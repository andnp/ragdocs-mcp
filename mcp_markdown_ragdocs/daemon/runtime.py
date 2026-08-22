from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads
from mcp_markdown_ragdocs.context import ApplicationContext
from mcp_markdown_ragdocs.coordination.queue import QueueRuntime, build_queue_runtime
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.daemon import RuntimePaths, read_daemon_metadata
from mcp_markdown_ragdocs.daemon.health import DaemonHealthServer
from mcp_markdown_ragdocs.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from mcp_markdown_ragdocs.indexing.tasks import (
    index_document_retry_policy,
    register_tasks,
)
from mcp_markdown_ragdocs.worker.process import HueyWorkerProcess

BuildAdminOverviewPayload = Callable[[ApplicationContext, RuntimePaths, QueueRuntime, bool, int | None, str], dict[str, object]]
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
    queue_runtime = build_queue_runtime(runtime_paths.queue_db_path)
    huey = queue_runtime.huey
    task_lease_store = TaskLeaseStore(runtime_paths.queue_db_path)
    work_intent_store = WorkIntentStore(
        runtime_paths.queue_db_path, retry_policy=index_document_retry_policy
    )

    def schedule_vocabulary_catch_up() -> bool:
        return False

    task_runtime = register_tasks(
        huey,
        ctx.index_manager,
        task_lease_store,
        work_intent_store,
        task_backpressure_limit=ctx.config.indexing.task_backpressure_limit,
        embedding_cache_prune_cooldown_seconds=ctx.config.indexing.embedding_cache_prune_cooldown_seconds,
        bootstrap_index_path=ctx.index_path,
        bootstrap_documents_roots=ctx.documents_roots,
        schedule_vocabulary_catch_up=schedule_vocabulary_catch_up,
    )
    ctx.attach_task_runtime(task_runtime)
    queue_runtime = task_runtime.queue_runtime
    worker = HueyWorkerProcess(runtime_paths=runtime_paths)
    health_server = DaemonHealthServer(
        socket_path=runtime_paths.socket_path,
        metadata_provider=lambda: read_daemon_metadata(runtime_paths.metadata_path),
        request_handler=build_daemon_request_handler(
            DaemonRequestRouterDependencies(
                ctx=ctx,
                coordinator=coordinator,
                runtime_root=runtime_paths.root,
                queue_runtime=queue_runtime,
                socket_path=runtime_paths.socket_path,
                index_db_path=runtime_paths.index_db_path,
                get_worker_running=lambda: worker.is_running,
                get_worker_pid=lambda: worker.pid,
                build_admin_overview_payload=lambda current_ctx, worker_running, worker_pid, lifecycle: build_admin_overview_payload(
                    cast(ApplicationContext, current_ctx),
                    runtime_paths,
                    queue_runtime,
                    worker_running,
                    worker_pid,
                    lifecycle,
                ),
                build_index_stats_payload=build_index_stats_payload,
                build_queue_status_payload=build_queue_status_payload,
                submit_record_batch=task_runtime.submission.submit_record_batch,
                task_submission=task_runtime.submission,
            )
        ),
    )
    return DaemonRuntime(ctx=ctx, worker=worker, health_server=health_server)
