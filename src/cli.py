import asyncio
import concurrent.futures
import contextlib
import errno
import json
import logging
import os
import sys
import signal
import time
from pathlib import Path

# Prevent tokenizers parallelism warning when forking worker process.
# Must be set before any HuggingFace/sentence-transformers imports.
# See: https://github.com/huggingface/tokenizers/issues/993
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Disable HuggingFace/tqdm progress bars to prevent stdout pollution in JSON output
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import click
import uvicorn
from rich.console import Console
from rich.table import Table

from src.config import ensure_runtime_project_registered, load_config
from src.daemon.queue_status import get_queue_stats
from src.daemon import DaemonMetadata, RuntimePaths, read_daemon_metadata
from src.daemon.health import (
    DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
    DaemonHealthServer,
    request_daemon_socket,
)
from src.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from src.daemon.management import (
    DaemonInspection,
    acquire_boot_lock,
    inspect_daemon,
    restart_daemon,
    start_daemon,
    stop_daemon,
    wait_for_daemon_ready,
)
from src.context import ApplicationContext
from src.coordination.queue import get_huey
from src.indexing.rebuild_service import (
    REBUILD_ACTIVE_STATUSES,
    REBUILD_TERMINAL_STATUSES,
)
from src.indexing.tasks import register_tasks
from src.lifecycle import LifecycleCoordinator, LifecycleState
from src.utils import should_include_file
from src.worker.consumer import HueyWorker
from src.worker.process import (
    HueyWorkerProcess,
    _worker_status_path,
    is_expected_daemon_parent,
)
from src.cli_utils.validators import (
    validate_range,
    validate_timestamp_range,
)
from src.cli_utils.formatters import print_result_panel, print_debug_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100
_DAEMON_OVERVIEW_TIMEOUT_SECONDS = 1.0
_REBUILD_POLL_INTERVAL_SECONDS = 0.2
_GLOBAL_DAEMON_PROJECT_OPTION_HELP = (
    "Accepted for backward compatibility but ignored; daemon runtime is global "
    "and project is request metadata only."
)
_DAEMON_PENDING_READY_STATUSES = {"starting", "initializing"}


def _create_query_context(project: str | None) -> ApplicationContext:
    logging.getLogger().setLevel(logging.WARNING)
    return ApplicationContext.create(
        project_override=project,
        enable_watcher=False,
        lazy_embeddings=False,
    )


def _ignore_daemon_startup_project_option(project: str | None) -> None:
    """Keep legacy daemon --project options as explicit no-ops."""

    _ = project


def _ignore_daemon_runtime_root_option(runtime_root: Path | None) -> None:
    """Accept runtime-root markers used for daemon process identification."""

    _ = runtime_root


def _ensure_runtime_auto_registration(project_override: str | None) -> None:
    registration = ensure_runtime_project_registered(
        cwd=Path.cwd(),
        project_override=project_override,
    )
    if registration.changed:
        logger.info(
            "Auto-registered project '%s' at %s",
            registration.project_name,
            registration.project_path,
        )

        inspection = inspect_daemon()
        if inspection.running:
            logger.info("Restarting running daemon to load auto-registered corpus")
            restart_daemon(
                cwd=Path.cwd(),
                project_override=project_override,
            )


def _should_include_file(
    file_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
):
    return should_include_file(
        file_path, include_patterns, exclude_patterns, exclude_hidden_dirs
    )


def _create_daemon_runtime(runtime_paths: RuntimePaths):
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
    worker = HueyWorkerProcess(
        runtime_paths=runtime_paths,
    )
    return ctx, worker


def _parent_process_alive(
    parent_pid: int,
    parent_start_time: int | None = None,
) -> bool:
    return is_expected_daemon_parent(parent_pid, parent_start_time)


async def _run_worker_forever_async(
    project: str | None,
    queue_db: Path,
    index_root: Path,
    parent_pid: int,
    parent_start_time: int | None = None,
) -> None:
    worker_loop = asyncio.get_running_loop()

    def _schedule_worker_vocabulary_catch_up() -> bool:
        result: concurrent.futures.Future[bool] = concurrent.futures.Future()

        def _schedule() -> None:
            try:
                result.set_result(ctx.schedule_vocabulary_catch_up())
            except Exception as exc:  # pragma: no cover - defensive handoff
                result.set_exception(exc)

        worker_loop.call_soon_threadsafe(_schedule)
        return result.result(timeout=5.0)

    ctx = ApplicationContext.create(
        project_override=project,
        enable_watcher=True,
        lazy_embeddings=True,
        use_tasks=True,
        index_path_override=index_root,
        global_runtime=True,
    )
    try:
        ctx.index_manager.load()
    except Exception:
        logger.info("Worker runtime starting with fresh indices", exc_info=True)

    huey = get_huey(queue_db)
    register_tasks(
        huey,
        ctx.index_manager,
        commit_indexer=ctx.commit_indexer,
        task_backpressure_limit=ctx.config.indexing.task_backpressure_limit,
        bootstrap_index_path=ctx.index_path,
        bootstrap_documents_roots=ctx.documents_roots,
        schedule_vocabulary_catch_up=_schedule_worker_vocabulary_catch_up,
    )
    worker = HueyWorker(huey)

    stop_requested = False

    def _handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        worker.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    git_watcher = None
    if ctx.watcher is not None:
        try:
            ctx.watcher.start()
        except OSError as e:
            if e.errno != errno.EMFILE:
                raise
            logger.warning(
                "Worker file watcher disabled after hitting the file descriptor limit",
                exc_info=True,
            )
            await ctx.watcher.stop()

    if ctx.commit_indexer is not None and ctx.config.git_indexing.watch_enabled:
        from src.git.watcher import GitWatcher

        repos = await asyncio.to_thread(ctx.discover_git_repositories)
        if repos:
            git_watcher = GitWatcher(
                git_repos=repos,
                commit_indexer=ctx.commit_indexer,
                config=ctx.config,
                poll_interval=ctx.config.git_indexing.poll_interval_seconds,
                use_tasks=True,
            )
            git_watcher.start()

    worker_status_path = _worker_status_path(RuntimePaths.resolve())

    def _write_worker_status(status: str) -> None:
        worker_status_path.parent.mkdir(parents=True, exist_ok=True)
        worker_status_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "status": status,
                    "heartbeat": time.time(),
                }
            ),
            encoding="utf-8",
        )

    worker.start()
    _write_worker_status("ready")
    try:
        while worker.is_running and not stop_requested:
            if not _parent_process_alive(parent_pid, parent_start_time):
                worker.stop()
                break
            _write_worker_status("ready")
            await asyncio.sleep(0.2)
    finally:
        worker.stop()
        with contextlib.suppress(OSError):
            worker_status_path.unlink()
        if git_watcher is not None:
            await git_watcher.stop()
        if ctx.watcher is not None:
            await ctx.watcher.stop()


def _build_queue_status_payload(
    *,
    queue_path: Path,
    worker_running: bool,
    backpressure_limit: int | None = None,
) -> dict[str, object]:
    huey = get_huey(queue_path)
    stats = get_queue_stats(
        huey,
        worker_running=worker_running,
        backpressure_limit=backpressure_limit,
    )
    payload = stats.to_dict()
    payload["queue_db_path"] = str(queue_path)
    return payload


def _build_admin_overview_payload(
    ctx: ApplicationContext,
    *,
    runtime_paths: RuntimePaths,
    worker_running: bool,
    worker_pid: int | None,
    lifecycle: str,
) -> dict[str, object]:
    index_payload = _build_index_stats_payload(ctx)
    task_payload = _build_queue_status_payload(
        queue_path=runtime_paths.queue_db_path,
        worker_running=worker_running,
        backpressure_limit=ctx.config.indexing.task_backpressure_limit,
    )
    watcher_stats = ctx.watcher.get_stats().to_dict() if ctx.watcher else None
    return {
        "status": "ok",
        "pid": os.getpid(),
        "lifecycle": lifecycle,
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": len(ctx.documents_roots),
        "documents_roots": [str(root) for root in ctx.documents_roots],
        "worker_health": "healthy" if worker_running else "dead",
        "worker_pid": worker_pid,
        "socket_path": str(runtime_paths.socket_path),
        "endpoint": f"ipc://{runtime_paths.socket_path}",
        "index_db_path": str(runtime_paths.index_db_path),
        "queue_db_path": str(runtime_paths.queue_db_path),
        "indexed_documents": index_payload["indexed_documents"],
        "indexed_chunks": index_payload["indexed_chunks"],
        "git_commits": index_payload["git_commits"],
        "git_repositories": index_payload["git_repositories"],
        "pending_count": task_payload["pending_count"],
        "scheduled_count": task_payload["scheduled_count"],
        "running_count": task_payload["running_count"],
        "failed_count": task_payload["failed_count"],
        "worker_running": task_payload["worker_running"],
        "queue_stats": task_payload,
        "watcher_stats": watcher_stats,
        "index_state": ctx.get_index_state().to_dict(),
    }


def _render_initializing_search_response(
    console: Console,
    payload: dict[str, object],
    *,
    include_git_metadata: bool = False,
) -> None:
    lifecycle = str(payload.get("lifecycle", "unknown"))
    configured_root_count = payload.get("configured_root_count")
    index_state = payload.get("index_state", {})
    status = "unknown"
    indexed_count = 0
    total_count = 0
    if isinstance(index_state, dict):
        status = str(index_state.get("status", "unknown"))
        indexed_count = int(index_state.get("indexed_count", 0) or 0)
        total_count = int(index_state.get("total_count", 0) or 0)

    console.print("[yellow]Search service is initializing.[/yellow]")
    console.print(f"[dim]Lifecycle:[/dim] {lifecycle}")
    if isinstance(configured_root_count, int):
        console.print(f"[dim]Configured roots:[/dim] {configured_root_count}")
    console.print(
        f"[dim]Index state:[/dim] {status} ({indexed_count}/{total_count})"
    )
    if include_git_metadata:
        console.print(
            f"[dim]Total commits indexed:[/dim] {int(payload.get('total_commits_indexed', 0) or 0)}"
        )
    console.print("[dim]Results will appear once background initialization completes.[/dim]")


def _request_daemon_overview(
    inspection: DaemonInspection,
    *,
    runtime_paths: RuntimePaths,
) -> dict[str, object] | None:
    metadata = inspection.metadata
    if metadata is None or not inspection.running or not metadata.socket_path:
        return None

    response = request_daemon_socket(
        Path(metadata.socket_path),
        "/api/admin/overview",
        {},
        timeout_seconds=_DAEMON_OVERVIEW_TIMEOUT_SECONDS,
    )
    if response.get("status") == "error":
        return None
    return response


@click.group()
def cli():
    pass


@cli.command("worker-run", hidden=True)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option("--queue-db", type=click.Path(path_type=Path), required=True)
@click.option("--index-root", type=click.Path(path_type=Path), required=True)
@click.option("--parent-pid", type=int, required=True)
@click.option("--parent-start-time", type=int, default=None)
def worker_run(
    project: str | None,
    queue_db: Path,
    index_root: Path,
    parent_pid: int,
    parent_start_time: int | None,
):
    """Run the Huey task worker in a dedicated subprocess."""
    try:
        asyncio.run(
            _run_worker_forever_async(
                project,
                queue_db,
                index_root,
                parent_pid,
                parent_start_time,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Failed to run worker: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _apply_project_detection(config, project_override: str | None = None):
    from src.config import detect_project, resolve_index_path, resolve_documents_path

    detected_project = detect_project(
        projects=config.projects, project_override=project_override
    )
    index_path = resolve_index_path(config)

    explicit_documents_path: Path | None = None
    if project_override:
        override_path = Path(project_override).expanduser()
        if override_path.exists():
            explicit_documents_path = override_path.resolve()
        elif detected_project:
            for project in config.projects:
                if project.name == detected_project:
                    explicit_documents_path = Path(project.path).resolve()
                    break

    documents_path = (
        str(explicit_documents_path)
        if explicit_documents_path is not None
        else resolve_documents_path(config)
    )

    config.indexing.index_path = str(index_path)
    config.indexing.documents_path = documents_path
    config.detected_project = detected_project
    return config


def _resolve_rebuild_project_scope(
    *,
    project: str | None,
    all_projects: bool,
) -> str | None:
    if project is not None:
        return project

    return None


def _request_rebuild_submit_payload(
    *,
    project_override: str | None,
) -> dict[str, object]:
    payload = _request_daemon_json(
        "/api/admin/rebuild/submit",
        {"project": project_override},
        project_override=project_override,
        auto_start=True,
        allow_error=True,
    )
    if payload is None or payload.get("status") == "error":
        _raise_daemon_request_error(payload)
    return payload


def _request_rebuild_status_payload(
    *,
    project_override: str | None,
) -> dict[str, object]:
    payload = _request_daemon_json(
        "/api/admin/rebuild/status",
        {},
        project_override=project_override,
        auto_start=False,
        allow_error=True,
    )
    if payload is None or payload.get("status") == "error":
        _raise_daemon_request_error(payload)
    return payload


def _render_rebuild_messages(
    payload: dict[str, object],
    *,
    printed_count: int,
) -> int:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return printed_count

    normalized_messages = [item for item in messages if isinstance(item, str)]
    for message in normalized_messages[printed_count:]:
        click.echo(message)
    return len(normalized_messages)


@cli.command()
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def mcp(project: str | None):
    """Run MCP server with stdio transport (for VS Code integration)."""
    try:
        # Import here to avoid importing mcp when not needed
        from src.mcp import MCPServer

        # Create and run the server
        async def _run():
            server = MCPServer(project_override=project)
            await server.run()

        asyncio.run(_run())
    except KeyboardInterrupt:
        pass  # Graceful shutdown handled
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


async def _run_daemon_forever() -> None:
    lock = await asyncio.to_thread(acquire_boot_lock, timeout_seconds=5.0)
    lock_released = False
    runtime_paths = RuntimePaths.resolve()
    health_server_started = False
    coordinator = LifecycleCoordinator()
    loop = asyncio.get_running_loop()
    coordinator.install_signal_handlers(loop)

    ctx, huey_worker = await asyncio.to_thread(
        _create_daemon_runtime,
        runtime_paths,
    )
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
            get_worker_running=lambda: huey_worker.is_running,
            get_worker_pid=lambda: huey_worker.pid,
            build_admin_overview_payload=lambda current_ctx, runtime_root, worker_running, worker_pid, lifecycle: _build_admin_overview_payload(
                current_ctx,
                runtime_paths=RuntimePaths(
                    root=runtime_root,
                    index_db_path=runtime_paths.index_db_path,
                    queue_db_path=runtime_paths.queue_db_path,
                    metadata_path=runtime_paths.metadata_path,
                    lock_path=runtime_paths.lock_path,
                    socket_path=runtime_paths.socket_path,
                ),
                worker_running=worker_running,
                worker_pid=worker_pid,
                lifecycle=lifecycle,
            ),
            build_index_stats_payload=_build_index_stats_payload,
            build_queue_status_payload=_build_queue_status_payload,
        )
        ),
    )

    try:
        try:
            await coordinator.start(
                ctx,
                background_index=True,
                db_manager=ctx.db_manager,
                huey_worker=huey_worker,
            )
            await health_server.start()
            health_server_started = True
            await asyncio.to_thread(lock.release)
            lock_released = True
            while coordinator.state not in (
                LifecycleState.SHUTTING_DOWN,
                LifecycleState.TERMINATED,
            ):
                await asyncio.sleep(0.2)
        finally:
            await coordinator.shutdown()
            if health_server_started:
                await health_server.stop()
    finally:
        if not lock_released:
            await asyncio.to_thread(lock.release)


@cli.group("daemon")
def daemon_group():
    """Manage the long-lived Ragdocs daemon."""


@cli.group("queue")
def queue_group():
    """Inspect task queue state."""


@daemon_group.command("run")
@click.option(
    "--project",
    default=None,
    help=_GLOBAL_DAEMON_PROJECT_OPTION_HELP,
)
def daemon_run(project: str | None):
    """Run the daemon in the foreground."""
    try:
        ensure_runtime_project_registered(
            cwd=Path.cwd(),
            project_override=project,
        )
        _ignore_daemon_startup_project_option(project)
        asyncio.run(_run_daemon_forever())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Failed to run daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("daemon-internal-run", hidden=True)
@click.option(
    "--project",
    default=None,
    help=_GLOBAL_DAEMON_PROJECT_OPTION_HELP,
)
@click.option(
    "--runtime-root",
    type=click.Path(path_type=Path),
    default=None,
    hidden=True,
)
def daemon_internal_run(project: str | None, runtime_root: Path | None):
    """Run the daemon in the foreground for internal start/restart flows."""
    _ignore_daemon_startup_project_option(project)
    _ignore_daemon_runtime_root_option(runtime_root)
    daemon_run.callback(None)


@daemon_group.command("start")
@click.option(
    "--project",
    default=None,
    help=_GLOBAL_DAEMON_PROJECT_OPTION_HELP,
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    help="Seconds to wait for daemon metadata",
)
def daemon_start(project: str | None, timeout: float):
    """Start the daemon in the background."""
    try:
        _ignore_daemon_startup_project_option(project)
        metadata = start_daemon(
            cwd=Path.cwd(),
            project_override=project,
            timeout_seconds=timeout,
        )
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(_format_daemon_startup_result("started", metadata))


@daemon_group.command("status")
@click.option("--json", "output_json", is_flag=True, help="Output daemon status as JSON")
def daemon_status(output_json: bool):
    """Print current daemon status."""
    inspection = inspect_daemon()
    runtime_paths = RuntimePaths.resolve()
    if inspection.metadata is None:
        if output_json:
            click.echo(
                json.dumps(
                    {
                        "status": "not_running",
                        "metadata_path": str(runtime_paths.metadata_path),
                        "lock_path": str(runtime_paths.lock_path),
                        "socket_path": str(runtime_paths.socket_path),
                    },
                    indent=2,
                )
            )
            return

        click.echo("Daemon status: not running")
        click.echo(f"Metadata path: {runtime_paths.metadata_path}")
        click.echo(f"Lock path: {runtime_paths.lock_path}")
        return

    metadata_ready = (
        inspection.metadata is not None
        and inspection.metadata.status in {"ready", "ready_primary", "ready_replica"}
    )
    state = (
        "running"
        if inspection.ready or (inspection.running and metadata_ready)
        else "starting" if inspection.running else "stale"
    )
    started_at = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(inspection.metadata.started_at)
    )

    payload = {
        "status": state,
        "pid": inspection.metadata.pid,
        "lifecycle": inspection.metadata.status,
        "daemon_scope": inspection.metadata.daemon_scope,
        "started_at": started_at,
        "metadata_path": str(runtime_paths.metadata_path),
        "lock_path": str(runtime_paths.lock_path),
        "socket_path": inspection.metadata.socket_path
        or str(runtime_paths.socket_path),
        "index_db_path": inspection.metadata.index_db_path,
        "queue_db_path": inspection.metadata.queue_db_path,
        "endpoint": inspection.metadata.transport_endpoint,
    }

    overview = _request_daemon_overview(inspection, runtime_paths=runtime_paths)
    if overview is not None:
        payload.update(
            {
                key: overview[key]
                for key in (
                    "indexed_documents",
                    "indexed_chunks",
                    "git_commits",
                    "git_repositories",
                    "worker_health",
                    "worker_pid",
                    "pending_count",
                    "scheduled_count",
                    "running_count",
                    "failed_count",
                    "worker_running",
                    "configured_root_count",
                    "documents_roots",
                    "project_context_mode",
                )
                if key in overview
            }
        )

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"Daemon status: {state}")
    click.echo(f"PID: {inspection.metadata.pid}")
    click.echo(f"Lifecycle: {inspection.metadata.status}")
    click.echo(f"Scope: {payload['daemon_scope']}")
    click.echo(f"Started: {started_at}")
    click.echo(f"Metadata path: {runtime_paths.metadata_path}")
    click.echo(f"Lock path: {runtime_paths.lock_path}")
    if inspection.metadata.index_db_path:
        click.echo(f"Index DB: {inspection.metadata.index_db_path}")
    if inspection.metadata.queue_db_path:
        click.echo(f"Queue DB: {inspection.metadata.queue_db_path}")
    if inspection.metadata.transport_endpoint:
        click.echo(f"Endpoint: {inspection.metadata.transport_endpoint}")
    configured_root_count = payload.get("configured_root_count")
    if isinstance(configured_root_count, int):
        click.echo(f"Documents roots: {configured_root_count}")
    if "indexed_documents" in payload:
        click.echo(f"Indexed documents: {payload['indexed_documents']}")
        click.echo(f"Indexed chunks: {payload['indexed_chunks']}")
        click.echo(f"Indexed git commits: {payload['git_commits']}")
    if "pending_count" in payload:
        click.echo(f"Pending tasks: {payload['pending_count']}")
        click.echo(f"Failed tasks: {payload['failed_count']}")


@daemon_group.command("stop")
@click.option(
    "--timeout",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds to wait before forcing stop",
)
def daemon_stop(timeout: float):
    """Stop the daemon if it is running."""
    try:
        metadata = stop_daemon(timeout_seconds=timeout)
    except Exception as e:
        logger.error(f"Failed to stop daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if metadata is None:
        click.echo("Daemon status: not running")
        return

    click.echo(f"Daemon stopped (pid={metadata.pid})")


@daemon_group.command("restart")
@click.option(
    "--project",
    default=None,
    help=_GLOBAL_DAEMON_PROJECT_OPTION_HELP,
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    help="Seconds to wait for daemon metadata after restart",
)
def daemon_restart(project: str | None, timeout: float):
    """Restart the daemon."""
    try:
        _ignore_daemon_startup_project_option(project)
        metadata = restart_daemon(
            cwd=Path.cwd(),
            project_override=project,
            start_timeout_seconds=timeout,
        )
    except Exception as e:
        logger.error(f"Failed to restart daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(_format_daemon_startup_result("restarted", metadata))


def _format_daemon_startup_result(action: str, metadata: DaemonMetadata) -> str:
    if metadata.status in _DAEMON_PENDING_READY_STATUSES:
        return (
            f"Daemon {action} (pid={metadata.pid}, lifecycle={metadata.status}, "
            "socket readiness pending)"
        )

    return f"Daemon {action} (pid={metadata.pid}, lifecycle={metadata.status})"


@cli.group("index")
def index_group():
    """Inspect indexed corpus state."""


@index_group.command("stats")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option("--json", "output_json", is_flag=True, help="Output index stats as JSON")
def index_stats(project: str | None, output_json: bool):
    """Print document and git index statistics for the active project context."""
    try:
        payload = _request_daemon_json(
            "/api/admin/index",
            {},
            project_override=project,
            auto_start=False,
            allow_error=True,
        )
        if payload is None or payload.get("status") == "error":
            _raise_daemon_request_error(payload)

        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        click.echo("Index stats")
        click.echo(f"Common documents root: {payload['documents_common_root']}")
        documents_roots = payload.get("documents_roots", [])
        if isinstance(documents_roots, list):
            click.echo(f"Documents roots: {len(documents_roots)}")
            for root in documents_roots:
                click.echo(f"  - {root}")
        _render_index_stats_table(payload)
        click.echo(f"Index path: {payload['index_path']}")
        click.echo(f"Manifest present: {payload['manifest_exists']}")
        click.echo(f"Indexed documents: {payload['indexed_documents']}")
        click.echo(f"Indexed chunks: {payload['indexed_chunks']}")
        click.echo(f"Discovered files: {payload['discovered_files']}")
        click.echo(f"Remaining estimate: {payload['remaining_estimate']}")
        click.echo(f"Git repositories: {payload['git_repositories']}")
        click.echo(f"Indexed git commits: {payload['git_commits']}")
        index_state = payload.get("index_state")
        if isinstance(index_state, dict):
            click.echo(f"Index state: {index_state.get('status', 'unknown')}")
        watcher_stats = payload.get("watcher_stats")
        if isinstance(watcher_stats, dict):
            click.echo(
                f"Watcher events: {watcher_stats.get('events_received', 0)} received / {watcher_stats.get('events_processed', 0)} processed"
            )
    except Exception as e:
        logger.error(f"Failed to inspect index stats: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _build_index_stats_payload(ctx: ApplicationContext) -> dict[str, object]:
    manifest_path = ctx.index_path / "index.manifest.json"
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        try:
            ctx.index_manager.load()
        except TimeoutError as exc:
            if "Failed to acquire shared lock" not in str(exc):
                raise
            logger.warning(
                "Index stats refresh skipped after shared-lock timeout; using current in-memory snapshot",
                exc_info=True,
            )

    docs_root = Path(ctx.config.indexing.documents_path).resolve()
    discovered_files = ctx.discover_files() if docs_root.exists() else []
    repo_count = len(ctx.discover_git_repositories())
    git_commit_count = (
        ctx.commit_indexer.get_total_commits() if ctx.commit_indexer is not None else 0
    )

    indexed_documents = 0
    indexed_chunks = 0
    if manifest_exists:
        indexed_descriptions = ctx.index_manager.vector.describe_documents()
        indexed_documents = len(indexed_descriptions)
        indexed_chunks = sum(
            int(description.get("chunk_count", 0) or 0)
            for description in indexed_descriptions
        )

    per_root_rows, unattributed_indexed_documents, unattributed_indexed_chunks = (
        _build_per_root_index_rows(
            ctx,
            discovered_files=discovered_files,
            common_root=docs_root,
            include_indexed_estimates=manifest_exists,
        )
    )
    remaining_estimate = max(len(discovered_files) - indexed_documents, 0)

    return {
        "documents_path": str(docs_root),
        "documents_common_root": str(docs_root),
        "documents_path_kind": "common_root",
        "documents_roots": [str(root) for root in ctx.documents_roots],
        "index_path": str(ctx.index_path),
        "index_db_path": str(ctx.index_path / "index.db"),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "indexed_documents": indexed_documents,
        "indexed_chunks": indexed_chunks,
        "discovered_files": len(discovered_files),
        "remaining_estimate": remaining_estimate,
        "per_root": per_root_rows,
        "per_root_counts_are_estimates": True,
        "unattributed_indexed_documents": unattributed_indexed_documents,
        "unattributed_indexed_chunks": unattributed_indexed_chunks,
        "git_commits": git_commit_count,
        "git_repositories": repo_count,
        "index_state": ctx.get_index_state().to_dict(),
        "watcher_stats": ctx.watcher.get_stats().to_dict() if ctx.watcher else None,
    }


@queue_group.command("status")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option("--json", "output_json", is_flag=True, help="Output queue stats as JSON")
def queue_status(project: str | None, output_json: bool):
    """Print queue depth and recent task failures."""
    try:
        payload = _request_daemon_json(
            "/api/admin/tasks",
            {},
            project_override=project,
            auto_start=False,
            allow_error=True,
        )
        if payload is None or payload.get("status") == "error":
            _raise_daemon_request_error(payload)

        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        click.echo("Queue status")
        click.echo(f"Queue DB: {payload['queue_db_path']}")
        click.echo(f"Pending tasks: {payload['pending_count']}")
        click.echo(f"Scheduled tasks: {payload['scheduled_count']}")
        click.echo(f"Running tasks: {payload['running_count']}")
        click.echo(f"Failed tasks: {payload['failed_count']}")
        click.echo(f"Worker running: {'yes' if payload['worker_running'] else 'no'}")
        if payload.get("backpressure_limit") is not None:
            click.echo(f"Backpressure limit: {payload['backpressure_limit']}")
        if payload.get("backpressure_utilization") is not None:
            click.echo(
                f"Backpressure utilization: {float(payload['backpressure_utilization']):.2f}"
            )

        task_counts = payload.get("task_counts", {})
        if isinstance(task_counts, dict) and task_counts:
            click.echo("Task counts:")
            for task_name, count in task_counts.items():
                click.echo(f"  {task_name}: {count}")

        failures = payload.get("recent_failures", [])
        if isinstance(failures, list) and failures:
            click.echo("Recent failures:")
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                task_name = failure.get("task_name") or "unknown"
                click.echo(
                    f"  {task_name} ({failure.get('task_id', 'unknown')}): {failure.get('error', 'unknown error')}"
                )
        else:
            click.echo("Recent failures: none")
    except Exception as e:
        logger.error(f"Failed to inspect queue status: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _request_daemon_json(
    path: str,
    payload: dict[str, object],
    *,
    project_override: str | None,
    auto_start: bool,
    allow_error: bool = False,
) -> dict[str, object] | None:
    runtime_paths = RuntimePaths.resolve()
    response: dict[str, object] | None = None

    if auto_start:
        metadata = start_daemon(
            cwd=Path.cwd(),
            project_override=project_override,
            paths=runtime_paths,
        )
        if metadata.status in _DAEMON_PENDING_READY_STATUSES:
            metadata = wait_for_daemon_ready(paths=runtime_paths)
    else:
        inspection = inspect_daemon(runtime_paths)
        metadata = inspection.metadata if inspection.running else None

    if metadata is not None and metadata.socket_path:
        response = request_daemon_socket(
            Path(metadata.socket_path),
            path,
            payload,
            timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
        )
        if not _should_retry_daemon_request(response) or not auto_start:
            if response.get("status") == "error" and not allow_error:
                return None
            return response

    if response is None:
        return None

    if response.get("status") == "error" and not allow_error:
        return None
    return response


def _should_retry_daemon_request(response: dict[str, object]) -> bool:
    return response.get("status") == "error" and response.get("error") in {
        "daemon_request_timed_out",
        "daemon_socket_unavailable",
        "empty_response",
        "invalid_response",
    }


def _raise_daemon_request_error(response: dict[str, object] | None) -> None:
    if response is None:
        raise RuntimeError(
            "Daemon unavailable. Start it with 'ragdocs daemon start' and retry."
        )

    error = str(response.get("error", "unknown_error"))
    details = response.get("details")

    if error == "git_indexing_unavailable":
        raise RuntimeError(
            "Git history search is not available (git binary not found or disabled in config)"
        )

    if error == "daemon_request_timed_out":
        raise RuntimeError(
            "Daemon request timed out while waiting for a response. The daemon may still be initializing or performing a long-running operation."
        )

    if isinstance(details, str) and details:
        raise RuntimeError(details)

    raise RuntimeError(f"Daemon request failed: {error}")


def _resolve_stats_file_path(
    file_path: str | None,
    *,
    common_root: Path,
) -> Path | None:
    if not file_path:
        return None

    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    return (common_root / candidate).resolve()


def _match_documents_root(
    file_path: Path,
    documents_roots: list[Path],
) -> int | None:
    for index, root in enumerate(documents_roots):
        try:
            file_path.relative_to(root)
            return index
        except ValueError:
            continue

    return None


def _build_per_root_index_rows(
    ctx: ApplicationContext,
    *,
    discovered_files: list[str],
    common_root: Path,
    include_indexed_estimates: bool,
) -> tuple[list[dict[str, object]], int, int]:
    documents_roots = [root.resolve() for root in ctx.documents_roots]
    rows = [
        {
            "root_path": str(root),
            "discovered_files": 0,
            "indexed_documents_estimate": 0,
            "indexed_chunks_estimate": 0,
            "remaining_estimate": 0,
        }
        for root in documents_roots
    ]

    for discovered_file in discovered_files:
        resolved_path = Path(discovered_file).expanduser().resolve()
        root_index = _match_documents_root(resolved_path, documents_roots)
        if root_index is None:
            continue
        rows[root_index]["discovered_files"] += 1

    unattributed_indexed_documents = 0
    unattributed_indexed_chunks = 0
    if include_indexed_estimates:
        for description in ctx.index_manager.vector.describe_documents():
            raw_file_path = description.get("file_path")
            resolved_path = _resolve_stats_file_path(
                raw_file_path if isinstance(raw_file_path, str) else None,
                common_root=common_root,
            )
            chunk_count = int(description.get("chunk_count", 0) or 0)
            if resolved_path is None:
                unattributed_indexed_documents += 1
                unattributed_indexed_chunks += chunk_count
                continue

            root_index = _match_documents_root(resolved_path, documents_roots)
            if root_index is None:
                unattributed_indexed_documents += 1
                unattributed_indexed_chunks += chunk_count
                continue

            rows[root_index]["indexed_documents_estimate"] += 1
            rows[root_index]["indexed_chunks_estimate"] += chunk_count

    for row in rows:
        row["remaining_estimate"] = max(
            int(row["discovered_files"]) - int(row["indexed_documents_estimate"]),
            0,
        )

    return rows, unattributed_indexed_documents, unattributed_indexed_chunks


def _render_index_stats_table(payload: dict[str, object]) -> None:
    console = Console()
    per_root_rows = payload.get("per_root")
    if not isinstance(per_root_rows, list):
        per_root_rows = []

    table = Table(title="Indexed corpus by root", show_footer=True)
    table.add_column("Root", style="cyan", footer="Total")
    table.add_column(
        "Discovered",
        justify="right",
        footer=str(int(payload.get("discovered_files", 0) or 0)),
    )
    table.add_column(
        "Indexed docs≈",
        justify="right",
        footer=str(int(payload.get("indexed_documents", 0) or 0)),
    )
    table.add_column(
        "Indexed chunks≈",
        justify="right",
        footer=str(int(payload.get("indexed_chunks", 0) or 0)),
    )
    table.add_column(
        "Remaining≈",
        justify="right",
        footer=str(int(payload.get("remaining_estimate", 0) or 0)),
    )

    for row in per_root_rows:
        if not isinstance(row, dict):
            continue
        table.add_row(
            str(row.get("root_path", "(unknown)")),
            str(int(row.get("discovered_files", 0) or 0)),
            str(int(row.get("indexed_documents_estimate", 0) or 0)),
            str(int(row.get("indexed_chunks_estimate", 0) or 0)),
            str(int(row.get("remaining_estimate", 0) or 0)),
        )

    caption_parts = []
    if payload.get("per_root_counts_are_estimates"):
        caption_parts.append(
            "≈ per-root indexed counts are estimated from indexed file paths; aggregate indexed totals remain exact."
        )
    unattributed_documents = int(payload.get("unattributed_indexed_documents", 0) or 0)
    unattributed_chunks = int(payload.get("unattributed_indexed_chunks", 0) or 0)
    if unattributed_documents > 0 or unattributed_chunks > 0:
        caption_parts.append(
            f"Unattributed indexed items: {unattributed_documents} docs / {unattributed_chunks} chunks."
        )
    if caption_parts:
        table.caption = " ".join(caption_parts)

    console.print(table)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option(
    "--port", default=8000, type=int, show_default=True, help="Port to bind to"
)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def run(host: str, port: int, project: str | None):
    try:
        _ensure_runtime_auto_registration(project)
        config = load_config()
        config = _apply_project_detection(config, project)

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            "src.server:create_app",
            host=host,
            port=port,
            factory=True,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


@cli.command("rebuild-index")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option(
    "--all-projects",
    is_flag=True,
    default=False,
    help="Rebuild the full global corpus instead of narrowing to the detected current project.",
)
def rebuild_index_cmd(project: str | None, all_projects: bool):
    try:
        if all_projects and project is not None:
            raise click.UsageError("--all-projects cannot be used with --project")

        _ensure_runtime_auto_registration(project)

        effective_project = _resolve_rebuild_project_scope(
            project=project,
            all_projects=all_projects,
        )
        submit_payload = _request_rebuild_submit_payload(
            project_override=effective_project,
        )
        if bool(submit_payload.get("already_running")):
            click.echo("ℹ️  Rebuild already in progress; attaching to daemon-owned status")

        printed_messages = 0
        while True:
            status_payload = _request_rebuild_status_payload(
                project_override=effective_project,
            )
            printed_messages = _render_rebuild_messages(
                status_payload,
                printed_count=printed_messages,
            )

            rebuild_status = str(status_payload.get("status", "idle"))
            if rebuild_status in REBUILD_TERMINAL_STATUSES:
                if rebuild_status != "succeeded":
                    raise RuntimeError(
                        str(status_payload.get("error", "Daemon rebuild failed"))
                    )
                return

            if rebuild_status not in REBUILD_ACTIVE_STATUSES:
                raise RuntimeError(
                    f"Unexpected daemon rebuild status: {rebuild_status}"
                )

            time.sleep(_REBUILD_POLL_INTERVAL_SECONDS)

    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("check-config")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def check_config_cmd(project: str | None):
    try:
        logger.info("Loading configuration")
        config = load_config()
        config = _apply_project_detection(config, project)

        console = Console()

        table = Table(title="Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Documents Path", config.indexing.documents_path)
        table.add_row("Index Path", config.indexing.index_path)

        if config.projects:
            table.add_row("", "")
            table.add_row(
                "[bold]Registered Projects[/bold]", f"{len(config.projects)} project(s)"
            )
            for proj in config.projects:
                table.add_row(f"  • {proj.name}", proj.path)

            from src.config import detect_project

            detected = detect_project(
                projects=config.projects, project_override=project
            )
            if detected:
                table.add_row("", "")
                override_indicator = " (via --project)" if project else ""
                table.add_row(
                    "[bold]Active Project[/bold]", f"✅ {detected}{override_indicator}"
                )
            else:
                table.add_row("", "")
                table.add_row(
                    "[bold]Active Project[/bold]",
                    "⚠️  None detected (using local index)",
                )

        if config.config_warnings:
            table.add_row("", "")
            table.add_row(
                "[bold yellow]Warnings[/bold yellow]",
                f"{len(config.config_warnings)} warning(s)",
            )
            for warning in config.config_warnings:
                table.add_row("  •", warning)

        table.add_row("", "")
        table.add_row("Semantic Weight", str(config.search.semantic_weight))
        table.add_row("Keyword Weight", str(config.search.keyword_weight))
        table.add_row("Recency Bias", str(config.search.recency_bias))

        table.add_row("", "")
        table.add_row("Embedding Model", config.llm.embedding_model)

        console.print(table)

        console.print("\n[bold green]✅ Configuration is valid[/bold green]")
        if config.config_warnings:
            console.print("[bold yellow]⚠️  Configuration warnings detected[/bold yellow]")

        index_path = Path(config.indexing.index_path)
        if index_path.exists():
            manifest_path = index_path / "index.manifest.json"
            if manifest_path.exists():
                console.print(f"📊 Index exists at: {index_path}")
            else:
                console.print(
                    f"⚠️  Index directory exists but no manifest found: {index_path}"
                )
        else:
            console.print(
                f"📭 No index found (will be created on first run): {index_path}"
            )

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        click.echo(f"❌ Configuration Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query_text")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--top-n", default=5, type=int, help="Maximum number of results (default: 5)"
)
@click.option("--debug", is_flag=True, help="Display intermediate search statistics")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option(
    "--project-filter",
    multiple=True,
    help="Explicitly restrict results to one or more project IDs",
)
def query(
    query_text: str,
    output_json: bool,
    top_n: int,
    debug: bool,
    project: str | None,
    project_filter: tuple[str, ...],
):
    try:
        console = Console()
        validate_range(top_n, MIN_TOP_N, MAX_TOP_N, "--top-n")

        daemon_payload = _request_daemon_json(
            "/api/search/query",
            {
                "query": query_text,
                "top_n": top_n,
                "project_filter": list(project_filter),
                "project_context": project,
            },
            project_override=project,
            auto_start=True,
            allow_error=True,
        )
        if daemon_payload is None or daemon_payload.get("status") == "error":
            _raise_daemon_request_error(daemon_payload)

        if output_json:
            click.echo(json.dumps(daemon_payload, indent=2))
            return

        if daemon_payload.get("status") == "initializing":
            _render_initializing_search_response(console, daemon_payload)
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")
        if debug:
            from src.models import CompressionStats, SearchStrategyStats

            strategy_stats = SearchStrategyStats(
                **daemon_payload.get("strategy_stats", {})
            )
            compression_stats = CompressionStats(
                **daemon_payload.get(
                    "compression_stats",
                    {
                        "original_count": 0,
                        "after_threshold": 0,
                        "after_content_dedup": 0,
                        "after_ngram_dedup": 0,
                        "after_dedup": 0,
                        "after_doc_limit": 0,
                        "clusters_merged": 0,
                    },
                )
            )
            print_debug_stats(
                console,
                strategy_stats,
                compression_stats,
                0.02,
            )

        results = daemon_payload.get("results", [])
        if results:
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")
            for idx, result in enumerate(results, 1):
                panel_content = [
                    f"[yellow]Document:[/yellow] {result.get('doc_id', '')}",
                    f"[magenta]Section:[/magenta] {result.get('header_path') or '(no section)'}",
                    f"[blue]File:[/blue] {result.get('file_path') or '(unknown)'}",
                    "",
                    result.get("content", ""),
                ]
                print_result_panel(
                    console,
                    idx,
                    float(result.get("score", 0.0)),
                    panel_content,
                    is_last=(idx == len(results)),
                )
        else:
            console.print("[yellow]No results found.[/yellow]")
        return

    except FileNotFoundError as e:
        logger.error(f"Indices not found: {e}")
        click.echo(
            "Error: No indices found. Run 'mcp-markdown-ragdocs rebuild-index' first.",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("search-commits")
@click.argument("query_text")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--top-n", default=5, type=int, help="Maximum number of results (default: 5)"
)
@click.option("--debug", is_flag=True, help="Display intermediate search statistics")
@click.option(
    "--files-glob",
    default=None,
    help="Glob pattern for file filtering (e.g., 'src/**/*.py')",
)
@click.option(
    "--after",
    "after_timestamp",
    default=None,
    type=int,
    help="Unix timestamp (lower bound)",
)
@click.option(
    "--before",
    "before_timestamp",
    default=None,
    type=int,
    help="Unix timestamp (upper bound)",
)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option(
    "--project-filter",
    multiple=True,
    help="Explicitly restrict results to one or more project IDs",
)
def search_commits(
    query_text: str,
    output_json: bool,
    top_n: int,
    debug: bool,
    files_glob: str | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
    project: str | None,
    project_filter: tuple[str, ...],
):
    """Search git commit history using natural language queries."""
    try:
        console = Console()
        validate_range(top_n, MIN_TOP_N, MAX_TOP_N, "--top-n")
        validate_timestamp_range(after_timestamp, before_timestamp)

        daemon_payload = _request_daemon_json(
            "/api/search/git-history",
            {
                "query": query_text,
                "top_n": top_n,
                "files_glob": files_glob,
                "after_timestamp": after_timestamp,
                "before_timestamp": before_timestamp,
                "project_filter": list(project_filter),
                "project_context": project,
            },
            project_override=project,
            auto_start=True,
            allow_error=True,
        )
        if daemon_payload is None or daemon_payload.get("status") == "error":
            _raise_daemon_request_error(daemon_payload)

        if output_json:
            click.echo(json.dumps(daemon_payload, indent=2))
            return

        if daemon_payload.get("status") == "initializing":
            _render_initializing_search_response(
                console,
                daemon_payload,
                include_git_metadata=True,
            )
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")
        console.print(
            f"[dim]Total commits indexed: {daemon_payload.get('total_commits_indexed', 0)}[/dim]\n"
        )
        results = daemon_payload.get("results", [])
        if results:
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")
            from datetime import datetime, timezone

            for idx, commit in enumerate(results, 1):
                commit_date = datetime.fromtimestamp(commit["timestamp"], timezone.utc)
                date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S UTC")
                panel_content = [
                    f"[yellow]Commit:[/yellow] {commit['hash'][:8]}",
                    f"[cyan]Author:[/cyan] {commit['author']}",
                    f"[blue]Date:[/blue] {date_str}",
                    "",
                    commit["title"],
                ]
                if len(commit.get("files_changed", [])) > 0:
                    panel_content.append("")
                    panel_content.append(
                        f"[magenta]Files Changed ({len(commit['files_changed'])}):[/magenta]"
                    )
                    for file_path in commit["files_changed"][:5]:
                        panel_content.append(f"  • {file_path}")
                print_result_panel(
                    console,
                    idx,
                    float(commit.get("score", 0.0)),
                    panel_content,
                    is_last=(idx == len(results)),
                )
        else:
            console.print("[yellow]No results found.[/yellow]")
        return

    except Exception as e:
        logger.error(f"Git commit search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
