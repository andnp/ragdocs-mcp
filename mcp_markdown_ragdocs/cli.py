import asyncio
import concurrent.futures
import contextlib
import errno
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

# Prevent tokenizers parallelism warning when forking worker process.
# Must be set before any HuggingFace/sentence-transformers imports.
# See: https://github.com/huggingface/tokenizers/issues/993
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Disable HuggingFace/tqdm progress bars to prevent stdout pollution in JSON output
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

_CLI_REEXEC_GUARD = "MCP_MARKDOWN_RAGDOCS_SKIP_REEXEC"


def _repo_venv_python() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def _should_reexec_into_repo_venv() -> bool:
    if os.environ.get(_CLI_REEXEC_GUARD) == "1":
        return False

    repo_python = _repo_venv_python()
    if not repo_python.exists():
        return False

    current_python = Path(sys.executable).resolve()
    return current_python != repo_python.resolve()


def _reexec_into_repo_venv() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_python = _repo_venv_python()
    env = os.environ.copy()
    env[_CLI_REEXEC_GUARD] = "1"

    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    argv = [str(repo_python), "-m", "mcp_markdown_ragdocs.cli", *sys.argv[1:]]
    os.execve(str(repo_python), argv, env)


if _should_reexec_into_repo_venv():
    _reexec_into_repo_venv()

from datetime import UTC
from typing import Any

import click
import uvicorn
from rich.console import Console
from rich.table import Table

from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads
from mcp_markdown_ragdocs.cli_utils.formatters import (
    _render_index_stats_table,
    _render_initializing_search_response,
    print_debug_stats,
    print_result_panel,
)
from mcp_markdown_ragdocs.cli_utils.project_context import (
    _apply_project_detection,
)
from mcp_markdown_ragdocs.cli_utils.queue_output import (
    _QUEUE_DETAIL_STATES,
    _coerce_dict,
    _coerce_int,
    _coerce_list_of_dicts,
    _emit_git_refresh_progress,
    _emit_queue_task_section,
    _filter_queue_status_payload,
)
from mcp_markdown_ragdocs.cli_utils.validators import (
    _should_include_file,  # noqa: F401 -- re-exported for tests.unit.test_file_filtering
    validate_range,
    validate_timestamp_range,
)
from mcp_markdown_ragdocs.config import load_config
from mcp_markdown_ragdocs.daemon import RuntimePaths
from mcp_markdown_ragdocs.daemon.admin_payloads import (
    _build_admin_overview_payload,
    _build_index_stats_payload,
    _build_queue_status_payload,
)
from mcp_markdown_ragdocs.daemon.client import (
    call_with_supported_kwargs,
    raise_daemon_request_error,
    request_daemon_json_with_dependencies,
)
from mcp_markdown_ragdocs.daemon.health import (
    request_daemon_socket,
)
from mcp_markdown_ragdocs.daemon.management import (
    acquire_boot_lock,
    inspect_daemon,
    restart_daemon,
    start_daemon,
    stop_daemon,
    wait_for_daemon_ready,
)
from mcp_markdown_ragdocs.daemon.rebuild_commands import run_rebuild_command
from mcp_markdown_ragdocs.daemon.status import (
    build_daemon_status_payload,
    format_daemon_startup_result,
    request_daemon_overview,
)
from mcp_markdown_ragdocs.lifecycle import LifecycleCoordinator, LifecycleState
from mcp_markdown_ragdocs.runtime_logging import configure_file_logging
from mcp_markdown_ragdocs.worker.process import (
    is_expected_daemon_parent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100
_GLOBAL_DAEMON_PROJECT_OPTION_HELP = (
    "Accepted for backward compatibility but ignored; daemon runtime is global "
    "and project is request metadata only."
)


def _ignore_daemon_startup_project_option(project: str | None) -> None:
    """Keep legacy daemon --project options as explicit no-ops."""

    _ = project


def _ignore_daemon_runtime_root_option(runtime_root: Path | None) -> None:
    """Accept runtime-root markers used for daemon process identification."""

    _ = runtime_root


def _parent_process_alive(
    parent_pid: int,
    parent_start_time: int | None = None,
) -> bool:
    return is_expected_daemon_parent(parent_pid, parent_start_time)


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


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
                result.set_result(False)
            except Exception as exc:  # noqa: BLE001 -- must forward any error to the awaiting future
                result.set_exception(exc)

        worker_loop.call_soon_threadsafe(_schedule)
        return result.result(timeout=5.0)

    from mcp_markdown_ragdocs.context import ApplicationContext
    from mcp_markdown_ragdocs.coordination.queue import build_queue_runtime
    from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
    from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
    from mcp_markdown_ragdocs.indexing.tasks import (
        TaskEmbeddingCacheAdapter,
        index_document_retry_policy,
        register_tasks,
    )
    from mcp_markdown_ragdocs.worker.consumer import HueyWorker

    config = load_config()
    configure_file_logging(index_root / "worker.log", config.logging)
    configure_runtime_threads(config)
    ctx = ApplicationContext.create(
        project_override=project,
        enable_watcher=True,
        lazy_embeddings=True,
        use_tasks=True,
        index_path_override=index_root,
        global_runtime=True,
    )
    services = getattr(ctx, "services", None)
    indexing: Any = getattr(services, "indexing", None) or ctx.index_manager
    task_target = (
        indexing.task_target()
        if hasattr(indexing, "task_target")
        else indexing
    )
    try:
        indexing.load()
    except Exception:
        logger.info("Worker runtime starting with fresh indices", exc_info=True)

    queue_runtime = build_queue_runtime(queue_db)
    huey = queue_runtime.huey
    task_lease_store = TaskLeaseStore(queue_db)
    work_intent_store = WorkIntentStore(queue_db, retry_policy=index_document_retry_policy)
    task_runtime = register_tasks(
        huey,
        task_target,
        task_lease_store,
        work_intent_store,
        config=config,
        task_backpressure_limit=ctx.config.indexing.task_backpressure_limit,
        embedding_cache_prune_cooldown_seconds=ctx.config.indexing.embedding_cache_prune_cooldown_seconds,
        bootstrap_index_path=ctx.index_path,
        bootstrap_documents_roots=ctx.documents_roots,
        schedule_vocabulary_catch_up=_schedule_worker_vocabulary_catch_up,
        embedding_cache=TaskEmbeddingCacheAdapter(
            indexing.ingestor.embedding_cache,
            indexing.storage.iter_records,
            indexing.ingestor.encoder_namespace or "",
            lambda: {
                "embedding_cache_hits": indexing.ingestor.embedding_cache.metrics.hits,
                "embedding_cache_misses": indexing.ingestor.embedding_cache.metrics.misses,
                "embedding_writes": indexing.ingestor.embedding_cache.metrics.writes,
                "embedding_invalidations": indexing.ingestor.embedding_cache.metrics.invalidations,
            },
        ),
    )
    ctx.attach_task_runtime(task_runtime)
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

    if ctx.git_indexing_enabled and ctx.config.git_indexing.watch_enabled:
        from mcp_markdown_ragdocs.git.watcher import GitWatcher

        repos = await asyncio.to_thread(ctx.discover_git_repositories)
        if repos:
            git_watcher = GitWatcher(
                git_repos=repos,
                index_manager=task_target,
                config=ctx.config,
                poll_interval=ctx.config.git_indexing.poll_interval_seconds,
                use_tasks=True,
                task_submission=task_runtime.submission,
            )
            git_watcher.start()

    worker_status_path = index_root / "worker.json"

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
        logger.exception("Failed to run worker")
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
    return request_daemon_json_with_dependencies(
        path,
        payload,
        project_override=project_override,
        auto_start=auto_start,
        allow_error=allow_error,
        runtime_paths_resolver=RuntimePaths.resolve,
        start_daemon_fn=start_daemon,
        wait_for_daemon_ready_fn=wait_for_daemon_ready,
        inspect_daemon_fn=inspect_daemon,
        request_daemon_socket_fn=request_daemon_socket,
        cwd_provider=Path.cwd,
    )


def _require_daemon_payload(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    if payload is None:
        raise_daemon_request_error(payload)
    assert payload is not None
    if payload.get("status") == "error":
        raise_daemon_request_error(payload)
    return payload


@cli.command()
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def mcp(project: str | None):
    """Run MCP server with stdio transport (for VS Code integration)."""
    try:
        # Import here to avoid importing mcp when not needed
        from mcp_markdown_ragdocs.mcp import MCPServer

        # Create and run the server
        async def _run():
            server = MCPServer(project_override=project)
            await server.run()

        asyncio.run(_run())
    except KeyboardInterrupt:
        pass  # Graceful shutdown handled
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


async def _run_daemon_forever() -> None:
    from mcp_markdown_ragdocs.daemon.runtime import create_daemon_runtime

    runtime_paths = RuntimePaths.resolve()
    config = load_config()
    configure_file_logging(runtime_paths.root / "daemon.log", config.logging)
    lock = await asyncio.to_thread(acquire_boot_lock, timeout_seconds=5.0)
    lock_released = False
    health_server_started = False
    coordinator = LifecycleCoordinator()
    loop = asyncio.get_running_loop()
    coordinator.install_signal_handlers(loop)

    runtime = await asyncio.to_thread(
        create_daemon_runtime,
        runtime_paths,
        coordinator=coordinator,
        build_admin_overview_payload=_build_admin_overview_payload,
        build_index_stats_payload=_build_index_stats_payload,
        build_queue_status_payload=_build_queue_status_payload,
    )
    ctx = runtime.ctx
    huey_worker = runtime.worker
    health_server = runtime.health_server

    try:
        try:
            # Bind transport before initialization so health and cold-start
            # search requests remain available while indices are loading.
            await health_server.start()
            health_server_started = True
            await coordinator.start(
                ctx,
                background_index=True,
                db_manager=ctx.db_manager,
                huey_worker=huey_worker,
            )
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


@cli.group("records")
def records_group():
    """Maintain indexed records directly (bypassing normal ingestion)."""


@daemon_group.command("run")
@click.option(
    "--project",
    default=None,
    help=_GLOBAL_DAEMON_PROJECT_OPTION_HELP,
)
def daemon_run(project: str | None):
    """Run the daemon in the foreground."""
    try:
        _ignore_daemon_startup_project_option(project)
        asyncio.run(_run_daemon_forever())
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
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
    callback = daemon_run.callback
    if callback is None:
        raise RuntimeError("daemon run callback is unavailable")
    callback(None)


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
        metadata = call_with_supported_kwargs(
            start_daemon,
            cwd=Path.cwd(),
            project_override=project,
            timeout_seconds=timeout,
        )
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to start daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(format_daemon_startup_result("started", metadata))


@daemon_group.command("status")
@click.option("--json", "output_json", is_flag=True, help="Output daemon status as JSON")
def daemon_status(output_json: bool):
    """Print current daemon status."""
    inspection = inspect_daemon()
    runtime_paths = RuntimePaths.resolve()
    overview = request_daemon_overview(inspection, runtime_paths=runtime_paths)
    payload = build_daemon_status_payload(
        inspection,
        runtime_paths=runtime_paths,
        overview=overview,
    )

    if payload["status"] == "not_running":
        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        click.echo("Daemon status: not running")
        click.echo(f"Metadata path: {runtime_paths.metadata_path}")
        click.echo(f"Lock path: {runtime_paths.lock_path}")
        return

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"Daemon status: {payload['status']}")
    click.echo(f"PID: {payload['pid']}")
    click.echo(f"Lifecycle: {payload['lifecycle']}")
    click.echo(f"Scope: {payload['daemon_scope']}")
    click.echo(f"Started: {payload['started_at']}")
    click.echo(f"Metadata path: {runtime_paths.metadata_path}")
    click.echo(f"Lock path: {runtime_paths.lock_path}")
    if payload.get("index_db_path"):
        click.echo(f"Index DB: {payload['index_db_path']}")
    if payload.get("queue_db_path"):
        click.echo(f"Queue DB: {payload['queue_db_path']}")
    if payload.get("endpoint"):
        click.echo(f"Endpoint: {payload['endpoint']}")
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
    _emit_git_refresh_progress(payload.get("git_refresh_progress"))


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
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
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
        metadata = call_with_supported_kwargs(
            restart_daemon,
            cwd=Path.cwd(),
            project_override=project,
            start_timeout_seconds=timeout,
        )
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to restart daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(format_daemon_startup_result("restarted", metadata))


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
        payload = _require_daemon_payload(payload)

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
        if "pending_count" in payload:
            click.echo(f"Pending tasks: {payload['pending_count']}")
            click.echo(f"Running tasks: {payload['running_count']}")
            click.echo(f"Failed tasks: {payload['failed_count']}")
        index_state = payload.get("index_state")
        if isinstance(index_state, dict):
            click.echo(f"Index state: {index_state.get('status', 'unknown')}")
        watcher_stats = payload.get("watcher_stats")
        if isinstance(watcher_stats, dict):
            click.echo(
                f"Watcher events: {watcher_stats.get('events_received', 0)} received / {watcher_stats.get('events_processed', 0)} processed"
            )
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to inspect index stats: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@index_group.command("reindex")
@click.argument(
    "operation",
    required=False,
    type=click.Choice(["start", "status", "rollback", "contract"]),
    default="start",
)
@click.option(
    "--model",
    required=False,
    help="Target embedding model name (e.g., 'Qwen3-Embedding-0.6B')",
)
@click.option(
    "--truncate-dim",
    type=int,
    default=None,
    help="Optional dimension truncation (must be <= target model dim)",
)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option(
    "--old-model",
    default=None,
    help="Expected source model for contract or rollback",
)
@click.option("--status", "status_flag", is_flag=True, help="Show migration status")
@click.option("--rollback", "rollback_flag", is_flag=True, help="Roll back migration")
@click.option("--contract", "contract_flag", is_flag=True, help="Delete the old model")
@click.option("--json", "output_json", is_flag=True, help="Output JSON")
def reindex_cmd(
    operation: str,
    model: str | None,
    truncate_dim: int | None,
    project: str | None,
    old_model: str | None,
    status_flag: bool,
    rollback_flag: bool,
    contract_flag: bool,
    output_json: bool,
):
    """Migrate embeddings to a new model without data loss.

    START queues expand, backfill, validate, and flip in the daemon worker.
    STATUS reads durable progress. ROLLBACK restores the old model and
    CONTRACT removes the old model after validation.
    """
    try:
        flags = [status_flag, rollback_flag, contract_flag]
        if sum(flags) > 1:
            raise click.UsageError("choose only one of --status, --rollback, or --contract")
        if any(flags) and operation != "start":
            raise click.UsageError(
                "operation argument cannot be combined with an operation flag"
            )
        if status_flag:
            operation = "status"
        elif rollback_flag:
            operation = "rollback"
        elif contract_flag:
            operation = "contract"

        if operation == "status":
            payload = _request_daemon_json(
                "/api/admin/reindex/status",
                {},
                project_override=project,
                auto_start=False,
                allow_error=True,
            )
        else:
            if operation == "start" and not model:
                raise click.UsageError("--model is required for reindex start")
            payload = _request_daemon_json(
                "/api/admin/reindex/submit",
                {
                    "operation": operation,
                    "model": model,
                    "truncate_dim": truncate_dim,
                    "old_model": old_model,
                },
                project_override=project,
                auto_start=True,
                allow_error=True,
            )
        payload = _require_daemon_payload(payload)

        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        if operation == "status":
            click.echo("Reindex status")
            click.echo(f"State: {payload.get('status', 'idle')}")
            click.echo(f"Phase: {payload.get('phase', 'idle')}")
            click.echo(f"Checkpoint: {payload.get('checkpoint', 0)}/{payload.get('total_records', 0)}")
            if payload.get("error"):
                click.echo(f"Error: {payload['error']}")
            return

        queued = payload.get("reindex", payload)
        if isinstance(queued, dict):
            click.echo(
                f"Reindex {operation} queued "
                f"(request_id={queued.get('request_id', 'unknown')})"
            )
        else:
            click.echo(f"Reindex {operation} queued")

    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to initiate reindex: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@queue_group.command("status")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option(
    "--state",
    type=click.Choice(_QUEUE_DETAIL_STATES, case_sensitive=False),
    default="all",
    show_default=True,
    help="Filter task detail output by queue state.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of detail entries to render per selected section.",
)
@click.option(
    "--details",
    is_flag=True,
    help="Render expanded task detail fields in human output.",
)
@click.option("--json", "output_json", is_flag=True, help="Output queue stats as JSON")
def queue_status(
    project: str | None,
    state: str,
    limit: int | None,
    details: bool,
    output_json: bool,
):
    """Print queue depth and recent task failures."""
    try:
        payload = _request_daemon_json(
            "/api/admin/tasks",
            {},
            project_override=project,
            auto_start=False,
            allow_error=True,
        )
        payload = _require_daemon_payload(payload)

        filtered_payload = _filter_queue_status_payload(
            payload,
            state=state.lower(),
            limit=limit,
        )

        if output_json:
            click.echo(json.dumps(filtered_payload, indent=2))
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
                f"Backpressure utilization: {float(payload['backpressure_utilization']) if isinstance(payload['backpressure_utilization'], (int, float)) else 0.0:.2f}"
            )
        _emit_git_refresh_progress(payload.get("git_refresh_progress"))

        task_counts = payload.get("task_counts", {})
        if isinstance(task_counts, dict) and task_counts:
            click.echo("Task counts:")
            for task_name, count in task_counts.items():
                click.echo(f"  {task_name}: {count}")

        pending_tasks = _coerce_list_of_dicts(filtered_payload.get("pending_tasks"))
        _emit_queue_task_section(
            "Pending task detail:",
            pending_tasks,
            details=details,
        )

        scheduled_tasks = _coerce_list_of_dicts(filtered_payload.get("scheduled_tasks"))
        _emit_queue_task_section(
            "Scheduled task detail:",
            scheduled_tasks,
            details=details,
        )

        failures = filtered_payload.get("recent_failures", [])
        if isinstance(failures, list) and failures:
            click.echo("Recent failures:")
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                task_name = failure.get("task_name") or "unknown"
                click.echo(
                    f"  {task_name} ({failure.get('task_id', 'unknown')}): {failure.get('error', 'unknown error')}"
                )
        elif state.lower() in ("all", "failed"):
            click.echo("Recent failures: none")
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to inspect queue status: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@queue_group.command("purge")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option(
    "--state",
    type=click.Choice(_QUEUE_DETAIL_STATES, case_sensitive=False),
    default="all",
    show_default=True,
    help="Select which queue state to purge.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Confirm the destructive purge operation.",
)
@click.option("--json", "output_json", is_flag=True, help="Output purge result as JSON")
def queue_purge(
    project: str | None,
    state: str,
    yes: bool,
    output_json: bool,
):
    """Purge daemon-owned queue state for explicit admin recovery."""
    if not yes:
        raise click.UsageError("Refusing to purge queue state without --yes.")

    try:
        payload = _request_daemon_json(
            "/api/admin/tasks/purge",
            {
                "state": state.lower(),
                "confirm": True,
            },
            project_override=project,
            auto_start=False,
            allow_error=True,
        )
        payload = _require_daemon_payload(payload)

        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        click.echo(f"Queue purge complete ({payload['purged_state']})")
        purged_counts = _coerce_dict(payload.get("purged_counts"))
        click.echo(
            "Purged tasks: "
            f"pending={_coerce_int(purged_counts.get('pending'))}, "
            f"scheduled={_coerce_int(purged_counts.get('scheduled'))}, "
            f"failed={_coerce_int(purged_counts.get('failed'))}"
        )
        click.echo(f"Queue DB: {payload['queue_db_path']}")
        click.echo(f"Pending tasks: {payload['pending_count']}")
        click.echo(f"Scheduled tasks: {payload['scheduled_count']}")
        click.echo(f"Failed tasks: {payload['failed_count']}")
        click.echo(f"Worker running: {'yes' if payload['worker_running'] else 'no'}")
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to purge queue state: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@records_group.command("purge")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option("--workspace", required=True, help="Workspace/project id to match.")
@click.option(
    "--source-kind", required=True, help="Record source_kind to match (e.g. git_commit)."
)
@click.option(
    "--yes",
    is_flag=True,
    help="Actually delete. Without this, only a matching-record count is shown.",
)
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON")
def records_purge(
    project: str | None,
    workspace: str,
    source_kind: str,
    yes: bool,
    output_json: bool,
):
    """Delete records matching a workspace and source_kind.

    Without --yes, reports how many records would be deleted and deletes
    nothing.
    """
    try:
        payload = _request_daemon_json(
            "/api/admin/records/purge",
            {
                "workspace_id": workspace,
                "source_kind": source_kind,
                "confirm": yes,
            },
            project_override=project,
            auto_start=False,
            allow_error=True,
        )
        payload = _require_daemon_payload(payload)

        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        if yes:
            click.echo(f"Deleted {payload['deleted']} record(s).")
        else:
            click.echo(
                f"Would delete {payload['would_delete']} record(s) "
                f"(workspace={workspace}, source_kind={source_kind}). "
                "Pass --yes to delete."
            )
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to purge records: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@records_group.command("vacuum-enable")
@click.option(
    "--yes",
    is_flag=True,
    help="Confirm this one-time, disk-rewriting migration.",
)
def records_vacuum_enable(yes: bool):
    """One-time migration: enable auto_vacuum=INCREMENTAL and reclaim space now.

    Rewrites the whole database file via VACUUM, so the daemon must be
    stopped first. Only after this has run does the periodic incremental
    vacuum have anything to reclaim.
    """
    if not yes:
        raise click.UsageError("Refusing to vacuum without --yes.")

    runtime_paths = RuntimePaths.resolve()
    if inspect_daemon(runtime_paths).running:
        click.echo(
            "Error: stop the daemon first (`mcp-markdown-ragdocs daemon stop`).",
            err=True,
        )
        sys.exit(1)

    index_db_path = runtime_paths.index_db_path
    if not index_db_path.exists():
        click.echo(f"Error: index database not found at {index_db_path}", err=True)
        sys.exit(1)

    before_size = index_db_path.stat().st_size
    connection = sqlite3.connect(str(index_db_path))
    try:
        current_mode = connection.execute("PRAGMA auto_vacuum").fetchone()[0]
        connection.execute("PRAGMA auto_vacuum = INCREMENTAL")
        connection.execute("VACUUM")
    finally:
        connection.close()
    after_size = index_db_path.stat().st_size

    click.echo(f"auto_vacuum: {current_mode} -> 2 (incremental)")
    click.echo(f"Database size: {before_size:,} -> {after_size:,} bytes")


@records_group.command("prune-old-git-diffs")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
@click.option("--workspace", required=True, help="Workspace/project id to match.")
@click.option(
    "--yes",
    is_flag=True,
    help="Actually delete. Without this, only a matching-record count is shown.",
)
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON")
def records_prune_old_git_diffs(
    project: str | None,
    workspace: str,
    yes: bool,
    output_json: bool,
):
    """Delete diff chunks from commits older than the configured age window.

    This is recurring maintenance, not a one-time migration: commits keep
    aging past git_indexing.git_diff_embedding_days continuously, so old
    diffs keep accumulating again. Without --yes, reports how many chunks
    would be deleted and deletes nothing.
    """
    try:
        payload = _request_daemon_json(
            "/api/admin/records/prune-old-git-diffs",
            {
                "workspace_id": workspace,
                "confirm": yes,
            },
            project_override=project,
            auto_start=False,
            allow_error=True,
        )
        payload = _require_daemon_payload(payload)

        if output_json:
            click.echo(json.dumps(payload, indent=2))
            return

        if yes:
            click.echo(f"Deleted {payload['deleted']} old diff chunk(s).")
        else:
            click.echo(
                f"Would delete {payload['would_delete']} old diff chunk(s) "
                f"(workspace={workspace}, max_age_days={payload['max_age_days']}). "
                "Pass --yes to delete."
            )
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to prune old git diffs: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
        config = load_config()
        config = _apply_project_detection(config, project)

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            "mcp_markdown_ragdocs.server:create_app",
            host=host,
            port=port,
            factory=True,
        )
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
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
        run_rebuild_command(
            project=project,
            all_projects=all_projects,
            emit=click.echo,
        )

    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
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

            from mcp_markdown_ragdocs.config import detect_project

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
        table.add_row(
            "Legacy Search Policy",
            f"{len(config.search.deprecated_policy_fields())} settings ignored",
        )

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

    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Failed to load configuration: {e}")
        click.echo(f"❌ Configuration Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query_text")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--top-n", default=5, type=int, help="Maximum number of results (default: 5)"
)
@click.option(
    "--min-score",
    default=None,
    type=float,
    help="Optional raw RRF score floor; calibrate before relying on abstention",
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
    min_score: float | None,
    debug: bool,
    project: str | None,
    project_filter: tuple[str, ...],
):
    try:
        console = Console()
        validate_range(top_n, MIN_TOP_N, MAX_TOP_N, "--top-n")
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError("--min-score must be between 0.0 and 1.0")

        request_payload: dict[str, object] = {
            "query": query_text,
            "top_n": top_n,
            "project_filter": list(project_filter),
            "project_context": project,
        }
        if min_score is not None:
            request_payload["min_score"] = min_score
        daemon_payload = _request_daemon_json(
            "/api/search/query",
            request_payload,
            project_override=project,
            auto_start=True,
            allow_error=True,
        )
        daemon_payload = _require_daemon_payload(daemon_payload)

        if output_json:
            click.echo(json.dumps(daemon_payload, indent=2))
            return

        if daemon_payload.get("status") == "initializing":
            _render_initializing_search_response(console, daemon_payload)
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")
        if debug:
            from mcp_markdown_ragdocs.models import (
                CompressionStats,
                SearchStrategyStats,
            )

            strategy_payload = _coerce_dict(daemon_payload.get("strategy_stats"))
            compression_payload = _coerce_dict(
                daemon_payload.get("compression_stats")
            )
            strategy_stats = SearchStrategyStats(
                vector_count=_coerce_int(strategy_payload.get("vector_count")),
                keyword_count=_coerce_int(strategy_payload.get("keyword_count")),
                graph_count=_coerce_int(strategy_payload.get("graph_count")),
                tag_expansion_count=_coerce_int(
                    strategy_payload.get("tag_expansion_count")
                ),
            )
            compression_stats = CompressionStats(
                original_count=_coerce_int(compression_payload.get("original_count")),
                after_threshold=_coerce_int(
                    compression_payload.get("after_threshold")
                ),
                after_content_dedup=_coerce_int(
                    compression_payload.get("after_content_dedup")
                ),
                after_ngram_dedup=_coerce_int(
                    compression_payload.get("after_ngram_dedup")
                ),
                after_dedup=_coerce_int(compression_payload.get("after_dedup")),
                after_doc_limit=_coerce_int(
                    compression_payload.get("after_doc_limit")
                ),
                clusters_merged=_coerce_int(
                    compression_payload.get("clusters_merged")
                ),
            )
            query_stats = {
                key: value
                for key, value in _coerce_dict(
                    daemon_payload.get("query_execution_stats")
                ).items()
                if isinstance(value, (int, float))
            }
            print_debug_stats(
                console,
                strategy_stats,
                compression_stats,
                0.02,
                query_stats,
            )

        results = _coerce_list_of_dicts(daemon_payload.get("results"))
        if results:
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")
            for idx, result in enumerate(results, 1):
                panel_content: list[str] = [
                    f"[yellow]Document:[/yellow] {result.get('doc_id', '')}",
                    f"[magenta]Section:[/magenta] {result.get('header_path') or '(no section)'}",
                    f"[blue]File:[/blue] {result.get('file_path') or '(unknown)'}",
                    "",
                    _as_text(result.get("content")),
                ]
                print_result_panel(
                    console,
                    idx,
                    _as_float(result.get("score")),
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
    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
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
        daemon_payload = _require_daemon_payload(daemon_payload)

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
        results = _coerce_list_of_dicts(daemon_payload.get("results"))
        if results:
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")
            from datetime import datetime

            for idx, commit in enumerate(results, 1):
                commit_date = datetime.fromtimestamp(
                    _as_float(commit.get("timestamp")),
                    UTC,
                )
                date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S UTC")
                raw_files_changed = commit.get("files_changed")
                files_changed = (
                    [str(item) for item in raw_files_changed]
                    if isinstance(raw_files_changed, list)
                    else []
                )
                panel_content = [
                    f"[yellow]Commit:[/yellow] {str(commit.get('hash', ''))[:8]}",
                    f"[cyan]Author:[/cyan] {commit.get('author', 'Unknown')}",
                    f"[blue]Date:[/blue] {date_str}",
                    "",
                    str(commit.get("title", "")),
                ]
                if files_changed:
                    panel_content.append("")
                    panel_content.append(
                        f"[magenta]Files Changed ({len(files_changed)}):[/magenta]"
                    )
                    for file_path in files_changed[:5]:
                        panel_content.append(f"  • {file_path}")
                print_result_panel(
                    console,
                    idx,
                    _as_float(commit.get("score")),
                    panel_content,
                    is_last=(idx == len(results)),
                )
        else:
            console.print("[yellow]No results found.[/yellow]")
        return

    except Exception as e:  # noqa: BLE001 -- CLI command boundary; must catch anything and report cleanly
        logger.error(f"Git commit search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
