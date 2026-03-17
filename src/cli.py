import asyncio
import json
import logging
import os
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

import click
import uvicorn
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.config import load_config
from src.daemon.queue_status import get_queue_stats
from src.daemon import RuntimePaths, read_daemon_metadata
from src.daemon.health import (
    DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
    DaemonHealthServer,
    request_daemon_socket,
)
from src.daemon.management import (
    acquire_boot_lock,
    inspect_daemon,
    restart_daemon,
    start_daemon,
    stop_daemon,
    wait_for_daemon_ready,
)
from src.git.repository import (
    discover_git_repositories,
    get_commits_after_timestamp,
    is_git_available,
)
from src.git.parallel_indexer import (
    ParallelIndexingConfig,
    index_commits_parallel_sync,
)
from src.context import ApplicationContext
from src.coordination.queue import get_huey
from src.indexing.manifest import IndexManifest, save_manifest
from src.indexing.tasks import register_tasks
from src.indexing.reconciler import build_indexed_files_map
from src.lifecycle import LifecycleCoordinator, LifecycleState
from src.utils import should_include_file
from src.worker.consumer import HueyWorker
from src.cli_utils.validators import (
    validate_range,
    validate_timestamp_range,
)
from src.cli_utils.formatters import print_result_panel, print_debug_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100
_DAEMON_QUERY_READY_WAIT_SECONDS = 120.0


def _create_query_context(project: str | None) -> ApplicationContext:
    logging.getLogger().setLevel(logging.WARNING)
    return ApplicationContext.create(
        project_override=project,
        enable_watcher=False,
        lazy_embeddings=False,
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


def _create_daemon_runtime(project: str | None, runtime_paths: RuntimePaths):
    ctx = ApplicationContext.create(
        project_override=project,
        enable_watcher=True,
        lazy_embeddings=True,
        use_tasks=True,
        index_path_override=runtime_paths.root,
    )
    huey = get_huey(runtime_paths.queue_db_path)
    register_tasks(huey, ctx.index_manager, commit_indexer=ctx.commit_indexer)
    worker = HueyWorker(huey)
    return ctx, worker


def _build_queue_status_payload(
    *,
    queue_path: Path,
    worker_running: bool,
) -> dict[str, object]:
    huey = get_huey(queue_path)
    stats = get_queue_stats(huey, worker_running=worker_running)
    payload = stats.to_dict()
    payload["queue_db_path"] = str(queue_path)
    return payload


@click.group()
def cli():
    pass


def _apply_project_detection(config, project_override: str | None = None):
    from src.config import detect_project, resolve_index_path, resolve_documents_path

    detected_project = detect_project(
        projects=config.projects, project_override=project_override
    )
    index_path = resolve_index_path(config, detected_project)
    documents_path = resolve_documents_path(config, detected_project, config.projects)

    config.indexing.index_path = str(index_path)
    config.indexing.documents_path = documents_path
    return config


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


async def _run_daemon_forever(project: str | None) -> None:
    lock = await asyncio.to_thread(acquire_boot_lock, timeout_seconds=5.0)
    runtime_paths = RuntimePaths.resolve()

    async def _handle_daemon_request(
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if path == "/api/mcp/tools":
            from src.mcp.tools.document_tools import get_document_tools

            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in get_document_tools()
                ]
            }
        if path == "/api/mcp/tool":
            import src.mcp.tools.document_tools  # noqa: F401

            from mcp.types import TextContent

            from src.mcp.handlers import HandlerContext, get_handler

            tool_name = str(payload.get("name", ""))
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                return {"status": "error", "error": "tool_arguments_must_be_object"}

            handler = get_handler(tool_name)
            if handler is None:
                return {"status": "error", "error": f"unknown_tool:{tool_name}"}

            hctx = HandlerContext(lambda: ctx, coordinator)
            contents = await handler(hctx, arguments)
            return {
                "contents": [
                    {"type": content.type, "text": content.text}
                    for content in contents
                    if isinstance(content, TextContent)
                ]
            }
        if path == "/api/admin/index-stats":
            return _build_index_stats_payload(ctx)
        if path == "/api/admin/queue-status":
            return _build_queue_status_payload(
                queue_path=runtime_paths.queue_db_path,
                worker_running=huey_worker.is_running,
            )
        if path == "/api/search/query":
            await coordinator.wait_ready(timeout=60.0)
            query_text = str(payload.get("query", ""))
            top_n = int(payload.get("top_n", 5))
            top_k = max(20, top_n * 4)
            results, compression_stats, strategy_stats = await ctx.orchestrator.query(
                query_text,
                top_k=top_k,
                top_n=top_n,
            )
            await ctx.orchestrator.drain_reindex()
            return {
                "query": query_text,
                "results": [result.to_dict() for result in results],
                "compression_stats": compression_stats.to_dict(),
                "strategy_stats": strategy_stats.to_dict(),
            }
        if path == "/api/search/git-history":
            await coordinator.wait_ready(timeout=60.0)
            if ctx.commit_indexer is None:
                return {"status": "error", "error": "git_indexing_unavailable"}

            from src.git.commit_search import search_git_history

            response = search_git_history(
                commit_indexer=ctx.commit_indexer,
                query=str(payload.get("query", "")),
                top_n=int(payload.get("top_n", 5)),
                files_glob=str(payload["files_glob"]) if payload.get("files_glob") else None,
                after_timestamp=int(payload["after_timestamp"]) if payload.get("after_timestamp") is not None else None,
                before_timestamp=int(payload["before_timestamp"]) if payload.get("before_timestamp") is not None else None,
            )
            return {
                "query": response.query,
                "total_commits_indexed": response.total_commits_indexed,
                "results": [
                    {
                        "hash": r.hash,
                        "title": r.title,
                        "author": r.author,
                        "committer": r.committer,
                        "timestamp": r.timestamp,
                        "message": r.message,
                        "files_changed": r.files_changed,
                        "delta_truncated": r.delta_truncated,
                        "score": r.score,
                        "repo_path": r.repo_path,
                    }
                    for r in response.results
                ],
            }
        return {"status": "error", "error": "unknown_request_path", "path": path}

    health_server = DaemonHealthServer(
        socket_path=runtime_paths.socket_path,
        metadata_provider=lambda: read_daemon_metadata(runtime_paths.metadata_path),
        request_handler=_handle_daemon_request,
    )
    coordinator = LifecycleCoordinator()
    loop = asyncio.get_running_loop()
    coordinator.install_signal_handlers(loop)

    ctx, huey_worker = await asyncio.to_thread(
        _create_daemon_runtime,
        project,
        runtime_paths,
    )

    try:
        await health_server.start()
        try:
            await coordinator.start(
                ctx,
                background_index=True,
                db_manager=ctx.db_manager,
                huey_worker=huey_worker,
            )
            while coordinator.state not in (
                LifecycleState.SHUTTING_DOWN,
                LifecycleState.TERMINATED,
            ):
                await asyncio.sleep(0.2)
        finally:
            await coordinator.shutdown()
            await health_server.stop()
    finally:
        await asyncio.to_thread(lock.release)


@cli.group("daemon")
def daemon_group():
    """Manage the long-lived Ragdocs daemon."""


@cli.group("queue")
def queue_group():
    """Inspect task queue state."""


@daemon_group.command("run")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def daemon_run(project: str | None):
    """Run the daemon in the foreground."""
    try:
        asyncio.run(_run_daemon_forever(project))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Failed to run daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("daemon-internal-run", hidden=True)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def daemon_internal_run(project: str | None):
    """Run the daemon in the foreground for internal start/restart flows."""
    daemon_run.callback(project)


@daemon_group.command("start")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
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
        metadata = start_daemon(
            project_override=project,
            timeout_seconds=timeout,
        )
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Daemon running (pid={metadata.pid}, status={metadata.status})")


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
        "started_at": started_at,
        "metadata_path": str(runtime_paths.metadata_path),
        "lock_path": str(runtime_paths.lock_path),
        "socket_path": inspection.metadata.socket_path
        or str(runtime_paths.socket_path),
        "index_db_path": inspection.metadata.index_db_path,
        "queue_db_path": inspection.metadata.queue_db_path,
        "endpoint": inspection.metadata.transport_endpoint,
    }

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"Daemon status: {state}")
    click.echo(f"PID: {inspection.metadata.pid}")
    click.echo(f"Lifecycle: {inspection.metadata.status}")
    click.echo(f"Started: {started_at}")
    click.echo(f"Metadata path: {runtime_paths.metadata_path}")
    click.echo(f"Lock path: {runtime_paths.lock_path}")
    if inspection.metadata.index_db_path:
        click.echo(f"Index DB: {inspection.metadata.index_db_path}")
    if inspection.metadata.queue_db_path:
        click.echo(f"Queue DB: {inspection.metadata.queue_db_path}")
    if inspection.metadata.transport_endpoint:
        click.echo(f"Endpoint: {inspection.metadata.transport_endpoint}")


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
    "--project", default=None, help="Override project detection (name or path)"
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
        metadata = restart_daemon(
            project_override=project,
            start_timeout_seconds=timeout,
        )
    except Exception as e:
        logger.error(f"Failed to restart daemon: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Daemon restarted (pid={metadata.pid}, status={metadata.status})")


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
            "/api/admin/index-stats",
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
        click.echo(f"Documents path: {payload['documents_path']}")
        click.echo(f"Index path: {payload['index_path']}")
        click.echo(f"Manifest present: {payload['manifest_exists']}")
        click.echo(f"Indexed documents: {payload['indexed_documents']}")
        click.echo(f"Indexed chunks: {payload['indexed_chunks']}")
        click.echo(f"Discovered files: {payload['discovered_files']}")
        click.echo(f"Git repositories: {payload['git_repositories']}")
        click.echo(f"Indexed git commits: {payload['git_commits']}")
    except Exception as e:
        logger.error(f"Failed to inspect index stats: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _build_index_stats_payload(ctx: ApplicationContext) -> dict[str, object]:
    manifest_path = ctx.index_path / "index.manifest.json"
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        ctx.index_manager.load()

    docs_root = Path(ctx.config.indexing.documents_path)
    discovered_files = ctx.discover_files() if docs_root.exists() else []
    repo_count = len(
        discover_git_repositories(
            docs_root,
            ctx.config.indexing.exclude,
            ctx.config.indexing.exclude_hidden_dirs,
        )
    )
    git_commit_count = (
        ctx.commit_indexer.get_total_commits() if ctx.commit_indexer is not None else 0
    )

    return {
        "documents_path": str(docs_root),
        "index_path": str(ctx.index_path),
        "index_db_path": str(ctx.index_path / "index.db"),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "indexed_documents": ctx.index_manager.get_document_count()
        if manifest_exists
        else 0,
        "indexed_chunks": len(ctx.index_manager.vector) if manifest_exists else 0,
        "discovered_files": len(discovered_files),
        "git_commits": git_commit_count,
        "git_repositories": repo_count,
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
            "/api/admin/queue-status",
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
    inspection = inspect_daemon(runtime_paths)
    metadata = inspection.metadata if inspection.responsive else None
    if metadata is None and auto_start:
        metadata = start_daemon(
            project_override=project_override,
            paths=runtime_paths,
        )
    if (
        auto_start
        and metadata is not None
        and metadata.status not in {"ready", "ready_primary", "ready_replica"}
    ):
        metadata = wait_for_daemon_ready(
            timeout_seconds=_DAEMON_QUERY_READY_WAIT_SECONDS,
            paths=runtime_paths,
        )
    if metadata is None or not metadata.socket_path:
        return None

    response = request_daemon_socket(
        Path(metadata.socket_path),
        path,
        payload,
        timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
    )
    if response.get("status") == "error":
        if allow_error:
            return response
        return None
    return response


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

    if isinstance(details, str) and details:
        raise RuntimeError(details)

    raise RuntimeError(f"Daemon request failed: {error}")


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
def rebuild_index_cmd(project: str | None):
    try:
        ctx = ApplicationContext.create(
            project_override=project,
            enable_watcher=False,
            lazy_embeddings=False,
        )

        docs_path = Path(ctx.config.indexing.documents_path)
        files_to_index = ctx.discover_files()
        total_files = len(files_to_index)

        ctx.index_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Indexing documents...", total=total_files)

            for file_path in files_to_index:
                try:
                    rel_path = Path(file_path).relative_to(docs_path)
                    display_path = str(rel_path)
                except ValueError:
                    display_path = file_path

                progress.update(
                    task, description=f"[bold blue]Indexing: {display_path}"
                )
                ctx.index_manager.index_document(file_path)
                progress.advance(task)

        ctx.index_manager.persist()

        current_manifest = IndexManifest(
            spec_version="1.0.0",
            embedding_model=ctx.config.llm.embedding_model,
            chunking_config={
                "strategy": ctx.config.chunking.strategy,
                "min_chunk_chars": ctx.config.chunking.min_chunk_chars,
                "max_chunk_chars": ctx.config.chunking.max_chunk_chars,
                "overlap_chars": ctx.config.chunking.overlap_chars,
            },
            indexed_files=build_indexed_files_map(files_to_index, docs_path),
        )
        save_manifest(ctx.index_path, current_manifest)

        click.echo(f"✅ Successfully rebuilt index: {total_files} documents indexed")

        # Git commit indexing phase
        if ctx.config.git_indexing.enabled and ctx.commit_indexer is not None:
            if not is_git_available():
                logger.warning("Git binary not available, skipping git commit indexing")
                click.echo("⚠️  Git binary not available, skipping git commit indexing")
            else:
                try:
                    click.echo("Clearing git commit index...")
                    ctx.commit_indexer.clear()

                    repos = discover_git_repositories(
                        docs_path,
                        ctx.config.indexing.exclude,
                        ctx.config.indexing.exclude_hidden_dirs,
                    )

                    if repos:
                        # Count total commits across all repos
                        total_commits = 0
                        repo_commits_map: dict[Path, list[str]] = {}
                        for repo_path in repos:
                            try:
                                last_timestamp = (
                                    ctx.commit_indexer.get_last_indexed_timestamp(
                                        str(repo_path.parent)
                                    )
                                )
                                commit_hashes = get_commits_after_timestamp(
                                    repo_path, last_timestamp
                                )
                                repo_commits_map[repo_path] = commit_hashes
                                total_commits += len(commit_hashes)
                            except Exception as e:
                                logger.error(
                                    f"Failed to get commits from {repo_path}: {e}"
                                )
                                continue

                        if total_commits > 0:
                            parallel_config = ParallelIndexingConfig()

                            with Progress(
                                TextColumn("[bold blue]{task.description}"),
                                BarColumn(),
                                TaskProgressColumn(),
                                TimeRemainingColumn(),
                            ) as progress:
                                task = progress.add_task(
                                    "Indexing git commits...",
                                    total=len(repo_commits_map),
                                )

                                indexed_count = 0
                                for (
                                    repo_path,
                                    commit_hashes,
                                ) in repo_commits_map.items():
                                    if not commit_hashes:
                                        progress.advance(task)
                                        continue

                                    try:
                                        indexed = index_commits_parallel_sync(
                                            commit_hashes,
                                            repo_path,
                                            ctx.commit_indexer,
                                            parallel_config,
                                            200,
                                        )
                                        indexed_count += indexed
                                        progress.advance(task)
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to process repository {repo_path}: {e}"
                                        )
                                        progress.advance(task)
                                        continue

                            click.echo(
                                f"✅ Successfully indexed {indexed_count} git commits from {len(repos)} repositories"
                            )
                        else:
                            click.echo("ℹ️  No new git commits to index")
                    else:
                        click.echo("ℹ️  No git repositories found")
                except Exception as e:
                    logger.error(f"Git indexing failed: {e}")
                    click.echo(f"⚠️  Git indexing failed: {e}", err=True)

        # Concept vocabulary building phase
        try:
            click.echo("Building concept vocabulary...")
            ctx.index_manager.vector.build_concept_vocabulary(
                max_terms=2000,
                min_frequency=3,
            )
            ctx.index_manager.persist()
            vocab_size = len(ctx.index_manager.vector._concept_vocabulary)
            click.echo(f"✅ Successfully built concept vocabulary: {vocab_size} terms")
        except Exception as e:
            logger.error(f"Concept vocabulary building failed: {e}")
            click.echo(f"⚠️  Concept vocabulary building failed: {e}", err=True)

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

        table.add_row("", "")
        table.add_row("Semantic Weight", str(config.search.semantic_weight))
        table.add_row("Keyword Weight", str(config.search.keyword_weight))
        table.add_row("Recency Bias", str(config.search.recency_bias))

        table.add_row("", "")
        table.add_row("Embedding Model", config.llm.embedding_model)

        console.print(table)

        console.print("\n[bold green]✅ Configuration is valid[/bold green]")

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
def query(
    query_text: str, output_json: bool, top_n: int, debug: bool, project: str | None
):
    try:
        console = Console()
        validate_range(top_n, MIN_TOP_N, MAX_TOP_N, "--top-n")

        daemon_payload = _request_daemon_json(
            "/api/search/query",
            {"query": query_text, "top_n": top_n},
            project_override=project,
            auto_start=True,
            allow_error=True,
        )
        if daemon_payload is None or daemon_payload.get("status") == "error":
            _raise_daemon_request_error(daemon_payload)

        if output_json:
            click.echo(json.dumps(daemon_payload, indent=2))
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

        if output_json:
            output = {
                "query": query_text,
                "results": [result.to_dict() for result in results],
            }
            click.echo(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")

        if debug:
            print_debug_stats(console, strategy_stats, compression_stats, 0.02)

        if results:
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")

            for idx, result in enumerate(results, 1):
                panel_content = [
                    f"[yellow]Document:[/yellow] {result.doc_id}",
                    f"[magenta]Section:[/magenta] {result.header_path or '(no section)'}",
                    f"[blue]File:[/blue] {result.file_path or '(unknown)'}",
                    "",
                    result.content,
                ]
                print_result_panel(
                    console,
                    idx,
                    result.score,
                    panel_content,
                    is_last=(idx == len(results)),
                )
        else:
            console.print("[yellow]No results found.[/yellow]")

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
def search_commits(
    query_text: str,
    output_json: bool,
    top_n: int,
    debug: bool,
    files_glob: str | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
    project: str | None,
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

        if output_json:
            output = {
                "query": response.query,
                "total_commits_indexed": response.total_commits_indexed,
                "results": [
                    {
                        "hash": r.hash,
                        "title": r.title,
                        "author": r.author,
                        "committer": r.committer,
                        "timestamp": r.timestamp,
                        "message": r.message,
                        "files_changed": r.files_changed,
                        "delta_truncated": r.delta_truncated,
                        "score": r.score,
                        "repo_path": r.repo_path,
                    }
                    for r in response.results
                ],
            }
            click.echo(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")
        console.print(
            f"[dim]Total commits indexed: {response.total_commits_indexed}[/dim]\n"
        )

        if response.results:
            console.print(f"[bold]Found {len(response.results)} results:[/bold]\n")

            from datetime import datetime, timezone

            for idx, commit in enumerate(response.results, 1):
                commit_date = datetime.fromtimestamp(commit.timestamp, timezone.utc)
                date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S UTC")

                panel_content = [
                    f"[yellow]Commit:[/yellow] {commit.hash[:8]}",
                    f"[cyan]Author:[/cyan] {commit.author}",
                    f"[blue]Date:[/blue] {date_str}",
                    "",
                    commit.title,
                ]

                if len(commit.files_changed) > 0:
                    panel_content.append("")
                    panel_content.append(
                        f"[magenta]Files Changed ({len(commit.files_changed)}):[/magenta]"
                    )
                    for file_path in commit.files_changed[:5]:
                        panel_content.append(f"  • {file_path}")
                    if len(commit.files_changed) > 5:
                        panel_content.append(
                            f"  ... and {len(commit.files_changed) - 5} more"
                        )

                print_result_panel(
                    console,
                    idx,
                    commit.score,
                    panel_content,
                    is_last=(idx == len(response.results)),
                )
        else:
            console.print("[yellow]No results found.[/yellow]")

    except Exception as e:
        logger.error(f"Git commit search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
