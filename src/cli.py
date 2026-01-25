import asyncio
import json
import logging
import sys
from pathlib import Path

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
from src.indexing.manifest import IndexManifest, save_manifest
from src.indexing.reconciler import build_indexed_files_map
from src.utils import should_include_file
from src.cli_utils.validators import validate_range, validate_timestamp_range, validate_non_negative
from src.cli_utils.formatters import print_result_panel, print_debug_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100


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


@cli.command()
@click.option("--host", default=None, help="Override host from config")
@click.option("--port", default=None, type=int, help="Override port from config")
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def run(host: str | None, port: int | None, project: str | None):
    try:
        config = load_config()
        config = _apply_project_detection(config, project)

        server_host = host or config.server.host
        server_port = port or config.server.port

        logger.info(f"Starting server on {server_host}:{server_port}")
        uvicorn.run(
            "src.server:create_app",
            host=server_host,
            port=server_port,
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
            parsers=ctx.config.parsers,
            chunking_config={
                "strategy": ctx.config.document_chunking.strategy,
                "min_chunk_chars": ctx.config.document_chunking.min_chunk_chars,
                "max_chunk_chars": ctx.config.document_chunking.max_chunk_chars,
                "overlap_chars": ctx.config.document_chunking.overlap_chars,
            },
            indexed_files=build_indexed_files_map(files_to_index, docs_path)
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
                            parallel_config = ParallelIndexingConfig(
                                max_workers=ctx.config.git_indexing.parallel_workers,
                                batch_size=ctx.config.git_indexing.batch_size,
                                embed_batch_size=ctx.config.git_indexing.embed_batch_size,
                            )

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
                                            ctx.config.git_indexing.delta_max_lines,
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
        if ctx.config.search.query_expansion_enabled:
            try:
                click.echo("Building concept vocabulary...")
                ctx.index_manager.vector.build_concept_vocabulary(
                    max_terms=ctx.config.search.query_expansion_max_terms,
                    min_frequency=ctx.config.search.query_expansion_min_frequency,
                )
                ctx.index_manager.persist()
                vocab_size = len(ctx.index_manager.vector._concept_vocabulary)
                click.echo(
                    f"✅ Successfully built concept vocabulary: {vocab_size} terms"
                )
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

        table.add_row("Server Host", config.server.host)
        table.add_row("Server Port", str(config.server.port))

        table.add_row("Documents Path", config.indexing.documents_path)
        table.add_row("Index Path", config.indexing.index_path)
        table.add_row("Recursive", str(config.indexing.recursive))

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
@click.option(
    "--debug", is_flag=True, help="Display intermediate search statistics"
)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def query(query_text: str, output_json: bool, top_n: int, debug: bool, project: str | None):
    try:
        console = Console()
        ctx = _create_query_context(project)

        # Check if manifest exists (indicates a valid index)
        manifest_path = ctx.index_path / "index.manifest.json"
        if not manifest_path.exists():
            click.echo("Error: No index found. Run 'rebuild-index' first.", err=True)
            sys.exit(1)

        ctx.index_manager.load()
        validate_range(top_n, MIN_TOP_N, MAX_TOP_N, "--top-n")

        with console.status("[bold green]Searching documents..."):
            top_k = max(20, top_n * 4)
            async def _run_query_with_healing(query: str, top_k_value: int, top_n_value: int):
                results, compression_stats, strategy_stats = await ctx.orchestrator.query(
                    query,
                    top_k=top_k_value,
                    top_n=top_n_value,
                )
                await ctx.orchestrator.drain_reindex()
                return results, compression_stats, strategy_stats

            results, compression_stats, strategy_stats = asyncio.run(
                _run_query_with_healing(query_text, top_k, top_n)
            )

        if output_json:
            output = {
                "query": query_text,
                "results": [result.to_dict() for result in results],
            }
            click.echo(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")

        if debug:
            print_debug_stats(console, strategy_stats, compression_stats, ctx.config.search.score_calibration_threshold)

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
                    console, idx, result.score, panel_content, is_last=(idx == len(results))
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
@click.option(
    "--debug", is_flag=True, help="Display intermediate search statistics"
)
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
        ctx = _create_query_context(project)

        # Check git indexing enabled
        if not ctx.config.git_indexing.enabled:
            click.echo(
                "Error: Git indexing is not enabled. Enable it in config.toml", err=True
            )
            sys.exit(1)

        # Check commit indexer exists
        if ctx.commit_indexer is None:
            click.echo(
                "Error: Git indexing unavailable. Run 'rebuild-index' to enable git search.",
                err=True,
            )
            sys.exit(1)
        assert ctx.commit_indexer is not None  # Narrowing for type checker

        validate_range(top_n, MIN_TOP_N, MAX_TOP_N, "--top-n")
        validate_timestamp_range(after_timestamp, before_timestamp)

        with console.status("[bold green]Searching git commits..."):
            from src.git.commit_search import search_git_history

            response = search_git_history(
                commit_indexer=ctx.commit_indexer,
                query=query_text,
                top_n=top_n,
                files_glob=files_glob,
                after_timestamp=after_timestamp,
                before_timestamp=before_timestamp,
            )

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
                    console, idx, commit.score, panel_content, is_last=(idx == len(response.results))
                )
        else:
            console.print("[yellow]No results found.[/yellow]")

    except Exception as e:
        logger.error(f"Git commit search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("search-memory")
@click.argument("query_text")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--limit", default=5, type=int, help="Maximum number of results (default: 5)"
)
@click.option(
    "--debug", is_flag=True, help="Display intermediate search statistics"
)
@click.option(
    "--type",
    "memory_type",
    default=None,
    help="Memory type filter (plan|journal|fact|observation|reflection)",
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
    "--relative-days",
    default=None,
    type=int,
    help="Last N days (overrides absolute timestamps)",
)
@click.option(
    "--full", "load_full_memory", is_flag=True, help="Load full memory content"
)
@click.option(
    "--project", default=None, help="Override project detection (name or path)"
)
def search_memory(
    query_text: str,
    output_json: bool,
    limit: int,
    debug: bool,
    memory_type: str | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
    relative_days: int | None,
    load_full_memory: bool,
    project: str | None,
):
    """Search AI memory bank using natural language queries."""
    try:
        console = Console()
        ctx = _create_query_context(project)

        # Check memory system enabled
        if not ctx.config.memory.enabled:
            click.echo(
                "Error: Memory system is not enabled. Enable it in config.toml",
                err=True,
            )
            sys.exit(1)

        # Check memory components exist
        if ctx.memory_manager is None or ctx.memory_search is None:
            click.echo(
                "Error: Memory system unavailable. Check configuration.", err=True
            )
            sys.exit(1)
        assert ctx.memory_manager is not None  # Narrowing for type checker
        assert ctx.memory_search is not None

        validate_range(limit, MIN_TOP_N, MAX_TOP_N, "--limit")

        if memory_type is not None:
            valid_types = ["plan", "journal", "fact", "observation", "reflection"]
            if memory_type not in valid_types:
                click.echo(
                    f"Error: --type must be one of: {', '.join(valid_types)}", err=True
                )
                sys.exit(1)

        validate_timestamp_range(after_timestamp, before_timestamp)
        validate_non_negative(relative_days, "--relative-days")

        # Load memory index
        ctx.memory_manager.load()

        # Capture for closure type narrowing
        memory_search = ctx.memory_search

        with console.status("[bold green]Searching memories..."):
            async def _run_memory_search_with_healing():
                results = await memory_search.search_memories(
                    query=query_text,
                    limit=limit,
                    filter_type=memory_type,
                    load_full_memory=load_full_memory,
                    after_timestamp=after_timestamp,
                    before_timestamp=before_timestamp,
                    relative_days=relative_days,
                )
                await memory_search.drain_reindex()
                return results

            results = asyncio.run(_run_memory_search_with_healing())

        if output_json:
            output = {
                "query": query_text,
                "results": [
                    {
                        "memory_id": r.memory_id,
                        "score": r.score,
                        "content": r.content,
                        "type": r.frontmatter.type,
                        "status": r.frontmatter.status,
                        "tags": r.frontmatter.tags,
                        "created_at": r.frontmatter.created_at.isoformat()
                        if r.frontmatter.created_at
                        else None,
                        "file_path": r.file_path,
                        "header_path": r.header_path,
                    }
                    for r in results
                ],
            }
            click.echo(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")

        if results:
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")

            for idx, memory in enumerate(results, 1):
                tags_str = (
                    ", ".join(memory.frontmatter.tags)
                    if memory.frontmatter.tags
                    else "(none)"
                )
                created_str = ""
                if memory.frontmatter.created_at:
                    created_str = memory.frontmatter.created_at.strftime(
                        "%Y-%m-%d %H:%M UTC"
                    )

                panel_content = [
                    f"[yellow]Memory:[/yellow] {memory.memory_id}",
                    f"[cyan]Type:[/cyan] {memory.frontmatter.type} | [magenta]Tags:[/magenta] {tags_str}",
                ]

                if created_str:
                    panel_content.append(f"[blue]Created:[/blue] {created_str}")

                panel_content.append("")

                # Truncate content for display unless --full
                content_display = memory.content
                if not load_full_memory and len(content_display) > 500:
                    content_display = content_display[:500] + "..."

                panel_content.append(content_display)

                print_result_panel(
                    console, idx, memory.score, panel_content, is_last=(idx == len(results))
                )
        else:
            console.print("[yellow]No results found.[/yellow]")

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()
