import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

from src.config import load_config
from src.git.repository import discover_git_repositories, get_commits_after_timestamp, is_git_available
from src.git.commit_parser import parse_commit, build_commit_document
from src.context import ApplicationContext
from src.indexing.manifest import IndexManifest, save_manifest
from src.indexing.reconciler import build_indexed_files_map
from src.utils import should_include_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100


def _should_include_file(
    file_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
):
    return should_include_file(file_path, include_patterns, exclude_patterns, exclude_hidden_dirs)


@click.group()
def cli():
    pass


def _apply_project_detection(config, project_override: str | None = None):
    from src.config import detect_project, resolve_index_path, resolve_documents_path

    detected_project = detect_project(projects=config.projects, project_override=project_override)
    index_path = resolve_index_path(config, detected_project)
    documents_path = resolve_documents_path(config, detected_project, config.projects)

    config.indexing.index_path = str(index_path)
    config.indexing.documents_path = documents_path
    return config


@cli.command()
@click.option("--project", default=None, help="Override project detection (name or path)")
def mcp(project: str | None):
    """Run MCP server with stdio transport (for VS Code integration)."""
    try:
        # Import here to avoid importing mcp when not needed
        from src.mcp_server import main as mcp_main

        # Pass arguments explicitly to avoid argparse parsing sys.argv
        argv = []
        if project:
            argv.extend(["--project", project])

        asyncio.run(mcp_main(argv))
    except KeyboardInterrupt:
        pass  # Graceful shutdown handled
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


@cli.command()
@click.option("--host", default=None, help="Override host from config")
@click.option("--port", default=None, type=int, help="Override port from config")
@click.option("--project", default=None, help="Override project detection (name or path)")
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
@click.option("--project", default=None, help="Override project detection (name or path)")
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

                progress.update(task, description=f"[bold blue]Indexing: {display_path}")
                ctx.index_manager.index_document(file_path)
                progress.advance(task)

        ctx.index_manager.persist()

        current_manifest = IndexManifest(
            spec_version="1.0.0",
            embedding_model=ctx.config.llm.embedding_model,
            parsers=ctx.config.parsers,
            chunking_config={
                "strategy": ctx.config.chunking.strategy,
                "min_chunk_chars": ctx.config.chunking.min_chunk_chars,
                "max_chunk_chars": ctx.config.chunking.max_chunk_chars,
                "overlap_chars": ctx.config.chunking.overlap_chars,
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
                                last_timestamp = ctx.commit_indexer.get_last_indexed_timestamp(str(repo_path))
                                commit_hashes = get_commits_after_timestamp(repo_path, last_timestamp)
                                repo_commits_map[repo_path] = commit_hashes
                                total_commits += len(commit_hashes)
                            except Exception as e:
                                logger.error(f"Failed to get commits from {repo_path}: {e}")
                                continue

                        if total_commits > 0:
                            with Progress(
                                TextColumn("[bold blue]{task.description}"),
                                BarColumn(),
                                TaskProgressColumn(),
                                TimeRemainingColumn(),
                            ) as progress:
                                task = progress.add_task("Indexing git commits...", total=total_commits)

                                indexed_count = 0
                                for repo_path, commit_hashes in repo_commits_map.items():
                                    try:
                                        for commit_hash in commit_hashes:
                                            try:
                                                commit_data = parse_commit(
                                                    repo_path,
                                                    commit_hash,
                                                    ctx.config.git_indexing.delta_max_lines,
                                                )
                                                commit_doc = build_commit_document(commit_data)
                                                ctx.commit_indexer.add_commit(
                                                    hash=commit_data.hash,
                                                    timestamp=commit_data.timestamp,
                                                    author=commit_data.author,
                                                    committer=commit_data.committer,
                                                    title=commit_data.title,
                                                    message=commit_data.message,
                                                    files_changed=commit_data.files_changed,
                                                    delta_truncated=commit_data.delta_truncated,
                                                    commit_document=commit_doc,
                                                    repo_path=str(repo_path.parent),
                                                )
                                                indexed_count += 1
                                                progress.advance(task)
                                            except Exception as e:
                                                logger.error(f"Failed to index commit {commit_hash}: {e}")
                                                progress.advance(task)
                                                continue
                                    except Exception as e:
                                        logger.error(f"Failed to process repository {repo_path}: {e}")
                                        continue

                            click.echo(f"✅ Successfully indexed {indexed_count} git commits from {len(repos)} repositories")
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
                click.echo(f"✅ Successfully built concept vocabulary: {vocab_size} terms")
            except Exception as e:
                logger.error(f"Concept vocabulary building failed: {e}")
                click.echo(f"⚠️  Concept vocabulary building failed: {e}", err=True)

    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("check-config")
@click.option("--project", default=None, help="Override project detection (name or path)")
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
            table.add_row("[bold]Registered Projects[/bold]", f"{len(config.projects)} project(s)")
            for proj in config.projects:
                table.add_row(f"  • {proj.name}", proj.path)

            from src.config import detect_project
            detected = detect_project(projects=config.projects, project_override=project)
            if detected:
                table.add_row("", "")
                override_indicator = " (via --project)" if project else ""
                table.add_row("[bold]Active Project[/bold]", f"✅ {detected}{override_indicator}")
            else:
                table.add_row("", "")
                table.add_row("[bold]Active Project[/bold]", "⚠️  None detected (using local index)")

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
                console.print(f"⚠️  Index directory exists but no manifest found: {index_path}")
        else:
            console.print(f"📭 No index found (will be created on first run): {index_path}")

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        click.echo(f"❌ Configuration Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query_text")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option("--top-n", default=5, type=int, help="Maximum number of results (default: 5)")
@click.option("--project", default=None, help="Override project detection (name or path)")
def query(query_text: str, output_json: bool, top_n: int, project: str | None):
    try:
        logging.getLogger().setLevel(logging.WARNING)
        console = Console()

        ctx = ApplicationContext.create(
            project_override=project,
            enable_watcher=False,
            lazy_embeddings=False,
        )

        # Check if manifest exists (indicates a valid index)
        manifest_path = ctx.index_path / "index.manifest.json"
        if not manifest_path.exists():
            click.echo("Error: No index found. Run 'rebuild-index' first.", err=True)
            sys.exit(1)

        ctx.index_manager.load()

        if top_n < MIN_TOP_N or top_n > MAX_TOP_N:
            click.echo(f"Error: --top-n must be between {MIN_TOP_N} and {MAX_TOP_N}", err=True)
            sys.exit(1)

        with console.status("[bold green]Searching documents..."):
            top_k = max(20, top_n * 4)
            results, _ = asyncio.run(ctx.orchestrator.query(
                query_text,
                top_k=top_k,
                top_n=top_n
            ))

        if output_json:
            output = {
                "query": query_text,
                "results": [result.to_dict() for result in results]
            }
            click.echo(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}\n")

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

                result_panel = Panel(
                    "\n".join(panel_content),
                    title=f"[bold cyan]#{idx}[/bold cyan] [bold green]Score: {result.score:.4f}[/bold green]",
                    border_style="cyan",
                    padding=(0, 1),
                )
                console.print(result_panel)

                if idx < len(results):
                    console.print()
        else:
            console.print("[yellow]No results found.[/yellow]")

    except FileNotFoundError as e:
        logger.error(f"Indices not found: {e}")
        click.echo("Error: No indices found. Run 'mcp-markdown-ragdocs rebuild-index' first.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()
