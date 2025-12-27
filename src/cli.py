import asyncio
import fnmatch
import glob
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
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, save_manifest
from src.indexing.reconciler import build_indexed_files_map
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import QueryOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _should_include_file(file_path: str, include_patterns: list[str], exclude_patterns: list[str], exclude_hidden_dirs: bool = True):
    # Convert to forward slashes for consistent matching
    normalized_path = file_path.replace("\\", "/")

    # Check if file is in a hidden directory (if enabled)
    if exclude_hidden_dirs:
        path_parts = normalized_path.split("/")
        for part in path_parts:
            # Skip empty parts and check if any directory component starts with '.'
            if part and part.startswith("."):
                return False

    # Check if file matches any include pattern
    included = False
    for pattern in include_patterns:
        if fnmatch.fnmatch(normalized_path, pattern):
            included = True
            break

    if not included:
        return False

    # Check if file matches any exclude pattern (exclude takes precedence)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(normalized_path, pattern):
            return False

    return True


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
        config = load_config()
        config = _apply_project_detection(config, project)

        vector = VectorIndex()
        keyword = KeywordIndex()
        graph = GraphStore()

        manager = IndexManager(config, vector, keyword, graph)

        docs_path = Path(config.indexing.documents_path)
        index_path = Path(config.indexing.index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        pattern = str(docs_path / "**" / "*.md")
        all_files = glob.glob(pattern, recursive=config.indexing.recursive)

        # Filter files using include/exclude patterns
        files_to_index = [
            f for f in all_files
            if _should_include_file(f, config.indexing.include, config.indexing.exclude, config.indexing.exclude_hidden_dirs)
        ]
        total_files = len(files_to_index)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Indexing documents...", total=total_files)

            for file_path in files_to_index:
                # Show path relative to documents_path for cleaner output
                try:
                    rel_path = Path(file_path).relative_to(docs_path)
                    display_path = str(rel_path)
                except ValueError:
                    # If relative_to fails, use the full path
                    display_path = file_path

                progress.update(task, description=f"[bold blue]Indexing: {display_path}")
                manager.index_document(file_path)
                progress.advance(task)

        manager.persist()

        current_manifest = IndexManifest(
            spec_version="1.0.0",
            embedding_model=config.llm.embedding_model,
            parsers=config.parsers,
            chunking_config={
                "strategy": config.chunking.strategy,
                "min_chunk_chars": config.chunking.min_chunk_chars,
                "max_chunk_chars": config.chunking.max_chunk_chars,
                "overlap_chars": config.chunking.overlap_chars,
            },
            indexed_files=build_indexed_files_map(files_to_index, docs_path)
        )
        save_manifest(index_path, current_manifest)

        click.echo(f"✅ Successfully rebuilt index: {total_files} documents indexed")

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
        # Suppress logging output for clean CLI experience
        logging.getLogger().setLevel(logging.WARNING)

        console = Console()

        config = load_config()
        config = _apply_project_detection(config, project)

        vector = VectorIndex()
        keyword = KeywordIndex()
        graph = GraphStore()

        manager = IndexManager(config, vector, keyword, graph)

        index_path = Path(config.indexing.index_path)
        if not index_path.exists():
            click.echo("Error: No index found. Run 'rebuild-index' first.", err=True)
            sys.exit(1)

        manager.load()

        orchestrator = QueryOrchestrator(vector, keyword, graph, config, manager)

        if top_n < 1 or top_n > 100:
            click.echo("Error: --top-n must be between 1 and 100", err=True)
            sys.exit(1)

        with console.status("[bold green]Searching documents..."):
            top_k = max(20, top_n * 4)
            results, _ = asyncio.run(orchestrator.query(
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
