import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def print_result_panel(
    console: Console,
    idx: int,
    score: float,
    content_lines: list[str],
    is_last: bool = False,
) -> None:
    result_panel = Panel(
        "\n".join(content_lines),
        title=f"[bold cyan]#{idx}[/bold cyan] [bold green]Score: {score:.4f}[/bold green]",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(result_panel)
    if not is_last:
        console.print()


def print_debug_stats(
    console: Console,
    strategy_stats,
    compression_stats,
    min_confidence: float,
    query_execution_stats: dict[str, int | float] | None = None,
) -> None:
    from mcp_markdown_ragdocs.models import CompressionStats, SearchStrategyStats

    if isinstance(strategy_stats, SearchStrategyStats):
        strategy_table = Table(
            title="Search Strategy Results", show_header=True, title_style="bold yellow"
        )
        strategy_table.add_column("Strategy", style="cyan")
        strategy_table.add_column("Count", style="green", justify="right")

        if strategy_stats.vector_count is not None:
            strategy_table.add_row(
                "Vector (Semantic)", str(strategy_stats.vector_count)
            )
        if strategy_stats.keyword_count is not None:
            strategy_table.add_row("Keyword (BM25)", str(strategy_stats.keyword_count))
        if strategy_stats.graph_count is not None:
            strategy_table.add_row("Graph (PageRank)", str(strategy_stats.graph_count))
        if strategy_stats.tag_expansion_count is not None:
            strategy_table.add_row(
                "Tag Expansion", str(strategy_stats.tag_expansion_count)
            )

        console.print(strategy_table)
        console.print()

    if isinstance(compression_stats, CompressionStats):
        compression_table = Table(
            title="Compression Pipeline", show_header=True, title_style="bold yellow"
        )
        compression_table.add_column("Stage", style="cyan")
        compression_table.add_column("Count", style="green", justify="right")
        compression_table.add_column("Removed", style="red", justify="right")

        compression_table.add_row(
            "Original (RRF Fusion)", str(compression_stats.original_count), "-"
        )

        removed_threshold = (
            compression_stats.original_count - compression_stats.after_threshold
        )
        compression_table.add_row(
            f"After Confidence Filter (≥{min_confidence:.2f})",
            str(compression_stats.after_threshold),
            str(removed_threshold) if removed_threshold > 0 else "-",
        )

        removed_content = (
            compression_stats.after_threshold - compression_stats.after_content_dedup
        )
        compression_table.add_row(
            "After Content Dedup",
            str(compression_stats.after_content_dedup),
            str(removed_content) if removed_content > 0 else "-",
        )

        removed_ngram = (
            compression_stats.after_content_dedup - compression_stats.after_ngram_dedup
        )
        compression_table.add_row(
            "After N-gram Dedup",
            str(compression_stats.after_ngram_dedup),
            str(removed_ngram) if removed_ngram > 0 else "-",
        )

        removed_dedup = (
            compression_stats.after_ngram_dedup - compression_stats.after_dedup
        )
        dedup_label = "After Semantic Dedup"
        if compression_stats.clusters_merged > 0:
            dedup_label += f" ({compression_stats.clusters_merged} clusters merged)"
        compression_table.add_row(
            dedup_label,
            str(compression_stats.after_dedup),
            str(removed_dedup) if removed_dedup > 0 else "-",
        )

        removed_doc_limit = (
            compression_stats.after_dedup - compression_stats.after_doc_limit
        )
        compression_table.add_row(
            "After Doc Limit",
            str(compression_stats.after_doc_limit),
            str(removed_doc_limit) if removed_doc_limit > 0 else "-",
        )

        console.print(compression_table)
        console.print()

    if isinstance(query_execution_stats, dict) and query_execution_stats:
        timing_table = Table(
            title="Query Phase Timings", show_header=True, title_style="bold yellow"
        )
        timing_table.add_column("Phase", style="cyan")
        timing_table.add_column("Time (ms)", style="green", justify="right")

        phase_labels = {
            "vector_search_ms": "Vector Search",
            "keyword_search_ms": "Keyword Search",
            "tag_expansion_ms": "Tag Expansion",
            "graph_expansion_ms": "Graph Expansion",
            "fusion_ms": "Fusion + Ranking",
            "pipeline_ms": "Compression Pipeline",
            "parent_expansion_ms": "Parent Expansion",
            "materialization_ms": "Result Hydration",
            "total_query_ms": "Total Query",
        }
        for key, label in phase_labels.items():
            raw_value = query_execution_stats.get(key)
            if isinstance(raw_value, (int, float)) and raw_value > 0:
                timing_table.add_row(label, f"{float(raw_value):.3f}")

        if timing_table.row_count > 0:
            console.print(timing_table)
            console.print()

        cache_table = Table(
            title="Query Execution Cache Stats",
            show_header=True,
            title_style="bold yellow",
        )
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Count", style="green", justify="right")

        cache_labels = {
            "metadata_lookups": "Metadata Lookups",
            "metadata_cache_hits": "Metadata Cache Hits",
            "content_lookups": "Content Lookups",
            "content_cache_hits": "Content Cache Hits",
            "embedding_fetches": "Embedding Fetches",
            "embedding_cache_hits": "Embedding Cache Hits",
            "parent_lookups": "Parent Lookups",
            "parent_cache_hits": "Parent Cache Hits",
        }
        for key, label in cache_labels.items():
            raw_value = query_execution_stats.get(key)
            if isinstance(raw_value, int):
                cache_table.add_row(label, str(raw_value))

        console.print(cache_table)
        console.print()


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
        indexed_count = _as_int(index_state.get("indexed_count"))
        total_count = _as_int(index_state.get("total_count"))

    console.print("[yellow]Search service is initializing.[/yellow]")
    console.print(f"[dim]Lifecycle:[/dim] {lifecycle}")
    if isinstance(configured_root_count, int):
        console.print(f"[dim]Configured roots:[/dim] {configured_root_count}")
    console.print(
        f"[dim]Index state:[/dim] {status} ({indexed_count}/{total_count})"
    )
    if include_git_metadata:
        console.print(
            f"[dim]Total commits indexed:[/dim] {_as_int(payload.get('total_commits_indexed'))}"
        )
    console.print("[dim]Results will appear once background initialization completes.[/dim]")


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
        footer=str(_as_int(payload.get("discovered_files"))),
    )
    table.add_column(
        "Indexed docs≈",
        justify="right",
        footer=str(_as_int(payload.get("indexed_documents"))),
    )
    table.add_column(
        "Indexed chunks≈",
        justify="right",
        footer=str(_as_int(payload.get("indexed_chunks"))),
    )
    table.add_column(
        "Remaining≈",
        justify="right",
        footer=str(_as_int(payload.get("remaining_estimate"))),
    )

    for row in per_root_rows:
        if not isinstance(row, dict):
            continue
        table.add_row(
            str(row.get("root_path", "(unknown)")),
            str(_as_int(row.get("discovered_files"))),
            str(_as_int(row.get("indexed_documents_estimate"))),
            str(_as_int(row.get("indexed_chunks_estimate"))),
            str(_as_int(row.get("remaining_estimate"))),
        )

    caption_parts = []
    if payload.get("per_root_counts_are_estimates"):
        caption_parts.append(
            "≈ per-root indexed counts are estimated from indexed file paths; aggregate indexed totals remain exact."
        )
    unattributed_documents = _as_int(payload.get("unattributed_indexed_documents"))
    unattributed_chunks = _as_int(payload.get("unattributed_indexed_chunks"))
    if unattributed_documents > 0 or unattributed_chunks > 0:
        caption_parts.append(
            f"Unattributed indexed items: {unattributed_documents} docs / {unattributed_chunks} chunks."
        )
    if caption_parts:
        table.caption = " ".join(caption_parts)

    console.print(table)
