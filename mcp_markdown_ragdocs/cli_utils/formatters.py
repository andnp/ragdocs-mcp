import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


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
