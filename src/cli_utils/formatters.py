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
) -> None:
    from src.models import SearchStrategyStats, CompressionStats

    if isinstance(strategy_stats, SearchStrategyStats):
        strategy_table = Table(title="Search Strategy Results", show_header=True, title_style="bold yellow")
        strategy_table.add_column("Strategy", style="cyan")
        strategy_table.add_column("Count", style="green", justify="right")

        if strategy_stats.vector_count is not None:
            strategy_table.add_row("Vector (Semantic)", str(strategy_stats.vector_count))
        if strategy_stats.keyword_count is not None:
            strategy_table.add_row("Keyword (BM25)", str(strategy_stats.keyword_count))
        if strategy_stats.graph_count is not None:
            strategy_table.add_row("Graph (PageRank)", str(strategy_stats.graph_count))
        if strategy_stats.code_count is not None:
            strategy_table.add_row("Code Search", str(strategy_stats.code_count))
        if strategy_stats.tag_expansion_count is not None:
            strategy_table.add_row("Tag Expansion", str(strategy_stats.tag_expansion_count))

        console.print(strategy_table)
        console.print()

    if isinstance(compression_stats, CompressionStats):
        compression_table = Table(title="Compression Pipeline", show_header=True, title_style="bold yellow")
        compression_table.add_column("Stage", style="cyan")
        compression_table.add_column("Count", style="green", justify="right")
        compression_table.add_column("Removed", style="red", justify="right")

        compression_table.add_row("Original (RRF Fusion)", str(compression_stats.original_count), "-")
        
        removed_threshold = compression_stats.original_count - compression_stats.after_threshold
        compression_table.add_row(
            f"After Confidence Filter (≥{min_confidence:.2f})",
            str(compression_stats.after_threshold),
            str(removed_threshold) if removed_threshold > 0 else "-"
        )
        
        removed_content = compression_stats.after_threshold - compression_stats.after_content_dedup
        compression_table.add_row(
            "After Content Dedup",
            str(compression_stats.after_content_dedup),
            str(removed_content) if removed_content > 0 else "-"
        )
        
        removed_ngram = compression_stats.after_content_dedup - compression_stats.after_ngram_dedup
        compression_table.add_row(
            "After N-gram Dedup",
            str(compression_stats.after_ngram_dedup),
            str(removed_ngram) if removed_ngram > 0 else "-"
        )
        
        removed_dedup = compression_stats.after_ngram_dedup - compression_stats.after_dedup
        dedup_label = "After Semantic Dedup"
        if compression_stats.clusters_merged > 0:
            dedup_label += f" ({compression_stats.clusters_merged} clusters merged)"
        compression_table.add_row(
            dedup_label,
            str(compression_stats.after_dedup),
            str(removed_dedup) if removed_dedup > 0 else "-"
        )
        
        removed_doc_limit = compression_stats.after_dedup - compression_stats.after_doc_limit
        compression_table.add_row(
            "After Doc Limit",
            str(compression_stats.after_doc_limit),
            str(removed_doc_limit) if removed_doc_limit > 0 else "-"
        )

        console.print(compression_table)
        console.print()
