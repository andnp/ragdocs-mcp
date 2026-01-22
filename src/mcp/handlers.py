from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Awaitable

from mcp.types import TextContent

from src.mcp.validation import (
    ValidationError,
    validate_query,
    validate_integer_range,
    validate_float_range,
    validate_string_list,
    validate_boolean,
)
from src.search.pipeline import SearchPipelineConfig
from src.search.utils import classify_query_type, truncate_content

if TYPE_CHECKING:
    from src.context import ApplicationContext
    from src.lifecycle import LifecycleCoordinator

logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100

ToolHandler = Callable[["HandlerContext", dict], Awaitable[list[TextContent]]]

_TOOL_HANDLERS: dict[str, ToolHandler] = {}


def tool_handler(name: str):
    def decorator(func: ToolHandler) -> ToolHandler:
        _TOOL_HANDLERS[name] = func
        return func
    return decorator


def get_handler(name: str) -> ToolHandler | None:
    return _TOOL_HANDLERS.get(name)


def get_all_handlers() -> dict[str, ToolHandler]:
    return _TOOL_HANDLERS.copy()


class HandlerContext:
    def __init__(
        self,
        ctx: ApplicationContext | None,
        coordinator: LifecycleCoordinator,
    ):
        self.ctx = ctx
        self.coordinator = coordinator

    def require_ctx(self) -> ApplicationContext:
        if self.ctx is None:
            raise RuntimeError("Server not initialized")
        return self.ctx

    async def wait_for_ready(self, timeout: float = 60.0) -> None:
        if self.ctx and not self.ctx.is_ready():
            logger.info("Query received while initializing, waiting for indices...")
            await self.coordinator.wait_ready(timeout=timeout)


async def _query_documents_impl(
    hctx: HandlerContext,
    arguments: dict,
    max_chunks_per_doc: int,
    result_header: str,
) -> list[TextContent]:
    """Query documents implementation with comprehensive validation."""
    try:
        # Validate required parameters
        query = validate_query(arguments, "query")

        # Validate optional parameters with proper defaults and ranges
        top_n = validate_integer_range(arguments, "top_n", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N)
        min_score = validate_float_range(arguments, "min_score", default=0.3, min_val=0.0, max_val=1.0)
        similarity_threshold = validate_float_range(
            arguments, "similarity_threshold", default=0.85, min_val=0.5, max_val=1.0
        )
        show_stats = validate_boolean(arguments, "show_stats", default=False)

        # Validate list parameters
        excluded_files_raw = validate_string_list(arguments, "excluded_files", default=[])

    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]

    await hctx.wait_for_ready()

    ctx = hctx.require_ctx()

    # Normalize excluded file paths
    excluded_files = None
    if excluded_files_raw:
        from src.search.path_utils import normalize_path
        docs_root = ctx.orchestrator.documents_path
        excluded_files = {normalize_path(f, docs_root) for f in excluded_files_raw}

    top_k = max(20, top_n * 4)

    pipeline_config = SearchPipelineConfig(
        min_confidence=min_score,
        max_chunks_per_doc=max_chunks_per_doc,
        dedup_enabled=True,
        dedup_threshold=similarity_threshold,
        rerank_enabled=False,
    )

    results, stats, _ = await ctx.orchestrator.query(
        query,
        top_k=top_k,
        top_n=top_n,
        pipeline_config=pipeline_config,
        excluded_files=excluded_files,
    )

    query_type = classify_query_type(query)

    results_text = "\n\n".join([
        f"[{i+1}] {r.file_path or 'unknown'} § {r.header_path or '(no section)'} ({r.score:.2f})\n"
        f"{truncate_content(r.content, 200) if query_type == 'factual' else r.content}"
        for i, r in enumerate(results)
    ])

    filtering_occurred = stats.original_count > stats.after_dedup
    if show_stats or filtering_occurred:
        stats_parts = [
            f"- Original results: {stats.original_count}",
            f"- After score filter (≥{min_score}): {stats.after_threshold}",
            f"- After deduplication: {stats.after_dedup}",
        ]
        if max_chunks_per_doc == 1:
            stats_parts.append(f"- After document limit (1 per doc): {stats.after_doc_limit}")
        stats_parts.append(f"- Clusters merged: {stats.clusters_merged}")
        stats_text = "\n".join(stats_parts)
        response = f"# {result_header}\n\n{results_text}\n\n# Compression Stats\n\n{stats_text}"
    else:
        response = f"# {result_header}\n\n{results_text}"

    return [TextContent(type="text", text=response)]


@tool_handler("query_documents")
async def handle_query_documents(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    # Handle optional uniqueness_mode parameter (Priority B unification)
    uniqueness_mode = arguments.get("uniqueness_mode", "allow_multiple")
    
    if uniqueness_mode == "one_per_document":
        max_chunks_per_doc = 1
        result_header = "Search Results (Unique Documents)"
    elif uniqueness_mode == "allow_multiple":
        max_chunks_per_doc = 0
        result_header = "Search Results"
    else:
        raise ValueError(f"Invalid uniqueness_mode: {uniqueness_mode}. Must be 'allow_multiple' or 'one_per_document'")
    
    return await _query_documents_impl(
        hctx,
        arguments,
        max_chunks_per_doc=max_chunks_per_doc,
        result_header=result_header,
    )


@tool_handler("query_unique_documents")
async def handle_query_unique_documents(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    # DEPRECATED: Maintained for backward compatibility
    # New code should use query_documents with uniqueness_mode="one_per_document"
    return await _query_documents_impl(
        hctx,
        arguments,
        max_chunks_per_doc=1,
        result_header="Search Results (Unique Documents)",
    )


@tool_handler("search_with_hypothesis")
async def handle_search_with_hypothesis(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    """Search with hypothesis (HyDE technique) with comprehensive validation."""
    try:
        # Validate required parameters
        hypothesis = validate_query(arguments, "hypothesis")

        # Validate optional parameters with proper defaults and ranges
        top_n = validate_integer_range(arguments, "top_n", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N)

        # Validate list parameters
        excluded_files_raw = validate_string_list(arguments, "excluded_files", default=[])

    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]

    await hctx.wait_for_ready()

    ctx = hctx.require_ctx()

    # Normalize excluded file paths
    excluded_files = None
    if excluded_files_raw:
        from src.search.path_utils import normalize_path
        docs_root = ctx.orchestrator.documents_path
        excluded_files = {normalize_path(f, docs_root) for f in excluded_files_raw}

    top_k = max(20, top_n * 4)

    results, _, _ = await ctx.orchestrator.query_with_hypothesis(
        hypothesis,
        top_k=top_k,
        top_n=top_n,
        excluded_files=excluded_files,
    )

    results_text = "\n\n".join([
        f"[{i+1}] {r.file_path or 'unknown'} § {r.header_path or '(no section)'} ({r.score:.2f})\n{r.content}"
        for i, r in enumerate(results)
    ])

    response = f"# HyDE Search Results\n\n{results_text}"
    return [TextContent(type="text", text=response)]


@tool_handler("search_git_history")
async def handle_search_git_history(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    if ctx.commit_indexer is None:
        return [TextContent(
            type="text",
            text="Git history search is not available (git binary not found or disabled in config)"
        )]

    from src.git.commit_search import search_git_history

    query = arguments["query"]
    top_n = max(MIN_TOP_N, min(arguments.get("top_n", 5), MAX_TOP_N))
    files_glob = arguments.get("files_glob")
    after_timestamp = arguments.get("after_timestamp")
    before_timestamp = arguments.get("before_timestamp")

    response = await asyncio.to_thread(
        search_git_history,
        ctx.commit_indexer,
        query,
        top_n,
        files_glob,
        after_timestamp,
        before_timestamp,
    )

    output_lines = [
        "# Git History Search Results",
        "",
        f"**Query:** {response.query}",
        f"**Total Commits Indexed:** {response.total_commits_indexed}",
        f"**Results Returned:** {len(response.results)}",
        "",
    ]

    for i, commit in enumerate(response.results, 1):
        commit_date = datetime.fromtimestamp(commit.timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        output_lines.extend([
            f"## {i}. {commit.title}",
            "",
            f"**Commit:** `{commit.hash[:8]}`",
            f"**Author:** {commit.author}",
            f"**Date:** {commit_date}",
            f"**Score:** {commit.score:.3f}",
            "",
        ])

        if commit.message:
            output_lines.extend([
                "### Message",
                "",
                commit.message,
                "",
            ])

        if commit.files_changed:
            output_lines.extend([
                f"### Files Changed ({len(commit.files_changed)})",
                "",
            ])

            for file_path in commit.files_changed[:10]:
                output_lines.append(f"- `{file_path}`")

            if len(commit.files_changed) > 10:
                output_lines.append(f"- ... and {len(commit.files_changed) - 10} more")

            output_lines.append("")

        if commit.delta_truncated:
            delta_display = commit.delta_truncated[:1000]
            if len(commit.delta_truncated) > 1000:
                delta_display += "\n... (truncated for display)"

            output_lines.extend([
                "### Delta (truncated)",
                "",
                "```diff",
                delta_display,
                "```",
                "",
            ])

        output_lines.extend(["---", ""])

    return [TextContent(type="text", text="\n".join(output_lines))]
