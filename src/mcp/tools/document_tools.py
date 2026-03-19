"""Document query and search tools for MCP."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from mcp.types import Tool, TextContent

from src.mcp.handlers import (
    HandlerContext,
    format_search_status_text,
    tool_handler,
    MIN_TOP_N,
    MAX_TOP_N,
)
from src.mcp.validation import (
    ValidationError,
    validate_query,
    validate_integer_range,
    validate_float_range,
    validate_string_list,
    validate_optional_string,
    validate_boolean,
)
from src.search.pipeline import SearchPipelineConfig
from src.search.utils import classify_query_type, truncate_content

logger = logging.getLogger(__name__)


def get_document_tools() -> list[Tool]:
    """Return tool schema definitions for document search tools."""
    return [
        Tool(
            name="query_documents",
            description=(
                "Search local documentation using hybrid search (semantic + keyword + graph). "
                + "Returns ranked document chunks with relevance scores. "
                + "Use for discovering relevant documentation sections in a large corpus. "
                + "Supports optional uniqueness_mode parameter for document-unique results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query or question about the documentation",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": f"Maximum number of results to return (default: 5, max: {MAX_TOP_N})",
                        "default": 5,
                        "minimum": MIN_TOP_N,
                        "maximum": MAX_TOP_N,
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum relevance score threshold (default: 0.3)",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Cosine similarity threshold for deduplication (default: 0.85)",
                        "default": 0.85,
                        "minimum": 0.5,
                        "maximum": 1.0,
                    },
                    "show_stats": {
                        "type": "boolean",
                        "description": "Whether to show compression stats in response (default: false)",
                        "default": False,
                    },
                    "excluded_files": {
                        "type": "array",
                        "description": "List of file paths to exclude from results (supports filename, relative path, or absolute path)",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "project_filter": {
                        "type": "array",
                        "description": "Optional list of project IDs to explicitly filter results to",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "project_context": {
                        "type": "string",
                        "description": "Optional active project context used for bounded ranking uplift",
                    },
                    "uniqueness_mode": {
                        "type": "string",
                        "enum": ["allow_multiple", "one_per_document"],
                        "description": "Result uniqueness mode: 'allow_multiple' (default) returns multiple chunks per document, 'one_per_document' returns at most one chunk per document for breadth across files",
                        "default": "allow_multiple",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_with_hypothesis",
            description=(
                "Search documentation using a hypothesis about what the answer might look like. "
                + "Useful for vague queries where you can describe the expected documentation content. "
                + "The hypothesis is embedded and used directly for semantic search (HyDE technique)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "A hypothesis describing what the expected documentation content looks like",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": f"Maximum number of results to return (default: 5, max: {MAX_TOP_N})",
                        "default": 5,
                        "minimum": MIN_TOP_N,
                        "maximum": MAX_TOP_N,
                    },
                    "excluded_files": {
                        "type": "array",
                        "description": "List of file paths to exclude from results",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "project_filter": {
                        "type": "array",
                        "description": "Optional list of project IDs to explicitly filter results to",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "project_context": {
                        "type": "string",
                        "description": "Optional active project context used for bounded ranking uplift",
                    },
                },
                "required": ["hypothesis"],
            },
        ),
        Tool(
            name="search_git_history",
            description=(
                "Search git commit history using natural language queries. "
                + "Returns relevant commits with metadata, message, and diff context. "
                + "Supports filtering by file glob patterns and timestamp ranges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing commits to find",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": f"Maximum number of commits to return (default: 5, max: {MAX_TOP_N})",
                        "default": 5,
                        "minimum": MIN_TOP_N,
                        "maximum": MAX_TOP_N,
                    },
                    "files_glob": {
                        "type": "string",
                        "description": "Optional glob pattern to filter by changed files (e.g., 'src/**/*.py')",
                    },
                    "after_timestamp": {
                        "type": "integer",
                        "description": "Optional Unix timestamp to filter commits after this date",
                    },
                    "before_timestamp": {
                        "type": "integer",
                        "description": "Optional Unix timestamp to filter commits before this date",
                    },
                    "project_filter": {
                        "type": "array",
                        "description": "Optional list of project IDs to explicitly filter results to",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "project_context": {
                        "type": "string",
                        "description": "Optional active project context used for bounded ranking uplift",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


async def _query_documents_impl(
    hctx: HandlerContext,
    arguments: dict,
    max_chunks_per_doc: int,
    result_header: str,
) -> list[TextContent]:
    """Query documents implementation with comprehensive validation."""
    try:
        query = validate_query(arguments, "query")
        top_n = validate_integer_range(
            arguments, "top_n", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N
        )
        min_score = validate_float_range(
            arguments, "min_score", default=0.3, min_val=0.0, max_val=1.0
        )
        similarity_threshold = validate_float_range(
            arguments, "similarity_threshold", default=0.85, min_val=0.5, max_val=1.0
        )
        show_stats = validate_boolean(arguments, "show_stats", default=False)
        excluded_files_raw = validate_string_list(
            arguments, "excluded_files", default=[]
        )
        project_filter = validate_string_list(arguments, "project_filter", default=[])
        project_context = validate_optional_string(arguments, "project_context")

    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]

    cold_start_payload = hctx.get_nonblocking_search_payload(query=query)
    if cold_start_payload is not None:
        return [
            TextContent(
                type="text",
                text=format_search_status_text(result_header, cold_start_payload),
            )
        ]

    await hctx.wait_for_ready()

    ctx = hctx.require_ctx()

    excluded_files = None
    if excluded_files_raw:
        from src.search.path_utils import normalize_path

        docs_root = ctx.orchestrator.documents_path
        excluded_files = {normalize_path(f, docs_root) for f in excluded_files_raw}

    top_k = max(20, top_n * 4)
    if project_filter:
        top_k = max(top_k, top_n * 10)

    pipeline_config = SearchPipelineConfig(
        min_confidence=min_score,
        max_chunks_per_doc=max_chunks_per_doc,
        dedup_threshold=similarity_threshold,
    )

    results, stats, _ = await ctx.orchestrator.query(
        query,
        top_k=top_k,
        top_n=top_n,
        pipeline_config=pipeline_config,
        excluded_files=excluded_files,
        project_filter=project_filter,
        project_context=project_context,
    )

    query_type = classify_query_type(query)

    results_text = "\n\n".join(
        [
            f"[{i + 1}] {r.file_path or 'unknown'} § {r.header_path or '(no section)'} ({r.score:.2f})\n"
            f"{truncate_content(r.content, 200) if query_type == 'factual' else r.content}"
            for i, r in enumerate(results)
        ]
    )

    filtering_occurred = stats.original_count > stats.after_dedup
    if show_stats or filtering_occurred:
        stats_parts = [
            f"- Original results: {stats.original_count}",
            f"- After score filter (≥{min_score}): {stats.after_threshold}",
            f"- After deduplication: {stats.after_dedup}",
        ]
        if max_chunks_per_doc == 1:
            stats_parts.append(
                f"- After document limit (1 per doc): {stats.after_doc_limit}"
            )
        stats_parts.append(f"- Clusters merged: {stats.clusters_merged}")
        stats_text = "\n".join(stats_parts)
        response = f"# {result_header}\n\n{results_text}\n\n# Compression Stats\n\n{stats_text}"
    else:
        response = f"# {result_header}\n\n{results_text}"

    return [TextContent(type="text", text=response)]


@tool_handler("query_documents")
async def handle_query_documents(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    uniqueness_mode = arguments.get("uniqueness_mode", "allow_multiple")

    if uniqueness_mode == "one_per_document":
        max_chunks_per_doc = 1
        result_header = "Search Results (Unique Documents)"
    elif uniqueness_mode == "allow_multiple":
        max_chunks_per_doc = 0
        result_header = "Search Results"
    else:
        raise ValueError(
            f"Invalid uniqueness_mode: {uniqueness_mode}. Must be 'allow_multiple' or 'one_per_document'"
        )

    return await _query_documents_impl(
        hctx,
        arguments,
        max_chunks_per_doc=max_chunks_per_doc,
        result_header=result_header,
    )


@tool_handler("search_with_hypothesis")
async def handle_search_with_hypothesis(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    """Search with hypothesis (HyDE technique) with comprehensive validation."""
    try:
        hypothesis = validate_query(arguments, "hypothesis")
        top_n = validate_integer_range(
            arguments, "top_n", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N
        )
        excluded_files_raw = validate_string_list(
            arguments, "excluded_files", default=[]
        )
        project_filter = validate_string_list(arguments, "project_filter", default=[])
        project_context = validate_optional_string(arguments, "project_context")

    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]

    await hctx.wait_for_ready()

    ctx = hctx.require_ctx()

    excluded_files = None
    if excluded_files_raw:
        from src.search.path_utils import normalize_path

        docs_root = ctx.orchestrator.documents_path
        excluded_files = {normalize_path(f, docs_root) for f in excluded_files_raw}

    top_k = max(20, top_n * 4)
    if project_filter:
        top_k = max(top_k, top_n * 10)

    results, _, _ = await ctx.orchestrator.query_with_hypothesis(
        hypothesis,
        top_k=top_k,
        top_n=top_n,
        excluded_files=excluded_files,
        project_filter=project_filter,
        project_context=project_context,
    )

    results_text = "\n\n".join(
        [
            f"[{i + 1}] {r.file_path or 'unknown'} § {r.header_path or '(no section)'} ({r.score:.2f})\n{r.content}"
            for i, r in enumerate(results)
        ]
    )

    response = f"# HyDE Search Results\n\n{results_text}"
    return [TextContent(type="text", text=response)]


@tool_handler("search_git_history")
async def handle_search_git_history(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    try:
        query = validate_query(arguments, "query")
        top_n = validate_integer_range(
            arguments, "top_n", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N
        )
        files_glob = validate_optional_string(arguments, "files_glob")
        after_timestamp = arguments.get("after_timestamp")
        before_timestamp = arguments.get("before_timestamp")
        project_filter = validate_string_list(arguments, "project_filter", default=[])
        project_context = validate_optional_string(arguments, "project_context")
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]

    cold_start_payload = hctx.get_nonblocking_search_payload(
        query=query,
        include_git_metadata=True,
    )
    if cold_start_payload is not None:
        return [
            TextContent(
                type="text",
                text=format_search_status_text(
                    "Git History Search Results",
                    cold_start_payload,
                    include_git_metadata=True,
                ),
            )
        ]

    await hctx.wait_for_ready()
    ctx = hctx.require_ctx()

    if ctx.commit_indexer is None:
        return [
            TextContent(
                type="text",
                text="Git history search is not available (git binary not found or disabled in config)",
            )
        ]

    from src.git.commit_search import search_git_history

    response = await asyncio.to_thread(
        search_git_history,
        ctx.commit_indexer,
        query,
        top_n,
        files_glob,
        after_timestamp,
        before_timestamp,
        project_filter,
        project_context,
        ctx.config,
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
        commit_date = datetime.fromtimestamp(commit.timestamp, timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        output_lines.extend(
            [
                f"## {i}. {commit.title}",
                "",
                f"**Commit:** `{commit.hash[:8]}`",
                f"**Author:** {commit.author}",
                f"**Date:** {commit_date}",
                f"**Score:** {commit.score:.3f}",
                "",
            ]
        )

        if commit.message:
            output_lines.extend(
                [
                    "### Message",
                    "",
                    commit.message,
                    "",
                ]
            )

        if commit.files_changed:
            output_lines.extend(
                [
                    f"### Files Changed ({len(commit.files_changed)})",
                    "",
                ]
            )

            for file_path in commit.files_changed[:10]:
                output_lines.append(f"- `{file_path}`")

            if len(commit.files_changed) > 10:
                output_lines.append(f"- ... and {len(commit.files_changed) - 10} more")

            output_lines.append("")

        if commit.delta_truncated:
            delta_display = commit.delta_truncated[:1000]
            if len(commit.delta_truncated) > 1000:
                delta_display += "\n... (truncated for display)"

            output_lines.extend(
                [
                    "### Delta (truncated)",
                    "",
                    "```diff",
                    delta_display,
                    "```",
                    "",
                ]
            )

        output_lines.extend(["---", ""])

    return [TextContent(type="text", text="\n".join(output_lines))]


