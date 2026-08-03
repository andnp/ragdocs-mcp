"""Document query and search tools for MCP."""

from __future__ import annotations

import logging
import json
from pathlib import Path

from mcp.types import TextContent, Tool

from mcp_markdown_ragdocs.app.search import SearchQuery
from mcp_markdown_ragdocs.mcp.handlers import (
    MAX_TOP_N,
    MIN_TOP_N,
    HandlerContext,
    tool_handler,
)
from mcp_markdown_ragdocs.mcp.tools.document_request import (
    NormalizedQueryDocumentsRequest,
    normalize_query_documents_request,
)
from mcp_markdown_ragdocs.mcp.tools.document_response import (
    build_query_documents_response_envelope,
    build_query_documents_status_envelope,
    build_query_documents_validation_error,
    build_compact_document_results_response,
    build_compact_error_response,
)
from mcp_markdown_ragdocs.mcp.validation import (
    ValidationError,
    validate_integer_range,
    validate_boolean,
    validate_optional_string,
    validate_query,
    validate_string_list,
)
from mcp_markdown_ragdocs.models import ChunkResult
from searchkernel.api import classify_query_type, normalize_path

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
                        "description": "Minimum canonical search score threshold (default: 0.0)",
                        "default": 0.0,
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
                    "excluded_files": {
                        "type": "array",
                        "description": "List of file paths to exclude from results (supports filename, relative path, or absolute path)",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "scope_mode": {
                        "type": "string",
                        "enum": ["global", "active_project", "explicit_projects"],
                        "description": "Canonical scope mode for agents: 'global' keeps search corpus-wide, 'active_project' applies bounded uplift for the active or preferred project, and 'explicit_projects' hard-filters to scope_projects.",
                        "default": "global",
                    },
                    "scope_projects": {
                        "type": "array",
                        "description": "Canonical project IDs for explicit scoping. Used when scope_mode is 'explicit_projects'.",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "preferred_project": {
                        "type": "string",
                        "description": "Canonical preferred project used for bounded ranking uplift when scope_mode is 'active_project' or 'explicit_projects'.",
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
                    "include_diff": {
                        "type": "boolean",
                        "description": "Include matched diff text in each commit result (default: false)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


async def _query_documents_impl(
    hctx: HandlerContext,
    request: NormalizedQueryDocumentsRequest,
) -> list[TextContent]:
    """Query documents implementation with comprehensive validation."""
    cold_start_payload = hctx.get_nonblocking_search_payload(query=request.query)
    if cold_start_payload is not None:
        status = str(cold_start_payload.get("status", "initializing"))
        response = build_query_documents_status_envelope(
            request,
            status=status,
            payload=cold_start_payload,
        ).render_text()
        return [
            TextContent(
                type="text",
                text=response,
            )
        ]

    await hctx.wait_for_ready()

    ctx = hctx.require_ctx()

    excluded_files = None
    if request.excluded_files_raw:
        docs_roots = tuple(getattr(ctx, "documents_roots", ())) or (
            ctx.orchestrator.documents_path,
        )
        excluded_files = {
            normalize_path(f, docs_root)
            for f in request.excluded_files_raw
            for docs_root in docs_roots
        }

    project_context = request.project_context
    if request.scope_mode == "active_project" and project_context is None:
        project_context = getattr(
            getattr(ctx, "config", None),
            "detected_project",
            None,
        )

    search_use_case = getattr(ctx, "search_use_case", None)
    if search_use_case is None:
        search_use_case = getattr(ctx.orchestrator, "search_use_case", None)
    if search_use_case is not None:
        execution = await search_use_case.execute(
            SearchQuery(
                query=request.query,
                top_n=request.top_n,
                project_filter=tuple(request.project_filter),
                project_context=project_context,
                excluded_files=frozenset(excluded_files or ()),
                min_score=request.min_score,
                similarity_threshold=request.similarity_threshold,
                max_chunks_per_doc=request.max_chunks_per_doc,
            )
        )
        results = execution.results
        stats = execution.compression_stats
        strategy_stats = execution.strategy_stats
    else:
        top_k = max(20, request.top_n * 4)
        if request.project_filter:
            top_k = max(top_k, request.top_n * 10)
        results, stats, strategy_stats = await ctx.orchestrator.query(
            request.query,
            top_k=top_k,
            top_n=request.top_n,
            pipeline_config=None,
            excluded_files=excluded_files,
            project_filter=request.project_filter,
            project_context=project_context,
            min_score=request.min_score,
            similarity_threshold=request.similarity_threshold,
            max_chunks_per_doc=request.max_chunks_per_doc,
        )
    results = [
        result if isinstance(result, ChunkResult) else ChunkResult.from_domain(result)
        for result in results
    ]

    query_type = classify_query_type(request.query)
    response = build_query_documents_response_envelope(
        request,
        query_type=query_type,
        results=results,
        strategy_stats=strategy_stats,
        compression_stats=stats,
        effective_project_context=project_context,
    ).render_text()

    return [TextContent(type="text", text=response)]


@tool_handler("query_documents")
async def handle_query_documents(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    try:
        request = normalize_query_documents_request(arguments)
    except (ValidationError, ValueError) as e:
        raw_query = arguments.get("query")
        query = raw_query if isinstance(raw_query, str) else ""
        response = build_query_documents_validation_error(
            query=query,
            message=str(e),
        ).render_text()
        return [TextContent(type="text", text=response)]

    return await _query_documents_impl(hctx, request)


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
        return [
            TextContent(
                type="text",
                text=build_compact_error_response(f"Validation error: {e}"),
            )
        ]

    await hctx.wait_for_ready()

    ctx = hctx.require_ctx()

    excluded_files = None
    if excluded_files_raw:
        docs_roots = tuple(getattr(ctx, "documents_roots", ())) or (
            ctx.orchestrator.documents_path,
        )
        excluded_files = {
            normalize_path(f, docs_root)
            for f in excluded_files_raw
            for docs_root in docs_roots
        }

    search_use_case = getattr(ctx, "search_use_case", None)
    if search_use_case is None:
        search_use_case = getattr(ctx.orchestrator, "search_use_case", None)
    if search_use_case is not None:
        execution = await search_use_case.execute(
            SearchQuery(
                query=hypothesis,
                top_n=top_n,
                project_filter=tuple(project_filter),
                project_context=project_context,
                excluded_files=frozenset(excluded_files or ()),
                retrieval_mode="semantic_only",
            )
        )
        results = execution.results
    else:
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
        results = [
            result if isinstance(result, ChunkResult) else ChunkResult.from_domain(result)
            for result in results
        ]

    return [
        TextContent(type="text", text=build_compact_document_results_response(results))
    ]


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
        include_diff = validate_boolean(arguments, "include_diff", default=False)
    except ValidationError as e:
        return [TextContent(type="text", text=build_compact_error_response(str(e)))]

    cold_start_payload = hctx.get_nonblocking_search_payload(
        query=query,
        include_git_metadata=True,
    )
    if cold_start_payload is not None:
        return [TextContent(type="text", text=json.dumps(
            {"status": cold_start_payload.get("status", "unknown"), "results": [],
             **({"message": cold_start_payload.get("message")} if cold_start_payload.get("message") else {}),
             **({"error": cold_start_payload.get("error")} if cold_start_payload.get("error") else {})},
            ensure_ascii=False, separators=(",", ":")))]

    await hctx.wait_for_ready()
    ctx = hctx.require_ctx()

    if not ctx.git_indexing_enabled:
        return [TextContent(type="text", text=json.dumps(
            {"status": "error", "error": "git_history_unavailable", "results": []},
            separators=(",", ":")))]

    # Over-fetch: a downstream files_glob/timestamp filter narrows the pool,
    # and each commit may span multiple chunks under source_filter=["git_commit"].
    overfetch_multiplier = 10 if (files_glob or after_timestamp or before_timestamp) else 4
    results, _, _ = await ctx.orchestrator.query(
        query,
        top_k=max(20, top_n * overfetch_multiplier),
        top_n=top_n * overfetch_multiplier,
        project_filter=project_filter,
        project_context=project_context,
        source_filter=["git_commit"],
    )
    results = [
        result if isinstance(result, ChunkResult) else ChunkResult.from_domain(result)
        for result in results
    ]

    commits = _filter_commit_results(results, files_glob, after_timestamp, before_timestamp)[
        :top_n
    ]

    compact_commits = []
    for result in commits:
        metadata = result.metadata
        commit_hash = result.doc_id.removeprefix("git:")
        files_changed = metadata.get("files_changed") or []
        item = {
            "hash": commit_hash,
            "title": metadata.get("title", ""),
            "author": metadata.get("author", "Unknown"),
            "timestamp": metadata.get("timestamp"),
            "score": result.score,
            "files_changed": files_changed,
        }
        if metadata.get("committer") is not None:
            item["committer"] = metadata["committer"]
        if metadata.get("chunk_section") == "body" and result.content:
            item["message"] = result.content
        if include_diff and metadata.get("chunk_section") == "diff" and result.content:
            item["diff"] = result.content
        compact_commits.append(item)

    return [TextContent(type="text", text=json.dumps(
        {"status": "ok", "results": compact_commits},
        ensure_ascii=False, separators=(",", ":")))]


def _filter_commit_results(
    results: list,
    files_glob: str | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
) -> list:
    """Post-filter git_commit ChunkResults by files_glob/timestamp bounds.

    SearchOrchestrator.query's source_filter narrows by source_kind only; the
    files_glob/after/before filters git history search has always exposed are
    applied here against the same commit metadata GitContentSource attaches.
    """
    if not (files_glob or after_timestamp is not None or before_timestamp is not None):
        return results

    filtered = []
    for result in results:
        metadata = result.metadata
        timestamp = metadata.get("timestamp")
        if after_timestamp is not None and (timestamp is None or timestamp <= after_timestamp):
            continue
        if before_timestamp is not None and (timestamp is None or timestamp >= before_timestamp):
            continue
        if files_glob:
            files_changed = metadata.get("files_changed") or []
            if not any(Path(f).match(files_glob) for f in files_changed):
                continue
        filtered.append(result)
    return filtered
