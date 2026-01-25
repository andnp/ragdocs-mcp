"""Memory CRUD and search tools for MCP."""

from __future__ import annotations

from mcp.types import Tool, TextContent

from src.mcp.handlers import HandlerContext, tool_handler, MIN_TOP_N, MAX_TOP_N
from src.mcp.validation import (
    ValidationError,
    validate_query,
    validate_integer_range,
    validate_enum,
    validate_boolean,
    validate_timestamp,
)
from src.memory import tools as memory_tools_impl


def _text_response(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def get_memory_tools() -> list[Tool]:
    """Return tool schema definitions for memory management tools."""
    return [
        Tool(
            name="create_memory",
            description=(
                "Create a new memory file in the Memory Bank. "
                + "Memories are persistent notes that AI assistants can use for long-term storage. "
                + "Fails if the file already exists. "
                + "IMPORTANT: The system automatically generates YAML frontmatter with type, status, tags, and created_at. "
                + "DO NOT include frontmatter in the content parameter - provide only the body text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the memory file (without .md extension)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The body content of the memory in markdown format (NO frontmatter - system auto-generates it)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorizing the memory (will be added to auto-generated frontmatter)",
                        "default": [],
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["plan", "journal", "fact", "observation", "reflection"],
                        "description": "Type of memory (default: journal)",
                        "default": "journal",
                    },
                },
                "required": ["filename", "content"],
            },
        ),
        Tool(
            name="append_memory",
            description="Append content to an existing memory file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the memory file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append",
                    },
                },
                "required": ["filename", "content"],
            },
        ),
        Tool(
            name="read_memory",
            description="Read the full content of a memory file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the memory file",
                    },
                },
                "required": ["filename"],
            },
        ),
        Tool(
            name="update_memory",
            description="Replace the entire content of a memory file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the memory file",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content to replace the file with",
                    },
                },
                "required": ["filename", "content"],
            },
        ),
        Tool(
            name="delete_memory",
            description="Delete a memory file (moves to .trash/ for safety).",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the memory file to delete",
                    },
                },
                "required": ["filename"],
            },
        ),
        Tool(
            name="search_memories",
            description=(
                "Search the Memory Bank using hybrid search (semantic + keyword). "
                + "Returns memories ranked by relevance with recency boost applied."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    },
                    "filter_type": {
                        "type": "string",
                        "enum": ["plan", "journal", "fact", "observation", "reflection"],
                        "description": "Only return memories of this type",
                    },
                    "load_full_memory": {
                        "type": "boolean",
                        "description": "Load complete memory file content instead of just matching chunks (default: true)",
                        "default": True,
                    },
                    "after_timestamp": {
                        "type": "integer",
                        "description": "Unix timestamp: only return memories created/modified after this time",
                    },
                    "before_timestamp": {
                        "type": "integer",
                        "description": "Unix timestamp: only return memories created/modified before this time",
                    },
                    "relative_days": {
                        "type": "integer",
                        "description": "Only return memories from last N days (overrides absolute timestamps)",
                        "minimum": 0,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@tool_handler("create_memory")
async def handle_create_memory(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")
    tags = arguments.get("tags", [])
    memory_type = arguments.get("memory_type", "journal")

    result = await memory_tools_impl.create_memory(
        ctx, filename, content, tags, memory_type
    )
    return _text_response(str(result))


@tool_handler("append_memory")
async def handle_append_memory(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")

    result = await memory_tools_impl.append_memory(ctx, filename, content)
    return _text_response(str(result))


@tool_handler("read_memory")
async def handle_read_memory(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    result = await memory_tools_impl.read_memory(ctx, filename)

    if "error" in result:
        return _text_response(str(result))
    return _text_response(result.get("content", ""))


@tool_handler("update_memory")
async def handle_update_memory(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")

    result = await memory_tools_impl.update_memory(ctx, filename, content)
    return _text_response(str(result))


@tool_handler("delete_memory")
async def handle_delete_memory(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    result = await memory_tools_impl.delete_memory(ctx, filename)
    return _text_response(str(result))


@tool_handler("search_memories")
async def handle_search_memories(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    """Search memories with comprehensive input validation."""
    ctx = hctx.require_ctx()

    try:
        query = validate_query(arguments, "query")
        limit = validate_integer_range(
            arguments, "limit", default=5, min_val=MIN_TOP_N, max_val=MAX_TOP_N
        )
        load_full_memory = validate_boolean(
            arguments, "load_full_memory", default=False
        )
        filter_type = validate_enum(
            arguments,
            "filter_type",
            allowed_values={"plan", "journal", "fact", "observation", "reflection"},
            default=None,
        )
        after_timestamp = validate_timestamp(arguments, "after_timestamp", default=None)
        before_timestamp = validate_timestamp(
            arguments, "before_timestamp", default=None
        )
        relative_days = arguments.get("relative_days")

        if after_timestamp is not None and before_timestamp is not None:
            if after_timestamp >= before_timestamp:
                raise ValidationError(
                    f"after_timestamp ({after_timestamp}) must be less than "
                    f"before_timestamp ({before_timestamp})"
                )

    except ValidationError as e:
        return _text_response(f"Validation error: {e}")

    results = await memory_tools_impl.search_memories(
        ctx,
        query,
        limit=limit,
        filter_type=filter_type,
        load_full_memory=load_full_memory,
        after_timestamp=after_timestamp,
        before_timestamp=before_timestamp,
        relative_days=relative_days,
    )

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = ["# Memory Search Results", ""]

    for i, r in enumerate(results, 1):
        output_lines.extend(
            [
                f"## {i}. {r.get('memory_id', 'unknown')} (score: {r.get('score', 0):.3f})",
                f"**Type:** {r.get('type', 'unknown')} | **Tags:** {', '.join(r.get('tags', []))}",
                "",
                r.get("content", "")[:500],
                "",
                "---",
                "",
            ]
        )

    return _text_response("\n".join(output_lines))
