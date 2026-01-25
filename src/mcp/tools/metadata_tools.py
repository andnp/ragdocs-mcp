"""Memory metadata and relationship tools for MCP."""

from __future__ import annotations

from mcp.types import Tool, TextContent

from src.mcp.handlers import HandlerContext, tool_handler
from src.memory import tools as memory_tools_impl


def _text_response(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def get_metadata_tools() -> list[Tool]:
    """Return tool schema definitions for memory metadata tools."""
    return [
        Tool(
            name="get_memory_stats",
            description="Get statistics about the Memory Bank (count, size, tags, types).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="merge_memories",
            description=(
                "Merge multiple memory files into a new summary file. "
                + "Source files are moved to .trash/ after merge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of memory filenames to merge",
                    },
                    "target_file": {
                        "type": "string",
                        "description": "Name of the new merged memory file",
                    },
                    "summary_content": {
                        "type": "string",
                        "description": "Content for the merged file (including frontmatter)",
                    },
                },
                "required": ["source_files", "target_file", "summary_content"],
            },
        ),
        Tool(
            name="search_by_tag_cluster",
            description=(
                "Find memories via tag traversal with configurable depth. "
                + "Discovers memories that share tags or are connected through tag relationships."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Tag to start cluster search from",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Traversal depth (default: 2, max: 3)",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["tag"],
            },
        ),
        Tool(
            name="get_tag_graph",
            description=(
                "Return tag nodes and co-occurrence counts across all memories. "
                + "Useful for understanding tag relationships and clusters."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="suggest_related_tags",
            description=(
                "Suggest related tags based on co-occurrence patterns. "
                + "Finds tags that frequently appear together with the specified tag."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Tag to find related tags for",
                    },
                },
                "required": ["tag"],
            },
        ),
        Tool(
            name="get_memory_relationships",
            description=(
                "Get memory relationships by type (supersedes/depends_on/contradicts). "
                + "Returns version history, dependencies, or contradictions for a memory. "
                + "Use [[memory:filename]] links with context keywords ('supersedes', 'depends on', 'contradicts') to create relationships."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Memory filename to query relationships for",
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": ["supersedes", "depends_on", "contradicts"],
                        "description": "Type of relationship to query. If omitted, returns all relationships.",
                    },
                },
                "required": ["filename"],
            },
        ),
        Tool(
            name="get_memory_versions",
            description=(
                "**DEPRECATED: Use get_memory_relationships(filename, 'supersedes') instead.** "
                + "Show version history by following SUPERSEDES chain. "
                + "Use [[memory:filename]] with 'supersedes' in context to create version links."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Memory filename to get version history for",
                    },
                },
                "required": ["filename"],
            },
        ),
        Tool(
            name="get_memory_dependencies",
            description=(
                "**DEPRECATED: Use get_memory_relationships(filename, 'depends_on') instead.** "
                + "Show dependencies by finding DEPENDS_ON links. "
                + "Use [[memory:filename]] with 'depends on' in context to create dependency links."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Memory filename to get dependencies for",
                    },
                },
                "required": ["filename"],
            },
        ),
        Tool(
            name="detect_contradictions",
            description=(
                "**DEPRECATED: Use get_memory_relationships(filename, 'contradicts') instead.** "
                + "Find conflicting memories by detecting CONTRADICTS links. "
                + "Use [[memory:filename]] with 'contradicts' in context to mark conflicts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Memory filename to detect contradictions for",
                    },
                },
                "required": ["filename"],
            },
        ),
    ]


@tool_handler("get_memory_stats")
async def handle_get_memory_stats(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    stats = await memory_tools_impl.get_memory_stats(ctx)

    if "error" in stats:
        return _text_response(str(stats))

    output_lines = [
        "# Memory Bank Statistics",
        "",
        f"**Total Memories:** {stats.get('count', 0)}",
        f"**Total Size:** {stats.get('total_size', '0B')}",
        f"**Storage Path:** `{stats.get('memory_path', '')}`",
        "",
        "## Tags",
        "",
    ]

    tags = stats.get("tags", {})
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        output_lines.append(f"- `{tag}`: {count}")

    output_lines.extend(["", "## Types", ""])

    types = stats.get("types", {})
    for mem_type, count in sorted(types.items(), key=lambda x: -x[1]):
        output_lines.append(f"- `{mem_type}`: {count}")

    return _text_response("\n".join(output_lines))


@tool_handler("merge_memories")
async def handle_merge_memories(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    source_files = arguments.get("source_files", [])
    target_file = arguments.get("target_file", "")
    summary_content = arguments.get("summary_content", "")

    result = await memory_tools_impl.merge_memories(
        ctx, source_files, target_file, summary_content
    )
    return _text_response(str(result))


@tool_handler("search_by_tag_cluster")
async def handle_search_by_tag_cluster(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    tag = arguments.get("tag", "")
    depth = arguments.get("depth", 2)
    limit = arguments.get("limit", 10)

    results = await memory_tools_impl.search_by_tag_cluster(ctx, tag, depth, limit)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Tag Cluster Search: {tag}", ""]

    for i, r in enumerate(results, 1):
        output_lines.extend(
            [
                f"## {i}. {r.get('memory_id', 'unknown')}",
                f"**Type:** {r.get('type', 'unknown')} | **Tags:** {', '.join(r.get('tags', []))}",
                "",
                r.get("content", "")[:500],
                "",
                "---",
                "",
            ]
        )

    return _text_response("\n".join(output_lines))


@tool_handler("get_tag_graph")
async def handle_get_tag_graph(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    result = await memory_tools_impl.get_tag_graph(ctx)

    if "error" in result:
        return _text_response(str(result))

    output_lines = ["# Tag Graph", "", "## Tag Frequencies", ""]

    frequencies = result.get("tag_frequencies", {})
    for tag, count in sorted(frequencies.items(), key=lambda x: -x[1]):
        output_lines.append(f"- `{tag}`: {count}")

    output_lines.extend(["", "## Tag Co-occurrences", ""])

    co_occurrences = result.get("co_occurrences", [])
    for co in co_occurrences[:20]:
        output_lines.append(f"- `{co['tag']}` ↔ `{co['related_tag']}`: {co['count']}")

    return _text_response("\n".join(output_lines))


@tool_handler("suggest_related_tags")
async def handle_suggest_related_tags(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    tag = arguments.get("tag", "")
    result = await memory_tools_impl.suggest_related_tags(ctx, tag)

    if "error" in result:
        return _text_response(str(result))

    output_lines = [f"# Related Tags for `{tag}`", ""]

    related = result.get("related_tags", [])
    for item in related:
        output_lines.append(f"- `{item['tag']}`: {item['count']}")

    return _text_response("\n".join(output_lines))


@tool_handler("get_memory_relationships")
async def handle_get_memory_relationships(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    relationship_type = arguments.get("relationship_type")

    result = await memory_tools_impl.get_memory_relationships(
        ctx, filename, relationship_type
    )

    if "error" in result:
        return _text_response(str(result))

    output_lines = [f"# Relationships for `{filename}`", ""]

    if "supersedes" in result:
        version_data = result["supersedes"]
        output_lines.extend(["## Version History (SUPERSEDES)", ""])
        chain = version_data.get("version_chain", [])
        for i, version in enumerate(chain, 1):
            output_lines.extend(
                [
                    f"{i}. `{version['memory_id']}`",
                    f"   Path: {version['file_path']}",
                    "",
                ]
            )

    if "depends_on" in result:
        deps = result["depends_on"]
        output_lines.extend(["## Dependencies (DEPENDS_ON)", ""])
        if not deps:
            output_lines.append("No dependencies found.")
        else:
            for i, dep in enumerate(deps, 1):
                output_lines.extend(
                    [
                        f"{i}. `{dep['memory_id']}`",
                        f"   Path: {dep['file_path']}",
                        f"   Context: {dep['context'][:100]}",
                        "",
                    ]
                )

    if "contradicts" in result:
        contras = result["contradicts"]
        output_lines.extend(["## Contradictions (CONTRADICTS)", ""])
        if not contras:
            output_lines.append("No contradictions detected.")
        else:
            for i, contra in enumerate(contras, 1):
                output_lines.extend(
                    [
                        f"{i}. `{contra['memory_id']}`",
                        f"   Path: {contra['file_path']}",
                        f"   Context: {contra['context'][:100]}",
                        "",
                    ]
                )

    return _text_response("\n".join(output_lines))


@tool_handler("get_memory_versions")
async def handle_get_memory_versions(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    result = await memory_tools_impl.get_memory_versions(ctx, filename)

    if "error" in result:
        return _text_response(str(result))

    output_lines = [f"# Version History for `{filename}`", ""]

    chain = result.get("version_chain", [])
    for i, version in enumerate(chain, 1):
        output_lines.extend(
            [
                f"{i}. `{version['memory_id']}`",
                f"   Path: {version['file_path']}",
                "",
            ]
        )

    return _text_response("\n".join(output_lines))


@tool_handler("get_memory_dependencies")
async def handle_get_memory_dependencies(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    results = await memory_tools_impl.get_memory_dependencies(ctx, filename)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Dependencies for `{filename}`", ""]

    for i, dep in enumerate(results, 1):
        output_lines.extend(
            [
                f"{i}. `{dep['memory_id']}`",
                f"   Path: {dep['file_path']}",
                f"   Context: {dep['context'][:100]}",
                "",
            ]
        )

    return _text_response("\n".join(output_lines))


@tool_handler("detect_contradictions")
async def handle_detect_contradictions(
    hctx: HandlerContext, arguments: dict
) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    results = await memory_tools_impl.detect_contradictions(ctx, filename)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Contradictions for `{filename}`", ""]

    if not results:
        output_lines.append("No contradictions detected.")
    else:
        for i, contradiction in enumerate(results, 1):
            output_lines.extend(
                [
                    f"{i}. `{contradiction['memory_id']}`",
                    f"   Path: {contradiction['file_path']}",
                    f"   Context: {contradiction['context'][:100]}",
                    "",
                ]
            )

    return _text_response("\n".join(output_lines))
