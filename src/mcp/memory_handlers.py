from __future__ import annotations

from mcp.types import TextContent

from src.mcp.handlers import HandlerContext, tool_handler
from src.memory import tools as memory_tools


def _text_response(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


@tool_handler("create_memory")
async def handle_create_memory(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")
    tags = arguments.get("tags", [])
    memory_type = arguments.get("memory_type", "journal")

    result = await memory_tools.create_memory(ctx, filename, content, tags, memory_type)
    return _text_response(str(result))


@tool_handler("append_memory")
async def handle_append_memory(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")

    result = await memory_tools.append_memory(ctx, filename, content)
    return _text_response(str(result))


@tool_handler("read_memory")
async def handle_read_memory(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    result = await memory_tools.read_memory(ctx, filename)

    if "error" in result:
        return _text_response(str(result))
    return _text_response(result.get("content", ""))


@tool_handler("update_memory")
async def handle_update_memory(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")

    result = await memory_tools.update_memory(ctx, filename, content)
    return _text_response(str(result))


@tool_handler("delete_memory")
async def handle_delete_memory(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    result = await memory_tools.delete_memory(ctx, filename)
    return _text_response(str(result))


@tool_handler("search_memories")
async def handle_search_memories(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    query = arguments.get("query", "")
    limit = arguments.get("limit", 5)
    filter_tags = arguments.get("filter_tags")
    filter_type = arguments.get("filter_type")
    load_full_memory = arguments.get("load_full_memory", False)

    results = await memory_tools.search_memories(
        ctx, query, limit, filter_tags, filter_type, load_full_memory
    )

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = ["# Memory Search Results", ""]

    for i, r in enumerate(results, 1):
        output_lines.extend([
            f"## {i}. {r.get('memory_id', 'unknown')} (score: {r.get('score', 0):.3f})",
            f"**Type:** {r.get('type', 'unknown')} | **Tags:** {', '.join(r.get('tags', []))}",
            "",
            r.get("content", "")[:500],
            "",
            "---",
            "",
        ])

    return _text_response("\n".join(output_lines))


@tool_handler("search_linked_memories")
async def handle_search_linked_memories(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    query = arguments.get("query", "")
    target_document = arguments.get("target_document", "")
    limit = arguments.get("limit", 5)

    results = await memory_tools.search_linked_memories(ctx, query, target_document, limit)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Memories Linked to `{target_document}`", ""]

    for i, r in enumerate(results, 1):
        output_lines.extend([
            f"## {i}. {r.get('memory_id', 'unknown')} (score: {r.get('score', 0):.3f})",
            f"**Edge Type:** {r.get('edge_type', 'unknown')}",
            f"**Anchor Context:** {r.get('anchor_context', '')}",
            "",
            r.get("content", "")[:500],
            "",
            "---",
            "",
        ])

    return _text_response("\n".join(output_lines))


@tool_handler("get_memory_stats")
async def handle_get_memory_stats(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    stats = await memory_tools.get_memory_stats(ctx)

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
async def handle_merge_memories(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    source_files = arguments.get("source_files", [])
    target_file = arguments.get("target_file", "")
    summary_content = arguments.get("summary_content", "")

    result = await memory_tools.merge_memories(ctx, source_files, target_file, summary_content)
    return _text_response(str(result))


@tool_handler("search_by_tag_cluster")
async def handle_search_by_tag_cluster(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    tag = arguments.get("tag", "")
    depth = arguments.get("depth", 2)
    limit = arguments.get("limit", 10)

    results = await memory_tools.search_by_tag_cluster(ctx, tag, depth, limit)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Tag Cluster Search: {tag}", ""]

    for i, r in enumerate(results, 1):
        output_lines.extend([
            f"## {i}. {r.get('memory_id', 'unknown')}",
            f"**Type:** {r.get('type', 'unknown')} | **Tags:** {', '.join(r.get('tags', []))}",
            "",
            r.get("content", "")[:500],
            "",
            "---",
            "",
        ])

    return _text_response("\n".join(output_lines))


@tool_handler("get_tag_graph")
async def handle_get_tag_graph(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    result = await memory_tools.get_tag_graph(ctx)

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
async def handle_suggest_related_tags(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    tag = arguments.get("tag", "")
    result = await memory_tools.suggest_related_tags(ctx, tag)

    if "error" in result:
        return _text_response(str(result))

    output_lines = [f"# Related Tags for `{tag}`", ""]

    related = result.get("related_tags", [])
    for item in related:
        output_lines.append(f"- `{item['tag']}`: {item['count']}")

    return _text_response("\n".join(output_lines))


@tool_handler("get_memory_versions")
async def handle_get_memory_versions(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    result = await memory_tools.get_memory_versions(ctx, filename)

    if "error" in result:
        return _text_response(str(result))

    output_lines = [f"# Version History for `{filename}`", ""]

    chain = result.get("version_chain", [])
    for i, version in enumerate(chain, 1):
        output_lines.extend([
            f"{i}. `{version['memory_id']}`",
            f"   Path: {version['file_path']}",
            "",
        ])

    return _text_response("\n".join(output_lines))


@tool_handler("get_memory_dependencies")
async def handle_get_memory_dependencies(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    results = await memory_tools.get_memory_dependencies(ctx, filename)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Dependencies for `{filename}`", ""]

    for i, dep in enumerate(results, 1):
        output_lines.extend([
            f"{i}. `{dep['memory_id']}`",
            f"   Path: {dep['file_path']}",
            f"   Context: {dep['context'][:100]}",
            "",
        ])

    return _text_response("\n".join(output_lines))


@tool_handler("detect_contradictions")
async def handle_detect_contradictions(hctx: HandlerContext, arguments: dict) -> list[TextContent]:
    ctx = hctx.require_ctx()

    filename = arguments.get("filename", "")
    results = await memory_tools.detect_contradictions(ctx, filename)

    if results and "error" in results[0]:
        return _text_response(str(results[0]))

    output_lines = [f"# Contradictions for `{filename}`", ""]

    if not results:
        output_lines.append("No contradictions detected.")
    else:
        for i, contradiction in enumerate(results, 1):
            output_lines.extend([
                f"{i}. `{contradiction['memory_id']}`",
                f"   Path: {contradiction['file_path']}",
                f"   Context: {contradiction['context'][:100]}",
                "",
            ])

    return _text_response("\n".join(output_lines))
