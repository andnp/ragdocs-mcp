from __future__ import annotations

from collections.abc import Callable

from mcp.types import TextContent

import mcp_markdown_ragdocs.mcp.tools.document_tools  # noqa: F401 - registers handlers
from mcp_markdown_ragdocs.mcp.handlers import (
    ApplicationContextLike,
    HandlerContext,
    LifecycleCoordinatorLike,
    get_handler,
)
from mcp_markdown_ragdocs.mcp.tools.document_tools import get_document_tools


def build_mcp_tools_payload() -> dict[str, object]:
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in get_document_tools()
        ]
    }


async def handle_mcp_tool_call(
    *,
    ctx_getter: Callable[[], ApplicationContextLike | None],
    coordinator: LifecycleCoordinatorLike,
    payload: dict[str, object],
) -> dict[str, object]:
    tool_name = str(payload.get("name", ""))
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        return {"status": "error", "error": "tool_arguments_must_be_object"}

    handler = get_handler(tool_name)
    if handler is None:
        return {"status": "error", "error": f"unknown_tool:{tool_name}"}

    hctx = HandlerContext(ctx_getter, coordinator)
    contents = await handler(hctx, arguments)
    return {
        "contents": [
            {"type": content.type, "text": content.text}
            for content in contents
            if isinstance(content, TextContent)
        ]
    }