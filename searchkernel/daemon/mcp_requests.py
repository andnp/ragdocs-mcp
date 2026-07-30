from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mcp.types import TextContent

import searchkernel.mcp.tools.document_tools  # noqa: F401 - registers handlers
from searchkernel.mcp.handlers import HandlerContext, get_handler
from searchkernel.mcp.tools.document_tools import get_document_tools

if TYPE_CHECKING:
    from searchkernel.context import ApplicationContext
    from searchkernel.lifecycle import LifecycleCoordinator


def build_mcp_tools_payload() -> dict[str, object]:
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in get_document_tools()
        ]
    }


async def handle_mcp_tool_call(
    *,
    ctx_getter: Callable[[], ApplicationContext],
    coordinator: LifecycleCoordinator,
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