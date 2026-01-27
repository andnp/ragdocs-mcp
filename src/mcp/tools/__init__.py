"""MCP tool implementations organized by category."""

from mcp.types import TextContent


def text_response(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]
