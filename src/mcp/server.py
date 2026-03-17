import asyncio
import logging
import os
from pathlib import Path

# Prevent tokenizers parallelism warning.
# Must be set before any HuggingFace/sentence-transformers imports.
# See: https://github.com/huggingface/tokenizers/issues/993
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Disable HuggingFace/tqdm progress bars to prevent stdout pollution in JSON output
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.daemon.health import (
    DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
    request_daemon_socket,
)
from src.daemon.management import start_daemon
from src.mcp.tools.document_tools import get_document_tools
import src.mcp.tools.document_tools  # noqa: F401 - registers handlers

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self, project_override: str | None = None):
        self.project_override = project_override
        self.server = Server("mcp-markdown-ragdocs")

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return await self._get_remote_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            return await self._call_remote_tool(name, arguments)

    async def _get_remote_tools(self) -> list[Tool]:
        try:
            metadata = await asyncio.to_thread(
                start_daemon,
                project_override=self.project_override,
            )
            if not metadata.socket_path:
                raise RuntimeError("Daemon did not provide a socket path")
            response = await asyncio.to_thread(
                request_daemon_socket,
                Path(metadata.socket_path),
                "/api/mcp/tools",
                {},
                timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
            )
            if response.get("status") == "error":
                raise RuntimeError(str(response.get("details") or response.get("error")))
            tools = response.get("tools")
            if not isinstance(tools, list):
                raise RuntimeError("Daemon returned an invalid MCP tool payload")
            return [
                Tool(
                    name=str(tool.get("name", "")),
                    description=str(tool.get("description", "")),
                    inputSchema=tool.get("inputSchema", {}),
                )
                for tool in tools
                if isinstance(tool, dict)
            ]
        except Exception as exc:
            logger.error("Failed to fetch tools from daemon", exc_info=True)
            raise RuntimeError(f"Daemon unavailable for MCP tools: {exc}") from exc

    async def _call_remote_tool(
        self,
        name: str,
        arguments: dict,
    ) -> list[TextContent]:
        try:
            metadata = await asyncio.to_thread(
                start_daemon,
                project_override=self.project_override,
            )
            if not metadata.socket_path:
                raise RuntimeError("Daemon did not provide a socket path")
            response = await asyncio.to_thread(
                request_daemon_socket,
                Path(metadata.socket_path),
                "/api/mcp/tool",
                {"name": name, "arguments": arguments},
                timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
            )
            if response.get("status") == "error":
                raise RuntimeError(str(response.get("details") or response.get("error")))
            contents = response.get("contents")
            if not isinstance(contents, list):
                raise RuntimeError("Daemon returned an invalid MCP tool response")
            return [
                TextContent(
                    type=str(content.get("type", "text")),
                    text=str(content.get("text", "")),
                )
                for content in contents
                if isinstance(content, dict)
            ]
        except Exception as exc:
            logger.error("Failed to call daemon MCP tool %s", name, exc_info=True)
            raise RuntimeError(f"Daemon unavailable for MCP tool '{name}': {exc}") from exc

    async def shutdown(self) -> None:
        logger.info("MCP thin client shutdown complete")

    async def run(self) -> None:
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        except asyncio.CancelledError:
            logger.info("Server run cancelled")
        finally:
            await self.shutdown()


async def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="MCP stdio server for markdown documentation search"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Override project detection by specifying project name or path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = MCPServer(project_override=args.project)

    try:
        await server.run()
    except asyncio.CancelledError:
        logger.info("Main task cancelled")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
