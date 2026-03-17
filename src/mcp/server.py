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

from src.context import ApplicationContext
from src.daemon.health import (
    DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
    request_daemon_socket,
)
from src.daemon.management import start_daemon
from src.lifecycle import LifecycleCoordinator, LifecycleState
from src.mcp.handlers import HandlerContext, get_handler
from src.mcp.tools.document_tools import get_document_tools
import src.mcp.tools.document_tools  # noqa: F401 - registers handlers

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(
        self, project_override: str | None = None, ctx: ApplicationContext | None = None
    ):
        self.ctx: ApplicationContext | None = ctx
        self.project_override = project_override
        self.server = Server("mcp-markdown-ragdocs")
        self._coordinator = LifecycleCoordinator(_manage_daemon_metadata=False)
        self._local_start_lock = asyncio.Lock()

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            remote_tools = await self._maybe_get_remote_tools()
            if remote_tools is not None:
                return remote_tools
            return [*get_document_tools()]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            remote_result = await self._maybe_call_remote_tool(name, arguments)
            if remote_result is not None:
                return remote_result

            await self._ensure_local_runtime_started()

            handler = get_handler(name)
            if handler is None:
                raise ValueError(f"Unknown tool: {name}")
            hctx = HandlerContext(lambda: self.ctx, self._coordinator)
            return await handler(hctx, arguments)

    async def _maybe_get_remote_tools(self) -> list[Tool] | None:
        try:
            metadata = await asyncio.to_thread(
                start_daemon,
                project_override=self.project_override,
            )
            if not metadata.socket_path:
                return None
            response = await asyncio.to_thread(
                request_daemon_socket,
                Path(metadata.socket_path),
                "/api/mcp/tools",
                {},
                timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
            )
            tools = response.get("tools")
            if not isinstance(tools, list):
                return None
            return [
                Tool(
                    name=str(tool.get("name", "")),
                    description=str(tool.get("description", "")),
                    inputSchema=tool.get("inputSchema", {}),
                )
                for tool in tools
                if isinstance(tool, dict)
            ]
        except Exception:
            logger.debug("Falling back to in-process tool list", exc_info=True)
            return None

    async def _maybe_call_remote_tool(
        self,
        name: str,
        arguments: dict,
    ) -> list[TextContent] | None:
        try:
            metadata = await asyncio.to_thread(
                start_daemon,
                project_override=self.project_override,
            )
            if not metadata.socket_path:
                return None
            response = await asyncio.to_thread(
                request_daemon_socket,
                Path(metadata.socket_path),
                "/api/mcp/tool",
                {"name": name, "arguments": arguments},
                timeout_seconds=DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
            )
            contents = response.get("contents")
            if not isinstance(contents, list):
                return None
            return [
                TextContent(
                    type=str(content.get("type", "text")),
                    text=str(content.get("text", "")),
                )
                for content in contents
                if isinstance(content, dict)
            ]
        except Exception:
            logger.debug("Falling back to in-process tool handling", exc_info=True)
            return None

    async def _ensure_context(self) -> None:
        if self.ctx is not None:
            return

        # ApplicationContext.create() is synchronous and may do slow work
        # (config loading, model path resolution).  Run in a thread so the
        # event loop stays responsive for MCP protocol handling.
        self.ctx = await asyncio.to_thread(
            ApplicationContext.create,
            project_override=self.project_override,
            enable_watcher=False,
            lazy_embeddings=True,
        )

    async def _init_and_start(self) -> None:
        """Background task: create context then start index loading.

        Runs after stdio_server() opens so the MCP handshake can complete
        immediately.  Tool handlers call wait_for_ready() to block until
        indices are available.
        """
        try:
            await self._ensure_context()

            if not isinstance(self.ctx, ApplicationContext):
                raise RuntimeError("Expected ApplicationContext")
            await self._coordinator.start(self.ctx, background_index=True)
        except Exception as e:
            logger.error("Background initialization failed: %s", e, exc_info=True)
            self._coordinator.record_init_error(e)
            raise

    async def _ensure_local_runtime_started(self) -> None:
        if self.ctx is not None and self._coordinator.state != LifecycleState.UNINITIALIZED:
            return

        async with self._local_start_lock:
            if self.ctx is not None and self._coordinator.state != LifecycleState.UNINITIALIZED:
                return
            await self._init_and_start()

    async def startup(self) -> None:
        """Initialize context and start index loading.

        Unlike run(), this blocks until initialization completes (suitable
        for callers that explicitly await startup before accepting requests).
        """
        logger.info("Starting MCP server initialization")
        await self._ensure_local_runtime_started()
        logger.info("MCP server initialization complete")

    async def shutdown(self) -> None:
        if self.ctx is None:
            return
        logger.info("Shutting down MCP server")
        if self._coordinator.state != LifecycleState.UNINITIALIZED:
            await self._coordinator.shutdown()
        self.ctx = None
        logger.info("Shutdown complete")

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._coordinator.install_signal_handlers(loop)

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
