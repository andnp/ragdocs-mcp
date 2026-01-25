import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.context import ApplicationContext
from src.lifecycle import LifecycleCoordinator
from src.mcp.handlers import HandlerContext, get_handler
from src.mcp.tools.document_tools import get_document_tools
from src.mcp.tools.memory_tools import get_memory_tools
from src.mcp.tools.metadata_tools import get_metadata_tools
import src.mcp.tools.document_tools  # noqa: F401 - registers handlers
import src.mcp.tools.memory_tools  # noqa: F401 - registers handlers
import src.mcp.tools.metadata_tools  # noqa: F401 - registers handlers

if TYPE_CHECKING:
    from src.reader.context import ReadOnlyContext

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self, project_override: str | None = None, ctx: ApplicationContext | None = None):
        self.ctx: ApplicationContext | ReadOnlyContext | None = ctx
        self.project_override = project_override
        self.server = Server("mcp-markdown-ragdocs")
        self._coordinator = LifecycleCoordinator()
        self._use_worker: bool = False

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                *get_document_tools(),
                *get_memory_tools(),
                *get_metadata_tools(),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            handler = get_handler(name)
            if handler is None:
                raise ValueError(f"Unknown tool: {name}")
            hctx = HandlerContext(self.ctx, self._coordinator)
            return await handler(hctx, arguments)

    async def _ensure_context(self) -> None:
        if self.ctx is not None:
            return

        from src.reader.context import ReadOnlyContext as ROContext

        app_ctx = ApplicationContext.create(
            project_override=self.project_override,
            enable_watcher=True,
            lazy_embeddings=True,
        )
        config = app_ctx.config
        self._use_worker = config.worker.enabled

        if self._use_worker:
            snapshot_base = Path(config.indexing.index_path) / "snapshots"
            self.ctx = await ROContext.create(config, snapshot_base)
        else:
            self.ctx = app_ctx

    async def _start_background_init(self) -> asyncio.Task[None]:
        from src.reader.context import ReadOnlyContext as ROContext

        if self._use_worker:
            if not isinstance(self.ctx, ROContext):
                raise RuntimeError("Worker mode requires ReadOnlyContext")
            return asyncio.create_task(self._coordinator.start_with_worker(self.ctx))
        else:
            if not isinstance(self.ctx, ApplicationContext):
                raise RuntimeError("Non-worker mode requires ApplicationContext")
            return asyncio.create_task(self._coordinator.start(self.ctx, background_index=True))

    async def startup(self) -> None:
        logger.info("Starting MCP server initialization")

        await self._ensure_context()

        if self._use_worker:
            from src.reader.context import ReadOnlyContext as ROContext

            if isinstance(self.ctx, ROContext):
                await self._coordinator.start_with_worker(self.ctx)
        else:
            if isinstance(self.ctx, ApplicationContext):
                await self.ctx.start(background_index=True)

        logger.info("MCP server initialization complete")

    async def shutdown(self) -> None:
        if not self.ctx:
            return
        logger.info("Shutting down MCP server")
        ctx = self.ctx
        self.ctx = None
        await ctx.stop()
        logger.info("Shutdown complete")

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._coordinator.install_signal_handlers(loop)

        try:
            await self._ensure_context()

            async with stdio_server() as (read_stream, write_stream):
                init_task = await self._start_background_init()

                try:
                    await self.server.run(
                        read_stream,
                        write_stream,
                        self.server.create_initialization_options(),
                    )
                finally:
                    if not init_task.done():
                        init_task.cancel()
                        try:
                            await init_task
                        except asyncio.CancelledError:
                            pass
        except asyncio.CancelledError:
            logger.info("Server run cancelled")
        finally:
            await self._coordinator.shutdown()


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
