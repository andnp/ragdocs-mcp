"""
MCP stdio server implementation for mcp-markdown-ragdocs.

Provides a long-lived MCP server with stdio transport that VS Code
can manage. Includes file watching and automatic index updates.
"""

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.context import ApplicationContext
from src.lifecycle import LifecycleCoordinator
from src.search.pipeline import SearchPipelineConfig

logger = logging.getLogger(__name__)

MIN_TOP_N = 1
MAX_TOP_N = 100


class MCPServer:
    def __init__(self, project_override: str | None = None, ctx: ApplicationContext | None = None):
        self.ctx = ctx
        self.project_override = project_override
        self.server = Server("mcp-markdown-ragdocs")
        self._coordinator = LifecycleCoordinator()

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="query_documents",
                    description=(
                        "Search local documentation using hybrid search (semantic + keyword + graph). " +
                        "Returns ranked document chunks with relevance scores. " +
                        "Use for discovering relevant documentation sections in a large corpus."
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
                                "description": "Minimum relevance score threshold (default: 0.3)",
                                "default": 0.3,
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
                            "show_stats": {
                                "type": "boolean",
                                "description": "Whether to show compression stats in response (default: false)",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "query_documents":
                return await self._handle_query_documents(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def startup(self) -> None:
        logger.info("Starting MCP server initialization")

        if self.ctx is None:
            self.ctx = ApplicationContext.create(
                project_override=self.project_override,
                enable_watcher=True,
                lazy_embeddings=True,
            )

        await self.ctx.start(background_index=True)
        logger.info("MCP server initialization complete")

    async def _handle_query_documents(self, arguments: dict):
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        # Wait for indices to be ready (blocks only if still initializing)
        if self.ctx and not self.ctx.is_ready():
            logger.info("Query received while initializing, waiting for indices...")
            await self._coordinator.wait_ready(timeout=60.0)

        top_n = arguments.get("top_n", 5)
        if not isinstance(top_n, int) or top_n < MIN_TOP_N or top_n > MAX_TOP_N:
            raise ValueError(f"top_n must be an integer between {MIN_TOP_N} and {MAX_TOP_N}")

        min_score = arguments.get("min_score", 0.3)
        if not isinstance(min_score, (int, float)) or min_score < 0.0 or min_score > 1.0:
            raise ValueError("min_score must be a number between 0.0 and 1.0")

        similarity_threshold = arguments.get("similarity_threshold", 0.85)
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0.5 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be a number between 0.5 and 1.0")

        show_stats = arguments.get("show_stats", False)
        if not isinstance(show_stats, bool):
            raise ValueError("show_stats must be a boolean")

        if not self.ctx:
            raise RuntimeError("Server not initialized")

        top_k = max(20, top_n * 4)

        pipeline_config = SearchPipelineConfig(
            min_confidence=min_score,
            max_chunks_per_doc=0,
            dedup_enabled=True,
            dedup_threshold=similarity_threshold,
            rerank_enabled=False,
        )

        results, stats = await self.ctx.orchestrator.query(
            query,
            top_k=top_k,
            top_n=top_n,
            pipeline_config=pipeline_config,
        )

        results_text = "\n\n".join([
            f"**Result {i+1}** (Score: {r.score:.4f})\n"
            f"File: {r.file_path or 'unknown'}\n"
            f"Section: {r.header_path or '(no section)'}\n\n"
            f"{r.content}"
            for i, r in enumerate(results)
        ])

        filtering_occurred = stats.original_count > stats.after_dedup
        if show_stats or filtering_occurred:
            stats_text = (
                f"- Original results: {stats.original_count}\n"
                f"- After score filter (≥{min_score}): {stats.after_threshold}\n"
                f"- After deduplication: {stats.after_dedup}\n"
                f"- Clusters merged: {stats.clusters_merged}"
            )
            response = f"# Search Results\n\n{results_text}\n\n# Compression Stats\n\n{stats_text}"
        else:
            response = f"# Search Results\n\n{results_text}"

        return [TextContent(type="text", text=response)]

    async def shutdown(self) -> None:
        if not self.ctx:
            return
        logger.info("Shutting down MCP server")
        ctx = self.ctx
        self.ctx = None
        await ctx.stop()
        logger.info("Shutdown complete")

    async def run(self):
        loop = asyncio.get_running_loop()
        self._coordinator.install_signal_handlers(loop)

        try:
            # Create minimal context BEFORE entering stdio (lightweight, no model loading)
            if self.ctx is None:
                self.ctx = ApplicationContext.create(
                    project_override=self.project_override,
                    enable_watcher=True,
                    lazy_embeddings=True,
                )

            # Enter MCP protocol loop IMMEDIATELY to respond to initialize handshake fast
            async with stdio_server() as (read_stream, write_stream):
                # Start background initialization AFTER entering stdio
                # This allows MCP handshake to complete while indices load
                init_task = asyncio.create_task(
                    self._coordinator.start(self.ctx, background_index=True)
                )

                try:
                    await self.server.run(
                        read_stream,
                        write_stream,
                        self.server.create_initialization_options(),
                    )
                finally:
                    # Ensure init task completes or is cancelled
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
