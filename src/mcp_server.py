import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.context import ApplicationContext
from src.lifecycle import LifecycleCoordinator
from src.mcp.handlers import HandlerContext, get_handler, MIN_TOP_N, MAX_TOP_N
import src.mcp.handlers  # noqa: F401 - registers search handlers
import src.mcp.memory_handlers  # noqa: F401 - registers memory handlers

logger = logging.getLogger(__name__)


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
                            "excluded_files": {
                                "type": "array",
                                "description": "List of file paths to exclude from results (supports filename, relative path, or absolute path)",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="query_unique_documents",
                    description=(
                        "Search local documentation using hybrid search with strict document uniqueness. " +
                        "Returns at most ONE chunk per document, ensuring results span multiple files. " +
                        "Use when you want breadth across documentation rather than depth within files."
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
                                "description": f"Maximum number of unique documents to return (default: 5, max: {MAX_TOP_N})",
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
                            "excluded_files": {
                                "type": "array",
                                "description": "List of file paths to exclude from results (supports filename, relative path, or absolute path)",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_git_history",
                    description=(
                        "Search git commit history using natural language queries. " +
                        "Returns relevant commits with metadata, message, and diff context. " +
                        "Supports filtering by file glob patterns and timestamp ranges."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query describing commits to find",
                            },
                            "top_n": {
                                "type": "integer",
                                "description": f"Maximum number of commits to return (default: 5, max: {MAX_TOP_N})",
                                "default": 5,
                                "minimum": MIN_TOP_N,
                                "maximum": MAX_TOP_N,
                            },
                            "files_glob": {
                                "type": "string",
                                "description": "Optional glob pattern to filter by changed files (e.g., 'src/**/*.py')",
                            },
                            "after_timestamp": {
                                "type": "integer",
                                "description": "Optional Unix timestamp to filter commits after this date",
                            },
                            "before_timestamp": {
                                "type": "integer",
                                "description": "Optional Unix timestamp to filter commits before this date",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="create_memory",
                    description=(
                        "Create a new memory file in the Memory Bank. " +
                        "Memories are persistent notes that AI assistants can use for long-term storage. " +
                        "Fails if the file already exists. " +
                        "IMPORTANT: The system automatically generates YAML frontmatter with type, status, tags, and created_at. " +
                        "DO NOT include frontmatter in the content parameter - provide only the body text."
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
                        "Search the Memory Bank using hybrid search (semantic + keyword). " +
                        "Returns memories ranked by relevance with recency boost applied."
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
                            "filter_tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Only return memories with these tags",
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
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_linked_memories",
                    description=(
                        "Find memories that link to a specific document. " +
                        "Uses graph traversal to find memories containing [[target_document]] links. " +
                        "Returns the anchor context explaining why each memory links to the target."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Optional query to filter/rank linked memories",
                            },
                            "target_document": {
                                "type": "string",
                                "description": "Document path to find links to (e.g., 'src/server.py')",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5,
                            },
                        },
                        "required": ["query", "target_document"],
                    },
                ),
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
                        "Merge multiple memory files into a new summary file. " +
                        "Source files are moved to .trash/ after merge."
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
                    name="search_with_hypothesis",
                    description=(
                        "Search documentation using a hypothesis about what the answer might look like. " +
                        "Useful for vague queries where you can describe the expected documentation content. " +
                        "The hypothesis is embedded and used directly for semantic search (HyDE technique)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "string",
                                "description": "A hypothesis describing what the expected documentation content looks like",
                            },
                            "top_n": {
                                "type": "integer",
                                "description": f"Maximum number of results to return (default: 5, max: {MAX_TOP_N})",
                                "default": 5,
                                "minimum": MIN_TOP_N,
                                "maximum": MAX_TOP_N,
                            },
                            "excluded_files": {
                                "type": "array",
                                "description": "List of file paths to exclude from results",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["hypothesis"],
                    },
                ),
                Tool(
                    name="search_by_tag_cluster",
                    description=(
                        "Find memories via tag traversal with configurable depth. " +
                        "Discovers memories that share tags or are connected through tag relationships."
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
                        "Return tag nodes and co-occurrence counts across all memories. " +
                        "Useful for understanding tag relationships and clusters."
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
                        "Suggest related tags based on co-occurrence patterns. " +
                        "Finds tags that frequently appear together with the specified tag."
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
                    name="get_memory_versions",
                    description=(
                        "Show version history by following SUPERSEDES chain. " +
                        "Use [[memory:filename]] with 'supersedes' in context to create version links."
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
                        "Show dependencies by finding DEPENDS_ON links. " +
                        "Use [[memory:filename]] with 'depends on' in context to create dependency links."
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
                        "Find conflicting memories by detecting CONTRADICTS links. " +
                        "Use [[memory:filename]] with 'contradicts' in context to mark conflicts."
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

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            handler = get_handler(name)
            if handler is None:
                raise ValueError(f"Unknown tool: {name}")
            hctx = HandlerContext(self.ctx, self._coordinator)
            return await handler(hctx, arguments)

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
