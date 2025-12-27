"""
MCP stdio server implementation for mcp-markdown-ragdocs.

Provides a long-lived MCP server with stdio transport that VS Code
can manage. Includes file watching and automatic index updates.
"""

import asyncio
import glob
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.compression.deduplication import deduplicate_results, get_embeddings_for_chunks
from src.compression.thresholding import filter_by_score
from src.config import load_config
from src.models import CompressionStats
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, load_manifest, save_manifest, should_rebuild
from src.indexing.watcher import FileWatcher
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import QueryOrchestrator

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self, project_override: str | None = None):
        self.config = None
        self.manager: IndexManager | None = None
        self.orchestrator: QueryOrchestrator | None = None
        self.watcher: FileWatcher | None = None
        self.server = Server("mcp-markdown-ragdocs")
        self.project_override = project_override

        # Register tool handlers
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="query_documents",
                    description="Search local Markdown documentation using hybrid search (semantic, keyword, graph traversal). Returns relevant document chunks and a synthesized answer.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query or question about the documentation",
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5, max: 100)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="query_documents_compressed",
                    description="Search documents with context compression. Filters low-relevance results and removes semantic duplicates to reduce context size.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query or question about the documentation",
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5, max: 100)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 100,
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
                        },
                        "required": ["query"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "query_documents":
                return await self._handle_query_documents(arguments)
            elif name == "query_documents_compressed":
                return await self._handle_query_documents_compressed(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def startup(self):
        """Initialize indices, load or build index, start file watcher."""
        logger.info("Starting MCP server initialization")

        self.config = load_config()

        from src.config import detect_project, resolve_index_path, resolve_documents_path

        detected_project = detect_project(projects=self.config.projects, project_override=self.project_override)

        # Reload config in case detect_project() persisted a new project
        # (the projects list needs to include the newly-registered project for resolve_documents_path)
        if detected_project and self.project_override:
            self.config = load_config()

        index_path = resolve_index_path(self.config, detected_project)
        documents_path = resolve_documents_path(self.config, detected_project, self.config.projects)

        self.config.indexing.index_path = str(index_path)
        self.config.indexing.documents_path = documents_path

        # Initialize indices
        vector = VectorIndex()
        keyword = KeywordIndex()
        graph = GraphStore()

        self.manager = IndexManager(self.config, vector, keyword, graph)
        self.orchestrator = QueryOrchestrator(vector, keyword, graph, self.config, self.manager)

        # Ensure index directory exists
        index_path_obj = Path(self.config.indexing.index_path)
        index_path_obj.mkdir(parents=True, exist_ok=True)

        # Check if rebuild needed
        current_manifest = IndexManifest(
            spec_version="1.0.0",
            embedding_model=self.config.llm.embedding_model,
            parsers=self.config.parsers,
            chunking_config={
                "strategy": self.config.chunking.strategy,
                "min_chunk_chars": self.config.chunking.min_chunk_chars,
                "max_chunk_chars": self.config.chunking.max_chunk_chars,
                "overlap_chars": self.config.chunking.overlap_chars,
            },
        )

        saved_manifest = load_manifest(index_path_obj)
        needs_rebuild = should_rebuild(current_manifest, saved_manifest)

        if needs_rebuild:
            logger.info("Index rebuild required - will index in background")
            # Schedule background indexing task
            asyncio.create_task(self._background_index(index_path_obj, current_manifest))
        else:
            logger.info("Loading existing indices")
            # Load indices in thread pool to avoid blocking
            await asyncio.to_thread(self.manager.load)

        # Start file watcher
        self.watcher = FileWatcher(
            documents_path=self.config.indexing.documents_path,
            index_manager=self.manager,
        )
        self.watcher.start()
        logger.info("File watcher started")
        logger.info("MCP server initialization complete")

    async def _background_index(self, index_path_obj: Path, current_manifest: IndexManifest):
        """Index documents in background without blocking server initialization."""
        try:
            logger.info("Starting background indexing")

            # Type assertions for properties set in startup()
            assert self.config is not None
            assert self.manager is not None

            docs_path = Path(self.config.indexing.documents_path)
            pattern = str(docs_path / "**" / "*.md")
            files = glob.glob(pattern, recursive=self.config.indexing.recursive)

            # Index files in thread pool to release GIL
            for file_path in files:
                await asyncio.to_thread(self.manager.index_document, file_path)

            await asyncio.to_thread(self.manager.persist)
            await asyncio.to_thread(save_manifest, index_path_obj, current_manifest)
            logger.info(f"Background indexing complete: {len(files)} documents indexed")
        except Exception as e:
            logger.error(f"Background indexing failed: {e}")

    async def _handle_query_documents(self, arguments: dict):
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        top_n = arguments.get("top_n", 5)
        if not isinstance(top_n, int) or top_n < 1 or top_n > 100:
            raise ValueError("top_n must be an integer between 1 and 100")

        if not self.orchestrator:
            raise RuntimeError("Server not initialized")

        top_k = max(20, top_n * 4)
        results, _ = await self.orchestrator.query(query, top_k=top_k, top_n=top_n)
        chunk_ids = [result.chunk_id for result in results]
        answer = await self.orchestrator.synthesize_answer(query, chunk_ids)

        results_text = "\n\n".join([
            f"**Result {i+1}** (Score: {r.score:.4f})\n"
            f"File: {r.file_path or 'unknown'}\n"
            f"Section: {r.header_path or '(no section)'}\n\n"
            f"{r.content}"
            for i, r in enumerate(results)
        ])

        response = f"# Answer\n\n{answer}\n\n# Source Documents\n\n{results_text}"
        return [TextContent(type="text", text=response)]

    async def _handle_query_documents_compressed(self, arguments: dict):
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        top_n = arguments.get("top_n", 5)
        if not isinstance(top_n, int) or top_n < 1 or top_n > 100:
            raise ValueError("top_n must be an integer between 1 and 100")

        min_score = arguments.get("min_score", 0.3)
        if not isinstance(min_score, (int, float)) or min_score < 0.0 or min_score > 1.0:
            raise ValueError("min_score must be a number between 0.0 and 1.0")

        similarity_threshold = arguments.get("similarity_threshold", 0.85)
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0.5 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be a number between 0.5 and 1.0")

        if not self.orchestrator:
            raise RuntimeError("Server not initialized")

        top_k = max(20, top_n * 3)
        results, compression_stats = await self.orchestrator.query(query, top_k=top_k, top_n=top_k)
        original_count = compression_stats.original_count

        filtered = filter_by_score(results, min_score)
        after_threshold = len(filtered)

        if len(filtered) > 1:
            embedding_model = self.orchestrator._vector._embedding_model
            embeddings = await asyncio.to_thread(
                get_embeddings_for_chunks, filtered, embedding_model
            )
            dedup_result = deduplicate_results(filtered, embeddings, similarity_threshold)
            final_results = dedup_result.results[:top_n]
            clusters_merged = dedup_result.clusters_merged
        else:
            final_results = filtered[:top_n]
            clusters_merged = 0

        stats = CompressionStats(
            original_count=original_count,
            after_threshold=after_threshold,
            after_doc_limit=after_threshold,
            after_dedup=len(final_results),
            clusters_merged=clusters_merged,
        )

        chunk_ids = [result.chunk_id for result in final_results]
        answer = await self.orchestrator.synthesize_answer(query, chunk_ids)

        stats_text = (
            f"- Original results: {stats.original_count}\n"
            f"- After score filter (≥{min_score}): {stats.after_threshold}\n"
            f"- After deduplication: {stats.after_dedup}\n"
            f"- Clusters merged: {stats.clusters_merged}"
        )

        results_text = "\n\n".join([
            f"**Result {i+1}** (Score: {r.score:.4f})\n"
            f"File: {r.file_path or 'unknown'}\n"
            f"Section: {r.header_path or '(no section)'}\n\n"
            f"{r.content}"
            for i, r in enumerate(final_results)
        ])

        response = f"# Answer\n\n{answer}\n\n# Compression Stats\n\n{stats_text}\n\n# Source Documents\n\n{results_text}"
        return [TextContent(type="text", text=response)]

    async def shutdown(self):
        """Stop file watcher and persist indices."""
        logger.info("Shutting down MCP server")
        if self.watcher:
            await self.watcher.stop()
        if self.manager:
            self.manager.persist()
        logger.info("Shutdown complete")

    async def run(self):
        """Run the MCP server with stdio transport."""
        try:
            await self.startup()

            # Run stdio server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        finally:
            await self.shutdown()


async def main(argv: list[str] | None = None):
    """Entry point for MCP stdio server."""
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
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
