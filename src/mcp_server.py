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
                            "after_timestamp": {
                                "type": "integer",
                                "description": "Unix timestamp: only return memories created/modified after this time",
                            },
                            "before_timestamp": {
                                "type": "integer",
                                "description": "Unix timestamp: only return memories created/modified before this time",
                            },
                            "relative_days": {
                                "type": "integer",
                                "description": "Only return memories from last N days (overrides absolute timestamps)",
                                "minimum": 0,
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


    async def _query_documents_impl(
        self,
        arguments: dict,
        max_chunks_per_doc: int,
        result_header: str,
    ):
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

        excluded_files_raw = arguments.get("excluded_files", [])
        excluded_files = None
        if excluded_files_raw:
            from pathlib import Path
            from src.search.path_utils import normalize_path
            docs_root = Path(self.ctx.config.indexing.documents_path)
            excluded_files = {normalize_path(f, docs_root) for f in excluded_files_raw}

        top_k = max(20, top_n * 4)

        pipeline_config = SearchPipelineConfig(
            min_confidence=min_score,
            max_chunks_per_doc=max_chunks_per_doc,
            dedup_enabled=True,
            dedup_threshold=similarity_threshold,
            rerank_enabled=False,
        )

        results, stats = await self.ctx.orchestrator.query(
            query,
            top_k=top_k,
            top_n=top_n,
            pipeline_config=pipeline_config,
            excluded_files=excluded_files,
        )

        query_type = classify_query_type(query)

        results_text = "\n\n".join([
            f"[{i+1}] {r.file_path or 'unknown'} § {r.header_path or '(no section)'} ({r.score:.2f})\n"
            f"{truncate_content(r.content, 200) if query_type == 'factual' else r.content}"
            for i, r in enumerate(results)
        ])

        filtering_occurred = stats.original_count > stats.after_dedup
        if show_stats or filtering_occurred:
            stats_parts = [
                f"- Original results: {stats.original_count}",
                f"- After score filter (≥{min_score}): {stats.after_threshold}",
                f"- After deduplication: {stats.after_dedup}",
            ]
            if max_chunks_per_doc == 1:
                stats_parts.append(f"- After document limit (1 per doc): {stats.after_doc_limit}")
            stats_parts.append(f"- Clusters merged: {stats.clusters_merged}")
            stats_text = "\n".join(stats_parts)
            response = f"# {result_header}\n\n{results_text}\n\n# Compression Stats\n\n{stats_text}"
        else:
            response = f"# {result_header}\n\n{results_text}"

        return [TextContent(type="text", text=response)]

    async def _handle_query_documents(self, arguments: dict):
        return await self._query_documents_impl(
            arguments,
            max_chunks_per_doc=0,
            result_header="Search Results",
        )

    async def _handle_query_unique_documents(self, arguments: dict):
        return await self._query_documents_impl(
            arguments,
            max_chunks_per_doc=1,
            result_header="Search Results (Unique Documents)",
        )

    async def _handle_search_with_hypothesis(self, arguments: dict) -> list[TextContent]:
        hypothesis = arguments.get("hypothesis")
        if not hypothesis:
            raise ValueError("Missing required parameter: hypothesis")

        if self.ctx and not self.ctx.is_ready():
            logger.info("HyDE query received while initializing, waiting for indices...")
            await self._coordinator.wait_ready(timeout=60.0)

        top_n = arguments.get("top_n", 5)
        if not isinstance(top_n, int) or top_n < MIN_TOP_N or top_n > MAX_TOP_N:
            raise ValueError(f"top_n must be an integer between {MIN_TOP_N} and {MAX_TOP_N}")

        if not self.ctx:
            raise RuntimeError("Server not initialized")

        excluded_files_raw = arguments.get("excluded_files", [])
        excluded_files = None
        if excluded_files_raw:
            from pathlib import Path
            from src.search.path_utils import normalize_path
            docs_root = Path(self.ctx.config.indexing.documents_path)
            excluded_files = {normalize_path(f, docs_root) for f in excluded_files_raw}

        top_k = max(20, top_n * 4)

        results, stats = await self.ctx.orchestrator.query_with_hypothesis(
            hypothesis,
            top_k=top_k,
            top_n=top_n,
            excluded_files=excluded_files,
        )

        results_text = "\n\n".join([
            f"[{i+1}] {r.file_path or 'unknown'} § {r.header_path or '(no section)'} ({r.score:.2f})\n{r.content}"
            for i, r in enumerate(results)
        ])

        response = f"# HyDE Search Results\n\n{results_text}"
        return [TextContent(type="text", text=response)]

    async def _handle_search_git_history(self, arguments: dict) -> list[TextContent]:
        """Handle search_git_history tool call."""
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        if self.ctx.commit_indexer is None:
            return [TextContent(
                type="text",
                text="Git history search is not available (git binary not found or disabled in config)"
            )]

        from datetime import datetime, timezone
        from src.git.commit_search import search_git_history

        query = arguments["query"]
        top_n = arguments.get("top_n", 5)
        files_glob = arguments.get("files_glob")
        after_timestamp = arguments.get("after_timestamp")
        before_timestamp = arguments.get("before_timestamp")

        # Validate top_n
        top_n = max(MIN_TOP_N, min(top_n, MAX_TOP_N))

        # Execute search
        response = await asyncio.to_thread(
            search_git_history,
            self.ctx.commit_indexer,
            query,
            top_n,
            files_glob,
            after_timestamp,
            before_timestamp,
        )

        # Format response
        output_lines = [
            "# Git History Search Results",
            "",
            f"**Query:** {response.query}",
            f"**Total Commits Indexed:** {response.total_commits_indexed}",
            f"**Results Returned:** {len(response.results)}",
            "",
        ]

        for i, commit in enumerate(response.results, 1):
            commit_date = datetime.fromtimestamp(commit.timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            output_lines.extend([
                f"## {i}. {commit.title}",
                "",
                f"**Commit:** `{commit.hash[:8]}`",
                f"**Author:** {commit.author}",
                f"**Date:** {commit_date}",
                f"**Score:** {commit.score:.3f}",
                "",
            ])

            if commit.message:
                output_lines.extend([
                    "### Message",
                    "",
                    commit.message,
                    "",
                ])

            if commit.files_changed:
                output_lines.extend([
                    f"### Files Changed ({len(commit.files_changed)})",
                    "",
                ])

                for file_path in commit.files_changed[:10]:
                    output_lines.append(f"- `{file_path}`")

                if len(commit.files_changed) > 10:
                    output_lines.append(f"- ... and {len(commit.files_changed) - 10} more")

                output_lines.append("")

            if commit.delta_truncated:
                # Truncate delta for display
                delta_display = commit.delta_truncated[:1000]
                if len(commit.delta_truncated) > 1000:
                    delta_display += "\n... (truncated for display)"

                output_lines.extend([
                    "### Delta (truncated)",
                    "",
                    "```diff",
                    delta_display,
                    "```",
                    "",
                ])

            output_lines.extend(["---", ""])

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_create_memory(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")
        content = arguments.get("content", "")
        tags = arguments.get("tags", [])
        memory_type = arguments.get("memory_type", "journal")

        result = await memory_tools.create_memory(
            self.ctx, filename, content, tags, memory_type
        )

        return [TextContent(type="text", text=str(result))]

    async def _handle_append_memory(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")
        content = arguments.get("content", "")

        result = await memory_tools.append_memory(self.ctx, filename, content)
        return [TextContent(type="text", text=str(result))]

    async def _handle_read_memory(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")
        result = await memory_tools.read_memory(self.ctx, filename)

        if "error" in result:
            return [TextContent(type="text", text=str(result))]

        return [TextContent(type="text", text=result.get("content", ""))]

    async def _handle_update_memory(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")
        content = arguments.get("content", "")

        result = await memory_tools.update_memory(self.ctx, filename, content)
        return [TextContent(type="text", text=str(result))]

    async def _handle_delete_memory(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")
        result = await memory_tools.delete_memory(self.ctx, filename)
        return [TextContent(type="text", text=str(result))]

    async def _handle_search_memories(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        filter_tags = arguments.get("filter_tags")
        filter_type = arguments.get("filter_type")
        load_full_memory = arguments.get("load_full_memory", False)
        after_timestamp = arguments.get("after_timestamp")
        before_timestamp = arguments.get("before_timestamp")
        relative_days = arguments.get("relative_days")

        results = await memory_tools.search_memories(
            self.ctx, query, limit, filter_tags, filter_type, load_full_memory,
            after_timestamp, before_timestamp, relative_days
        )

        if results and "error" in results[0]:
            return [TextContent(type="text", text=str(results[0]))]

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

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_search_linked_memories(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        query = arguments.get("query", "")
        target_document = arguments.get("target_document", "")
        limit = arguments.get("limit", 5)

        results = await memory_tools.search_linked_memories(
            self.ctx, query, target_document, limit
        )

        if results and "error" in results[0]:
            return [TextContent(type="text", text=str(results[0]))]

        output_lines = [
            f"# Memories Linked to `{target_document}`",
            "",
        ]

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

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_get_memory_stats(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        stats = await memory_tools.get_memory_stats(self.ctx)

        if "error" in stats:
            return [TextContent(type="text", text=str(stats))]

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

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_merge_memories(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        source_files = arguments.get("source_files", [])
        target_file = arguments.get("target_file", "")
        summary_content = arguments.get("summary_content", "")

        result = await memory_tools.merge_memories(
            self.ctx, source_files, target_file, summary_content
        )

        return [TextContent(type="text", text=str(result))]

    async def _handle_search_by_tag_cluster(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        tag = arguments.get("tag", "")
        depth = arguments.get("depth", 2)
        limit = arguments.get("limit", 10)

        results = await memory_tools.search_by_tag_cluster(
            self.ctx, tag, depth, limit
        )

        if results and "error" in results[0]:
            return [TextContent(type="text", text=str(results[0]))]

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

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_get_tag_graph(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        result = await memory_tools.get_tag_graph(self.ctx)

        if "error" in result:
            return [TextContent(type="text", text=str(result))]

        output_lines = ["# Tag Graph", "", "## Tag Frequencies", ""]

        frequencies = result.get("tag_frequencies", {})
        for tag, count in sorted(frequencies.items(), key=lambda x: -x[1]):
            output_lines.append(f"- `{tag}`: {count}")

        output_lines.extend(["", "## Tag Co-occurrences", ""])

        co_occurrences = result.get("co_occurrences", [])
        for co in co_occurrences[:20]:
            output_lines.append(
                f"- `{co['tag']}` ↔ `{co['related_tag']}`: {co['count']}"
            )

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_suggest_related_tags(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        tag = arguments.get("tag", "")

        result = await memory_tools.suggest_related_tags(self.ctx, tag)

        if "error" in result:
            return [TextContent(type="text", text=str(result))]

        output_lines = [f"# Related Tags for `{tag}`", ""]

        related = result.get("related_tags", [])
        for item in related:
            output_lines.append(f"- `{item['tag']}`: {item['count']}")

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_get_memory_versions(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")

        result = await memory_tools.get_memory_versions(self.ctx, filename)

        if "error" in result:
            return [TextContent(type="text", text=str(result))]

        output_lines = [f"# Version History for `{filename}`", ""]

        chain = result.get("version_chain", [])
        for i, version in enumerate(chain, 1):
            output_lines.extend([
                f"{i}. `{version['memory_id']}`",
                f"   Path: {version['file_path']}",
                "",
            ])

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_get_memory_dependencies(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")

        results = await memory_tools.get_memory_dependencies(self.ctx, filename)

        if results and "error" in results[0]:
            return [TextContent(type="text", text=str(results[0]))]

        output_lines = [f"# Dependencies for `{filename}`", ""]

        for i, dep in enumerate(results, 1):
            output_lines.extend([
                f"{i}. `{dep['memory_id']}`",
                f"   Path: {dep['file_path']}",
                f"   Context: {dep['context'][:100]}",
                "",
            ])

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _handle_detect_contradictions(self, arguments: dict) -> list[TextContent]:
        if not self.ctx:
            raise RuntimeError("Server not initialized")

        from src.memory import tools as memory_tools

        filename = arguments.get("filename", "")

        results = await memory_tools.detect_contradictions(self.ctx, filename)

        if results and "error" in results[0]:
            return [TextContent(type="text", text=str(results[0]))]

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

        return [TextContent(type="text", text="\n".join(output_lines))]

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
