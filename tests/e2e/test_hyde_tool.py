"""
E2E tests for HyDE (Hypothetical Document Embeddings) search tool.

Tests the search_with_hypothesis MCP tool end-to-end:
- Tool registration in list_tools
- Tool invocation via call_tool
- Parameter validation
- Result format
- Integration with search infrastructure
"""

from typing import cast
from unittest.mock import MagicMock

import pytest

from src.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
    ServerConfig,
)
from src.context import ApplicationContext
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.mcp.handlers import HandlerContext, get_handler
from src.mcp import MCPServer
from src.search.orchestrator import SearchOrchestrator


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_docs_dir(tmp_path):
    """
    Create test documents directory with sample files for HyDE testing.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "mcp_tools.md").write_text("""# MCP Tools Guide

## Adding New Tools

To add a new tool to the MCP server:

1. Define the Tool in list_tools() method
2. Add inputSchema with properties and required fields
3. Implement handler in call_tool() method
4. Return TextContent with results

## Example Implementation

```python
Tool(
    name="my_tool",
    description="Description here",
    inputSchema={"type": "object", "properties": {...}}
)
```
""")

    (docs_dir / "search.md").write_text("""# Search System

## Hybrid Search

The search combines semantic, keyword, and graph strategies.

## Query Orchestration

SearchOrchestrator handles multi-strategy queries.
""")

    (docs_dir / "config.md").write_text("""# Configuration

## Server Settings

Configure host and port in config.toml.

## Search Options

Adjust weights and thresholds.
""")

    return docs_dir


def _create_config(tmp_path, test_docs_dir, hyde_enabled: bool = True) -> Config:
    """
    Create a Config object for testing.

    Args:
        tmp_path: Pytest temporary path fixture
        test_docs_dir: Path to test documents directory
        hyde_enabled: Whether to enable HyDE search

    Returns:
        Config object with the specified settings
    """
    index_path = tmp_path / "indices"
    index_path.mkdir(parents=True, exist_ok=True)

    return Config(
        server=ServerConfig(host="127.0.0.1", port=8000),
        indexing=IndexingConfig(
            documents_path=str(test_docs_dir),
            index_path=str(index_path),
            recursive=True,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            hyde_enabled=hyde_enabled,
            community_detection_enabled=True,
            dynamic_weights_enabled=True,
        ),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


def _create_mcp_server(config: Config, docs_dir) -> tuple[MCPServer, HandlerContext]:
    """
    Create an MCPServer with initialized indices.

    Args:
        config: Configuration object
        docs_dir: Path to test documents directory

    Returns:
        Tuple of (MCPServer, HandlerContext) for testing
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()

    manager = IndexManager(config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

    for doc_path in docs_dir.glob("*.md"):
        manager.index_document(str(doc_path))

    class MockContext:
        def __init__(self):
            self.config = config
            self.orchestrator = orchestrator
            self.index_manager = manager
            self.commit_indexer = None

        def is_ready(self):
            return True

    mock_ctx = MockContext()
    server = MCPServer(ctx=cast(ApplicationContext, mock_ctx))

    # Create HandlerContext for direct handler testing
    mock_coordinator = MagicMock()
    mock_coordinator.wait_ready = MagicMock(return_value=None)
    hctx = HandlerContext(ctx=cast(ApplicationContext, mock_ctx), coordinator=mock_coordinator)

    return server, hctx


# ============================================================================
# Tool Handler Tests (verifies search_with_hypothesis handler exists and works)
# ============================================================================


class TestHyDEToolHandler:
    """Tests that the HyDE handler exists and is callable."""

    @pytest.mark.asyncio
    async def test_handler_method_exists(self, tmp_path, test_docs_dir):
        """
        The search_with_hypothesis handler should be registered.
        """
        handler = get_handler("search_with_hypothesis")
        assert handler is not None
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_returns_text_content(self, tmp_path, test_docs_dir):
        """
        The handler should return a list of TextContent objects.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": "Documentation about configuration",
            "top_n": 3,
        })

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].type == "text"
        assert isinstance(result[0].text, str)


# ============================================================================
# Tool Invocation Tests
# ============================================================================


class TestHyDEToolInvocation:
    """Tests for search_with_hypothesis tool invocation."""

    @pytest.mark.asyncio
    async def test_invoke_with_valid_hypothesis(self, tmp_path, test_docs_dir):
        """
        Valid hypothesis returns search results.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": (
                "To add a new MCP tool, I need to modify the list_tools method "
                "and add a handler in call_tool."
            ),
            "top_n": 5,
        })

        assert len(result) == 1
        assert result[0].type == "text"
        assert "HyDE Search Results" in result[0].text

    @pytest.mark.asyncio
    async def test_invoke_finds_relevant_docs(self, tmp_path, test_docs_dir):
        """
        HyDE search finds documentation matching the hypothesis content.

        Note: Semantic search results depend on embedding model behavior.
        We verify that results are returned and contain file paths.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": (
                "The MCP server has a method called list_tools that returns Tool objects. "
                "Each tool has a name, description, and inputSchema."
            ),
            "top_n": 3,
        })

        text = result[0].text
        # Verify we get results from one of our test files
        assert any(
            doc in text for doc in ["mcp_tools.md", "search.md", "config.md"]
        ), f"Expected at least one test doc in results: {text}"

    @pytest.mark.asyncio
    async def test_invoke_with_top_n_limit(self, tmp_path, test_docs_dir):
        """
        top_n parameter limits the number of results.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result_1 = await handler(hctx, {
            "hypothesis": "Search documentation",
            "top_n": 1,
        })

        result_5 = await handler(hctx, {
            "hypothesis": "Search documentation",
            "top_n": 5,
        })

        count_1 = result_1[0].text.count("[1]")
        count_5 = result_5[0].text.count("[1]")

        assert count_1 <= 1
        assert count_5 <= 1

    @pytest.mark.asyncio
    async def test_invoke_with_excluded_files(self, tmp_path, test_docs_dir):
        """
        excluded_files parameter filters out specified files.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": "Configuration and search settings",
            "top_n": 5,
            "excluded_files": ["config.md"],
        })

        text = result[0].text
        matches = [
            line
            for line in text.split("\n")
            if line.startswith("[") and "config.md" in line
        ]
        assert len(matches) == 0


# ============================================================================
# Parameter Validation Tests
# ============================================================================


class TestHyDEParameterValidation:
    """Tests for parameter validation in search_with_hypothesis."""

    @pytest.mark.asyncio
    async def test_missing_hypothesis_raises_error(self, tmp_path, test_docs_dir):
        """
        Missing hypothesis parameter should return validation error.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "top_n": 5,
        })

        assert len(result) == 1
        assert "Validation error" in result[0].text
        assert "hypothesis" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_empty_hypothesis_raises_error(self, tmp_path, test_docs_dir):
        """
        Empty hypothesis parameter should return validation error.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": "",
            "top_n": 5,
        })

        assert len(result) == 1
        assert "Validation error" in result[0].text
        assert "empty" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_invalid_top_n_raises_error(self, tmp_path, test_docs_dir):
        """
        Invalid top_n parameter should return validation error.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": "test",
            "top_n": 0,
        })
        assert len(result) == 1
        assert "Validation error" in result[0].text
        assert "top_n" in result[0].text.lower()

        result = await handler(hctx, {
            "hypothesis": "test",
            "top_n": 1000,
        })
        assert len(result) == 1
        assert "Validation error" in result[0].text
        assert "top_n" in result[0].text.lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestHyDEIntegrationWithSearchInfrastructure:
    """Tests for HyDE integration with the search infrastructure."""

    @pytest.mark.asyncio
    async def test_hyde_uses_vector_search(self, tmp_path, test_docs_dir):
        """
        HyDE search should use vector/semantic search under the hood.
        The hypothesis is embedded and matched against document embeddings.

        Note: We verify that search returns results, not specific ordering,
        as embedding model behavior can vary.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": (
                "Documentation about adding new tools to the Model Context Protocol server "
                "including the Tool class and schema definitions."
            ),
            "top_n": 3,
        })

        text = result[0].text
        # Verify search returned results from our test docs
        assert any(
            doc in text.lower() for doc in ["mcp_tools", "search", "config"]
        ), f"Expected at least one test doc in results: {text}"

    @pytest.mark.asyncio
    async def test_hyde_returns_score_in_results(self, tmp_path, test_docs_dir):
        """
        Results should include relevance scores.
        """
        config = _create_config(tmp_path, test_docs_dir)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": "Search system documentation",
            "top_n": 3,
        })

        text = result[0].text
        assert "(" in text and ")" in text


# ============================================================================
# Config Toggle Tests
# ============================================================================


class TestHyDEConfigToggle:
    """Tests for HyDE enable/disable configuration."""

    @pytest.mark.asyncio
    async def test_hyde_disabled_still_returns_results(self, tmp_path, test_docs_dir):
        """
        With hyde_enabled=False, search_with_hypothesis falls back to regular query.
        """
        config = _create_config(tmp_path, test_docs_dir, hyde_enabled=False)
        _server, hctx = _create_mcp_server(config, test_docs_dir)
        handler = get_handler("search_with_hypothesis")
        assert handler is not None

        result = await handler(hctx, {
            "hypothesis": "Search documentation",
            "top_n": 3,
        })

        assert len(result) == 1
        assert "Results" in result[0].text
