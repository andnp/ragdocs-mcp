"""
Unit tests for HyDE (Hypothetical Document Embeddings) search.

Tests cover:
- HyDE search wrapper function
- Hypothesis text is passed through to vector search
- Parameter handling (top_k, excluded_files, docs_root)
"""

from pathlib import Path

from src.search.hyde import search_with_hypothesis


class MockVectorIndex:
    """Test double for VectorIndex that records calls."""

    def __init__(self, results: list[dict] | None = None):
        self.results = results or []
        self.search_calls: list[dict] = []
        self.embedding_calls: list[str] = []

    def search(
        self,
        query: str,
        top_k: int,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[dict]:
        self.search_calls.append({
            "query": query,
            "top_k": top_k,
            "excluded_files": excluded_files,
            "docs_root": docs_root,
        })
        return self.results

    def get_text_embedding(self, text: str) -> list[float]:
        self.embedding_calls.append(text)
        return [0.1] * 384


# ============================================================================
# Basic HyDE Search Tests
# ============================================================================


class TestSearchWithHypothesis:
    """Tests for the search_with_hypothesis function."""

    def test_hypothesis_passed_to_vector_search(self):
        """
        The hypothesis text should be passed directly to vector search.
        This is the core of HyDE - embedding the hypothesis, not the query.
        """
        mock_index = MockVectorIndex()
        hypothesis = "To add a tool, modify src/mcp_server.py and register in list_tools"

        search_with_hypothesis(mock_index, hypothesis, top_k=10)

        assert len(mock_index.search_calls) == 1
        assert mock_index.search_calls[0]["query"] == hypothesis

    def test_top_k_parameter_forwarded(self):
        """
        The top_k parameter should be forwarded to vector search.
        """
        mock_index = MockVectorIndex()

        search_with_hypothesis(mock_index, "test hypothesis", top_k=20)

        assert mock_index.search_calls[0]["top_k"] == 20

    def test_default_top_k(self):
        """
        Default top_k should be 10 if not specified.
        """
        mock_index = MockVectorIndex()

        search_with_hypothesis(mock_index, "test hypothesis")

        assert mock_index.search_calls[0]["top_k"] == 10

    def test_excluded_files_forwarded(self):
        """
        Excluded files should be forwarded to vector search.
        """
        mock_index = MockVectorIndex()
        excluded = {"config.md", "readme.md"}

        search_with_hypothesis(
            mock_index,
            "test hypothesis",
            excluded_files=excluded,
        )

        assert mock_index.search_calls[0]["excluded_files"] == excluded

    def test_docs_root_forwarded(self):
        """
        docs_root path should be forwarded to vector search.
        """
        mock_index = MockVectorIndex()
        docs_root = Path("/home/user/docs")

        search_with_hypothesis(
            mock_index,
            "test hypothesis",
            docs_root=docs_root,
        )

        assert mock_index.search_calls[0]["docs_root"] == docs_root

    def test_returns_vector_search_results(self):
        """
        Should return the results from vector search unchanged.
        """
        expected_results = [
            {"chunk_id": "doc1_chunk_0", "doc_id": "doc1", "score": 0.9},
            {"chunk_id": "doc2_chunk_0", "doc_id": "doc2", "score": 0.8},
        ]
        mock_index = MockVectorIndex(results=expected_results)

        results = search_with_hypothesis(mock_index, "test hypothesis")

        assert results == expected_results


# ============================================================================
# HyDE Hypothesis Content Tests
# ============================================================================


class TestHyDEHypothesisContent:
    """Tests for various hypothesis content scenarios."""

    def test_long_hypothesis(self):
        """
        Long hypotheses should be passed through without truncation.
        """
        mock_index = MockVectorIndex()
        long_hypothesis = (
            "The documentation likely describes how to implement a new MCP tool. "
            "The process involves modifying the MCPServer class in src/mcp_server.py, "
            "adding a Tool definition with name, description, and inputSchema, "
            "then registering a handler function in the call_tool method. "
            "The handler should validate parameters and return TextContent results."
        )

        search_with_hypothesis(mock_index, long_hypothesis)

        assert mock_index.search_calls[0]["query"] == long_hypothesis

    def test_technical_hypothesis(self):
        """
        Technical hypotheses with code references should be handled.
        """
        mock_index = MockVectorIndex()
        technical = "class SearchOrchestrator has method query_with_hypothesis"

        search_with_hypothesis(mock_index, technical)

        assert mock_index.search_calls[0]["query"] == technical

    def test_multiline_hypothesis(self):
        """
        Multiline hypotheses should be preserved.
        """
        mock_index = MockVectorIndex()
        multiline = """The configuration file contains:
- server settings (host, port)
- indexing paths
- search weights"""

        search_with_hypothesis(mock_index, multiline)

        assert mock_index.search_calls[0]["query"] == multiline

    def test_empty_hypothesis_handled(self):
        """
        Empty hypothesis should still be passed to search.
        (Validation should happen at higher level)
        """
        mock_index = MockVectorIndex()

        search_with_hypothesis(mock_index, "")

        assert mock_index.search_calls[0]["query"] == ""


# ============================================================================
# HyDE Edge Cases
# ============================================================================


class TestHyDEEdgeCases:
    """Tests for edge cases in HyDE search."""

    def test_none_excluded_files(self):
        """
        None excluded_files should be forwarded as None.
        """
        mock_index = MockVectorIndex()

        search_with_hypothesis(mock_index, "test", excluded_files=None)

        assert mock_index.search_calls[0]["excluded_files"] is None

    def test_empty_excluded_files(self):
        """
        Empty set of excluded files should be forwarded.
        """
        mock_index = MockVectorIndex()

        search_with_hypothesis(mock_index, "test", excluded_files=set())

        assert mock_index.search_calls[0]["excluded_files"] == set()

    def test_none_docs_root(self):
        """
        None docs_root should be forwarded as None.
        """
        mock_index = MockVectorIndex()

        search_with_hypothesis(mock_index, "test", docs_root=None)

        assert mock_index.search_calls[0]["docs_root"] is None

    def test_empty_results_returned(self):
        """
        Empty results from vector search should be returned as-is.
        """
        mock_index = MockVectorIndex(results=[])

        results = search_with_hypothesis(mock_index, "test")

        assert results == []

    def test_special_characters_in_hypothesis(self):
        """
        Special characters in hypothesis should be preserved.
        """
        mock_index = MockVectorIndex()
        special = "How to use [[wikilinks]] and #tags in docs? @mentions"

        search_with_hypothesis(mock_index, special)

        assert mock_index.search_calls[0]["query"] == special
