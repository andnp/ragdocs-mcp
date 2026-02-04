"""Tests for IndexManager.prune_document() method.

This module tests the document pruning logic that removes documents
from all indices (vector, keyword, graph) and updates the manifest.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import Config, IndexingConfig
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, save_manifest
from src.models import Chunk


def make_chunk(chunk_id: str, doc_id: str, content: str, index: int) -> Chunk:
    """Helper to create Chunk with required fields."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        metadata={},
        chunk_index=index,
        header_path="",
        start_pos=0,
        end_pos=len(content),
        file_path=f"/docs/{doc_id}.md",
        modified_time=datetime.now(timezone.utc),
    )


@pytest.fixture
def tmp_index_path(tmp_path):
    """Provide isolated index directory."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def base_config(tmp_path, tmp_index_path):
    """Config with minimal settings."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_dir),
            index_path=str(tmp_index_path),
            enable_delta_indexing=False,
        )
    )


@pytest.fixture
def manager(base_config, shared_embedding_model):
    """IndexManager with real indices."""
    return IndexManager(
        base_config,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )


class TestPruneDocumentSuccess:
    """Tests for successful document pruning."""

    def test_prune_removes_from_vector_index(self, manager: IndexManager, shared_embedding_model):
        """Verify prune_document removes chunks from vector index."""
        # Add a document with chunks
        doc_id = "test_doc"
        chunks = [make_chunk(f"{doc_id}_chunk_0", doc_id, "Test content for vector", 0)]
        manager.vector.add_chunks(chunks)

        # Verify chunk exists
        chunk_ids = manager.vector.get_chunk_ids_for_document(doc_id)
        assert len(chunk_ids) > 0, "Chunk should exist before prune"

        # Prune
        result = manager.prune_document(doc_id)

        assert result is True
        # After prune, doc should not be tracked
        remaining = manager.vector.get_chunk_ids_for_document(doc_id)
        assert len(remaining) == 0, "Chunks should be removed after prune"

    def test_prune_removes_from_keyword_index(self, manager: IndexManager):
        """Verify prune_document removes document from keyword index."""
        doc_id = "keyword_test"
        chunks = [make_chunk(f"{doc_id}_chunk_0", doc_id, "Unique keyword searchable", 0)]
        manager.keyword.add_chunks(chunks)

        # Verify searchable
        results = manager.keyword.search("Unique keyword")
        assert len(results) > 0

        result = manager.prune_document(doc_id)

        assert result is True
        results_after = manager.keyword.search("Unique keyword")
        assert len(results_after) == 0, "Keyword search should return empty after prune"

    def test_prune_removes_from_graph_store(self, manager: IndexManager):
        """Verify prune_document removes node from graph store."""
        doc_id = "graph_test"
        manager.graph.add_node(doc_id, {"title": "Test"})

        # Verify node exists via has_node (public API)
        assert manager.graph.has_node(doc_id), "Node should exist before prune"

        result = manager.prune_document(doc_id)

        assert result is True
        assert not manager.graph.has_node(doc_id), "Node should be removed from graph"

    def test_prune_updates_manifest(self, manager: IndexManager, tmp_index_path: Path):
        """Verify prune_document removes doc_id from manifest."""
        doc_id = "manifest_test"

        # Create manifest with the doc_id
        manifest = IndexManifest(
            spec_version="1.0",
            embedding_model="test",
            parsers={},
            chunking_config={},
            indexed_files={doc_id: "hash123"},
        )
        save_manifest(tmp_index_path, manifest)

        result = manager.prune_document(doc_id)

        assert result is True

        # Reload manifest and verify removal
        from src.indexing.manifest import load_manifest

        updated = load_manifest(tmp_index_path)
        assert updated is not None
        assert updated.indexed_files is not None
        assert doc_id not in updated.indexed_files

    def test_prune_returns_true_on_success(self, manager: IndexManager):
        """Verify prune_document returns True for successful operation."""
        doc_id = "success_test"
        # Add to graph to ensure something to prune
        manager.graph.add_node(doc_id, {})

        result = manager.prune_document(doc_id)

        assert result is True

    def test_prune_logs_reason_when_provided(self, manager: IndexManager, caplog):
        """Verify prune_document logs reason in output."""
        import logging

        doc_id = "reason_test"
        reason = "file_deleted"

        with caplog.at_level(logging.INFO, logger="src.indexing.manager"):
            manager.prune_document(doc_id, reason=reason)

        # Check logs contain reason
        assert any("file_deleted" in record.message for record in caplog.records)


class TestPruneDocumentEdgeCases:
    """Tests for edge cases and error handling."""

    def test_prune_nonexistent_document_succeeds(self, manager: IndexManager):
        """Verify pruning non-existent document returns True (idempotent)."""
        result = manager.prune_document("nonexistent_doc_id")

        # Should succeed (no-op for non-existent)
        assert result is True

    def test_prune_manifest_without_doc_id(self, manager: IndexManager, tmp_index_path: Path):
        """Verify prune handles manifest that doesn't contain doc_id."""
        doc_id = "not_in_manifest"

        # Create manifest without the doc_id
        manifest = IndexManifest(
            spec_version="1.0",
            embedding_model="test",
            parsers={},
            chunking_config={},
            indexed_files={"other_doc": "hash456"},
        )
        save_manifest(tmp_index_path, manifest)

        result = manager.prune_document(doc_id)

        assert result is True

        # Verify manifest unchanged
        from src.indexing.manifest import load_manifest

        updated = load_manifest(tmp_index_path)
        assert updated is not None
        assert updated.indexed_files is not None
        assert "other_doc" in updated.indexed_files

    def test_prune_with_no_manifest(self, manager: IndexManager):
        """Verify prune succeeds when no manifest exists."""
        result = manager.prune_document("doc_without_manifest")

        assert result is True

    def test_prune_with_code_index(self, base_config, shared_embedding_model):
        """Verify prune_document calls code index removal when present."""
        mock_code = MagicMock()

        manager = IndexManager(
            base_config,
            VectorIndex(embedding_model=shared_embedding_model),
            KeywordIndex(),
            GraphStore(),
            code=mock_code,
        )

        doc_id = "code_doc"
        manager.prune_document(doc_id)

        mock_code.remove_by_doc_id.assert_called_once_with(doc_id)


class TestPruneDocumentFailure:
    """Tests for failure scenarios."""

    def test_prune_returns_false_on_vector_error(self, base_config, shared_embedding_model):
        """Verify prune_document returns False when vector.prune raises."""
        mock_vector = MagicMock(spec=VectorIndex)
        mock_vector.prune_document.side_effect = RuntimeError("Vector prune failed")

        manager = IndexManager(
            base_config,
            mock_vector,
            KeywordIndex(),
            GraphStore(),
        )

        result = manager.prune_document("failing_doc")

        assert result is False

    def test_prune_exception_logged(self, base_config, shared_embedding_model, caplog):
        """Verify exceptions during prune are logged with exc_info."""
        mock_vector = MagicMock(spec=VectorIndex)
        mock_vector.prune_document.side_effect = RuntimeError("Prune explosion")

        manager = IndexManager(
            base_config,
            mock_vector,
            KeywordIndex(),
            GraphStore(),
        )

        manager.prune_document("error_doc")

        # Check error was logged
        assert any("Failed to prune" in record.message for record in caplog.records)
        assert any(record.levelname == "ERROR" for record in caplog.records)
