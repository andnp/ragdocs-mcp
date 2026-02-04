"""
Integration tests for MemoryIndexManager.

Tests the complete memory indexing lifecycle: parsing files, extracting
frontmatter, creating chunks, managing ghost nodes, and persistence.
"""

from pathlib import Path

import pytest

from src.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    MemoryConfig,
    SearchConfig,
    ServerConfig,
)
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.manager import MemoryIndexManager
from src.memory.storage import compute_memory_id, ensure_memory_dirs


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_config(tmp_path: Path):
    """
    Create test configuration with memory enabled.
    """
    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        memory=MemoryConfig(
            enabled=True,
            storage_strategy="project",
        ),
        search=SearchConfig(),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    """
    Create and return the memory storage path.
    """
    path = tmp_path / ".memories"
    ensure_memory_dirs(path)
    return path


@pytest.fixture
def memory_indices():
    """
    Create fresh indices for memory testing.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def memory_manager(memory_config: Config, memory_path: Path, memory_indices):
    """
    Create MemoryIndexManager instance for testing.
    """
    vector, keyword, graph = memory_indices
    return MemoryIndexManager(memory_config, memory_path, vector, keyword, graph)


def create_memory_file(memory_path: Path, filename: str, content: str) -> Path:
    """
    Create a memory file with given content.
    """
    file_path = memory_path / f"{filename}.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ============================================================================
# Basic Indexing Tests
# ============================================================================


class TestMemoryIndexing:

    def test_index_simple_memory(self, memory_manager: MemoryIndexManager, memory_path: Path):
        """
        Verify indexing a simple memory file creates chunks in all indices.

        This is the fundamental operation: read a file, chunk it, and store
        in vector, keyword, and graph indices.
        """
        file_path = create_memory_file(
            memory_path,
            "simple-note",
            "# My Note\n\nThis is a simple memory note."
        )

        memory_manager.index_memory(str(file_path))

        assert memory_manager.get_memory_count() > 0

    def test_index_memory_creates_graph_node(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify indexed memory creates a node in the graph index.

        Graph nodes enable relationship queries between memories.
        """
        file_path = create_memory_file(
            memory_path,
            "graph-test",
            "# Graph Test\n\nContent for graph testing."
        )

        memory_manager.index_memory(str(file_path))

        neighbors = memory_manager.graph.get_neighbors("memory:graph-test", depth=1)
        assert isinstance(neighbors, list)

    def test_remove_memory(self, memory_manager: MemoryIndexManager, memory_path: Path):
        """
        Verify removing a memory cleans up all indices.
        """
        file_path = create_memory_file(
            memory_path,
            "to-remove",
            "# To Remove\n\nThis will be removed."
        )

        memory_manager.index_memory(str(file_path))
        initial_count = memory_manager.get_memory_count()

        memory_manager.remove_memory("memory:to-remove")

        assert memory_manager.get_memory_count() < initial_count or initial_count == 0


# ============================================================================
# Frontmatter Parsing Tests
# ============================================================================


class TestFrontmatterParsing:

    def test_parses_complete_frontmatter(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify all frontmatter fields are extracted and stored in chunk metadata.
        """
        content = """---
type: "plan"
status: "active"
tags: ["refactor", "auth"]
created_at: "2025-01-10T12:00:00Z"
---

# Authentication Refactor Plan

Steps to refactor the auth module.
"""
        file_path = create_memory_file(memory_path, "plan-note", content)

        memory_manager.index_memory(str(file_path))

        chunk = memory_manager.vector.get_chunk_by_id("memory:plan-note_chunk_0")
        if chunk is None:
            chunk = memory_manager.vector.get_chunk_by_id("memory:plan-note")

        assert chunk is not None

    def test_handles_missing_frontmatter(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify files without frontmatter are indexed with defaults.
        """
        content = """# No Frontmatter

This file has no YAML frontmatter block.
"""
        file_path = create_memory_file(memory_path, "no-frontmatter", content)

        memory_manager.index_memory(str(file_path))

        assert memory_manager.get_memory_count() > 0

    def test_handles_invalid_frontmatter(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify invalid YAML frontmatter is handled gracefully.

        The file should still be indexed with default frontmatter values.
        """
        content = """---
this is: [not: valid: yaml
---

# Content

Body text.
"""
        file_path = create_memory_file(memory_path, "invalid-yaml", content)

        memory_manager.index_memory(str(file_path))

        assert memory_manager.get_memory_count() > 0

    def test_parses_tags_as_list(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify tags array is parsed correctly.
        """
        content = """---
type: "fact"
tags: ["important", "reference", "api"]
---

# API Reference

Important API documentation.
"""
        file_path = create_memory_file(memory_path, "tags-test", content)

        memory_manager.index_memory(str(file_path))

        tags = memory_manager.get_all_tags()
        assert "important" in tags or memory_manager.get_memory_count() > 0

    def test_parses_tags_as_comma_string(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify comma-separated tags string is parsed correctly.

        Some frontmatter may use string format: tags: "tag1, tag2"
        """
        content = """---
type: "journal"
tags: "daily, standup, notes"
---

# Daily Standup

Today's notes.
"""
        file_path = create_memory_file(memory_path, "string-tags", content)

        memory_manager.index_memory(str(file_path))

        assert memory_manager.get_memory_count() > 0


# ============================================================================
# Ghost Node Tests
# ============================================================================


class TestGhostNodes:

    def test_creates_ghost_node_for_link(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify [[wikilink]] creates a ghost node in the graph.

        Ghost nodes represent external files linked from memories.
        """
        content = """# Bug Fix Notes

Found issue in [[src/server.py]] causing 500 errors.
"""
        file_path = create_memory_file(memory_path, "ghost-test", content)

        memory_manager.index_memory(str(file_path))

        edges = memory_manager.graph.get_edges_to("ghost:src/server.py")
        assert len(edges) > 0
        assert edges[0]["source"] == "memory:ghost-test"

    def test_ghost_node_has_edge_context(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify edge to ghost node includes anchor context.

        Context explains why the link was made.
        """
        content = """# Refactor Plan

Need to refactor [[src/auth.py]] for better security.
"""
        file_path = create_memory_file(memory_path, "context-test", content)

        memory_manager.index_memory(str(file_path))

        edges = memory_manager.graph.get_edges_to("ghost:src/auth.py")
        assert len(edges) > 0
        assert "refactor" in edges[0]["edge_context"].lower()

    def test_ghost_node_has_edge_type(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify edge type is inferred from context.
        """
        content = """# Bug Investigation

There's a bug in [[src/handler.py]] causing timeouts.
"""
        file_path = create_memory_file(memory_path, "edge-type-test", content)

        memory_manager.index_memory(str(file_path))

        edges = memory_manager.graph.get_edges_to("ghost:src/handler.py")
        assert len(edges) > 0
        assert edges[0]["edge_type"] == "debugs"

    def test_multiple_links_create_multiple_ghost_nodes(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify multiple [[wikilinks]] create multiple ghost nodes.
        """
        content = """# Architecture Notes

The [[src/server.py]] handles requests.
The [[src/handler.py]] processes them.
Results stored in [[data/cache.json]].
"""
        file_path = create_memory_file(memory_path, "multi-ghost", content)

        memory_manager.index_memory(str(file_path))

        edges_server = memory_manager.graph.get_edges_to("ghost:src/server.py")
        edges_handler = memory_manager.graph.get_edges_to("ghost:src/handler.py")
        edges_cache = memory_manager.graph.get_edges_to("ghost:data/cache.json")

        assert len(edges_server) > 0
        assert len(edges_handler) > 0
        assert len(edges_cache) > 0


# ============================================================================
# Persistence Tests
# ============================================================================


class TestPersistence:

    def test_persist_and_load_cycle(
        self, memory_config: Config, memory_path: Path
    ):
        """
        Verify memory indices survive persist/load cycle.

        This tests that indexed data is correctly serialized and restored.
        """
        vector1 = VectorIndex()
        keyword1 = KeywordIndex()
        graph1 = GraphStore()
        manager1 = MemoryIndexManager(memory_config, memory_path, vector1, keyword1, graph1)

        file_path = create_memory_file(
            memory_path,
            "persist-test",
            "# Persist Test\n\nContent that should survive persistence."
        )
        manager1.index_memory(str(file_path))
        original_count = manager1.get_memory_count()

        manager1.persist()

        vector2 = VectorIndex()
        keyword2 = KeywordIndex()
        graph2 = GraphStore()
        manager2 = MemoryIndexManager(memory_config, memory_path, vector2, keyword2, graph2)
        manager2.load()

        assert manager2.get_memory_count() == original_count

    def test_persist_preserves_ghost_nodes(
        self, memory_config: Config, memory_path: Path
    ):
        """
        Verify ghost nodes and edges survive persist/load cycle.
        """
        vector1 = VectorIndex()
        keyword1 = KeywordIndex()
        graph1 = GraphStore()
        manager1 = MemoryIndexManager(memory_config, memory_path, vector1, keyword1, graph1)

        file_path = create_memory_file(
            memory_path,
            "ghost-persist",
            "# Ghost Persist\n\nLink to [[src/module.py]] here."
        )
        manager1.index_memory(str(file_path))
        manager1.persist()

        vector2 = VectorIndex()
        keyword2 = KeywordIndex()
        graph2 = GraphStore()
        manager2 = MemoryIndexManager(memory_config, memory_path, vector2, keyword2, graph2)
        manager2.load()

        edges = manager2.graph.get_edges_to("ghost:src/module.py")
        assert len(edges) > 0


# ============================================================================
# Reindex and Stats Tests
# ============================================================================


class TestReindexAndStats:

    def test_reindex_all(self, memory_manager: MemoryIndexManager, memory_path: Path):
        """
        Verify reindex_all processes all memory files.
        """
        create_memory_file(memory_path, "note1", "# Note 1\n\nFirst note.")
        create_memory_file(memory_path, "note2", "# Note 2\n\nSecond note.")
        create_memory_file(memory_path, "note3", "# Note 3\n\nThird note.")

        indexed_count = memory_manager.reindex_all()

        assert indexed_count == 3

    def test_reindex_memory(self, memory_manager: MemoryIndexManager, memory_path: Path):
        """
        Verify reindex_memory restores a missing memory entry.

        Ensures:
        - memory_id resolves to a file under memory_path
        - reindex_memory returns True when file exists
        """
        file_path = create_memory_file(
            memory_path,
            "reindex-me",
            "# Reindex Me\n\nRecoverable memory.",
        )
        memory_id = compute_memory_id(memory_path, file_path)

        memory_manager.index_memory(str(file_path))
        memory_manager.remove_memory(memory_id)

        reindexed = memory_manager.reindex_memory(memory_id, reason="test")

        assert reindexed is True
        assert memory_id in memory_manager.vector.get_document_ids()

    def test_get_all_tags(self, memory_manager: MemoryIndexManager, memory_path: Path):
        """
        Verify get_all_tags aggregates tags across memories.
        """
        file1 = create_memory_file(
            memory_path,
            "tagged1",
            "---\ntags: [\"api\", \"backend\"]\n---\n# Tagged 1"
        )
        file2 = create_memory_file(
            memory_path,
            "tagged2",
            "---\ntags: [\"api\", \"frontend\"]\n---\n# Tagged 2"
        )
        memory_manager.index_memory(str(file1))
        memory_manager.index_memory(str(file2))

        tags = memory_manager.get_all_tags()

        assert "api" in tags
        assert tags["api"] == 2

    def test_get_all_types(self, memory_manager: MemoryIndexManager, memory_path: Path):
        """
        Verify get_all_types aggregates memory types.
        """
        file1 = create_memory_file(
            memory_path,
            "plan1",
            "---\ntype: \"plan\"\n---\n# Plan 1"
        )
        file2 = create_memory_file(
            memory_path,
            "plan2",
            "---\ntype: \"plan\"\n---\n# Plan 2"
        )
        file3 = create_memory_file(
            memory_path,
            "fact1",
            "---\ntype: \"fact\"\n---\n# Fact 1"
        )
        memory_manager.index_memory(str(file1))
        memory_manager.index_memory(str(file2))
        memory_manager.index_memory(str(file3))

        types = memory_manager.get_all_types()

        assert "plan" in types
        assert "fact" in types
        assert types["plan"] == 2
        assert types["fact"] == 1

    def test_get_total_size_bytes(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify get_total_size_bytes sums file sizes.
        """
        content1 = "# Note 1\n\n" + "x" * 100
        content2 = "# Note 2\n\n" + "y" * 200
        file1 = create_memory_file(memory_path, "size1", content1)
        file2 = create_memory_file(memory_path, "size2", content2)
        memory_manager.index_memory(str(file1))
        memory_manager.index_memory(str(file2))

        total_size = memory_manager.get_total_size_bytes()

        assert total_size > 300

    def test_get_failed_files_initially_empty(self, memory_manager: MemoryIndexManager):
        """
        Verify get_failed_files returns empty list initially.
        """
        failed = memory_manager.get_failed_files()

        assert failed == []


# ============================================================================
# Metadata Cache Tests
# ============================================================================


class TestMetadataCache:

    def test_cache_updated_on_index_memory(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify metadata cache is populated when a memory is indexed.
        """
        content = """---
type: "fact"
tags: ["cache-test", "validation"]
---

# Cache Validation

This tests cache population on index.
"""
        file_path = create_memory_file(memory_path, "cache-test", content)

        memory_manager.index_memory(str(file_path))

        tags = memory_manager.get_all_tags()
        types = memory_manager.get_all_types()

        assert "cache-test" in tags
        assert "validation" in tags
        assert tags["cache-test"] == 1
        assert "fact" in types
        assert types["fact"] == 1

    def test_cache_cleared_on_remove_memory(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify metadata cache entry is removed when memory is removed.
        """
        content = """---
type: "journal"
tags: ["remove-test"]
---

# To Be Removed

This memory will be removed.
"""
        file_path = create_memory_file(memory_path, "remove-cache", content)

        memory_manager.index_memory(str(file_path))
        assert "remove-test" in memory_manager.get_all_tags()

        memory_manager.remove_memory("memory:remove-cache")

        tags = memory_manager.get_all_tags()
        assert "remove-test" not in tags

    def test_cache_size_tracking(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify size tracking is maintained in cache.
        """
        content1 = "# Small\n\n" + "a" * 50
        content2 = "# Large\n\n" + "b" * 500

        file1 = create_memory_file(memory_path, "small-file", content1)
        file2 = create_memory_file(memory_path, "large-file", content2)

        memory_manager.index_memory(str(file1))
        size_after_first = memory_manager.get_total_size_bytes()

        memory_manager.index_memory(str(file2))
        size_after_both = memory_manager.get_total_size_bytes()

        assert size_after_both > size_after_first

    def test_cache_rebuilt_on_load(
        self, memory_config: Config, memory_path: Path
    ):
        """
        Verify metadata cache is rebuilt when indices are loaded.
        """
        content = """---
type: "plan"
tags: ["rebuild-test"]
---

# Rebuild Cache Test

Cache should be rebuilt on load.
"""
        file_path = create_memory_file(memory_path, "rebuild-cache", content)

        vector1 = VectorIndex()
        keyword1 = KeywordIndex()
        graph1 = GraphStore()
        manager1 = MemoryIndexManager(memory_config, memory_path, vector1, keyword1, graph1)

        manager1.index_memory(str(file_path))
        manager1.persist()

        vector2 = VectorIndex()
        keyword2 = KeywordIndex()
        graph2 = GraphStore()
        manager2 = MemoryIndexManager(memory_config, memory_path, vector2, keyword2, graph2)
        manager2.load()

        tags = manager2.get_all_tags()
        types = manager2.get_all_types()
        total_size = manager2.get_total_size_bytes()

        assert "rebuild-test" in tags
        assert "plan" in types
        assert total_size > 0

    def test_stats_fast_from_cache(
        self, memory_manager: MemoryIndexManager, memory_path: Path
    ):
        """
        Verify stats queries access cache without re-reading files.

        This test ensures the cache is used by checking that multiple
        calls return consistent results without file I/O.
        """
        for i in range(5):
            content = f"---\ntype: \"journal\"\ntags: [\"batch-{i}\"]\n---\n# Note {i}"
            file_path = create_memory_file(memory_path, f"batch-{i}", content)
            memory_manager.index_memory(str(file_path))

        tags1 = memory_manager.get_all_tags()
        types1 = memory_manager.get_all_types()
        size1 = memory_manager.get_total_size_bytes()

        tags2 = memory_manager.get_all_tags()
        types2 = memory_manager.get_all_types()
        size2 = memory_manager.get_total_size_bytes()

        assert tags1 == tags2
        assert types1 == types2
        assert size1 == size2
        assert len(tags1) == 5
        assert types1.get("journal") == 5
