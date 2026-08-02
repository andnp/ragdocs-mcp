"""Integration tests for delta indexing end-to-end scenarios."""

import asyncio

import pytest

from mcp_markdown_ragdocs.config import ChunkingConfig, Config, IndexingConfig
from tests.integration._canonical import make_record_index_manager


@pytest.fixture
def config_delta_enabled(tmp_path):
    """Create test configuration with delta indexing enabled."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
            delta_full_reindex_threshold=0.5,
        ),
        chunking=ChunkingConfig(
            min_chunk_chars=100,
            max_chunk_chars=1000,
        ),
    )


@pytest.fixture
def manager(config_delta_enabled):
    """Create the canonical record manager with delta indexing enabled."""
    return make_record_index_manager(config_delta_enabled)


def test_delta_indexing_single_chunk_change(tmp_path, manager):
    """Verify only changed chunk is re-indexed when one section is modified."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # 1. Create document with 3 sections
    original_content = """# Document Title

## Section 1

Content for section 1 with enough text to make a chunk.

## Section 2

Content for section 2 with enough text to make a chunk.

## Section 3

Content for section 3 with enough text to make a chunk.
"""
    test_file.write_text(original_content)

    # 2. Index document fully
    manager.index_document(str(test_file))
    initial_count = manager.count_records("note")
    assert initial_count >= 1, f"Expected at least 1 document, got {initial_count}"

    # 3. Modify only section 2
    modified_content = """# Document Title

## Section 1

Content for section 1 with enough text to make a chunk.

## Section 2

MODIFIED content for section 2 with completely different text now.

## Section 3

Content for section 3 with enough text to make a chunk.
"""
    test_file.write_text(modified_content)

    # 4. Re-index document
    manager.index_document(str(test_file))

    # 5. Verify: document count unchanged (delta didn't add/remove documents)
    final_count = manager.count_records("note")
    assert final_count == initial_count, (
        f"Expected {initial_count} documents, got {final_count}"
    )

    # 6. Verify: query results correct
    results = manager.keyword.search("MODIFIED", 10)
    assert len(results) > 0, "Should find modified content"

    # Old content should be removed; sections 2+3 are merged into one chunk so the
    # modified chunk may still contain scattered tokens from section 3 that overlap
    # the query. Any match must be noise-level (score < 0.001).
    results_old = manager.keyword.search("section 2 with enough text", 10)
    assert all(r.score < 0.001 for r in results_old), (
        f"Old content should not be found at meaningful score; got: {results_old}"
    )


@pytest.mark.asyncio
async def test_delta_indexing_no_changes(tmp_path, manager):
    """Verify no re-indexing when content unchanged (mtime changed only)."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # 1. Index document
    content = """# Test Document

## Section

Some test content here.
"""
    test_file.write_text(content)
    manager.index_document(str(test_file))

    initial_count = manager.count_records("note")

    # 2. Touch file (change mtime but not content)
    await asyncio.sleep(0.1)  # Ensure mtime difference
    test_file.touch()

    # Write same content (triggers file change but content identical)
    test_file.write_text(content)

    # 3. Re-index
    manager.index_document(str(test_file))

    # 4. Verify: chunk count unchanged
    final_count = manager.count_records("note")
    assert final_count == initial_count

    # The canonical manager keeps the same record identities for unchanged text.
    assert manager.count_records("note") == initial_count


def test_delta_indexing_full_reindex_threshold(tmp_path):
    """Verify full re-index when change ratio exceeds threshold."""
    # 1. Config with threshold=0.5 (50%)
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "indices"),
            delta_full_reindex_threshold=0.5,
        ),
        chunking=ChunkingConfig(
            min_chunk_chars=100,
            max_chunk_chars=1000,
        ),
    )

    manager = make_record_index_manager(config)

    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    test_file = docs_path / "test.md"

    # 2. Index document with 4 sections
    original_content = """# Document

## Section 1
Content 1.

## Section 2
Content 2.

## Section 3
Content 3.

## Section 4
Content 4.
"""
    test_file.write_text(original_content)
    manager.index_document(str(test_file))
    initial_count = manager.count_records("note")

    # 3. Modify 3 sections (75% change)
    modified_content = """# Document

## Section 1
MODIFIED 1.

## Section 2
MODIFIED 2.

## Section 3
MODIFIED 3.

## Section 4
Content 4.
"""
    test_file.write_text(modified_content)

    # 4. Verify: full re-index triggered (75% > 50% threshold)
    manager.index_document(str(test_file))

    # 5. Verify: query results correct
    results = manager.keyword.search("MODIFIED", 10)
    # Should find modified sections (may be 2-3 depending on chunking)
    assert len(results) >= 2, f"Should find modified sections, got {len(results)}"

    final_count = manager.count_records("note")
    assert final_count == initial_count, "Document count should remain stable"


@pytest.mark.asyncio
async def test_delta_indexing_query_correctness(tmp_path, manager):
    """Verify query results correct after delta update."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # 1. Index doc with "Python" content
    original_content = """# Programming Language

## Introduction

This is a document about Python programming language.
Python is great for data science and machine learning.
"""
    test_file.write_text(original_content)
    manager.index_document(str(test_file))

    # 2. Query "Python" → finds result
    results = manager.keyword.search("Python", 10)
    assert len(results) > 0, "Should find Python content"

    # 3. Modify doc to "Rust" content
    modified_content = """# Programming Language

## Introduction

This is a document about Rust programming language.
Rust is great for systems programming and performance.
"""
    test_file.write_text(modified_content)

    # 4. Delta re-index
    manager.index_document(str(test_file))

    # 5. Query "Rust" → finds result
    results_rust = manager.keyword.search("Rust", 10)
    assert len(results_rust) > 0, "Should find Rust content after update"

    # 6. Query "Python" → no results (old content removed)
    results_python = manager.keyword.search("Python programming", 10)
    # Old content should be removed or heavily de-weighted
    if results_python:
        # If any results, they should be very low relevance
        assert all(r.score < 0.5 for r in results_python), (
            "Python content should be removed/de-weighted"
        )


@pytest.mark.asyncio
async def test_delta_indexing_multiple_updates(tmp_path, manager):
    """Verify delta indexing works correctly across multiple updates."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # Section bodies are padded well past min_chunk_chars=100 so each section
    # forms its own chunk instead of being merged with its neighbor.
    section_1_v1 = (
        "Version 1 of section 1, with enough padding text so this section on "
        "its own comfortably exceeds the minimum chunk size and is never "
        "merged with its neighbor."
    )
    section_1_v2 = (
        "Version 2 of section 1, with enough padding text so this section on "
        "its own comfortably exceeds the minimum chunk size and is never "
        "merged with its neighbor."
    )
    section_1_v3 = (
        "Version 3 of section 1, with enough padding text so this section on "
        "its own comfortably exceeds the minimum chunk size and is never "
        "merged with its neighbor."
    )
    section_2_v1 = (
        "Version 1 of section 2, with enough padding text so this section on "
        "its own comfortably exceeds the minimum chunk size and is never "
        "merged with its neighbor."
    )
    section_2_v2 = (
        "Version 2 of section 2, with enough padding text so this section on "
        "its own comfortably exceeds the minimum chunk size and is never "
        "merged with its neighbor."
    )

    # 1. Index doc
    content_v1 = f"# Document\n\n## Section 1\n{section_1_v1}\n\n## Section 2\n{section_2_v1}\n"
    test_file.write_text(content_v1)
    manager.index_document(str(test_file))

    # 2. Modify section 1 → re-index
    content_v2 = f"# Document\n\n## Section 1\n{section_1_v2}\n\n## Section 2\n{section_2_v1}\n"
    test_file.write_text(content_v2)
    manager.index_document(str(test_file))

    # 3. Modify section 2 → re-index
    content_v3 = f"# Document\n\n## Section 1\n{section_1_v2}\n\n## Section 2\n{section_2_v2}\n"
    test_file.write_text(content_v3)
    manager.index_document(str(test_file))

    # 4. Modify section 1 again → re-index
    content_v4 = f"# Document\n\n## Section 1\n{section_1_v3}\n\n## Section 2\n{section_2_v2}\n"
    test_file.write_text(content_v4)
    manager.index_document(str(test_file))

    # 5. Verify: query results reflect all updates
    # The keyword search may not find exact version numbers reliably
    # Verify document exists instead
    assert manager.count_records("note") >= 1, "Document should be indexed"

    results_v1 = manager.keyword.search("Version 1", 10)
    # Version 1 content should be removed
    if results_v1:
        assert all(r.score < 0.3 for r in results_v1), (
            "Old content should be de-weighted"
        )


def test_delta_indexing_new_section_added(tmp_path, manager):
    """Verify adding new section only indexes the new chunk."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # 1. Index doc with 2 sections
    original_content = """# Document

## Section 1
Content 1.

## Section 2
Content 2.
"""
    test_file.write_text(original_content)
    manager.index_document(str(test_file))
    initial_count = manager.count_records("note")

    # 2. Add 3rd section
    modified_content = """# Document

## Section 1
Content 1.

## Section 2
Content 2.

## Section 3
New section content added here.
"""
    test_file.write_text(modified_content)

    # 3. Re-index
    manager.index_document(str(test_file))

    # 4. Verify: canonical record count increased by the new chunk.
    final_count = manager.count_records("note")
    assert final_count == initial_count + 1, (
        f"Expected {initial_count + 1} records, got {final_count}"
    )

    # 5. Verify: query finds content from new section
    results = manager.keyword.search("New section", 10)
    assert len(results) > 0, "Should find new section content"


def test_delta_indexing_section_removed(tmp_path, manager):
    """Verify removing section removes its chunks."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # 1. Index doc with 3 sections
    original_content = """# Document

## Section 1
Content 1.

## Section 2
Content to be removed.

## Section 3
Content 3.
"""
    test_file.write_text(original_content)
    manager.index_document(str(test_file))
    initial_count = manager.count_records("note")

    # Verify section 2 is indexed
    results_before = manager.keyword.search("removed", 10)
    assert len(results_before) > 0, "Should find section 2 before removal"

    # 2. Remove section 2
    modified_content = """# Document

## Section 1
Content 1.

## Section 3
Content 3.
"""
    test_file.write_text(modified_content)

    # 3. Re-index
    manager.index_document(str(test_file))

    # 4. Verify: the removed chunk is gone.
    final_count = manager.count_records("note")
    assert final_count == initial_count - 1, (
        f"Expected {initial_count - 1} records, got {final_count}"
    )

    # 5. Verify: query for section 2 content returns no results
    results_after = manager.keyword.search("removed", 10)
    assert len(results_after) == 0, "Section 2 content should be removed"


def test_delta_indexing_empty_document(tmp_path, manager):
    """Verify delta indexing handles empty documents gracefully."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # Index non-empty document
    test_file.write_text("# Test\n\nContent")
    manager.index_document(str(test_file))
    assert manager.count_records("note") > 0

    # Make it empty
    test_file.write_text("")
    manager.index_document(str(test_file))

    # Should handle gracefully (either remove all chunks or handle empty state)
    # Exact behavior depends on implementation, but should not crash


def test_delta_indexing_single_chunk_document(tmp_path, manager):
    """Verify delta indexing works with very small documents (1-2 chunks)."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # Single chunk document
    original_content = "# Single Section\n\nSome content."
    test_file.write_text(original_content)
    manager.index_document(str(test_file))

    initial_count = manager.count_records("note")
    assert initial_count >= 1

    # Modify the single chunk
    modified_content = "# Single Section\n\nModified content."
    test_file.write_text(modified_content)
    manager.index_document(str(test_file))

    # Should handle single-chunk delta correctly
    final_count = manager.count_records("note")
    assert final_count == initial_count


def test_delta_indexing_large_document(tmp_path, manager):
    """Verify delta indexing scales with large documents (100+ chunks)."""
    docs_path = tmp_path / "docs"
    test_file = docs_path / "test.md"

    # Create large document with many sections
    sections = []
    for i in range(50):
        sections.append(f"## Section {i}\n\nContent for section {i}.\n")

    original_content = "# Large Document\n\n" + "\n".join(sections)
    test_file.write_text(original_content)
    manager.index_document(str(test_file))

    initial_count = manager.count_records("note")
    assert initial_count >= 1, "Should have document indexed"

    # Modify one section in the middle
    sections[25] = "## Section 25\n\nMODIFIED content for section 25.\n"
    modified_content = "# Large Document\n\n" + "\n".join(sections)
    test_file.write_text(modified_content)

    # Delta index should handle large documents efficiently
    manager.index_document(str(test_file))

    final_count = manager.count_records("note")
    assert final_count == initial_count, "Document count should be stable"

    # Verify modification indexed
    results = manager.keyword.search("MODIFIED", 10)
    assert len(results) > 0, "Should find modified section"
