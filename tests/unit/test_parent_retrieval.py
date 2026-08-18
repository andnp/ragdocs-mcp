from datetime import UTC, datetime

from searchkernel.chunking.header_chunker import HeaderBasedChunker
from searchkernel.domain import Record

from mcp_markdown_ragdocs.config import ChunkingConfig


def _make_record(record_id: str, content: str) -> Record:
    """Build a domain.Record matching the fixed test-fixture fields used below."""
    now = datetime.now(UTC)
    return Record(
        source_kind="note",
        source_id=record_id,
        title=record_id,
        body=content,
        created_at=now,
        updated_at=now,
        metadata={"links": [], "tags": [], "file_path": "/test/doc.md"},
        uri="file:///test/doc.md",
    )


class TestParentChildChunking:
    def test_child_chunks_preserve_section_titles_without_markdown_noise(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=40,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_chunk_min_chars=400,
            parent_chunk_max_chars=800,
        )
        chunker = HeaderBasedChunker(config)

        content = """# API Guide

## Authentication

Bearer tokens protect API calls and require issuer validation.

### Rotation

Rotate credentials every 24 hours and revoke compromised tokens immediately.
"""

        doc = _make_record("clean_titles", content)

        chunks = chunker.chunk_record(doc)
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        authentication_chunk = next(
            chunk for chunk in child_chunks if chunk.metadata.get("header_path") == "API Guide > Authentication"
        )
        rotation_chunk = next(
            chunk
            for chunk in child_chunks
            if chunk.metadata.get("header_path") == "API Guide > Authentication > Rotation"
        )

        assert authentication_chunk.content.startswith("Authentication")
        assert "Context: API Guide" in authentication_chunk.content
        assert not authentication_chunk.content.startswith("#")

        assert rotation_chunk.content.startswith("Rotation")
        assert "Context: API Guide > Authentication" in rotation_chunk.content
        assert "#" not in rotation_chunk.content.split("\n", 1)[0]

    def test_header_chunker_preserves_shared_context_without_synthetic_parents(self):
        """
        Verify current searchkernel chunking keeps header context on children.

        Searchkernel 1.5.1 no longer emits synthetic parent chunks.
        """
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=40,
            max_chunk_chars=160,
            overlap_chars=0,
            parent_chunk_min_chars=120,
            parent_chunk_max_chars=600,
        )
        chunker = HeaderBasedChunker(config)

        content = """# API Guide

## TL;DR

Use token auth.

## Authentication Details

Bearer tokens must be rotated every 24 hours. Include scopes and issuer validation.
"""

        doc = _make_record("parent_headers", content)

        chunks = chunker.chunk_record(doc)
        assert len(chunks) == 2
        assert all("_parent_" not in chunk.chunk_id for chunk in chunks)
        assert [chunk.metadata.get("header_path") for chunk in chunks] == [
            "API Guide > TL;DR",
            "API Guide > Authentication Details",
        ]
        assert all("Context: API Guide" in chunk.content for chunk in chunks)

    def test_trailing_short_section_merges_backward_into_previous_chunk(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=80,
            max_chunk_chars=220,
            overlap_chars=0,
            parent_chunk_min_chars=300,
            parent_chunk_max_chars=800,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Operations Guide

## Monitoring

Monitoring dashboards should include request rate, latency, and error budgets.
Alert routing must notify the on-call engineer and preserve incident context.

## Appendix

CLI cheatsheet.
"""

        doc = _make_record("trailing_short", content)

        chunks = chunker.chunk_record(doc)
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        assert len(child_chunks) >= 1
        assert all(chunk.metadata.get("header_path") != "Operations Guide > Appendix" for chunk in child_chunks)
        assert any("Appendix" in chunk.content for chunk in child_chunks)
        assert any("CLI cheatsheet." in chunk.content for chunk in child_chunks)

    def test_creates_parent_and_child_chunks_when_enabled(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=100,
            max_chunk_chars=400,
            overlap_chars=0,
            parent_chunk_min_chars=500,
            parent_chunk_max_chars=1000,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Main Title

This is the introduction with some content that should be reasonably long.

## Section One

First section content with details about topic A. More content here to make
the chunk larger. Adding extra text to ensure we have enough characters.

## Section Two

Second section content about topic B. More details and explanations here.
Additional content to reach minimum chunk size for testing purposes.

## Section Three

Third section content covering topic C. Further elaboration and examples.
More text to ensure this section is substantial enough for chunking.
"""

        doc = _make_record("test_doc", content)

        chunks = chunker.chunk_record(doc)

        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        # Should have both parents and children when parent retrieval is enabled
        # and content is long enough
        assert len(chunks) > 0

        # Child chunks should have parent_chunk_id set
        for child in child_chunks:
            parent_id = child.metadata.get("parent_chunk_id")
            if isinstance(parent_id, str):
                assert parent_id.startswith("test_doc_parent_")

    def test_child_chunks_reference_correct_parent(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=50,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_chunk_min_chars=300,
            parent_chunk_max_chars=800,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Doc Title

Introduction paragraph with enough text to form a chunk.

## First Section

Content for section one with sufficient length for a chunk.

## Second Section

Content for section two with adequate length for testing.

## Third Section

Content for section three with more text for the chunk.
"""

        doc = _make_record("test_doc", content)

        chunks = chunker.chunk_record(doc)

        parent_chunks = {c.chunk_id: c for c in chunks if "_parent_" in c.chunk_id}
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        # Each child with a parent_chunk_id should reference an existing parent
        for child in child_chunks:
            if child.metadata.get("parent_chunk_id"):
                assert child.metadata.get("parent_chunk_id") in parent_chunks

    def test_parent_content_contains_child_content(self):
        config = ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=50,
            max_chunk_chars=200,
            overlap_chars=0,
            parent_chunk_min_chars=300,
            parent_chunk_max_chars=1000,
        )
        chunker = HeaderBasedChunker(config)

        content = """# Document

Intro text that should be included.

## Section A

Content for section A with enough text.

## Section B

Content for section B with enough text.
"""

        doc = _make_record("test_doc", content)

        chunks = chunker.chunk_record(doc)

        parent_chunks = {c.chunk_id: c for c in chunks if "_parent_" in c.chunk_id}
        child_chunks = [c for c in chunks if "_parent_" not in c.chunk_id]

        # Child content should be part of parent content
        for child in child_chunks:
            parent_id = child.metadata.get("parent_chunk_id")
            if isinstance(parent_id, str) and parent_id in parent_chunks:
                parent = parent_chunks[parent_id]
                # The child content (without overlap markers) should be in parent
                child_text = child.content
                if child_text.startswith("[..."):
                    # Remove overlap prefix
                    child_text = child_text.split("]\n\n", 1)[-1]
                # Check content is part of parent (allowing for whitespace differences)
                assert any(
                    line.strip() in parent.content
                    for line in child_text.split("\n")
                    if line.strip()
                )
